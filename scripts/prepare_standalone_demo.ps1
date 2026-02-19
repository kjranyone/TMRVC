param(
    [string[]]$AudioFiles = @(),
    [string]$Checkpoint = "",
    [string]$OutputSpeaker = "models/demo_fewshot.tmrvc_speaker",
    [int]$Steps = 200,
    [string]$Device = "cpu",
    [switch]$SkipSmokeTest,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

function Resolve-RepoPath {
    param([string]$PathText)
    if ([System.IO.Path]::IsPathRooted($PathText)) {
        return (Resolve-Path $PathText).Path
    }
    return (Join-Path $repoRoot $PathText)
}

function Ensure-Cargo {
    $cargo = Join-Path $env:USERPROFILE ".cargo\\bin\\cargo.exe"
    if (-not (Test-Path $cargo)) {
        throw "cargo.exe not found at $cargo"
    }
    return $cargo
}

function Stop-StandaloneProcesses {
    $procs = Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessName -eq "tmrvc-rt" -or $_.ProcessName -eq "tmrvc_rt" }
    foreach ($proc in $procs) {
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
            Write-Host "Stopped running process: $($proc.ProcessName) (PID $($proc.Id))"
        } catch {
            Write-Warning "Failed to stop process $($proc.ProcessName) (PID $($proc.Id)): $_"
        }
    }
}
function Get-LoraNonZeroCount {
    param([string]$SpeakerPath)

    $bytes = [System.IO.File]::ReadAllBytes($SpeakerPath)
    if ($bytes.Length -lt 16) {
        throw "Invalid speaker file: too small"
    }

    $dSpeaker = 192
    $headerSize = 16
    $loraSize = [int][System.BitConverter]::ToUInt32($bytes, 12)
    $offset = $headerSize + ($dSpeaker * 4)
    $nonZero = 0

    for ($i = 0; $i -lt $loraSize; $i++) {
        $v = [System.BitConverter]::ToSingle($bytes, $offset + ($i * 4))
        if ([System.Math]::Abs($v) -gt 1e-8) {
            $nonZero += 1
        }
    }

    return $nonZero
}

Write-Host "[1/5] Validating model files..."
$requiredOnnx = @(
    "models/fp32/content_encoder.onnx",
    "models/fp32/converter.onnx",
    "models/fp32/ir_estimator.onnx",
    "models/fp32/vocoder.onnx"
)
foreach ($rel in $requiredOnnx) {
    $p = Resolve-RepoPath $rel
    if (-not (Test-Path $p)) {
        throw "Required file not found: $p"
    }
}

if (-not $AudioFiles -or $AudioFiles.Count -eq 0) {
    $defaultDir = Resolve-RepoPath "data/fewshot_test"
    $AudioFiles = @(Get-ChildItem -Path $defaultDir -Filter *.wav -File | Sort-Object Name | ForEach-Object { $_.FullName })
    if ($AudioFiles.Count -eq 0) {
        throw "No input wav found. Provide -AudioFiles or place wav files in data/fewshot_test"
    }
}

for ($i = 0; $i -lt $AudioFiles.Count; $i++) {
    $AudioFiles[$i] = Resolve-RepoPath $AudioFiles[$i]
    if (-not (Test-Path $AudioFiles[$i])) {
        throw "Audio file not found: $($AudioFiles[$i])"
    }
}

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $ckptCandidates = Get-ChildItem -Path (Resolve-RepoPath "checkpoints/distill") -Filter "distill_step*.pt" -File |
        Sort-Object LastWriteTime
    if ($ckptCandidates.Count -eq 0) {
        throw "No distill checkpoint found in checkpoints/distill"
    }
    $Checkpoint = $ckptCandidates[-1].FullName
} else {
    $Checkpoint = Resolve-RepoPath $Checkpoint
}
if (-not (Test-Path $Checkpoint)) {
    throw "Checkpoint not found: $Checkpoint"
}

$OutputSpeaker = Resolve-RepoPath $OutputSpeaker
$OutputDir = Split-Path -Parent $OutputSpeaker
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "[2/5] Running few-shot fine-tuning..."
$finetuneArgs = @(
    "run", "tmrvc-finetune",
    "--audio-files"
) + $AudioFiles + @(
    "--checkpoint", $Checkpoint,
    "--output", $OutputSpeaker,
    "--steps", $Steps.ToString(),
    "--device", $Device
)
& uv @finetuneArgs
if ($LASTEXITCODE -ne 0) {
    throw "tmrvc-finetune failed with code $LASTEXITCODE"
}

Write-Host "[3/5] Verifying generated speaker file..."
if (-not (Test-Path $OutputSpeaker)) {
    throw "Speaker file generation failed: $OutputSpeaker"
}
$nonZero = Get-LoraNonZeroCount -SpeakerPath $OutputSpeaker
Write-Host "    LoRA non-zero entries: $nonZero"

if (-not $SkipSmokeTest) {
    Write-Host "[4/5] Running runtime smoke test..."
    $cargo = Ensure-Cargo
    & $cargo test -p tmrvc-engine-rs smoke_ -- --ignored
    if ($LASTEXITCODE -ne 0) {
        throw "runtime smoke test failed"
    }
} else {
    Write-Host "[4/5] Skipped runtime smoke test"
}

if (-not $SkipBuild) {
    Write-Host "[5/5] Building standalone app (tmrvc-rt release)..."
    Stop-StandaloneProcesses
    $cargo = Ensure-Cargo
    & $cargo build -p tmrvc-rt --release
    if ($LASTEXITCODE -ne 0) {
        throw "tmrvc-rt build failed"
    }
} else {
    Write-Host "[5/5] Skipped standalone build"
}

Write-Host ""
Write-Host "Preparation complete."
Write-Host "Next command:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/run_standalone_demo.ps1 -OnnxDir models/fp32 -SpeakerPath $OutputSpeaker"


