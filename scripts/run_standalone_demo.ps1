param(
    [string]$OnnxDir = "models/fp32",
    [string]$SpeakerPath = "models/demo_fewshot.tmrvc_speaker",
    [switch]$Build
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

$onnxDirAbs = Resolve-RepoPath $OnnxDir
$speakerAbs = Resolve-RepoPath $SpeakerPath

if (-not (Test-Path $onnxDirAbs)) {
    throw "ONNX directory not found: $onnxDirAbs"
}
if (-not (Test-Path $speakerAbs)) {
    throw "Speaker file not found: $speakerAbs"
}

$cargo = Join-Path $env:USERPROFILE ".cargo\\bin\\cargo.exe"
if (-not (Test-Path $cargo)) {
    throw "cargo.exe not found at $cargo"
}

if ($Build) {
    Stop-StandaloneProcesses
    & $cargo build -p tmrvc-rt --release
    if ($LASTEXITCODE -ne 0) {
        throw "tmrvc-rt build failed"
    }
}

$exeCandidates = @(
    (Join-Path $repoRoot "target/release/tmrvc-rt.exe"),
    (Join-Path $repoRoot "target/release/tmrvc_rt.exe"),
    (Join-Path $repoRoot "tmrvc-rt/target/release/tmrvc-rt.exe"),
    (Join-Path $repoRoot "tmrvc-rt/target/release/tmrvc_rt.exe")
)
$exePath = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exePath) {
    throw "tmrvc-rt executable not found. Build first: cargo build -p tmrvc-rt --release"
}

$env:TMRVC_ONNX_DIR = $onnxDirAbs
$env:TMRVC_SPEAKER_PATH = $speakerAbs

Write-Host "Launching standalone demo..."
Write-Host "  TMRVC_ONNX_DIR=$env:TMRVC_ONNX_DIR"
Write-Host "  TMRVC_SPEAKER_PATH=$env:TMRVC_SPEAKER_PATH"

& $exePath
