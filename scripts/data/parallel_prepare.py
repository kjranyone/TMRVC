#!/usr/bin/env python3
"""Run prepare_dataset.py in parallel.

Splits audio files into N chunks and processes them in parallel.

Usage:
    uv run python scripts/parallel_prepare.py \
        --input data/moe_multispeaker_voices \
        --output data/cache \
        --name moe_multispeaker \
        --speaker-map data/moe_multispeaker_voices/_speaker_map.json \
        --n-jobs 4 \
        --device cuda
"""

import argparse
import subprocess
import sys
from pathlib import Path
import tempfile
import os

def main():
    parser = argparse.ArgumentParser(description="Parallel prepare_dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/cache"))
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--speaker-map", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="ja")
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=30.0)
    args = parser.parse_args()
    
    # Collect audio files
    audio_files = sorted(args.input.glob("*.wav"))
    n_files = len(audio_files)
    print(f"Found {n_files} audio files")
    
    if n_files == 0:
        print("No files found!")
        return 1
    
    # Split into chunks
    chunk_size = (n_files + args.n_jobs - 1) // args.n_jobs
    chunks = []
    for i in range(args.n_jobs):
        start = i * chunk_size
        end = min(start + chunk_size, n_files)
        if start < n_files:
            chunks.append([f.name for f in audio_files[start:end]])
    
    print(f"Split into {len(chunks)} chunks of ~{chunk_size} files each")
    
    # Create temp files for file lists
    temp_files = []
    for i, chunk in enumerate(chunks):
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write('\n'.join(chunk))
        temp_file.close()
        temp_files.append(temp_file.name)
        print(f"  Chunk {i}: {len(chunk)} files -> {temp_file.name}")
    
    # Launch parallel processes
    processes = []
    for i, temp_file in enumerate(temp_files):
        cmd = [
            sys.executable, "scripts/prepare_dataset.py",
            "--input", str(args.input),
            "--output", str(args.output),
            "--name", args.name,
            "--language", args.language,
            "--device", args.device,
            "--min-duration", str(args.min_duration),
            "--max-duration", str(args.max_duration),
            "--resume",
            "--file-list", temp_file,
        ]
        if args.speaker_map:
            cmd.extend(["--speaker-map", str(args.speaker_map)])
        
        print(f"Starting job {i}: {' '.join(cmd[:10])}...")
        
        log_file = open(f"parallel_job_{i}.log", "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((proc, log_file, temp_file))
    
    # Wait for all
    print(f"\nWaiting for {len(processes)} jobs...")
    failed = 0
    for i, (proc, log_file, temp_file) in enumerate(processes):
        ret = proc.wait()
        log_file.close()
        if ret != 0:
            print(f"Job {i} FAILED with code {ret}. See parallel_job_{i}.log")
            failed += 1
        else:
            print(f"Job {i} completed successfully")
        
        # Cleanup temp file
        os.unlink(temp_file)
    
    print(f"\nAll jobs finished. {len(processes) - failed}/{len(processes)} succeeded")
    return 1 if failed > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
