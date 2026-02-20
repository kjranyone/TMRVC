"""Profile a single Teacher training step to identify bottlenecks.

Usage:
    uv run python scripts/profile_teacher_step.py --device xpu
    uv run python scripts/profile_teacher_step.py --device cpu
"""

from __future__ import annotations

import argparse
import time

import torch

from tmrvc_core.constants import D_SPEAKER, D_WAVLM_LARGE, N_MELS
from tmrvc_core.types import TrainingBatch
from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.models.teacher_unet import TeacherUNet


def _make_batch(batch_size: int, n_frames: int, device: str) -> TrainingBatch:
    return TrainingBatch(
        content=torch.randn(batch_size, D_WAVLM_LARGE, n_frames, device=device),
        f0=torch.randn(batch_size, 1, n_frames, device=device),
        spk_embed=torch.randn(batch_size, D_SPEAKER, device=device),
        mel_target=torch.randn(batch_size, N_MELS, n_frames, device=device),
        lengths=torch.full((batch_size,), n_frames, dtype=torch.long),
        speaker_ids=[f"spk_{i}" for i in range(batch_size)],
    )


def _sync(device: str) -> None:
    if device == "xpu":
        torch.xpu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def profile_step(device: str, batch_size: int, n_frames: int) -> None:
    print(f"Device: {device}, Batch: {batch_size}, Frames: {n_frames}")
    print("=" * 60)

    # --- Model setup ---
    t0 = time.perf_counter()
    teacher = TeacherUNet().to(device)
    _sync(device)
    print(f"Model to device:  {time.perf_counter() - t0:.3f}s")

    scheduler = FlowMatchingScheduler()
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=2e-4)
    teacher.train()

    # --- Warmup (XPU kernel compilation) ---
    print("\nWarmup (3 steps)...")
    batch = _make_batch(batch_size, n_frames, device)
    for i in range(3):
        t0 = time.perf_counter()
        B = batch.mel_target.shape[0]
        t = torch.rand(B, 1, 1, device=device)
        x_t, v_target = scheduler.forward_process(batch.mel_target, t)
        v_pred = teacher(x_t, t.squeeze(-1).squeeze(-1),
                         batch.content, batch.f0, batch.spk_embed)
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _sync(device)
        print(f"  Warmup step {i}: {time.perf_counter() - t0:.3f}s")

    # --- Detailed profiling ---
    print(f"\nDetailed profiling (5 steps)...")
    timings: dict[str, list[float]] = {
        "data_prep": [], "forward_process": [], "teacher_forward": [],
        "loss": [], "backward": [], "optimizer": [], "total": [],
    }

    for step in range(5):
        batch = _make_batch(batch_size, n_frames, device)
        _sync(device)
        total_start = time.perf_counter()

        # (1) Data prep (already on device — measures tensor creation overhead)
        t0 = time.perf_counter()
        B = batch.mel_target.shape[0]
        mel_target = batch.mel_target
        content = batch.content
        f0 = batch.f0
        spk_embed = batch.spk_embed
        _sync(device)
        timings["data_prep"].append(time.perf_counter() - t0)

        # (2) Forward process (noise sampling + interpolation)
        t0 = time.perf_counter()
        t_step = torch.rand(B, 1, 1, device=device)
        x_t, v_target = scheduler.forward_process(mel_target, t_step)
        _sync(device)
        timings["forward_process"].append(time.perf_counter() - t0)

        # (3) Teacher forward
        t0 = time.perf_counter()
        v_pred = teacher(
            x_t, t_step.squeeze(-1).squeeze(-1),
            content, f0, spk_embed,
        )
        _sync(device)
        timings["teacher_forward"].append(time.perf_counter() - t0)

        # (4) Loss
        t0 = time.perf_counter()
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        _sync(device)
        timings["loss"].append(time.perf_counter() - t0)

        # (5) Backward
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        _sync(device)
        timings["backward"].append(time.perf_counter() - t0)

        # (6) Optimizer step
        t0 = time.perf_counter()
        optimizer.step()
        _sync(device)
        timings["optimizer"].append(time.perf_counter() - t0)

        timings["total"].append(time.perf_counter() - total_start)

    # --- Report ---
    print(f"\n{'Component':<20} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 48)
    for name, vals in timings.items():
        mean = sum(vals) / len(vals)
        mn = min(vals)
        mx = max(vals)
        print(f"{name:<20} {mean:>7.3f}s {mn:>7.3f}s {mx:>7.3f}s")

    total_mean = sum(timings["total"]) / len(timings["total"])
    print(f"\n=> 平均 step 時間: {total_mean:.3f}s")
    if total_mean > 1.0:
        # Identify bottleneck
        bottleneck = max(
            [(k, sum(v)/len(v)) for k, v in timings.items() if k != "total"],
            key=lambda x: x[1],
        )
        print(f"=> 最大ボトルネック: {bottleneck[0]} ({bottleneck[1]:.3f}s)")


def profile_data_loading(cache_dir: str, dataset: str, batch_size: int) -> None:
    """Profile data loading separately."""
    print("\n" + "=" * 60)
    print("Data loading profiling")
    print("=" * 60)

    try:
        from tmrvc_data.dataset import TMRVCDataset, create_dataloader
        ds = TMRVCDataset(cache_dir, dataset)
        print(f"Dataset size: {len(ds)} samples")

        # Single item load
        times = []
        for i in range(min(20, len(ds))):
            t0 = time.perf_counter()
            _ = ds[i]
            times.append(time.perf_counter() - t0)

        mean_t = sum(times) / len(times)
        print(f"Single item load: {mean_t*1000:.1f}ms (mean of {len(times)} samples)")
        print(f"Estimated per-batch: {mean_t * batch_size * 1000:.1f}ms (bs={batch_size})")

        # Dataloader iteration
        dl = create_dataloader(ds, batch_size=batch_size, num_workers=0)
        dl_times = []
        for i, batch in enumerate(dl):
            if i >= 5:
                break
            t0 = time.perf_counter()
            _ = next(iter(dl)) if i > 0 else None
            dl_times.append(time.perf_counter() - t0)

        if dl_times:
            print(f"Dataloader batch: {sum(dl_times)/len(dl_times)*1000:.1f}ms mean")

    except Exception as e:
        print(f"Data loading profiling skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dataset", default="vctk")
    args = parser.parse_args()

    # Check device availability
    dev = args.device
    if dev == "xpu" and not torch.xpu.is_available():
        print("XPU not available, falling back to CPU")
        dev = "cpu"
    elif dev == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        dev = "cpu"

    profile_step(dev, args.batch_size, args.frames)

    if args.cache_dir:
        profile_data_loading(args.cache_dir, args.dataset, args.batch_size)
