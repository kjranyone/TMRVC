"""Benchmark teacher training throughput across configurations.

Usage:
    uv run python scripts/benchmark_training.py
"""

from __future__ import annotations

import gc
import time
from functools import partial

import torch

from tmrvc_core.constants import D_CONTENT_VEC, D_SPEAKER, N_MELS
from tmrvc_core.types import FeatureSet, TrainingBatch
from tmrvc_data.dataset import (
    DEFAULT_BUCKET_BOUNDARIES,
    collate_fn,
    create_dataloader,
)
from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.losses import FlowMatchingLoss
from tmrvc_train.models.teacher_unet import TeacherUNet


def _sync(device: str) -> None:
    if device == "xpu":
        torch.xpu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _run_steps(
    teacher: TeacherUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: FlowMatchingScheduler,
    flow_loss: FlowMatchingLoss,
    dl_iter,
    device: str,
    n_warmup: int,
    n_steps: int,
    use_mask: bool,
    label: str,
) -> dict:
    """Run warmup + measured steps, return timing stats."""
    teacher.train()

    # Warmup
    for i in range(n_warmup):
        batch = next(dl_iter)
        B, T = batch.mel_target.shape[0], batch.mel_target.shape[-1]
        mel = batch.mel_target.to(device)
        content = batch.content.to(device)
        f0 = batch.f0.to(device)
        spk = batch.spk_embed.to(device)
        t = torch.rand(B, 1, 1, device=device)
        x_t, v_target = scheduler.forward_process(mel, t)
        v_pred = teacher(x_t, t.squeeze(-1).squeeze(-1), content, f0, spk)

        if use_mask:
            lengths = batch.lengths.to(device)
            mask = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1))
            mask = mask.unsqueeze(1).float()
            loss = flow_loss(v_pred, v_target, mask=mask)
        else:
            loss = flow_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        optimizer.step()
        _sync(device)

    # Measured steps
    step_times = []
    shapes = []
    data_times = []
    compute_times = []

    for i in range(n_steps):
        t0 = time.perf_counter()
        batch = next(dl_iter)
        t1 = time.perf_counter()

        B, T = batch.mel_target.shape[0], batch.mel_target.shape[-1]
        shapes.append(T)
        mel = batch.mel_target.to(device)
        content = batch.content.to(device)
        f0 = batch.f0.to(device)
        spk = batch.spk_embed.to(device)

        t_step = torch.rand(B, 1, 1, device=device)
        x_t, v_target = scheduler.forward_process(mel, t_step)
        v_pred = teacher(x_t, t_step.squeeze(-1).squeeze(-1), content, f0, spk)

        if use_mask:
            lengths = batch.lengths.to(device)
            mask = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1))
            mask = mask.unsqueeze(1).float()
            loss = flow_loss(v_pred, v_target, mask=mask)
        else:
            loss = flow_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        optimizer.step()
        _sync(device)
        t2 = time.perf_counter()

        data_times.append(t1 - t0)
        compute_times.append(t2 - t1)
        step_times.append(t2 - t0)

    mean_total = sum(step_times) / len(step_times)
    mean_data = sum(data_times) / len(data_times)
    mean_compute = sum(compute_times) / len(compute_times)
    unique_shapes = sorted(set(shapes))

    return {
        "label": label,
        "mean_total_ms": mean_total * 1000,
        "mean_data_ms": mean_data * 1000,
        "mean_compute_ms": mean_compute * 1000,
        "min_ms": min(step_times) * 1000,
        "max_ms": max(step_times) * 1000,
        "throughput": 1.0 / mean_total,
        "shapes": unique_shapes,
        "n_shapes": len(unique_shapes),
    }


def main() -> None:
    device = "xpu" if torch.xpu.is_available() else "cpu"
    B = 64
    N_WARMUP = 5
    N_STEPS = 15

    print("=" * 72)
    print(f"TMRVC Teacher Training Benchmark  (device={device}, batch={B})")
    print(f"Dataset: VCTK, warmup={N_WARMUP}, measured={N_STEPS} steps")
    print("=" * 72)

    configs = []

    # --- Config 1: Fine-grained buckets (many shapes — simulates old behavior) ---
    # Every 10 frames = ~150 distinct kernel shapes → frequent XPU recompilation
    configs.append({
        "label": "Variable T (fine-10)",
        "max_frames": 0,
        "bucket_boundaries": list(range(10, 1500, 10)),
        "use_mask": False,
    })

    # --- Config 2: Bucket batching (default boundaries) ---
    configs.append({
        "label": "Bucket [250,500,750,1000]",
        "max_frames": 0,
        "bucket_boundaries": DEFAULT_BUCKET_BOUNDARIES,
        "use_mask": True,
    })

    # --- Config 3: Bucket + loss mask + tighter buckets ---
    configs.append({
        "label": "Bucket [500,750,1000]",
        "max_frames": 0,
        "bucket_boundaries": [500, 750, 1000],
        "use_mask": True,
    })

    # --- Config 4: Fixed max_frames=750 ---
    configs.append({
        "label": "Fixed max_frames=750",
        "max_frames": 750,
        "bucket_boundaries": None,
        "use_mask": True,
    })

    results = []

    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")

        # Fresh model each time to avoid cache effects
        teacher = TeacherUNet(d_content=D_CONTENT_VEC).to(device)
        scheduler = FlowMatchingScheduler()
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=2e-4)
        flow_loss = FlowMatchingLoss()

        collate = partial(
            collate_fn,
            max_frames=cfg["max_frames"],
            bucket_boundaries=cfg["bucket_boundaries"],
        )

        from tmrvc_data.cache import FeatureCache
        from tmrvc_data.dataset import TMRVCDataset
        from torch.utils.data import DataLoader

        cache = FeatureCache("data/cache")
        ds = TMRVCDataset(cache=cache, dataset="vctk", split="train", cross_speaker_prob=0.5)
        dl = DataLoader(
            ds, batch_size=B, shuffle=True, num_workers=0,
            collate_fn=collate, pin_memory=True, drop_last=True,
        )
        dl_iter = iter(dl)

        result = _run_steps(
            teacher, optimizer, scheduler, flow_loss,
            dl_iter, device, N_WARMUP, N_STEPS,
            use_mask=cfg["use_mask"],
            label=cfg["label"],
        )
        results.append(result)
        print(f"  Shapes: {result['shapes']} ({result['n_shapes']} unique)")
        print(f"  Data:    {result['mean_data_ms']:.0f} ms")
        print(f"  Compute: {result['mean_compute_ms']:.0f} ms")
        print(f"  Total:   {result['mean_total_ms']:.0f} ms  "
              f"(min={result['min_ms']:.0f}, max={result['max_ms']:.0f})")
        print(f"  Throughput: {result['throughput']:.2f} steps/s")

        # Cleanup
        del teacher, optimizer, dl, ds, dl_iter
        gc.collect()
        if device == "xpu":
            torch.xpu.empty_cache()

    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Config':<30} {'Shapes':>7} {'Data':>7} {'Compute':>8} "
          f"{'Total':>8} {'steps/s':>8}")
    print("-" * 72)
    for r in results:
        print(f"{r['label']:<30} {r['n_shapes']:>7} {r['mean_data_ms']:>6.0f}ms "
              f"{r['mean_compute_ms']:>7.0f}ms {r['mean_total_ms']:>7.0f}ms "
              f"{r['throughput']:>7.2f}")

    # Speedup
    if len(results) >= 2:
        baseline = results[0]["mean_total_ms"]
        print()
        for r in results[1:]:
            speedup = baseline / r["mean_total_ms"]
            print(f"  {r['label']}: {speedup:.1f}x faster than {results[0]['label']}")


if __name__ == "__main__":
    main()
