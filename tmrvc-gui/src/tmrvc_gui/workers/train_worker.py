"""Training worker for TMRVC teacher and student models.

Wraps the ``tmrvc_train`` package in a :class:`BaseWorker` so that
training can be driven from the GUI with live progress, metrics, and
cancellation support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


class TrainWorker(BaseWorker):
    """Background worker that manages a training run.

    Parameters
    ----------
    config : dict
        Training configuration with the following keys:

        * **phase** (*str*) -- Training phase identifier, e.g.
          ``"0"``, ``"1a"``, ``"1b"``, ``"2"``.
        * **datasets** (*list[str]*) -- Dataset names to include,
          e.g. ``["vctk", "jvs"]``.
        * **batch_size** (*int*) -- Mini-batch size.
        * **lr** (*float*) -- Initial learning rate.
        * **total_steps** (*int*) -- Total optimisation steps.
        * **checkpoint_dir** (*Path*) -- Directory for saving checkpoints.
        * **cache_dir** (*Path*) -- Feature cache directory.
        * **resume_from** (*str | None*) -- Checkpoint path to resume from.
        * **mode** (*str*) -- ``"local"`` or ``"ssh"``.
        * **ssh_config** (*dict | None*) -- SSH connection parameters when
          *mode* is ``"ssh"`` (keys: ``host``, ``user``, ``key_path``,
          ``remote_dir``).
    parent : QObject, optional
        Parent Qt object.
    """

    def __init__(self, config: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.phase: str = config["phase"]
        self.datasets: list[str] = config["datasets"]
        self.batch_size: int = config["batch_size"]
        self.lr: float = config["lr"]
        self.total_steps: int = config["total_steps"]
        self.checkpoint_dir: Path = Path(config["checkpoint_dir"])
        self.cache_dir: Path = Path(config.get("cache_dir", "data/cache"))
        self.resume_from: str | None = config.get("resume_from")
        self.mode: str = config.get("mode", "local")
        self.ssh_config: dict[str, Any] | None = config.get("ssh_config")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the training loop.

        In local mode, this creates a TeacherTrainer and iterates using
        its ``train_iter()`` generator.  In SSH mode, this delegates to
        a remote machine (not yet implemented).
        """
        self.log_message.emit(
            f"[TrainWorker] Starting phase={self.phase} "
            f"datasets={self.datasets} bs={self.batch_size} "
            f"lr={self.lr} steps={self.total_steps} mode={self.mode}"
        )

        if self.mode == "ssh":
            self._run_ssh()
            return

        self._run_local()

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def _run_local(self) -> None:
        """Run training locally using TeacherTrainer."""
        try:
            import torch
            from torch.utils.data import DataLoader

            from tmrvc_data.dataset import TMRVCDataset, collate_fn
            from tmrvc_train.diffusion import FlowMatchingScheduler
            from tmrvc_train.models.teacher_unet import TeacherUNet
            from tmrvc_train.trainer import TeacherTrainer, TrainerConfig

            # Build config
            config = TrainerConfig(
                phase=self.phase,
                lr=self.lr,
                max_steps=self.total_steps,
                checkpoint_dir=str(self.checkpoint_dir),
            )

            # Build model
            self.log_message.emit("[TrainWorker] Building Teacher model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            teacher = TeacherUNet().to(device)

            param_count = sum(p.numel() for p in teacher.parameters())
            self.log_message.emit(
                f"[TrainWorker] Teacher: {param_count / 1e6:.1f}M params on {device}"
            )

            # Build dataset & dataloader
            self.log_message.emit("[TrainWorker] Loading dataset...")
            dataset = TMRVCDataset(
                cache_dir=self.cache_dir,
                datasets=self.datasets,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=2,
                pin_memory=(device == "cuda"),
            )

            # Optimizer & scheduler
            optimizer = torch.optim.AdamW(
                teacher.parameters(), lr=self.lr, weight_decay=0.01,
            )
            scheduler = FlowMatchingScheduler()

            # Build trainer
            trainer = TeacherTrainer(
                teacher=teacher,
                scheduler=scheduler,
                optimizer=optimizer,
                dataloader=dataloader,
                config=config,
            )

            # Resume if requested
            if self.resume_from:
                self.log_message.emit(
                    f"[TrainWorker] Resuming from {self.resume_from}"
                )
                trainer.load_checkpoint(self.resume_from)

            # Training loop using train_iter() generator
            self.log_message.emit("[TrainWorker] Starting training loop...")
            for step, losses in trainer.train_iter():
                if self.is_cancelled:
                    self.log_message.emit("[TrainWorker] Cancelled by user.")
                    trainer.save_checkpoint()
                    self.log_message.emit("[TrainWorker] Checkpoint saved on cancel.")
                    self.finished.emit(False, "Cancelled")
                    return

                # Emit progress
                self.progress.emit(step, self.total_steps)

                # Emit metrics
                for loss_name, loss_val in losses.items():
                    self.metric.emit(loss_name, loss_val, step)

                # Log periodically
                if step % config.log_every == 0:
                    loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
                    self.log_message.emit(
                        f"[TrainWorker] step {step}/{self.total_steps} {loss_str}"
                    )

            # Final checkpoint
            trainer.save_checkpoint()
            self.log_message.emit("[TrainWorker] Training complete.")
            self.finished.emit(True, "Training completed successfully")

        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit(False, str(exc))

    # ------------------------------------------------------------------
    # SSH training (not yet implemented)
    # ------------------------------------------------------------------

    def _run_ssh(self) -> None:
        """Run training on a remote machine via SSH (placeholder).

        TODO: Implement SSH-based training launch.
              - Connect to remote host using self.ssh_config
              - Upload config / sync dataset references
              - Launch remote training process (e.g. via tmux/screen)
              - Poll for log file updates and relay metrics back
              - Support cancellation by sending SIGINT to remote process
        """
        if self.ssh_config is None:
            self.error.emit("SSH config is required for ssh mode")
            self.finished.emit(False, "Missing SSH config")
            return

        host = self.ssh_config.get("host", "unknown")
        self.log_message.emit(
            f"[TrainWorker] SSH mode not yet implemented. "
            f"Would connect to {host}."
        )
        self.finished.emit(False, "SSH training not implemented")
