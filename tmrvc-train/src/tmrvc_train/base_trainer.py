"""BaseTrainer: abstract base class for all training loops."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BaseTrainerConfig:
    """Common configuration fields shared by all trainers."""

    lr: float = 1e-4
    max_steps: int = 100_000
    save_every: int = 10_000
    log_every: int = 100
    checkpoint_dir: str = "checkpoints"
    grad_clip: float = 1.0
    use_wandb: bool = False


class BaseTrainer(ABC):
    """Abstract base class for training loops.

    Provides the standard train_iter/train_epoch loop, checkpoint dir setup,
    optional WandB integration, and a backward_step helper.

    Subclasses must implement :meth:`train_step`, :meth:`save_checkpoint`,
    and :meth:`load_checkpoint`.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        dataloader: Any,
        config: BaseTrainerConfig,
        lr_scheduler: Any = None,
    ) -> None:
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.config = config
        self.lr_scheduler = lr_scheduler
        self.global_step = 0

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._wandb = None
        if config.use_wandb:
            self._init_wandb()

    # ------------------------------------------------------------------
    # WandB hooks (override for custom project/config)
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        """Initialize WandB logging. Override for custom project/config."""
        try:
            import wandb

            project = f"tmrvc-{self._wandb_project}" if self._wandb_project else "tmrvc"
            wandb.init(project=project, config=vars(self.config))
            self._wandb = wandb
            logger.info("WandB logging enabled (project=%s)", project)
        except ImportError:
            logger.warning("wandb not installed, skipping WandB logging")

    @property
    def _wandb_project(self) -> str:
        """WandB project suffix (e.g. ``"teacher"`` → project ``tmrvc-teacher``)."""
        return ""

    @property
    def _wandb_prefix(self) -> str:
        """WandB metric key prefix (e.g. ``"tts"`` → ``tts/loss``)."""
        return ""

    # ------------------------------------------------------------------
    # Training loop configuration
    # ------------------------------------------------------------------

    @property
    def _increment_step_in_train_step(self) -> bool:
        """If ``True``, :meth:`train_step` increments ``global_step`` itself."""
        return False

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _backward_step(
        self,
        loss: torch.Tensor,
        params: Any,
    ) -> None:
        """Common backward pass: zero_grad → backward → clip → step → lr_step."""
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(params, self.config.grad_clip)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _format_log(self, losses: dict[str, float]) -> str:
        """Format losses for logging. Override for custom format."""
        return ", ".join(f"{k}={v:.4f}" for k, v in losses.items())

    def _log_step(self, losses: dict[str, float]) -> None:
        """Log and optionally WandB-log a training step."""
        logger.info("Step %d: %s", self.global_step, self._format_log(losses))
        if self._wandb is not None:
            prefix = self._wandb_prefix
            if prefix:
                log_dict = {f"{prefix}/{k}": v for k, v in losses.items()}
            else:
                log_dict = losses
            self._wandb.log(log_dict, step=self.global_step)

    def _pre_train_setup(self) -> None:
        """Called once at the start of :meth:`train_iter`."""

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_iter(self) -> Iterator[tuple[int, dict[str, float]]]:
        """Iterate over training steps, yielding ``(step, losses)`` each step."""
        self._pre_train_setup()
        while self.global_step < self.config.max_steps:
            for batch in self.dataloader:
                if self.global_step >= self.config.max_steps:
                    return

                losses = self.train_step(batch)
                if not self._increment_step_in_train_step:
                    self.global_step += 1

                if self.global_step % self.config.log_every == 0:
                    self._log_step(losses)

                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()

                yield self.global_step, losses

    def train_epoch(self) -> None:
        """Train for one full epoch over the dataloader."""
        for _step, _losses in self.train_iter():
            if self.global_step >= self.config.max_steps:
                break

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train_step(self, batch: Any) -> dict[str, float]:
        """Execute a single training step. Returns dict of loss values."""
        ...

    @abstractmethod
    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save model checkpoint."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        ...
