"""Exponential moving average (EMA) of model weights, as a Lightning callback.

A shadow copy of the optimizer's parameters is updated after every optimizer step.
The shadow can be swapped in for validation/test and is written out as EMA-only
checkpoints (`last-ema.ckpt` / `epochNNNN-ema.ckpt`) alongside the regular ones.
When ``monitor`` is set, the best EMA weights by that validation metric are also
written to `best-ema.ckpt`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback


class EMA(Callback):
    def __init__(
        self,
        decay: float = 0.999,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
        checkpoint_every_n_epochs: int | None = None,
        monitor: str | None = None,
        mode: str = "min",
    ) -> None:
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must be in [0, 1]")
        if mode not in ("min", "max"):
            raise ValueError("EMA mode must be 'min' or 'max'")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self._shadow: list[torch.Tensor] | None = None
        self._device: torch.device | None = None
        self._last_step = -1
        self._best: float | None = None

    # --- parameter access ---
    @staticmethod
    def _params(trainer: pl.Trainer) -> list[torch.Tensor]:
        return [p for opt in trainer.optimizers for g in opt.param_groups for p in g["params"]]

    # --- shadow lifecycle ---
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._device = torch.device("cpu") if self.cpu_offload else pl_module.device
        if self._shadow is None:
            self._shadow = [p.detach().clone().to(self._device) for p in self._params(trainer)]
        else:  # resumed from checkpoint
            self._shadow = [s.to(self._device) for s in self._shadow]

    @torch.no_grad()
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args: Any) -> None:
        step = trainer.global_step
        if self._shadow is None or step == self._last_step or step % self.every_n_steps != 0:
            return
        self._last_step = step
        current = [p.data.to(self._device, non_blocking=True) for p in self._params(trainer)]
        torch._foreach_mul_(self._shadow, self.decay)
        torch._foreach_add_(self._shadow, current, alpha=1.0 - self.decay)

    # --- swap EMA <-> live weights (in place; its own inverse) ---
    @torch.no_grad()
    def _swap(self, trainer: pl.Trainer) -> None:
        for p, s in zip(self._params(trainer), self._shadow):
            tmp = p.data.detach().clone()
            p.data.copy_(s)
            s.copy_(tmp)

    def _maybe_swap(self, trainer: pl.Trainer, *args: Any) -> None:
        if self._shadow is not None and not self.validate_original_weights:
            self._swap(trainer)

    on_validation_start = _maybe_swap
    on_validation_end = _maybe_swap
    on_test_start = _maybe_swap
    on_test_end = _maybe_swap

    # --- EMA checkpoints ---
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        cb = getattr(trainer, "checkpoint_callback", None)
        dirpath = getattr(cb, "dirpath", None)
        if dirpath is None or self._shadow is None:
            return
        dirpath = Path(dirpath)

        self._swap(trainer)
        try:
            trainer.save_checkpoint(str(dirpath / "last-ema.ckpt"), weights_only=True)
            epoch = trainer.current_epoch + 1
            every = self.checkpoint_every_n_epochs
            if every is not None and epoch % every == 0:
                trainer.save_checkpoint(str(dirpath / f"epoch{epoch:04d}-ema.ckpt"), weights_only=True)
            self._maybe_save_best(trainer, dirpath)
        finally:
            self._swap(trainer)

    def _maybe_save_best(self, trainer: pl.Trainer, dirpath: Path) -> None:
        """Save ``best-ema.ckpt`` when the monitored metric improves.

        Runs while the EMA weights are swapped in, after validation has populated
        ``trainer.callback_metrics`` for the current epoch, so the saved weights
        match the metric that selected them.
        """
        if self.monitor is None:
            return
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        current = float(current)
        better = self._best is None or (current < self._best if self.mode == "min" else current > self._best)
        if better:
            self._best = current
            trainer.save_checkpoint(str(dirpath / "best-ema.ckpt"), weights_only=True)

    # --- resume support ---
    def state_dict(self) -> dict[str, Any]:
        return {"shadow": self._shadow, "last_step": self._last_step, "best": self._best}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        shadow = state_dict.get("shadow")
        self._shadow = list(shadow) if shadow is not None else None
        self._last_step = state_dict.get("last_step", -1)
        self._best = state_dict.get("best")
