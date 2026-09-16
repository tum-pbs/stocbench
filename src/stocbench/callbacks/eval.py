from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb

from ..results import save_artifact
from ..utils import Schedule
from ..utils.figures import fig_to_image
from ..utils.plotting import plot_stat_grid


class EvalCallback(pl.Callback):
    """Stats + rollout metrics over the eval loaders; saves artifacts on test, logs on validation.

    stats loader:   dicts {"_seed", "init", "mean", "std"[, "raw"]}; each init is inflated to an ensemble
    rollout loader: [B, 1, C, H, W] start states (free-running) or [B, T+1, C, H, W] GT windows (gt_rollout)
    """

    def __init__(
        self,
        metrics: list,
        *,
        schedule: str | tuple = ("u", 1),
        ensemble_size: int = 32,
        rollout_steps: int = 10,
        batch_size: int = 32,
        gt_rollout: bool = False,
        rollout_noise: float = 0.0,
        output_dir: str | Path | None = None,
        seed: int = 0,
        artifact_prefix: str = "",
    ) -> None:
        super().__init__()
        self.metrics = metrics
        self.schedule = Schedule.parse(schedule)
        self.ensemble_size = ensemble_size
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.gt_rollout = gt_rollout
        self.rollout_noise = rollout_noise
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.seed = seed
        self.artifact_prefix = artifact_prefix  # "r50_" for the long-rollout pass
        self._stats = [m for m in metrics if m.mode == "stats"]
        self._rollout = [m for m in metrics if m.mode == "rollout"]
        self._step_metrics = [m for m in self._stats if m.record_steps]
        self._record_traj = any(m.record_steps for m in self._rollout)

    # --- shared per-batch work (test and validation) ---

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        spectrum = getattr(trainer.datamodule, "reference_spectrum", None)
        for m in self.metrics:
            m.reset()
            m.to(pl_module.device)
            if hasattr(m, "bind") and spectrum is not None:
                m.bind(reference_spectrum=spectrum)
        if self._step_metrics and hasattr(pl_module, "sample_steps"):
            grid = pl_module.sample_steps(self.schedule.value, self.schedule.kind).detach().float()
            for m in self._step_metrics:
                m.grid = grid

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            self._stats_batch(pl_module, batch)
        else:
            self._rollout_batch(pl_module, batch)

    def _stats_batch(self, model, batch: dict) -> None:
        if not self._stats:
            return
        inflated = batch["init"].expand(self.ensemble_size, *batch["init"].shape[1:])
        samples, traj = self._sample(model, inflated, return_steps=bool(self._step_metrics))
        for m in self._stats:
            m.update(batch, samples.unsqueeze(0), traj if m.record_steps else None)

    def _rollout_batch(self, model, batch: torch.Tensor) -> None:
        if not self._rollout:
            return
        state = batch[:, 0]
        for step in range(self.rollout_steps):
            pred, traj = self._sample(model, state, return_steps=self._record_traj)
            gt = batch[:, step + 1] if self.gt_rollout else None
            for m in self._rollout:
                m.update(step, pred, target=gt[:, : pred.shape[1]] if gt is not None else None, traj=traj)
            # dither the fed-back state only (metrics see the raw prediction): probes whether
            # per-step noise re-excitation counteracts the sampler's spectral damping
            if self.rollout_noise > 0.0:
                pred = pred + self.rollout_noise * torch.randn_like(pred)
            # autoregress the predicted channels; exogenous channels (e.g. forcing) come from GT
            state = torch.cat([pred, gt[:, pred.shape[1] :]], dim=1) if gt is not None else pred

    # --- test: save artifacts ---

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # compute() syncs metric state across ranks (collective), so every rank must call it
        results = [(m.name, m.prefix, m.compute()) for m in self.metrics]
        if not trainer.is_global_zero or self.output_dir is None:
            return
        for name, prefix, payload in results:
            save_artifact(
                self.output_dir / f"artifacts/{self.artifact_prefix}{name}_{self.schedule.slug}.npz",
                seed=self.seed,
                **{f"{prefix}{k}": v for k, v in payload.items()},
            )

    # --- validation: log scalars + stat grids ---

    on_validation_start = on_test_start
    on_validation_batch_end = on_test_batch_end

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        outputs = {m.name: m.compute() for m in self.metrics}
        pl_module.log_dict(
            {f"val/{name}": out["value"] for name, out in outputs.items()},
            sync_dist=True,
            on_epoch=True,
        )
        for m in (m for m in self._stats if hasattr(m, "stat")):
            out = outputs[m.name]
            fig = plot_stat_grid(out["prediction"].cpu(), out["reference"].cpu(), out["seeds"].cpu(), m.stat)
            pl_module.logger.experiment.log({f"val/{m.name}_grid": wandb.Image(fig_to_image(fig))})
            plt.close(fig)

    def _sample(self, model, batch: torch.Tensor, return_steps: bool = False):
        chunks = batch.split(self.batch_size)
        if not return_steps:  # not every model accepts return_steps, so only pass it when needed
            return torch.cat([model.sample(c, sampling_schedule=self.schedule) for c in chunks]), None
        samples, trajs = zip(*(model.sample(c, sampling_schedule=self.schedule, return_steps=True) for c in chunks))
        return torch.cat(samples), torch.cat(trajs, dim=1)
