"""`stocbench eval`: evaluate a checkpoint over the budgets of an evaluate protocol and render its figures."""

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .results import write_config, write_summary
from .utils import load_model, register_resolvers
from .utils.hf import resolve_hf_path

torch.set_float32_matmul_precision("high")
register_resolvers()


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    experiment = HydraConfig.get().runtime.choices.experiment
    pl.seed_everything(cfg.seed, workers=True)
    trainer = hydra.utils.instantiate(cfg.trainer)

    cfg.data.root = resolve_hf_path(cfg.data.root, cfg.hf.data)
    model = load_model(cfg, resolve_hf_path(cfg.ckpt_path, cfg.hf.ckpt))
    metrics = [hydra.utils.instantiate(spec) for spec in cfg.metrics]
    if model.fixed_cost:  # the sampler ignores the schedule, so one budget covers the protocol
        cfg.evaluate.sampling_schedules = cfg.evaluate.sampling_schedules[:1]
    schedules = list(cfg.evaluate.sampling_schedules)
    long_steps = cfg.evaluate.get("long_rollout_steps")
    rollout_metrics = [m for m in metrics if m.mode == "rollout"] if long_steps else []

    output_dir = Path(cfg.output_dir).resolve()  # resolve "." so plots label the run by its dir name
    config = {
        "model": str(cfg.model.name),
        "dataset": str(cfg.data.root),
        "checkpoint": str(cfg.ckpt_path or ""),
        "metrics": [m.name for m in metrics] + [f"r{long_steps}_{m.name}" for m in rollout_metrics],
        **OmegaConf.to_container(cfg.evaluate, resolve=True),
        "seed": int(cfg.seed),
    }
    if trainer.is_global_zero:
        write_config(output_dir, config)

    def run_pass(schedules: list, metrics: list, rollout_steps: int, prefix: str = "") -> None:
        cfg.evaluate.rollout_steps = rollout_steps  # the rollout datasets take their window length from it
        data = hydra.utils.instantiate(cfg.data)
        for schedule in schedules:
            callback = hydra.utils.instantiate(
                cfg.callback,
                metrics=metrics,
                schedule=schedule,
                output_dir=output_dir,
                seed=cfg.seed,
                ensemble_size=cfg.evaluate.ensemble_size,
                rollout_steps=rollout_steps,
                batch_size=cfg.evaluate.batch_size,
                artifact_prefix=prefix,
            )
            trainer.callbacks.append(callback)
            try:
                trainer.test(model, datamodule=data)
            finally:
                trainer.callbacks.remove(callback)

    run_pass(schedules, metrics, cfg.evaluate.rollout_steps)
    if rollout_metrics:  # the rollout metrics again over a long rollout at the last budget: r<steps>_* artifacts
        run_pass(schedules[-1:], rollout_metrics, long_steps, prefix=f"r{long_steps}_")

    if trainer.is_global_zero:
        write_summary(output_dir, config)
        from .utils.plotting import CONFIGS, plot_runs

        plot_runs([output_dir], plots_config=CONFIGS / f"plotting/plots/{cfg.evaluate.figures}/{experiment}.yaml")


if __name__ == "__main__":
    main()
