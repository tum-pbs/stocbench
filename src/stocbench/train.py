"""`stocbench train`: train one model config on one experiment."""

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from .utils import load_model, register_resolvers
from .utils.hf import resolve_hf_path

torch.set_float32_matmul_precision("high")
register_resolvers()


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    ckpt_path = resolve_hf_path(cfg.ckpt_path, cfg.hf.ckpt)
    cfg.data.root = resolve_hf_path(cfg.data.root, cfg.hf.data)
    data = hydra.utils.instantiate(cfg.data)
    model = load_model(cfg)
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks.values()]
    loggers = [hydra.utils.instantiate(logger) for logger in cfg.tracking.values()]

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    trainer.fit(model, datamodule=data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
