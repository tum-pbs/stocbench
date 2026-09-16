from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import NamedTuple

import torch


class Schedule(NamedTuple):
    """Sampling schedule, e.g. ("u", 100) <-> "u:100" or ("at", [0, 2]) <-> "at:[0,2]"."""

    kind: str
    value: int | list

    @classmethod
    def parse(cls, s) -> Schedule:
        if not isinstance(s, str):
            return cls(*s)
        kind, _, value = s.rpartition(":")  # last colon: the kind itself may hold one ("mix:0.25:50")
        return cls(kind, ast.literal_eval(value))

    @property
    def steps(self) -> int:
        return self.value if isinstance(self.value, int) else len(self.value)

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.value}".replace(" ", "")

    @property
    def slug(self) -> str:
        return self.label.replace(":", "_").replace(",", "-")


def parse_artifact_name(path) -> tuple[str, Schedule]:
    # all identity lives in the filename: "mean_error_u_10.npz" -> ("mean_error", ("u", 10))
    metric, kind, value = Path(path).stem.rsplit("_", 2)
    return metric, Schedule.parse(f"{kind}:{value.replace('-', ',')}")


def register_resolvers() -> None:
    from omegaconf import ListConfig, OmegaConf

    OmegaConf.register_new_resolver("int", int, replace=True)

    # ListConfig so the value survives OmegaConf.resolve() materializing it into the tree
    OmegaConf.register_new_resolver(
        "range", lambda start, n: ListConfig(list(range(int(start), int(start) + int(n)))), replace=True
    )

    OmegaConf.register_new_resolver(
        "out", lambda channels, params, extra=0: len(channels) + len(params) + int(extra), replace=True
    )

    OmegaConf.register_new_resolver("add", lambda *xs: sum(int(x) for x in xs), replace=True)

    # slots: predicted-frame inputs appended to the context (1: the noisy target x_t; 0 for direct regressors)
    OmegaConf.register_new_resolver(
        "in",
        lambda cond_len, channels, params, extra=0, slots=1: (
            (cond_len + slots) * (len(channels) + len(params)) + cond_len * int(extra)
        ),
        replace=True,
    )


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint as a plain dict of tensors."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:  # older ckpts pickle omegaconf objects into hparams; ours are trusted
        return torch.load(path, map_location="cpu", weights_only=False)


def load_model(cfg, checkpoint: str | None = None):
    import hydra
    from omegaconf import OmegaConf

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model_dict.pop("name", None)
    model = hydra.utils.instantiate(model_dict)
    if checkpoint:
        model.load_state_dict(load_checkpoint(checkpoint)["state_dict"])
    return model
