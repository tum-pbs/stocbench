from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .det import DetRollout, DetStats
from .stoc import StocRollout, StocStats

__all__ = [
    "DetRollout",
    "DetStats",
    "DistributedSliceSampler",
    "IncnsData",
    "StocRollout",
    "StocStats",
    "TrajTrain",
]


class TrajTrain(Dataset):
    """Training windows from trajectory files (det and stoc).

    Frames keep all channels, incl. exogenous ones like the det forcing;
    models slice the channels they predict (see ``BaseGenerativeModel._unpack``).
    """

    def __init__(self, root: str, seeds: list[int], stride: int = 1, seq_len: int = 2) -> None:
        super().__init__()
        root = Path(root).expanduser()
        self.seq_len = int(seq_len)
        self.traj = {seed: np.load(root / f"traj_seed_{seed}.npy", mmap_mode="r") for seed in seeds}
        self.index = [
            (seed, sample_idx, start)
            for seed, traj in self.traj.items()
            for sample_idx in range(traj.shape[0])
            for start in range(0, traj.shape[1] - (self.seq_len - 1), int(stride))
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed, sample_idx, start = self.index[idx]
        seq = torch.tensor(self.traj[seed][sample_idx, start : start + self.seq_len], dtype=torch.float32)
        return seq[:1], seq[1:]


class DistributedSliceSampler(Sampler[int]):
    """Eval-time DDP sharding without the sample duplication of torch's DistributedSampler."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def _indices(self) -> range:
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return range(rank, len(self.dataset), world)

    def __iter__(self):
        return iter(self._indices())

    def __len__(self) -> int:
        return len(self._indices())


class IncnsData(pl.LightningDataModule):
    """INCNS datamodule: config-driven train dataset + named eval loaders (stats, rollout)."""

    allow_zero_length_dataloader_with_multiple_devices = True

    def __init__(
        self,
        root: str,
        batch_size: int = 256,
        num_workers: int = 4,
        *,
        sim_channels: list[str],
        sim_params: list[str],
        train_dataset,
        test_loaders=None,
        cond_extra_channels: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sim_channels = sim_channels
        self.sim_params = sim_params
        self.cond_extra_channels = cond_extra_channels
        self.train_dataset_cfg = train_dataset
        self.test_loader_cfgs = test_loaders or {}

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_set = hydra.utils.instantiate(self.train_dataset_cfg, root=str(self.root))
        if stage in (None, "fit", "validate", "test"):
            self.eval_sets = {
                name: hydra.utils.instantiate(spec["dataset"], root=str(self.root))
                for name, spec in self.test_loader_cfgs.items()
            }
            spectrum = self.root / "enstrophy_spec.npy"
            if spectrum.exists():
                self.reference_spectrum = torch.as_tensor(np.load(spectrum), dtype=torch.float32)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                self.eval_sets[name],
                batch_size=spec.get("batch_size", self.batch_size),
                sampler=DistributedSliceSampler(self.eval_sets[name]),
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
            for name, spec in self.test_loader_cfgs.items()
        ]

    def test_dataloader(self) -> list[DataLoader]:
        return self.val_dataloader()
