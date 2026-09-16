from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DetStats(Dataset):
    """Deterministic INCNS stats: inits from trajectories, reference mean is the GT next frame.

    The deterministic map (vorticity, forcing) -> next vorticity has no spread,
    so the reference std is zero by definition.
    """

    def __init__(self, root: str, seeds: list[int], n_inits: int) -> None:
        super().__init__()
        root = Path(root).expanduser()
        self.traj = [np.load(root / f"traj_seed_{s}.npy", mmap_mode="r") for s in seeds]
        shape = self.traj[0].shape
        if any(t.shape != shape for t in self.traj):
            raise ValueError("all stats trajectories must share the same shape")
        self.starts_per_sample = shape[1] - 1  # one-step windows need a next frame
        assert self.starts_per_sample > 0, "trajectory too short for stats"
        self.per_seed = shape[0] * self.starts_per_sample
        total = len(self.traj) * self.per_seed
        assert total >= n_inits, f"not enough samples: {total} < {n_inits}"
        self.order = np.random.default_rng(0).permutation(total)[: int(n_inits)]

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        flat = int(self.order[idx])
        seed_idx, j = divmod(flat, self.per_seed)
        sample_idx, start = divmod(j, self.starts_per_sample)
        window = torch.tensor(self.traj[seed_idx][sample_idx, start : start + 2], dtype=torch.float32)
        gt_next = window[1, :1]  # deterministic next vorticity
        return {"_seed": flat, "init": window[0], "mean": gt_next, "std": torch.zeros_like(gt_next)}


class DetRollout(Dataset):
    """Deterministic INCNS rollout (init + GT trajectory window)."""

    def __init__(self, root: str, seeds: list[int], rollout_len: int, ensemble_size: int) -> None:
        super().__init__()
        root = Path(root).expanduser()
        self.traj = [np.load(root / f"traj_seed_{s}.npy", mmap_mode="r") for s in seeds]
        shape = self.traj[0].shape
        if any(t.shape != shape for t in self.traj):
            raise ValueError("all rollout trajectories must share the same shape")
        self.rollout_len = int(rollout_len)
        self.starts_per_sample = shape[1] - self.rollout_len
        assert self.starts_per_sample > 0, "trajectory too short for rollout_len"
        self.per_seed = shape[0] * self.starts_per_sample
        total = len(self.traj) * self.per_seed
        assert total >= ensemble_size, f"not enough samples: {total} < {ensemble_size}"
        self.order = np.random.default_rng(0).permutation(total)[: int(ensemble_size)]

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seed_idx, j = divmod(int(self.order[idx]), self.per_seed)
        sample_idx, start = divmod(j, self.starts_per_sample)
        window = self.traj[seed_idx][sample_idx, start : start + self.rollout_len + 1]
        return torch.tensor(window, dtype=torch.float32)
