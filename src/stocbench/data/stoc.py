from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class StocStats(Dataset):
    """Stochastic INCNS test stats: one held-out seed per initial condition."""

    def __init__(self, root: str, seeds: list[int]) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.seeds = list(seeds)

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        seed = self.seeds[idx]
        with np.load(self.root / f"step_seed_{seed}.npz", allow_pickle=False) as stats:
            raw = torch.tensor(stats["raw"][:, 0], dtype=torch.float32)
            init = torch.tensor(stats["init"], dtype=torch.float32)
        return {
            "_seed": seed,
            "init": init,
            "mean": raw.mean(dim=0),
            "std": raw.std(dim=0, unbiased=False),
            "raw": raw,
        }


class StocRollout(Dataset):
    """Stochastic INCNS rollout contexts (free-running ensemble starts)."""

    def __init__(self, root: str, seeds: list[int], ensemble_size: int) -> None:
        super().__init__()
        root = Path(root).expanduser()
        self.traj = [np.load(root / f"traj_seed_{seed}.npy", mmap_mode="r") for seed in seeds]
        shape = self.traj[0].shape
        if any(traj.shape != shape for traj in self.traj):
            raise ValueError("all test trajectories must share the same shape")
        self.num_starts = shape[1]
        self.num_per_seed = shape[0] * shape[1]
        total = len(self.traj) * self.num_per_seed
        self.ensemble_size = int(ensemble_size)
        assert total >= self.ensemble_size, "not enough samples"
        self.order = np.random.default_rng(0).permutation(total)[: self.ensemble_size]

    def __len__(self) -> int:
        return self.ensemble_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        seed_idx, idx = divmod(int(self.order[idx]), self.num_per_seed)
        sample_idx, start = divmod(idx, self.num_starts)
        return torch.tensor(self.traj[seed_idx][sample_idx, start : start + 1], dtype=torch.float32)
