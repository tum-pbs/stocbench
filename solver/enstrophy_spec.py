"""Write the reference enstrophy spectrum `enstrophy_spec.npy` of a generated dataset.

The eval metric `EnstrophyError` compares a rollout's radial enstrophy spectrum against this
reference: the mean of `stocbench.metrics.radial_enstrophy` over the vorticity channel of the
training trajectories. Run after `create_incns_*_dataset.py`:

    python -m solver.enstrophy_spec data/incns_stoc --seeds 42 43 44
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from stocbench.metrics import radial_enstrophy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="dataset directory holding traj_seed_<s>.npy")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="training trajectory seeds")
    parser.add_argument("--stride", type=int, default=1, help="use every stride-th frame")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    total = None
    count = 0
    for seed in args.seeds:
        traj = np.load(args.root / f"traj_seed_{seed}.npy", mmap_mode="r")  # [samples, frames, C, H, W]
        fields = traj[:, :: args.stride, 0].reshape(-1, *traj.shape[-2:])  # vorticity channel only
        for start in range(0, fields.shape[0], args.batch_size):
            spec = radial_enstrophy(torch.as_tensor(np.asarray(fields[start : start + args.batch_size])))
            total = spec.sum(0) if total is None else total + spec.sum(0)
            count += spec.shape[0]
    out = args.root / "enstrophy_spec.npy"
    np.save(out, (total / count).numpy().astype(np.float32))
    print(f"wrote {out} ({count} frames, {total.shape[0]} bins)")


if __name__ == "__main__":
    main()
