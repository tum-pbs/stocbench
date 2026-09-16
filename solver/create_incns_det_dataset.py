#!/usr/bin/env python3
"""Generate the deterministic INCNS dataset: trajectories with the per-window forcing as a second channel.

Run from the repository root, e.g. `python -m solver.create_incns_det_dataset 42 43 44 45 --out data/incns_det`;
see `solver/__init__.py` for the published invocation and the output layout.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solver.incns_det import DetIncNSSolver  # noqa: E402
from solver.utils import save_preview_video  # noqa: E402


def worker(rank: int, num_gpus: int, args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    traj_seeds = args.traj_seeds[rank::num_gpus]
    if not traj_seeds:
        return

    solver_cfg = OmegaConf.load(ROOT / "solver" / "configs" / "incns_det.yaml")
    solver = DetIncNSSolver(hydra.utils.instantiate(solver_cfg))
    m = args.out_grid or solver.cfg.n

    for s in traj_seeds:
        with tempfile.TemporaryDirectory() as tmp:
            warmup = np.lib.format.open_memmap(
                Path(tmp) / "warmup.npy",
                mode="w+",
                dtype=np.float32,
                shape=(args.batch_size, args.warmup_frames, 2, m, m),
            )
            traj = np.lib.format.open_memmap(
                args.out / f"traj_seed_{s}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(args.batch_size, args.traj_frames, 2, m, m),
            )
            solver.simulate(
                warmup_out=warmup,
                traj_out=traj,
                warmup_frames=args.warmup_frames,
                traj_frames=args.traj_frames,
                seed=s,
                batch_size=args.batch_size,
                out_grid=args.out_grid,
                desc=f"GPU {rank} | Seed {s}",
                position=rank,
            )
            save_preview_video(
                args.out / f"traj_seed_{s}_preview.mp4",
                traj[0, :, 0],
                fps=args.preview_fps,
                max_frames=args.preview_frames,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traj_seeds", type=int, nargs="+", help="Seeds of the trajectories (one traj_seed_<s>.npy each).")
    p.add_argument("--out", type=Path, required=True, help="Output directory.")
    p.add_argument("--traj-frames", type=int, default=100)
    p.add_argument("--warmup-frames", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--out-grid", type=int, default=None, help="Block-mean downsample stored frames to this grid (must divide n)."
    )
    p.add_argument("--preview-frames", type=int, default=10)
    p.add_argument("--preview-fps", type=int, default=2)

    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gpus = max(1, torch.cuda.device_count())

    if torch.cuda.device_count() > 1:
        mp.spawn(worker, args=(gpus, args), nprocs=gpus)
    else:
        worker(0, 1, args)
