#!/usr/bin/env python3
"""Generate the stochastic INCNS dataset: training trajectories plus one-step test ensembles.

Run from the repository root, e.g. `python -m solver.create_incns_stoc_dataset 42 43 44 --out data/incns_stoc`;
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

from solver.incns_stoc import StocIncNSSolver  # noqa: E402
from solver.utils import save_preview_video  # noqa: E402


def worker(rank: int, num_gpus: int, args: argparse.Namespace, test_seeds: list[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    start = test_seeds[0]
    traj_seeds = args.traj_seeds[rank::num_gpus]
    test_seeds = test_seeds[rank::num_gpus]
    if not traj_seeds and not test_seeds:
        return

    solver_cfg = OmegaConf.load(ROOT / "solver" / "configs" / "incns_stoc.yaml")
    solver = StocIncNSSolver(hydra.utils.instantiate(solver_cfg))
    n = solver.cfg.n
    m = args.out_grid or n

    # generate trajectories
    for s in traj_seeds:
        with tempfile.TemporaryDirectory() as tmp:
            warmup = np.lib.format.open_memmap(
                Path(tmp) / "warmup.npy",
                mode="w+",
                dtype=np.float32,
                shape=(args.batch_size, args.warmup_frames, 1, m, m),
            )
            traj = np.lib.format.open_memmap(
                args.out / f"traj_seed_{s}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(args.batch_size, args.traj_frames, 1, m, m),
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

    # generate test cases
    if test_seeds:
        chunk = min(args.stats_samples, args.batch_size)
        chunks_per_seed = (args.stats_samples + chunk - 1) // chunk
        with tempfile.TemporaryDirectory() as tmp:
            warmup = np.lib.format.open_memmap(
                Path(tmp) / "warmup.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(test_seeds), args.warmup_frames, 1, n, n),
            )
            traj = np.lib.format.open_memmap(
                Path(tmp) / "traj.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(test_seeds), 1, 1, n, n),
            )
            solver.simulate(
                warmup_out=warmup,
                traj_out=traj,
                warmup_frames=args.warmup_frames,
                traj_frames=1,
                seed=test_seeds[0],
                batch_size=len(test_seeds),
                desc=f"GPU {rank} | Test warmup",
                position=rank,
            )

            for s, init in zip(test_seeds, traj[:, 0, 0], strict=True):
                offset = start + (s - start) * chunks_per_seed
                samples = np.concatenate(
                    [
                        np.array(
                            solver.step(
                                np.repeat(init[None], min(chunk, args.stats_samples - i), axis=0),
                                seed=offset + i // chunk,
                                desc=f"GPU {rank} | Test seed {s} step",
                                position=rank,
                            ),
                            dtype=np.float32,
                        )
                        for i in range(0, args.stats_samples, chunk)
                    ]
                )
                init_lo = np.array(solver._downsample(init[None], m), dtype=np.float32)
                samples_lo = np.array(solver._downsample(samples, m), dtype=np.float32)
                np.savez(
                    args.out / f"step_seed_{s}.npz",
                    init=init_lo,
                    mean=samples_lo.mean(0, keepdims=True),
                    std=samples_lo.std(0, keepdims=True),
                    raw=samples_lo[:, None, None],
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "traj_seeds", type=int, nargs="*", help="Seeds of the training trajectories (one traj_seed_<s>.npy each)."
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory.")
    p.add_argument("--traj-frames", type=int, default=100)
    p.add_argument("--warmup-frames", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--out-grid", type=int, default=None, help="Block-mean downsample stored frames to this grid (must divide n)."
    )
    p.add_argument("--stats-samples", type=int, default=1000, help="One-step draws per test initial condition.")
    p.add_argument(
        "--num-test-seeds", type=int, default=50, help="Test initial conditions, written as step_seed_100.npz onwards."
    )
    p.add_argument("--preview-frames", type=int, default=10)
    p.add_argument("--preview-fps", type=int, default=2)

    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    test_seeds = list(range(100, 100 + args.num_test_seeds))
    gpus = max(1, torch.cuda.device_count())

    if torch.cuda.device_count() > 1:
        mp.spawn(worker, args=(gpus, args, test_seeds), nprocs=gpus)
    else:
        worker(0, 1, args, test_seeds)
