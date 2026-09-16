"""JAX pseudo-spectral solver for 2D incompressible Navier-Stokes (vorticity) that generated the StocBench datasets.

`StocIncNSSolver` draws i.i.d. Fourier-mode forcing per step; `DetIncNSSolver` holds the forcing constant per
sample window and stores it as a second channel. Parameters: `solver/configs/incns_{stoc,det}.yaml`. Run from
the repository root:

    python -m solver.create_incns_stoc_dataset 42 43 44 --out data/incns_stoc --warmup-frames 100 \
        --traj-frames 200 --batch-size 500 --num-test-seeds 48 --stats-samples 5000 --out-grid 64
    python -m solver.create_incns_det_dataset 42 43 44 45 --out data/incns_det --warmup-frames 100 \
        --traj-frames 200 --batch-size 500 --out-grid 64
    python -m solver.enstrophy_spec data/incns_stoc

Output (fields normalised by the config's mean/std; the det forcing channel stored as is):

* `traj_seed_<s>.npy`: float32 `[batch_size, traj_frames, C, out_grid, out_grid]`, C = 1 (stoc) or 2 (det).
* stoc `step_seed_<s>.npz` (s = 100 ...): one-step test ensembles with `init` `[1, H, W]`,
  `raw` `[stats_samples, 1, 1, H, W]` and their `mean` / `std` `[1, H, W]`.
* `enstrophy_spec.npy`: reference radial enstrophy spectrum, written by `enstrophy_spec.py`.
"""

from .configs import IncnsConfig
from .incns_det import DetIncNSSolver
from .incns_stoc import StocIncNSSolver

__all__ = ["DetIncNSSolver", "IncnsConfig", "StocIncNSSolver"]
