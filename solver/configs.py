from __future__ import annotations

from dataclasses import dataclass
from math import pi


@dataclass
class IncnsConfig:
    """Shared config for stoc and det INCNS variants."""

    n: int = 256
    l: float = 2 * pi
    dt: float = 1e-4
    nu: float = 1e-3
    drag: float = -0.1
    sample_dt: float = 0.5
    epsilon: float = 1.0
    forcing_modes: tuple[tuple[float, float], ...] = (
        (6.0, 0.0),
        (7.0, 0.0),
        (5.0, 5.0),
        (8.0, 8.0),
    )
    mean: float = 0.0
    std: float = 1.0
