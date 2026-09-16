from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from matplotlib import colormaps


def save_preview_video(
    path: str | Path,
    frames: np.ndarray,
    fps: int = 2,
    max_frames: int = 10,
) -> None:
    """Save a short preview video."""
    frames = np.asarray(frames[:max_frames], dtype=np.float32)
    scale = max(float(frames.max() - frames.min()), 1e-6)
    frames = (frames - frames.min()) / scale
    rgb = (255 * colormaps["viridis"](frames)[..., :3]).astype(np.uint8)
    imageio.mimwrite(Path(path), rgb, fps=fps)
