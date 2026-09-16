from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import tqdm.auto

Array = jnp.ndarray


def _iters(n: int, *, desc=None, leave=False, position=0, show=False):
    return (
        tqdm.auto.trange(1, n + 1, desc=desc, leave=leave, position=position)
        if show or desc is not None
        else range(1, n + 1)
    )


class IncNSSpectralSolver:
    """Shared spectral solver. Subclasses define `_init_state` and `_step_hat`."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def _setup(self) -> None:
        assert self.cfg.sample_dt >= self.cfg.dt, "sample_dt >= dt"
        self.n = self.cfg.n
        self.dt = self.cfg.dt
        self.steps = int(round(self.cfg.sample_dt / self.dt))
        k = jnp.fft.fftfreq(self.n, d=1.0 / self.n)
        scale = 2 * jnp.pi / self.cfg.l
        self.kx, self.ky = scale * k[:, None], scale * k[None, :]
        self.k2 = self.kx**2 + self.ky**2
        self.dealias = (jnp.abs(k[:, None]) <= self.n / 3.0) & (jnp.abs(k[None, :]) <= self.n / 3.0)
        self.linear_hat = self.cfg.drag - self.cfg.nu * self.k2
        x = jnp.linspace(0.0, self.cfg.l, self.n, endpoint=False, dtype=jnp.float32)
        gx, gy = jnp.meshgrid(x, x, indexing="ij")
        modes = jnp.stack(
            [f(kx * gx + ky * gy) for kx, ky in self.cfg.forcing_modes for f in (jnp.sin, jnp.cos)],
            axis=0,
        )
        self.forcing_basis = self._fft(modes)
        self._configure()

    def _fft(self, x: Array) -> Array:
        return jnp.fft.fft2(x, axes=(-2, -1))

    def _ifft(self, x: Array) -> Array:
        return jnp.fft.ifft2(x, axes=(-2, -1)).real

    def _downsample(self, x: Array, m: int) -> Array:
        """Block-mean spatial downsample of the last two axes to m."""
        if m == self.n:
            return x
        f = self.n // m
        return x.reshape(*x.shape[:-2], m, f, m, f).mean(axis=(-3, -1))

    def _nonlinear_head(self, wh: Array) -> Array:
        wh = wh * self.dealias
        psi = jnp.where(self.k2 > 0, -wh / self.k2, 0.0)
        u = self._ifft(1j * self.ky * psi)
        v = self._ifft(-1j * self.kx * psi)
        wx = self._ifft(1j * self.kx * wh)
        wy = self._ifft(1j * self.ky * wh)
        return -self._fft(u * wx + v * wy) * self.dealias

    def _record(self, wh: Array, aux) -> Array:
        """Snapshot tensor [B, C, out_n, out_n] for the current sample boundary."""
        return self._downsample(self._ifft(wh), self.out_n)[..., None, :, :]

    def _rollout(self, w, *, steps, out, aux, desc=None, show_progress=False, leave=False, position=0):
        wh = self._fft(w)
        record = 0
        for i in _iters(steps, desc=desc, leave=leave, position=position, show=show_progress):
            wh, aux = self._step_hat(wh, aux)
            if i % self.steps == 0:
                out[:, record] = self._record(wh, aux)
                record += 1
        out.flush()
        return self._ifft(wh), aux

    def simulate(
        self,
        *,
        warmup_out: np.memmap,
        traj_out: np.memmap,
        warmup_frames: int = 50,
        traj_frames: int = 100,
        seed: int = 0,
        batch_size: int = 1,
        same_warmup: bool = False,
        out_grid: int | None = None,
        desc: str | None = None,
        show_progress: bool = False,
        position: int = 0,
    ):
        self._setup()
        assert out_grid is None or self.n % out_grid == 0, "out_grid must divide n"
        self.out_n = out_grid or self.n
        w = self._init_state(seed=seed, batch_size=batch_size, same_warmup=same_warmup)
        warm_key, traj_key = jax.random.split(jax.random.PRNGKey(seed))
        prog = show_progress or desc is not None

        if warmup_frames:
            w, _ = self._rollout(
                w,
                steps=warmup_frames * self.steps,
                out=warmup_out,
                aux=warm_key,
                desc=desc and f"{desc} warmup",
                show_progress=prog,
                leave=False,
                position=position,
            )
        else:
            warmup_out.flush()

        if same_warmup:
            w = jnp.repeat(w, batch_size, axis=0)

        self._rollout(
            w,
            steps=traj_frames * self.steps,
            out=traj_out,
            aux=traj_key,
            desc=desc and f"{desc} traj",
            show_progress=prog,
            leave=True,
            position=position,
        )
        return self._finalize(warmup_out, traj_out)

    def step(
        self, w: Array, *, seed: int = 0, normalized: bool = True, desc=None, show_progress=False, position=0
    ) -> Array:
        """Advance batched vorticity by one sample_dt."""
        w = jnp.asarray(w)
        assert w.ndim == 3, f"w: [B,N,N], got {tuple(w.shape)}"
        mean, std = self.cfg.mean, self.cfg.std
        if normalized:
            w = w * std + mean
        self._setup()
        wh = self._fft(w)
        key = jax.random.PRNGKey(seed)
        for _ in _iters(self.steps, desc=desc, leave=True, position=position, show=show_progress):
            wh, key = self._step_hat(wh, key)
        w = self._ifft(wh)
        return (w - mean) / std if normalized else w

    def _finalize(self, warmup, traj):
        mean, std = self.cfg.mean, self.cfg.std
        warmup[:] = (warmup - mean) / std
        traj[:] = (traj - mean) / std
        warmup.flush()
        traj.flush()
        return warmup, traj

    def _configure(self) -> None:
        raise NotImplementedError

    def _init_state(self, *, seed: int, batch_size: int, same_warmup: bool) -> Array:
        raise NotImplementedError

    def _step_hat(self, wh: Array, aux) -> tuple[Array, object]:
        raise NotImplementedError
