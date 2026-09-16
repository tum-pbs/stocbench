from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import Array, IncNSSpectralSolver


class StocIncNSSolver(IncNSSpectralSolver):
    """Per-step iid Fourier-mode forcing (SDE-style)."""

    def _configure(self) -> None:
        self.sqrt_dt = jnp.sqrt(self.dt)

    def _init_state(self, *, seed, batch_size, same_warmup) -> Array:
        return jnp.zeros((1 if same_warmup else batch_size, self.n, self.n))

    def _step_hat(self, wh: Array, key) -> tuple[Array, Array]:
        key, sub = jax.random.split(key)
        coeff = jax.random.normal(sub, (wh.shape[0], self.forcing_basis.shape[0]))
        forcing = jnp.tensordot(coeff, self.forcing_basis, axes=([1], [0]))
        wh = wh.at[..., 0, 0].set(0.0)
        wh = wh + self.dt * (self.linear_hat * wh + self._nonlinear_head(wh))
        wh = wh + self.cfg.epsilon * self.sqrt_dt * forcing
        return wh, key
