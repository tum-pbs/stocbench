from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import Array, IncNSSpectralSolver, _iters


class DetIncNSSolver(IncNSSpectralSolver):
    """Forcing held constant per snapshot window; saved as channel 1.

    Given (vort_t, forcing_{t->t+1}) the map to vort_{t+1} is deterministic.
    PRNG key and current forcing live on `self` (set by `_rollout` /
    `_sample_forcing`); this assumes the Python loop in `_rollout` — would
    break under `jit` / `lax.scan`.
    """

    _key: Array
    forcing_hat: Array

    def _configure(self) -> None:
        self.forcing_scale = self.cfg.epsilon * self.dt / jnp.sqrt(self.cfg.sample_dt)

    def _init_state(self, *, seed, batch_size, same_warmup) -> Array:
        return jnp.zeros((1 if same_warmup else batch_size, self.n, self.n))

    def _sample_forcing(self, batch: int) -> None:
        self._key, sub = jax.random.split(self._key)
        coeff = jax.random.normal(sub, (batch, self.forcing_basis.shape[0]))
        self.forcing_hat = jnp.tensordot(coeff, self.forcing_basis, axes=([1], [0]))

    def _step_hat(self, wh: Array, aux) -> tuple[Array, Array]:
        wh = wh.at[..., 0, 0].set(0.0)
        wh = wh + self.dt * (self.linear_hat * wh + self._nonlinear_head(wh))
        wh = wh + self.forcing_scale * self.forcing_hat
        return wh, aux

    def _record(self, wh: Array, aux) -> Array:
        self._sample_forcing(wh.shape[0])
        return self._downsample(jnp.stack([self._ifft(wh), self._ifft(self.forcing_hat)], axis=1), self.out_n)

    def _rollout(self, w, *, steps, out, aux, desc=None, show_progress=False, leave=False, position=0):
        wh = self._fft(w)
        self._key = aux
        record = 0
        for i in _iters(steps, desc=desc, leave=leave, position=position, show=show_progress):
            if (i - 1) % self.steps == 0 and record < out.shape[1]:
                out[:, record] = self._record(wh, aux)
                record += 1
            wh, aux = self._step_hat(wh, aux)
        out.flush()
        return self._ifft(wh), self._key

    def _finalize(self, warmup, traj):
        mean, std = self.cfg.mean, self.cfg.std
        warmup[:, :, 0] = (warmup[:, :, 0] - mean) / std
        traj[:, :, 0] = (traj[:, :, 0] - mean) / std
        warmup.flush()
        traj.flush()
        return warmup, traj
