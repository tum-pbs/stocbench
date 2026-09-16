import torch
import torch.nn as nn

from .base import BaseGenerativeModel


def _e(t: torch.Tensor) -> torch.Tensor:
    """[B] -> [B, 1, 1, 1]"""
    return t[:, None, None, None]


class StochasticInterpolation(BaseGenerativeModel):
    """Stochastic interpolation baseline (sigma-scaled interpolant, optional Foellmer process)."""

    def __init__(
        self,
        net: nn.Module,
        sigma_coef: float = 1.0,
        beta_fn: str = "t^2",
        t_train: tuple[float, float] = (0.0, 0.999),
        t_sample: tuple[float, float] = (0.0, 0.999),
        foellmer: bool = False,
        foellmer_coef: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        if beta_fn not in ("t", "t^2"):
            raise ValueError("beta_fn must be 't' or 't^2'")
        if foellmer and beta_fn != "t^2":
            raise ValueError("the Foellmer diffusion coefficient assumes beta_fn='t^2'")
        self.save_hyperparameters(ignore=["net"])
        self._assign_init_args(locals())

    # --- interpolant coefficients ---

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return t.pow(2) if self.beta_fn == "t^2" else t

    def _beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * t if self.beta_fn == "t^2" else torch.ones_like(t)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_coef * (1.0 - t)

    def _g(self, t: torch.Tensor, g_coef: float | None = None) -> torch.Tensor:
        """Diffusion coefficient; g_coef scales the excess variance g_F^2 - sigma^2."""
        sigma = self._sigma(t)
        if not self.foellmer:
            return sigma
        g_F = self.sigma_coef * torch.sqrt((3.0 - t) * (1.0 - t))
        lam = 1.0 if g_coef is None else g_coef
        g2 = (sigma**2 + lam * (g_F**2 - sigma**2)).clamp_min(0.0)
        return self.foellmer_coef * g2.sqrt()

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        """Drift-matching loss of the sigma-scaled interpolant.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        cond_flat, x1 = self._unpack(target, cond)
        x0 = cond[:, -1, : x1.shape[1]]

        lo, hi = self.t_train
        t = torch.rand(x1.shape[0], device=x1.device) * (hi - lo) + lo
        noise = torch.randn_like(x0)
        root_t = torch.sqrt(t.clamp_min(0.0))

        xt = _e(1.0 - t) * x0 + _e(self._beta(t)) * x1 + _e(self._sigma(t) * root_t) * noise
        drift_target = -x0 + _e(self._beta_dot(t)) * x1 - _e(self.sigma_coef * root_t) * noise

        drift_pred = self.net(torch.cat([cond_flat, xt], dim=1), t)[:, -xt.shape[1] :]
        return (drift_pred - drift_target).square().sum(dim=(1, 2, 3)).mean()

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        *,
        sampling_schedule: tuple[str, int] = ("u", 1),
        g_coef: float | None = None,
        return_steps: bool = False,
    ):
        """Sample next frame by SDE integration.

        cond: [B, C, H, W]
        """
        kind, num_steps = sampling_schedule
        if kind != "u":
            raise ValueError(f"{type(self).__name__} only supports 'u' schedules")

        B = cond.shape[0]
        x0 = cond[:, : self.net.out_channels]
        xt = x0.clone()
        ts = torch.linspace(*self.t_sample, num_steps + 1, device=cond.device, dtype=cond.dtype)

        steps = []
        for i, (t0, t1) in enumerate(zip(ts[:-1], ts[1:])):
            dt = (t1 - t0).to(cond.dtype)
            t = torch.full((B,), float(t0), device=cond.device, dtype=cond.dtype)
            drift = self.net(torch.cat([cond, xt], dim=1), t)[:, -xt.shape[1] :]
            sigma_t = _e(self._sigma(t)).to(xt.dtype)
            w = torch.randn_like(xt)

            if self.foellmer and i > 0:
                # score identity of the interpolant; singular at t=0, hence the i > 0 guard
                beta, beta_dot, sigma = self._beta(t), self._beta_dot(t), self._sigma(t)
                A = 1.0 / (t * sigma * (beta_dot * sigma + beta * self.sigma_coef))
                c = _e(beta_dot) * xt - _e(beta + beta_dot * (1.0 - t)) * x0
                score = _e(A) * (_e(beta) * drift - c)

                g_t = _e(self._g(t, g_coef)).to(xt.dtype)
                drift = drift + 0.5 * (g_t**2 - sigma_t**2) * score
                xt = xt + drift * dt + g_t * w * dt.sqrt()
            else:
                xt = xt + drift * dt + sigma_t * w * dt.sqrt()
            steps.append(xt)

        return (xt, torch.stack(steps)) if return_steps else xt
