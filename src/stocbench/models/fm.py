import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseGenerativeModel


class FlowMatchingModel(BaseGenerativeModel):
    """Conditional flow matching (optimal-transport path, ODE sampling)."""

    def __init__(
        self,
        net: nn.Module,
        sigma_min: float = 0.001,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        integrator: str = "euler",
    ):
        super().__init__()
        if integrator not in ("euler", "rk4"):
            raise ValueError(f"unknown integrator: {integrator}")
        self.save_hyperparameters(ignore=["net"])
        self._assign_init_args(locals())

    def phi_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolate x_0 -> x_1 at t; t: [B]."""
        t = t.view(-1, 1, 1, 1)
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1

    def v_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Target vector field at t."""
        return x_1 - (1 - self.sigma_min) * x_0

    def sample_x0(self, cond: torch.Tensor) -> torch.Tensor:
        """Draw from the base distribution. cond: [B, C, H, W]"""
        return torch.randn_like(cond[:, : self.net.out_channels])

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        """Flow matching loss.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        cond, x_1 = self._unpack(target, cond)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)

        x_t = self.phi_t(x_0, x_1, t)
        v_pred = self.net(torch.cat([cond, x_t], dim=1), t)[:, -x_t.shape[1] :]
        return F.mse_loss(v_pred, self.v_t(x_0, x_1, t))

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, *, sampling_schedule: tuple[str, int] = ("u", 1), return_steps: bool = False):
        """Sample next frame by ODE integration.

        cond: [B, C, H, W]
        """

        def v(t, x):
            return self.net(torch.cat([cond, x], dim=1), t.expand(x.shape[0]))[:, -x.shape[1] :]

        return self._integrate(v, self.sample_x0(cond), sampling_schedule, return_steps)

    def _integrate(self, v, x: torch.Tensor, sampling_schedule: tuple[str, int], return_steps: bool):
        """Integrate dx = v(t, x) dt from x at t=0 to t=1 on the schedule's time grid."""
        kind, num_steps = sampling_schedule
        ts = self._sample_times(num_steps, kind, x.device, x.dtype)
        steps = []
        for t0, t1 in zip(ts[:-1], ts[1:]):
            x = self._step(v, x, t0, t1 - t0)
            steps.append(x)
        return (x, torch.stack(steps)) if return_steps else x

    def _step(self, v, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """One euler / rk4 step of dx = v(t, x) dt."""
        if self.integrator == "euler":
            return x + dt * v(t, x)
        k1 = v(t, x)
        k2 = v(t + dt / 2, x + dt * k1 / 2)
        k3 = v(t + dt / 2, x + dt * k2 / 2)
        k4 = v(t + dt, x + dt * k3)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @staticmethod
    def _sample_times(num_steps: int, kind: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if kind == "u":
            return torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
        if kind == "l":
            tail = torch.logspace(-3, 0, num_steps, device=device, dtype=dtype)
            tail = (tail - tail[0]) / (tail[-1] - tail[0])
            return torch.cat([tail.new_zeros(1), tail])
        raise ValueError(f"unknown schedule: {kind}")
