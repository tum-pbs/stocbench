import math

import torch


class VP:
    """Continuous-time VP schedule."""

    def __init__(
        self,
        beta_schedule: str,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        cosine_s: float = 0.008,
        t_eps: float = 1e-4,
    ) -> None:
        assert beta_schedule in {"linear", "cosine"}, "unknown beta_schedule"
        assert beta_min > 0, "beta_min > 0"
        assert beta_max > 0, "beta_max > 0"
        assert cosine_s > 0, "cosine_s > 0"
        assert 0 < t_eps < 1, "0 < t_eps < 1"

        self.beta_schedule = beta_schedule
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cosine_s = cosine_s
        self.t_eps = t_eps
        self._cos0 = math.cos(self._theta(0.0)) ** 2

    def sample_t(self, batch: int, device: torch.device) -> torch.Tensor:
        """Sample training times."""
        return torch.rand(batch, device=device).mul_(1.0 - self.t_eps).add_(self.t_eps)

    def grid(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Reverse solver grid."""
        assert num_steps >= 1, "num_steps >= 1"
        return torch.linspace(1.0 - self.t_eps, 0.0, num_steps + 1, device=device)

    def grid_for(self, num_steps: int, kind: str, device: torch.device) -> torch.Tensor:
        """Reverse solver grid by kind: 'u' uniform-in-t, 'lsnr' uniform-in-log-SNR, 'alsnr' its
        mirror (concentrates steps at high noise as strongly as 'lsnr' does at low noise), 'mix:α' λ-interpolation."""
        if kind == "u":
            return self.grid(num_steps, device)
        lam_hi = self.lam(torch.tensor(1.0 - self.t_eps, device=device))
        lam_lo = self.lam(torch.tensor(self.t_eps, device=device))
        lam_lsnr = torch.linspace(lam_hi.item(), lam_lo.item(), num_steps + 1, device=device)
        if kind == "lsnr":
            return self.inverse_lambda(lam_lsnr)
        if kind == "alsnr":  # t -> 1-t mirror of the 'lsnr' grid ([t_eps, 1-t_eps] is symmetric about 1/2)
            return 1.0 - self.inverse_lambda(lam_lsnr).flip(0)
        if kind.startswith("mix:"):
            alpha = float(kind[4:])
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"mix alpha must be in [0, 1], got {alpha}")
            lam_u = self.lam(self.grid(num_steps, device))
            return self.inverse_lambda((1.0 - alpha) * lam_u + alpha * lam_lsnr)
        raise ValueError(f"unknown grid kind: {kind!r}")

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous beta."""
        t = t.clamp(0.0, 1.0)
        if self.beta_schedule == "linear":
            return self.beta_min + (self.beta_max - self.beta_min) * t
        elif self.beta_schedule == "cosine":
            theta = self._theta(t)
            return math.pi * torch.tan(theta) / (1.0 + self.cosine_s)

    def abar(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha bar."""
        t = t.clamp(0.0, 1.0)
        if self.beta_schedule == "linear":
            log_abar = -self.beta_min * t
            log_abar = log_abar - 0.5 * (self.beta_max - self.beta_min) * t.square()
            return torch.exp(log_abar)
        elif self.beta_schedule == "cosine":
            theta = self._theta(t)
            abar = torch.cos(theta).square() / self._cos0
            return abar.clamp_min(0.0)

    def rabar(self, t: torch.Tensor) -> torch.Tensor:
        """Root alpha bar."""
        return torch.sqrt(self.abar(t).clamp_min(1e-12))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise scale."""
        return torch.sqrt((1.0 - self.abar(t)).clamp_min(0.0))

    def sigma_hat(self, t: torch.Tensor) -> torch.Tensor:
        """Half-line noise level."""
        rabar = self.rabar(t)
        sigma = self.sigma(t)
        return sigma / rabar.clamp_min(1e-12)

    def lam(self, t: torch.Tensor) -> torch.Tensor:
        """LogSNR / 2."""
        rabar = self.rabar(t)
        sigma = self.sigma(t)
        return torch.log(rabar.clamp_min(1e-12)) - torch.log(sigma.clamp_min(1e-12))

    def inverse_lambda(self, lam: torch.Tensor) -> torch.Tensor:
        """Invert logSNR / 2 to time."""
        abar = torch.sigmoid(2.0 * lam).clamp(1e-12, 1.0 - 1e-12)
        if self.beta_schedule == "linear":
            if math.isclose(self.beta_max, self.beta_min):
                t = -torch.log(abar) / self.beta_min
            else:
                beta_span = self.beta_max - self.beta_min
                disc = self.beta_min**2 - 2.0 * beta_span * torch.log(abar)
                t = (torch.sqrt(disc.clamp_min(0.0)) - self.beta_min) / beta_span
        elif self.beta_schedule == "cosine":
            theta = torch.arccos(torch.sqrt((abar * self._cos0).clamp(0.0, 1.0)))
            t = (2.0 * (1.0 + self.cosine_s) / math.pi) * theta - self.cosine_s
        return t.clamp(0.0, 1.0)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Sample x_t from x0."""
        rabar = self.rabar(t).view(-1, 1, 1, 1)
        sigma = self.sigma(t).view(-1, 1, 1, 1)
        return rabar * x0 + sigma * eps

    def x0_from_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x0 from eps."""
        rabar = self.rabar(t).view(-1, 1, 1, 1)
        sigma = self.sigma(t).view(-1, 1, 1, 1)
        return (x - sigma * eps) / rabar.clamp_min(1e-12)

    def _theta(self, t: torch.Tensor | float) -> torch.Tensor | float:
        return ((t + self.cosine_s) / (1.0 + self.cosine_s)) * (math.pi / 2)
