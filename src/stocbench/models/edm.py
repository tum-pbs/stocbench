import torch
import torch.nn as nn

from .base import BaseGenerativeModel


class EDMModel(BaseGenerativeModel):
    """Elucidated diffusion model (EDM preconditioning, Karras sigma schedule)."""

    def __init__(
        self,
        net: nn.Module,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 1.0,
        rho: float = 7.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        use_heun: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self._assign_init_args(locals())

    # --- EDM preconditioning ---

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return 0.25 * torch.log(sigma)

    def _sigma_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(num_steps + 1, device=device, dtype=torch.float32)
        smax, smin = self.sigma_max ** (1 / self.rho), self.sigma_min ** (1 / self.rho)
        sigmas = (smax + idx / num_steps * (smin - smax)) ** self.rho
        sigmas[-1] = 0.0
        return sigmas

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Denoise x at noise level sigma.

        x: [B, C, H, W]
        sigma: [B]
        cond: [B, S*C, H, W]
        """
        x_in = torch.cat([cond, x * self.c_in(sigma)[:, None, None, None]], dim=1)
        f_theta = self.net(x_in, self.c_noise(sigma))[:, -x.shape[1] :]
        return x * self.c_skip(sigma)[:, None, None, None] + f_theta * self.c_out(sigma)[:, None, None, None]

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        """EDM weighted denoising loss.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        cond, x0 = self._unpack(target, cond)

        sigma = torch.exp(torch.randn(x0.shape[0], device=x0.device) * self.p_std + self.p_mean)
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)
        x = x0 + sigma[:, None, None, None] * torch.randn_like(x0)
        d = self.denoise(x, sigma, cond)

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        return ((d - x0).square().mean(dim=(1, 2, 3)) * weight).mean()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, *, sampling_schedule: tuple[str, int] = ("u", 1), return_steps: bool = False):
        """Sample next frame (Euler over the sigma schedule, optional Heun correction).

        cond: [B, C, H, W]
        """
        kind, num_steps = sampling_schedule
        if kind != "u":
            raise ValueError(f"{type(self).__name__} only supports 'u' schedules")

        B, _, H, W = cond.shape
        sigmas = self._sigma_schedule(num_steps, cond.device)
        x = torch.randn(B, self.net.out_channels, H, W, device=cond.device) * sigmas[0]

        steps = []
        for i in range(num_steps):
            sigma, sigma_next = sigmas[i].expand(B), sigmas[i + 1].expand(B)
            d = (x - self.denoise(x, sigma, cond)) / sigma[:, None, None, None]
            x_next = x + (sigma_next - sigma)[:, None, None, None] * d
            if self.use_heun and sigma_next[0] > 0:
                d_next = (x_next - self.denoise(x_next, sigma_next, cond)) / sigma_next[:, None, None, None]
                x_next = x + (sigma_next - sigma)[:, None, None, None] * (0.5 * d + 0.5 * d_next)
            x = x_next
            steps.append(x)

        return (x, torch.stack(steps)) if return_steps else x
