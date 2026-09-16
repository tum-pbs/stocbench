import torch


class DDIMSampler:
    """DDIM/DDPM sampler."""

    def __init__(self, eta: float = 0.0) -> None:
        assert 0.0 <= eta <= 1.0, "0 <= eta <= 1"
        self.eta = eta

    @staticmethod
    def _split_sigma(
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if eta == 0.0:
            zero = torch.zeros_like(sigma)
            return sigma_next, zero

        # sigma_up is eta*sqrt(Var(z_next|z_t,x_0))
        sigma_up = eta * torch.sqrt(
            (
                sigma_next.square()
                * (sigma.square() - sigma_next.square()).clamp_min(0.0)
                / sigma.square().clamp_min(1e-12)
            ).clamp_min(0.0)
        )
        sigma_up = torch.minimum(sigma_next, sigma_up)
        sigma_down = torch.sqrt((sigma_next.square() - sigma_up.square()).clamp_min(0.0))
        return sigma_down, sigma_up

    @torch.no_grad()
    def sample(
        self,
        model,
        cond: torch.Tensor,
        num_steps: int,
        *,
        schedule: str = "u",
        return_steps: bool = False,
    ):
        """Sample with DDIM/DDPM."""
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"

        sched = model.sched
        ts = sched.grid_for(num_steps, schedule, cond.device)
        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        t = ts[0].expand(B)
        rabar = sched.rabar(t).view(B, 1, 1, 1)
        z = torch.randn(B, model.net.out_channels, H, W, device=cond.device) / rabar
        steps = [] if return_steps else None

        for i in range(num_steps):
            t = ts[i].expand(B)
            x = sched.rabar(t).view(B, 1, 1, 1) * z
            x0 = model.predict_x0(x, t, cond_flat)

            if i == num_steps - 1:
                z = x0
                x_next = x0
            else:
                t_next = ts[i + 1].expand(B)
                sigma = sched.sigma_hat(t)
                sigma_next = sched.sigma_hat(t_next)
                sigma_down, sigma_up = self._split_sigma(
                    sigma,
                    sigma_next,
                    self.eta,
                )
                sigma = sigma.view(B, 1, 1, 1)
                sigma_down = sigma_down.view(B, 1, 1, 1)
                sigma_up = sigma_up.view(B, 1, 1, 1)
                z = x0 + (sigma_down / sigma.clamp_min(1e-12)) * (z - x0)
                if self.eta > 0.0:
                    z = z + sigma_up * torch.randn_like(z)
                x_next = sched.rabar(t_next).view(B, 1, 1, 1) * z

            if steps is not None:
                steps.append(x_next.detach())

        sample = z.unsqueeze(1)
        if steps is None:
            return sample
        return sample, torch.stack(steps, dim=0)
