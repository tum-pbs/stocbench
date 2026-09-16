import torch


class DDIM2Sampler:
    """Heun second-order DDIM sampler."""

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
        """Sample with Heun's DDIM."""
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"

        sched = model.sched
        ts = sched.grid_for(num_steps, schedule, cond.device)
        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        t = ts[0].expand(B)
        z = torch.randn(B, model.net.out_channels, H, W, device=cond.device) / sched.rabar(t).view(B, 1, 1, 1)
        steps = [] if return_steps else None

        for i in range(num_steps):
            t = ts[i].expand(B)
            sigma = sched.sigma_hat(t).view(B, 1, 1, 1)
            x0 = model.predict_x0(sched.rabar(t).view(B, 1, 1, 1) * z, t, cond_flat)

            if i == num_steps - 1:
                z = x0
                x_next = x0
            else:
                t_next = ts[i + 1].expand(B)
                sigma_next = sched.sigma_hat(t_next).view(B, 1, 1, 1)
                d = (z - x0) / sigma
                z_pred = z + (sigma_next - sigma) * d
                x0_next = model.predict_x0(sched.rabar(t_next).view(B, 1, 1, 1) * z_pred, t_next, cond_flat)
                d_next = (z_pred - x0_next) / sigma_next
                z = z + 0.5 * (sigma_next - sigma) * (d + d_next)
                x_next = sched.rabar(t_next).view(B, 1, 1, 1) * z

            if steps is not None:
                steps.append(x_next.detach())

        sample = z.unsqueeze(1)
        if steps is None:
            return sample
        return sample, torch.stack(steps, dim=0)
