import torch


class DPMppSampler:
    """DPM++ 2M sampler."""

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
        """Sample with DPM++."""
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"

        B, S, C, H, W = cond.shape
        sched = model.sched
        ts = sched.grid_for(num_steps, schedule, cond.device)
        cond_flat = cond.view(B, S * C, H, W)
        t = ts[0].expand(B)
        rabar = sched.rabar(t).view(B, 1, 1, 1)
        z = torch.randn(B, model.net.out_channels, H, W, device=cond.device) / rabar
        steps = [] if return_steps else None

        x0_prev = None
        h_prev = None

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
                ratio = (sigma_next / sigma).view(B, 1, 1, 1)

                if x0_prev is None:
                    z = ratio * z + (1.0 - ratio) * x0
                else:
                    h = torch.log(sigma.clamp_min(1e-12)) - torch.log(sigma_next.clamp_min(1e-12))
                    r = h_prev / h
                    r = r.view(B, 1, 1, 1)
                    x0_mid = (1.0 + 0.5 / r) * x0 - 0.5 / r * x0_prev
                    z = ratio * z + (1.0 - ratio) * x0_mid
                    h_prev = h

                if x0_prev is None:
                    h_prev = torch.log(sigma.clamp_min(1e-12)) - torch.log(sigma_next.clamp_min(1e-12))
                x_next = sched.rabar(t_next).view(B, 1, 1, 1) * z

            x0_prev = x0

            if steps is not None:
                steps.append(x_next.detach())

        sample = z.unsqueeze(1)
        if steps is None:
            return sample
        return sample, torch.stack(steps, dim=0)
