import torch


class ExpSampler:
    """Second-order exponential-integrator sampler (DPM-Solver-2 family).

    Time discretization is selected at call time via the schedule kind:
    'u' = uniform-in-t, 'lsnr' = uniform-in-log-SNR (λ), 'alsnr' = mirrored 'lsnr'.
    """

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
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"
        B, S, C, H, W = cond.shape
        sched = model.sched
        ts = sched.grid_for(num_steps, schedule, cond.device)
        cond_flat = cond.view(B, S * C, H, W)
        # target may be narrower than the conditioning (e.g. det INCNS conditions on
        # vorticity + an extra channel but only predicts vorticity), so size the noise
        # by the net's output channels, not the conditioning channels.
        x = torch.randn(B, model.net.out_channels, H, W, device=cond.device)
        steps = [] if return_steps else None

        for i in range(num_steps):
            s = ts[i].expand(B)
            eps = model.predict_eps(x, s, cond_flat)
            x0 = sched.x0_from_eps(x, s, eps)

            if i == num_steps - 1:
                x = x0
                x_next = x0
            else:
                t = ts[i + 1].expand(B)
                lam_s, lam_t = sched.lam(s), sched.lam(t)
                h = lam_t - lam_s
                s_mid = sched.inverse_lambda(lam_s + 0.5 * h)

                alpha_s = sched.rabar(s).view(B, 1, 1, 1)
                alpha_mid = sched.rabar(s_mid).view(B, 1, 1, 1)
                alpha_t = sched.rabar(t).view(B, 1, 1, 1)
                sigma_mid = sched.sigma(s_mid).view(B, 1, 1, 1)
                sigma_t = sched.sigma(t).view(B, 1, 1, 1)
                h = h.view(B, 1, 1, 1)

                x_mid = (alpha_mid / alpha_s) * x - sigma_mid * torch.expm1(0.5 * h) * eps
                eps_mid = model.predict_eps(x_mid, s_mid, cond_flat)
                x = (alpha_t / alpha_s) * x - sigma_t * torch.expm1(h) * eps_mid
                x_next = x

            if steps is not None:
                steps.append(x_next.detach())

        sample = x.unsqueeze(1)
        if steps is None:
            return sample
        return sample, torch.stack(steps, dim=0)
