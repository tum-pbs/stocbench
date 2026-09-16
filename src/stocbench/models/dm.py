import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from ..utils.figures import fig_to_image
from .base import BaseGenerativeModel


class DiffusionModel(BaseGenerativeModel):
    """Denoising diffusion model (eps-prediction, VP schedule, pluggable sampler)."""

    def __init__(
        self,
        net: nn.Module,
        sched,
        sampler,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "sched", "sampler"])
        self._assign_init_args(locals())

    def predict_eps(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict eps.

        x: [B, C, H, W]
        t: [B]
        cond: [B, S*C, H, W]
        """
        return self.net(torch.cat([cond, x], dim=1), t)

    def predict_x0(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict x0."""
        return self.sched.x0_from_eps(x, t, self.predict_eps(x, t, cond))

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        """Eps-matching loss.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        cond, x0 = self._unpack(target, cond)

        t = self.sched.sample_t(x0.shape[0], x0.device)
        eps = torch.randn_like(x0)
        x = self.sched.q_sample(x0, t, eps)
        return F.mse_loss(self.predict_eps(x, t, cond), eps)

    def sample(self, cond: torch.Tensor, *, sampling_schedule: tuple[str, int] = ("u", 1), return_steps: bool = False):
        """Sample next frame (delegated to the configured sampler).

        cond: [B, C, H, W]
        """
        kind, num_steps = sampling_schedule
        out = self.sampler.sample(self, cond[:, None], num_steps=num_steps, schedule=kind, return_steps=return_steps)
        return (out[0][:, 0], out[1]) if return_steps else out[:, 0]

    def on_train_start(self):
        """Log the noise schedule to W&B."""
        if self.logger is None:
            return
        t = torch.linspace(self.sched.t_eps, 1.0 - self.sched.t_eps, 256)
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in ("beta", "abar", "rabar", "sigma", "lam"):
            ax.plot(t.numpy(), getattr(self.sched, name)(t).numpy(), label=name)
        ax.set_title(f"Noise Schedule: {getattr(self.sched, 'beta_schedule', type(self.sched).__name__)}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.logger.experiment.log({"model/noise_schedule": wandb.Image(fig_to_image(fig))})
        plt.close(fig)
