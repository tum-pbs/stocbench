import copy

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from ..utils import load_checkpoint
from ..utils.hf import resolve_hf_path
from .base import BaseGenerativeModel


class ADDFlowMatching(BaseGenerativeModel):
    """Adversarial diffusion distillation (ADD) of a flow matching teacher."""

    def __init__(
        self,
        teacher: DictConfig,
        discriminator: DictConfig,
        teacher_ckpt: str,
        hf: DictConfig | None = None,
        student_steps: int = 4,
        teacher_steps: int = 10,
        distill_weight: float = 0.5,
        gen_rate: int = 10,
        r1_gamma: float = 1.0,
        lr_g: float = 1e-5,
        lr_d: float = 2e-5,
        betas_g: tuple[float, float] = (0.9, 0.99),
        betas_d: tuple[float, float] = (0.5, 0.999),
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._assign_init_args(locals())
        self.automatic_optimization = False

        teacher_cfg = dict(teacher)
        teacher_cfg.pop("name", None)
        self.teacher = hydra.utils.instantiate(teacher_cfg)
        self._load_weights(self.teacher, teacher_ckpt)
        self.teacher.eval().requires_grad_(False)

        self.student = copy.deepcopy(self.teacher)
        self.student.train().requires_grad_(True)
        self.discriminator = hydra.utils.instantiate(discriminator, original_unet=self.student.net)

    @property
    def net(self):
        return self.student.net

    def _load_weights(self, model, path: str) -> None:
        if self.hf is not None:
            path = resolve_hf_path(path, self.hf)
        ckpt = load_checkpoint(path)
        model.load_state_dict(ckpt.get("state_dict", ckpt))

    def _solve_euler(self, model, x: torch.Tensor, cond_flat: torch.Tensor, steps: int, return_steps: bool = False):
        """Euler integration of the model's flow from noise to prediction."""
        dt = 1.0 / steps
        B = x.shape[0]
        traj = []
        for i in range(steps):
            t = torch.full((B,), i * dt, device=x.device)
            v_pred = model.net(torch.cat([cond_flat, x], dim=1), t)[:, -x.shape[1] :]
            x = x + v_pred * dt
            traj.append(x)
        return (x, torch.stack(traj)) if return_steps else x

    def student_forward(self, cond: torch.Tensor, x_init: torch.Tensor | None = None, return_steps: bool = False):
        """One student pass (usually few-step).

        cond: [B, S, C, H, W]
        x_init: [B, C_out, H, W] optional noise
        """
        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        if x_init is None:
            x_init = torch.randn(B, self.net.out_channels, H, W, device=cond.device)
        return self._solve_euler(self.student, x_init, cond_flat, self.student_steps, return_steps=return_steps)

    def training_step(self, batch, batch_idx):
        cond, target = (t.detach() for t in batch)
        cond_flat, x_real = self._unpack(target, cond)
        B = x_real.shape[0]
        opt_g, opt_d = self.optimizers()
        t_zero = torch.zeros((B,), device=self.device, dtype=torch.long)

        def disc(x):
            return self.discriminator(torch.cat([cond_flat, x], dim=1), t_zero)

        # Sample the student once; keep its graph only if G uses it this step.
        gen_step = (batch_idx + 1) % self.gen_rate == 0
        noise = torch.randn_like(x_real)
        with torch.set_grad_enabled(gen_step):
            x_fake = self.student_forward(cond, x_init=noise)

        # ---- discriminator ----
        self.toggle_optimizer(opt_d)
        x_real.requires_grad = True
        logits_real, logits_fake = disc(x_real), disc(x_fake.detach())

        l_real = l_fake = 0.0
        for k in logits_real:
            self.log(f"scores/{k}/real", logits_real[k].mean(), on_step=True)
            self.log(f"scores/{k}/fake", logits_fake[k].mean(), on_step=True)
            l_real += F.relu(1.0 - logits_real[k]).mean()
            l_fake += F.relu(1.0 + logits_fake[k]).mean()

        grad_real = torch.autograd.grad(
            sum(l.sum() for l in logits_real.values()), x_real, create_graph=True, retain_graph=True
        )[0]
        r1 = grad_real.pow(2).view(B, -1).sum(1).mean()
        loss_d = l_real + l_fake + (self.r1_gamma / 2.0) * r1

        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
        self.untoggle_optimizer(opt_d)
        x_real.requires_grad = False
        self.log_dict({"train/d_loss": loss_d, "train/d_r1": r1}, on_step=True, on_epoch=True)

        if not gen_step:
            return

        # ---- student: adversarial + teacher distillation ----
        self.toggle_optimizer(opt_g)
        loss_adv = -sum(s.mean() for s in disc(x_fake).values())

        with torch.no_grad():
            x_teacher = self._solve_euler(self.teacher, noise, cond_flat, self.teacher_steps)
        t = torch.randint(0, self.student_steps, (B,), device=self.device).float() / self.student_steps
        x_t = (1 - t.view(B, 1, 1, 1)) * noise + t.view(B, 1, 1, 1) * x_teacher
        v_pred = self.student.net(torch.cat([cond_flat, x_t], dim=1), t)[:, -x_t.shape[1] :]
        loss_distill = F.mse_loss(v_pred, x_teacher - noise)

        loss_g = (1 - self.distill_weight) * loss_adv + self.distill_weight * loss_distill
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()
        self.untoggle_optimizer(opt_g)
        self.log_dict(
            {"train/g_loss": loss_g, "train/g_adv": loss_adv, "train/g_distill": loss_distill},
            on_step=True,
            on_epoch=True,
        )

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, *, sampling_schedule: tuple[str, int] = ("u", 1), return_steps: bool = False):
        """Sample next frame with the few-step student.

        cond: [B, C, H, W]
        """
        return self.student_forward(cond[:, None], return_steps=return_steps)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.student.parameters(), lr=self.lr_g, betas=tuple(self.betas_g), weight_decay=self.weight_decay
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(), lr=self.lr_d, betas=tuple(self.betas_d), weight_decay=self.weight_decay
        )
        return [opt_g, opt_d]
