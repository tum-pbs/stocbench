import copy

import hydra
import torch
from omegaconf import DictConfig

from ..utils import load_checkpoint
from ..utils.hf import resolve_hf_path
from .base import BaseGenerativeModel


class ConsistencyDistillationModel(BaseGenerativeModel):
    """Consistency distillation of an EDM teacher (CM-shifted preconditioning)."""

    def __init__(
        self,
        net: DictConfig,
        teacher: DictConfig,
        teacher_ckpt: str,
        hf: DictConfig | None = None,
        ema_rate: float = 0.999,
        student_steps: int = 50,
        snr_weighting: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        net = hydra.utils.instantiate(net)
        self.save_hyperparameters(ignore=["net"])
        self._assign_init_args(locals())
        assert student_steps >= 2, "student_steps >= 2"

        teacher_cfg = dict(teacher)
        teacher_cfg.pop("name", None)
        self.teacher = hydra.utils.instantiate(teacher_cfg)
        self._load_weights(self.teacher, teacher_ckpt)
        self.teacher.eval().requires_grad_(False)
        self.net.load_state_dict(self.teacher.net.state_dict())
        self.target_model = copy.deepcopy(self.net).eval().requires_grad_(False)

        self.sigma_min, self.sigma_max = float(self.teacher.sigma_min), float(self.teacher.sigma_max)
        self.sigma_data, self.rho = float(self.teacher.sigma_data), float(self.teacher.rho)

        # student sigma grid (Karras spacing over [sigma_max, sigma_min])
        idx = torch.arange(student_steps, dtype=torch.float32)
        smax, smin = self.sigma_max ** (1 / self.rho), self.sigma_min ** (1 / self.rho)
        self.register_buffer("sigmas", (smax + idx / (student_steps - 1) * (smin - smax)) ** self.rho)

    def _load_weights(self, model, path: str) -> None:
        if self.hf is not None:
            path = resolve_hf_path(path, self.hf)
        ckpt = load_checkpoint(path)
        model.load_state_dict(ckpt.get("state_dict", ckpt))

    @torch.no_grad()
    def _update_target_model(self) -> None:
        for tp, p in zip(self.target_model.parameters(), self.net.parameters()):
            tp.mul_(self.ema_rate).add_(p, alpha=1 - self.ema_rate)

    # --- CM preconditioning (boundary condition at sigma_min) ---

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data * (sigma - self.sigma_min) / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return 0.25 * torch.log(sigma)

    def consistency_function(
        self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor, use_target: bool = False
    ) -> torch.Tensor:
        """Consistency prediction at noise level sigma.

        x: [B, C, H, W]
        sigma: [B]
        cond: [B, S*C, H, W]
        """
        x_in = torch.cat([cond, x * self.c_in(sigma)[:, None, None, None]], dim=1)
        model = self.target_model if use_target else self.net
        f_theta = model(x_in, self.c_noise(sigma))[:, -x.shape[1] :]
        return x * self.c_skip(sigma)[:, None, None, None] + f_theta * self.c_out(sigma)[:, None, None, None]

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        """Consistency loss between adjacent points of the student sigma grid.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        cond, x0 = self._unpack(target, cond)
        B = x0.shape[0]

        t_idx = torch.randint(0, self.student_steps - 1, (B,), device=x0.device)
        sigma_t = self.sigmas[t_idx].to(dtype=x0.dtype)
        sigma_next = self.sigmas[t_idx + 1].to(dtype=x0.dtype)
        x = x0 + sigma_t[:, None, None, None] * torch.randn_like(x0)

        with torch.no_grad():
            x0_pred = self.teacher.denoise(x, sigma_t, cond)
            eps_pred = (x - x0_pred) / sigma_t[:, None, None, None]
            x_next = x0_pred + sigma_next[:, None, None, None] * eps_pred
            target_pred = self.consistency_function(x_next, sigma_next, cond, use_target=True)

        student_pred = self.consistency_function(x, sigma_t, cond)

        if self.snr_weighting:
            weight = (sigma_t**2 + self.sigma_data**2) / (sigma_t * self.sigma_data) ** 2
            return ((student_pred - target_pred).square().mean(dim=(1, 2, 3)) * weight).mean()
        return (student_pred - target_pred).square().mean()

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        *,
        sampling_schedule: tuple[str, int] = ("u", 1),
        use_ema: bool = True,
        return_steps: bool = False,
    ):
        """Sample next frame by iterated consistency steps.

        cond: [B, C, H, W]
        """
        kind, num_steps = sampling_schedule
        B, _, H, W = cond.shape
        indices = self.sample_steps(num_steps, schedule=kind, device=cond.device)
        sigmas = torch.cat([self.sigmas[indices].to(cond.device), self.sigmas.new_zeros(1, device=cond.device)])
        x = torch.randn(B, self.net.out_channels, H, W, device=cond.device, dtype=cond.dtype) * sigmas[0]

        steps = []
        for i in range(len(indices)):
            x = self.consistency_function(x, sigmas[i].expand(B), cond, use_target=use_ema)
            steps.append(x)
            if i < len(indices) - 1:
                noise_std = torch.sqrt((sigmas[i + 1] ** 2 - self.sigma_min**2).clamp_min(0.0))
                x = x + noise_std * torch.randn_like(x)

        return (x, torch.stack(steps)) if return_steps else x

    def sample_steps(
        self, num_steps: int | list[int], schedule: str = "u", device: torch.device | None = None
    ) -> torch.Tensor:
        """Sigma-grid indices for a sampling schedule ('at' passes explicit indices)."""
        total = self.student_steps - 1
        if schedule == "at":
            return torch.as_tensor(num_steps, device=device)
        if schedule == "u":
            positions = torch.linspace(0, total, num_steps + 1, device=device)[:-1]
        elif schedule == "l":
            positions = (
                torch.logspace(
                    0, torch.log10(torch.tensor(float(total + 1), device=device)), num_steps + 1, device=device
                )[:-1]
                - 1
            )
        else:
            raise ValueError(f"unknown schedule: {schedule}")
        return positions.round().clamp(0, total).to(torch.long)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_target_model()
