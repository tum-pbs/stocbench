from __future__ import annotations

import torch
from torchmetrics import Metric as _TMMetric


def _cat(xs, device, dtype=torch.float32):
    if isinstance(xs, torch.Tensor):
        return xs.to(device=device, dtype=dtype)
    return torch.cat(xs) if xs else torch.empty(0, device=device, dtype=dtype)


def radial_enstrophy(field: torch.Tensor) -> torch.Tensor:
    *batch, height, width = field.shape
    field = field.reshape(-1, height, width)
    spectrum = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
    power = (spectrum.real.square() + spectrum.imag.square()) / (height * width)
    yy = torch.arange(height, device=field.device) - height // 2
    xx = torch.arange(width, device=field.device) - width // 2
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    bins = torch.floor(torch.sqrt(grid_y.float().square() + grid_x.float().square())).long()
    spec = power.new_zeros(field.shape[0], int(bins.max()) + 1)
    spec.scatter_add_(1, bins.reshape(1, -1).expand(field.shape[0], -1), power.flatten(1))
    return spec.reshape(*batch, -1)


def reference_steps(reference: torch.Tensor, steps: int) -> torch.Tensor:
    return reference.expand(steps, -1) if reference.ndim == 1 else reference[:steps]


def pointwise_energy_distance(pred: torch.Tensor, ref: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    _, channels, height, width = pred.shape
    if channels == 1:
        pred = pred[:, 0].movedim(0, -1).reshape(-1, pred.shape[0])
        ref = ref[:, 0].movedim(0, -1).reshape(-1, ref.shape[0]).sort(1).values.contiguous()
        pred_sorted = pred.sort(1).values
        n, m = pred.shape[1], ref.shape[1]
        w_n = 2 * torch.arange(n, device=pred.device, dtype=pred.dtype) - n + 1
        w_m = 2 * torch.arange(m, device=pred.device, dtype=pred.dtype) - m + 1
        dxx = 2 * (pred_sorted * w_n).sum(1) / (n * n)
        dyy = 2 * (ref * w_m).sum(1) / (m * m)
        idx = torch.searchsorted(ref, pred.contiguous()).to(pred.dtype)
        csum = torch.cat([ref.new_zeros(ref.shape[0], 1), ref.cumsum(1)], 1)
        left = csum.gather(1, idx.long())
        dxy = (pred * idx - left + csum[:, -1:] - left - pred * (m - idx)).sum(1) / (n * m)
        return (2 * dxy - dxx - dyy).clamp_min(0).sqrt().reshape(height, width)

    pred = pred.permute(2, 3, 0, 1).reshape(-1, pred.shape[0], channels)
    ref = ref.permute(2, 3, 0, 1).reshape(-1, ref.shape[0], channels)
    return torch.cat(
        [
            (2.0 * torch.cdist(x, y).mean((1, 2)) - torch.cdist(x, x).mean((1, 2)) - torch.cdist(y, y).mean((1, 2)))
            .clamp_min(0)
            .sqrt()
            for x, y in zip(pred.split(chunk_size), ref.split(chunk_size))
        ]
    ).reshape(height, width)


class StatMetric(_TMMetric):
    """Stats metric over the generated ensemble, with optional per-sampler-step recording.

    Subclasses implement `_measure(batch, samples) -> [N]`, vectorised over the leading
    sample axis (N=1 for the final sample, N=K for the sampler trajectory). With
    `record_steps` the callback feeds the trajectory and binds the consistency-time `grid`,
    so the metric is also saved at every intermediary step (`sched`) for that grid.
    """

    full_state_update = False
    mode = "stats"

    def __init__(self, record_steps: bool = False) -> None:
        super().__init__()
        self.record_steps = record_steps
        self.grid: torch.Tensor | None = None  # consistency-time grid, bound by the callback
        self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("seeds", default=[], dist_reduce_fx="cat")
        self.add_state("sched", default=[], dist_reduce_fx="cat")

    def update(self, batch: dict[str, torch.Tensor], samples: torch.Tensor, traj: torch.Tensor | None = None) -> None:
        self.values.append(self._measure(batch, samples))
        self.seeds.append(batch["_seed"].to(self.device).long().reshape(-1))
        if self.record_steps and traj is not None:
            self.sched.append(self._measure(batch, traj).unsqueeze(0))  # [1, K]: one row per IC

    def compute(self) -> dict[str, torch.Tensor]:
        values = _cat(self.values, self.device)
        seeds = _cat(self.seeds, self.device, torch.long)
        order = seeds.argsort() if seeds.numel() else torch.empty(0, device=seeds.device, dtype=torch.long)
        out = {
            "value": values.mean(),
            "per_initial_condition": values[order],
            "seeds": seeds[order],
            **self._extra(order),
        }
        if self.grid is not None and len(self.sched):  # sched syncs to a tensor, so avoid bool()
            out["sched"] = _cat(self.sched, self.device).mean(0)  # average the per-step curve over ICs
            out["grid"] = self.grid.to(self.device)
        return out

    def _extra(self, order: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}


class EnsembleStatError(StatMetric):
    _REDUCE = {"mean": lambda s: s.mean(1), "std": lambda s: s.std(1, unbiased=False)}

    def __init__(self, stat: str, relative: bool = True, record_steps: bool = False) -> None:
        super().__init__(record_steps)
        if stat not in self._REDUCE:
            raise ValueError(f"unknown stat: {stat!r}")
        self.stat = stat
        self.relative = relative
        self.name = f"{stat}_error"
        self.prefix = f"{stat}_"
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("refs", default=[], dist_reduce_fx="cat")

    def _measure(self, batch: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:  # x: [N, ens, C, H, W] -> [N]
        pred = self._REDUCE[self.stat](x.detach().float())
        ref = batch[self.stat].to(pred.device).float()
        diff = torch.linalg.vector_norm((pred - ref).flatten(1), dim=1)
        return diff / torch.linalg.vector_norm(ref.flatten(1), dim=1).clamp_min(1e-8) if self.relative else diff

    def update(self, batch: dict[str, torch.Tensor], samples: torch.Tensor, traj: torch.Tensor | None = None) -> None:
        super().update(batch, samples, traj)
        self.preds.append(self._REDUCE[self.stat](samples.detach().float()))
        self.refs.append(batch[self.stat].to(self.device).float())

    def _extra(self, order: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"prediction": _cat(self.preds, self.device)[order], "reference": _cat(self.refs, self.device)[order]}


class EnergyDistance(StatMetric):
    name = "energy_dist"
    prefix = "energy_"

    def _measure(self, batch: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:  # x: [N, ens, C, H, W] -> [N]
        raw = batch["raw"].to(x.device).float()[0]  # [ens_raw, C, H, W]: this IC's reference ensemble
        return torch.stack([pointwise_energy_distance(s.detach().float(), raw).mean() for s in x])


class EnstrophyError(_TMMetric):
    full_state_update = False
    mode = "rollout"
    name = "enstrophy_error"
    prefix = "enstr_"
    record_steps = False

    def __init__(self, record_steps: bool = False) -> None:
        super().__init__()
        self.record_steps = record_steps
        self.reference_spectrum: torch.Tensor | None = None
        self.add_state("steps", default=[], dist_reduce_fx="cat")
        self.add_state("spectrum_sums", default=[], dist_reduce_fx="cat")
        self.add_state("counts", default=[], dist_reduce_fx="cat")
        self.add_state("sampler_step_sums", default=[], dist_reduce_fx="cat")

    def bind(self, *, reference_spectrum: torch.Tensor) -> None:
        self.reference_spectrum = reference_spectrum.float()

    def update(
        self, step: int, pred: torch.Tensor, target: torch.Tensor | None = None, traj: torch.Tensor | None = None
    ) -> None:
        spectrum = radial_enstrophy(pred[:, 0].detach().float())
        self.steps.append(torch.full((1,), int(step), device=pred.device, dtype=torch.long))
        self.spectrum_sums.append(spectrum.sum(0, keepdim=True))
        self.counts.append(torch.tensor([pred.shape[0]], device=pred.device, dtype=torch.float32))
        if self.record_steps and traj is not None:
            K, B, _, H, W = traj.shape
            spec = radial_enstrophy(traj[:, :, 0].detach().float().reshape(K * B, H, W)).reshape(K, B, -1)
            self.sampler_step_sums.append(spec.sum(1).unsqueeze(0))

    def compute(self) -> dict[str, torch.Tensor]:
        if self.reference_spectrum is None:
            raise RuntimeError("EnstrophyError.bind(reference_spectrum=...) must be called before compute()")
        steps = _cat(self.steps, self.device, torch.long)
        sums = _cat(self.spectrum_sums, self.device)
        counts = _cat(self.counts, self.device)
        unique = steps.unique(sorted=True)
        model = torch.stack([sums[steps == s].sum(0) / counts[steps == s].sum() for s in unique])
        ref = reference_steps(self.reference_spectrum.to(model.device), model.shape[0])
        per_step = (model - ref)[..., 1:].abs().mean(-1)
        payload = {
            "value": per_step.mean(),
            "per_rollout_step": per_step,
            "model_spectrum": model,
            "reference_spectrum": ref,
            "wavenumbers": torch.arange(model.shape[-1], device=model.device),
        }
        sampler_sums = _cat(self.sampler_step_sums, self.device)
        if sampler_sums.numel():
            payload["sampler_step_spectrum"] = torch.stack(
                [sampler_sums[steps == s].sum(0) / counts[steps == s].sum() for s in unique]
            )
        return payload


class RolloutRMSE(_TMMetric):
    full_state_update = False
    mode = "rollout"
    record_steps = False
    name = "rollout_rmse"
    prefix = "rmse_"

    def __init__(self) -> None:
        super().__init__()
        self.add_state("steps", default=[], dist_reduce_fx="cat")
        self.add_state("sq_sums", default=[], dist_reduce_fx="cat")
        self.add_state("counts", default=[], dist_reduce_fx="cat")

    def update(
        self, step: int, pred: torch.Tensor, target: torch.Tensor | None = None, traj: torch.Tensor | None = None
    ) -> None:
        err = (pred.detach().float() - target.float()).flatten(1)
        self.steps.append(torch.full((1,), int(step), device=pred.device, dtype=torch.long))
        self.sq_sums.append(err.pow(2).mean(1).sum().reshape(1))
        self.counts.append(torch.tensor([pred.shape[0]], device=pred.device, dtype=torch.float32))

    def compute(self) -> dict[str, torch.Tensor]:
        steps = _cat(self.steps, self.device, torch.long)
        sqs = _cat(self.sq_sums, self.device)
        counts = _cat(self.counts, self.device)
        unique = steps.unique(sorted=True)
        per_step = torch.stack([(sqs[steps == s].sum() / counts[steps == s].sum()).sqrt() for s in unique])
        return {
            "value": per_step.mean(),
            "per_rollout_step": per_step,
        }
