"""Figure drivers for `stocbench plot`; styling comes from configs/plotting/style.yaml."""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from cycler import cycler
from matplotlib import colors
from matplotlib.ticker import MaxNLocator, MultipleLocator

with contextlib.suppress(ModuleNotFoundError):
    import cmcrameri.cm  # noqa: F401  # registers the cmc.* scientific colormaps (e.g. cmc.vanimo)

from . import parse_artifact_name

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
STYLE = yaml.safe_load(
    (CONFIGS / "plotting/style.yaml").read_text(encoding="utf-8")
)  # figsize/alpha/cmaps; rc applied lazily by _apply_style
LEGEND_H = (
    0.5  # inches reserved for the top legend strip, kept constant so legend-to-plot spacing matches across layouts
)


@dataclass
class Run:
    path: Path
    label: str
    color: str | None  # None for runs absent from the registry -> matplotlib's palette cycle
    linestyle: str = "solid"
    marker: str | None = None
    nfe: int = 1  # model evals per solver step; >1 for higher-order solvers (scales the vs-NFE x axis)

    @property
    def style(self) -> dict:  # transparent line; markers are opt-in (added only by the vs-steps plots)
        line = colors.to_rgba(self.color, STYLE["line_alpha"]) if self.color else self.color
        return dict(color=line, linestyle=self.linestyle, label=self.label)

    @property
    def artifacts(self) -> Path:  # find artifacts/ at any depth, so results/<model> works
        return next(self.path.glob("**/artifacts"), self.path / "artifacts")


def plot_runs(
    paths,
    *,
    output_dir=None,
    style=CONFIGS / "plotting/style.yaml",
    runs_config=CONFIGS / "plotting/runs.yaml",
    plots_config=CONFIGS / "plotting/plots/stocbench/stoc.yaml",
) -> Path:
    _apply_style(style)
    registry = yaml.safe_load(Path(runs_config).read_text(encoding="utf-8"))["runs"]
    cfg = yaml.safe_load(
        Path(plots_config).read_text(encoding="utf-8")
    )  # explicit utf-8: config comments break under the C locale of compute nodes otherwise
    prefix = cfg.get("name_prefix", "")  # e.g. "stoc_" / "det_", so figure names carry the experiment
    overrides = cfg.get("runs", {})  # per-experiment registry tweaks, e.g. a label that only makes sense for this eval
    runs = []
    for path in map(Path, paths):
        s = {**registry.get(path.name, {}), **overrides.get(path.name, {})}
        runs.append(
            Run(
                path,
                s.get("label", path.name),
                s.get("color"),
                s.get("linestyle", "solid"),
                s.get("marker"),
                s.get("nfe", 1),
            )
        )
    out = Path(output_dir) if output_dir else (runs[0].path / "plots" if len(runs) == 1 else Path("plots"))

    # Each figure is produced only if its key is present in the plots config, so a config
    # can request just the plots relevant to its eval (e.g. det omits the spectral ones).
    figures = {
        "std_norm": plot_std_norm,
        "rmse_rollout": plot_rmse_rollout,
        "enstr_spec": plot_enstr_spec,
        "enstr_rollout": plot_enstr_rollout,
        "enstr_ratio": plot_enstr_ratio,
        "enstr_flow": plot_enstr_flow,
        "seed_fields": plot_seed_fields,
    }
    for name, spec in cfg.items():
        # a `metrics:` list makes a vs-steps panel grid under any figure name
        plot = plot_metrics_steps if isinstance(spec, dict) and "metrics" in spec else figures.get(name)
        if plot and (fig := plot(runs, spec)) is not None:
            _save(fig, out / f"{prefix}{name}" / f"{prefix}{name}")

    for run in runs:  # per-run figures, saved under each run's own plots/
        if "schedule" in cfg and (fig := plot_schedule(run, cfg["schedule"])) is not None:
            _save(fig, run.path / "plots" / f"{prefix}schedule" / f"{prefix}schedule")
        if "stat_grid" in cfg:
            for stat in ("mean", "std"):
                for art in sorted(run.artifacts.glob(f"{stat}_error_*.npz")):
                    with np.load(art) as d:
                        fig = plot_stat_grid(
                            d[f"{stat}_prediction"],
                            d[f"{stat}_reference"],
                            d[f"{stat}_seeds"],
                            stat,
                            cfg["stat_grid"]["cols"],
                            cfg["stat_grid"].get("max_seeds"),
                        )
                    _save(fig, run.path / "plots" / f"{prefix}stat_grid" / art.stem)
    return out


# --- shared pieces ---


def _pick(run, key, steps=None):
    """The run's `<key>_*.npz` at the largest inference-step budget <= `steps`, else its smallest one."""
    arts = {parse_artifact_name(p)[1].steps: p for p in run.artifacts.glob(f"{key}_*.npz")}
    usable = [s for s in sorted(arts) if steps is not None and s <= steps]
    return arts[usable[-1]] if usable else arts[min(arts)] if arts else None


def _load(run, key, steps, *names):
    """Arrays `names` from the run's `<key>` artifact at the requested budget; None entries for absent keys."""
    if (art := _pick(run, key, steps)) is None:
        return None
    with np.load(art) as d:
        return [np.asarray(d[n]) if n in d.files else None for n in names]


def _series(run, key, value):
    """Sorted (NFE, values) over the run's `<key>_*.npz` artifacts; `value` maps an open npz to a float."""
    points = []
    for p in run.artifacts.glob(f"{key}_*.npz"):
        with np.load(p) as d:
            n = parse_artifact_name(p)[1].steps
            points.append((run.nfe * n - (run.nfe - 1), value(d)))  # the final step is a plain x0 readout -> 1 eval
    return tuple(zip(*sorted(points))) if points else ((), ())


def _steps_line(ax, run, steps, values):
    """One vs-inference-steps line; fixed-step methods (single artifact, e.g. CD) become a horizontal baseline."""
    if len(steps) == 1:
        return ax.axhline(
            values[0], color=run.color, linestyle=run.linestyle, label=run.label, alpha=STYLE["line_alpha"]
        )
    (line,) = ax.plot(steps, values, **run.style, marker=run.marker, markerfacecolor=run.color, markeredgewidth=0)
    return line


def _spec(a):  # spectrum at the final rollout step when stored per-step
    return a[-1] if a.ndim == 2 else a


def _letter(i, label="") -> str:
    """Panel-letter prefix for multi-panel combined figures: '(a) Label'."""
    return f"({chr(97 + i)}) {label}".strip()


def _legend_top(fig, handles: dict, grid_h) -> None:
    """Shared legend strip above the axes; the rect keeps the legend-to-plot gap identical across layouts."""
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=min(len(handles), 4))
    fig.tight_layout(rect=(0, 0, 1, grid_h / (grid_h + LEGEND_H)))


def _legend(fig, ax, handles: dict, grid_h, legend) -> None:
    """`legend: box` (or a matplotlib loc) draws an in-axes legend; absent: a strip above the axes."""
    if not legend:
        return _legend_top(fig, handles, grid_h)
    _legend_box(ax, legend, handles)
    fig.tight_layout()


def _legend_box(ax, legend="box", handles: dict | None = None) -> None:
    """In-axes legend; `legend` may name a matplotlib loc, overriding the style default."""
    opts = {**STYLE["legend_box"], **({} if legend == "box" else {"loc": legend})}
    ax.legend(*([handles.values(), handles.keys()] if handles else []), **opts)


def _inset(ax, cfg) -> None:
    """Zoom inset (`bounds`/`xlim`/`ylim`) copying the parent's lines and axis scales."""
    axins = ax.inset_axes(cfg["bounds"])
    for line in ax.get_lines():
        if line.get_transform() is ax.transData:
            axins.plot(
                *line.get_data(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                marker=line.get_marker(),
                markerfacecolor=line.get_markerfacecolor(),
                markeredgewidth=0,
                alpha=STYLE["line_alpha"],
            )
        else:  # axhlines (ground truth, fixed-step baselines) store x in axes fractions, so redraw them as such
            axins.axhline(
                line.get_ydata()[0], color=line.get_color(), linestyle=line.get_linestyle(), alpha=STYLE["line_alpha"]
            )
    axins.set(xscale=ax.get_xscale(), yscale=ax.get_yscale(), xlim=cfg["xlim"], ylim=cfg["ylim"], xticks=[], yticks=[])
    ax.indicate_inset_zoom(axins, edgecolor="0.35")


def _axes(ax, spec: dict, **defaults) -> None:
    """ax.set(...) from the figure's `axes:` config, with defaults for keys it omits.
    Any figure thus supports yscale/ylim overrides; a log y-scale drops the default zero floor."""
    if spec.get("yscale") == "log" and "ylim" not in spec and "ylim" in defaults:
        defaults["ylim"] = (None, defaults["ylim"][1])
    ax.set(**{**defaults, **spec})


# --- one function per figure ---


def plot_metrics_steps(runs, cfg) -> plt.Figure:
    """Grid of scalar metrics vs inference steps, one panel per configured metric."""
    metrics = cfg["metrics"]
    nrows, ncols = math.ceil(len(metrics) / 2), min(len(metrics), 2)
    gw, gh = STYLE["figsize"]["grid"]
    w, grid_h = STYLE["figsize"]["single"] if len(metrics) == 1 else (gw * ncols / 2, gh * nrows / 2)
    box = cfg.get("legend")
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, grid_h + (0 if box else LEGEND_H)), squeeze=False)
    handles = {}
    for i, (ax, m) in enumerate(zip(axes.flat, metrics)):
        for run in runs:
            steps, values = _series(run, m["key"], lambda d: float(d[next(k for k in d.files if k.endswith("value"))]))
            if steps:
                handles.setdefault(run.label, _steps_line(ax, run, steps, values))
        title = _letter(i, m.get("title") or "") if len(metrics) > 1 else m.get("title")
        yscale = m.get("yscale", cfg.get("yscale", "linear"))  # per-metric override, else the figure-level setting
        ax.set(title=title, xlabel=cfg["xlabel"], ylabel=m["ylabel"], xlim=cfg.get("xlim", (0, None)), yscale=yscale)
        ax.set_ylim(
            m.get("ymin", None if yscale == "log" else 0), m["ymax"]
        )  # log can't start at 0; autoscale the bottom
    for ax in axes.flat[len(metrics) :]:
        ax.axis("off")
    _legend(fig, axes.flat[0], handles, grid_h, box)
    return fig


def plot_std_norm(runs, cfg) -> plt.Figure | None:
    """L2 norm of the predicted std field vs inference steps; the black line is the ground truth."""

    def norm(field):  # [N, C, H, W] -> mean over seeds of the per-seed spatial L2 norm
        field = np.asarray(field)
        return float(np.linalg.norm(field.reshape(field.shape[0], -1), axis=1).mean())

    w, grid_h = STYLE["figsize"]["single"]
    box = cfg.get("legend")
    fig, ax = plt.subplots(figsize=(w, grid_h + (0 if box else LEGEND_H)))
    handles = {}
    for run in runs:
        steps, values = _series(run, "std_error", lambda d: norm(d["std_prediction"]))
        if steps:
            handles.setdefault(run.label, _steps_line(ax, run, steps, values))
    if not handles:
        return None
    with np.load(next(p for r in runs for p in r.artifacts.glob("std_error_*.npz"))) as d:
        ref = norm(d["std_reference"])
    if ref > 0 or cfg["axes"].get("yscale") != "log":  # a zero reference (det) is off-scale on a log axis
        handles["Ground truth"] = ax.axhline(ref, color="black", alpha=STYLE["line_alpha"], label="Ground truth")
    _axes(ax, cfg["axes"], xlim=(0, None))
    if inset := cfg.get("inset"):
        _inset(ax, inset)
    _legend(fig, ax, handles, grid_h, box)
    return fig


def plot_rmse_rollout(runs, cfg) -> plt.Figure | None:
    """RMSE per rollout step at a fixed inference-step budget; framed like a single metrics_steps panel."""
    w, gh = STYLE["figsize"]["grid"]
    grid_h = gh / 2
    box = cfg.get("legend")
    fig, ax = plt.subplots(figsize=(w / 2, grid_h + (0 if box else LEGEND_H)))
    xmax = 0
    for run in runs:
        if data := _load(run, cfg.get("key", "rollout_rmse"), cfg["steps"], "rmse_per_rollout_step"):
            curve = np.atleast_1d(data[0])
            ax.plot(np.arange(1, len(curve) + 1), curve, **run.style)
            xmax = max(xmax, len(curve))
    if not xmax:
        return None
    _axes(ax, cfg["axes"], xlim=(1, xmax), ylim=(0, None))
    ax.xaxis.set_major_locator(
        MultipleLocator(1) if xmax <= 10 else MaxNLocator(integer=True)
    )  # per-step ticks only for short rollouts
    handles, labels = ax.get_legend_handles_labels()
    _legend(fig, ax, dict(zip(labels, handles)), grid_h, box)
    return fig


def plot_enstr_spec(runs, cfg) -> plt.Figure | None:
    """Enstrophy spectrum at the final rollout step, with an optional zoom inset."""
    fig, ax = plt.subplots(figsize=STYLE["figsize"]["wide"])
    ref = None
    for run in runs:
        if data := _load(
            run, cfg.get("key", "enstrophy_error"), cfg["steps"], "enstr_model_spectrum", "enstr_reference_spectrum"
        ):
            spec, ref = map(_spec, data)
            ax.plot(np.arange(1, len(spec)), spec[1:], **run.style)
    if ref is None:
        return None
    ax.plot(np.arange(1, len(ref)), ref[1:], color="black", label="Ground truth", alpha=STYLE["line_alpha"])
    _axes(ax, cfg["axes"])
    _legend_box(ax, cfg.get("legend", "box"))
    if inset := cfg.get("inset"):
        _inset(ax, inset)
    fig.tight_layout()
    return fig


def plot_enstr_rollout(runs, cfg) -> plt.Figure | None:
    """Enstrophy spectrum error per rollout step at a fixed inference-step budget."""
    fig, ax = plt.subplots(figsize=STYLE["figsize"]["single"])
    xmax = 0
    for run in runs:
        if data := _load(run, cfg.get("key", "enstrophy_error"), cfg["steps"], "enstr_per_rollout_step"):
            values = np.atleast_1d(data[0])  # 0-dim when rollout has a single step
            ax.plot(np.arange(1, len(values) + 1), values, **run.style)
            xmax = max(xmax, len(values))
    if not xmax:
        return None
    _axes(ax, cfg["axes"], xlim=(0, xmax), ylim=(0, None))
    _legend_box(ax, cfg.get("legend", "box"))
    fig.tight_layout()
    return fig


def plot_enstr_ratio(runs, cfg) -> plt.Figure | None:
    """Model / ground-truth enstrophy per wavenumber (1 = perfect) at the final rollout step."""
    k, ref = np.arange(1, cfg["kmax"] + 1), None
    fig, ax = plt.subplots(figsize=STYLE["figsize"]["single"])
    for run in runs:
        if data := _load(
            run, cfg.get("key", "enstrophy_error"), cfg["steps"], "enstr_model_spectrum", "enstr_reference_spectrum"
        ):
            spec, ref = map(_spec, data)
            ax.plot(k, spec[k] / ref[k], **run.style)
    if ref is None:
        return None
    ax.axhline(1.0, color="black", alpha=STYLE["line_alpha"])
    _axes(ax, cfg["axes"])
    _legend_box(ax, cfg.get("legend", "box"))
    fig.tight_layout()
    return fig


def plot_enstr_flow(runs, cfg) -> plt.Figure | None:
    """Spectrum-over-sampler-steps heatmap, one row per run plus the ground-truth rollout."""
    target, flows = None, []
    for run in runs:
        data = _load(
            run,
            cfg.get("key", "enstrophy_error"),
            cfg["steps"],
            "enstr_model_spectrum",
            "enstr_reference_spectrum",
            "enstr_sampler_step_spectrum",
        )
        if not data:
            continue
        spec, ref, traj = data
        if target is None:
            target = np.broadcast_to(ref, spec.shape) if ref.ndim == 1 else ref
        flows.append((traj.reshape(-1, traj.shape[-1]) if traj is not None else spec, run.label))
    if target is None:
        return None

    rows = [(target, "Ground truth"), *flows]
    rollout = target.shape[0]
    k_min, k_max = cfg["k_range"]
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(STYLE["figsize"]["single"][0] + 1.5, len(rows)), sharex=True, squeeze=False
    )
    norm = colors.PowerNorm(
        cfg["gamma"], vmin=min(np.nanmin(v) for v, _ in rows), vmax=max(np.nanmax(v) for v, _ in rows)
    )
    for i, (ax, (vals, label)) in enumerate(zip(axes[:, 0], rows)):
        x = np.linspace(0, rollout, vals.shape[0] + 1)
        y = np.arange(vals.shape[1]) + 0.5
        im = ax.pcolormesh(x, y, vals[:, 1:].T, cmap=cfg["cmap"], norm=norm, shading="flat")
        ax.set(yscale="log", ylim=(k_min, k_max or vals.shape[1] - 1), ylabel="k")
        ax.text(-0.14, 0.5, _letter(i, label), transform=ax.transAxes, ha="right", va="center", rotation=90)
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
    axes[-1, 0].set(xticks=np.arange(1, rollout + 1), xlabel="Rollout step")
    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.12, top=0.98, hspace=0.12)
    fig.colorbar(im, cax=fig.add_axes([0.88, 0.12, 0.02, 0.86]))
    return fig


def plot_seed_fields(runs, cfg) -> plt.Figure | None:
    """Paper grid: ground-truth mean & std maps on top, each method's signed error below, per seed."""
    want, steps = cfg.get("seeds", 3), cfg.get("steps")
    cmaps = {s: sns.color_palette(STYLE["cmaps"][s], as_cmap=True) for s in ("mean", "std")}
    diff = sns.color_palette(STYLE["cmaps"]["diff"], as_cmap=True)
    fields, seeds = {}, None  # fields[label][stat] -> (n_seed, H, W); reference shared under "Ground truth"
    for run in runs:
        for stat in ("mean", "std"):
            if data := _load(run, f"{stat}_error", steps, f"{stat}_prediction", f"{stat}_reference", f"{stat}_seeds"):
                pred, refs, seeds = (np.atleast_1d(a) for a in data)
                fields.setdefault(run.label, {})[stat] = pred[:, 0]
                fields.setdefault("Ground truth", {})[stat] = refs[:, 0]
    if seeds is None:
        return None
    order = (
        [np.where(seeds == s)[0][0] for s in want if s in seeds]
        if isinstance(want, (list, tuple))
        else np.argsort(seeds)[:want]
    )  # explicit seed list, else the n smallest seeds
    n, ref = len(order), fields["Ground truth"]
    rows = ["Ground truth", *(r for r in fields if r != "Ground truth")]  # GT maps on top, method errors below
    cols = [(stat, i) for stat in ("mean", "std") for i in order]  # 2 stats x n seeds
    widths = [1] * n + [0.2] + [1] * n  # central spacer column widens the mean|std divide
    axcols = [*range(n), *range(n + 1, 2 * n + 1)]  # axes column per data column, skipping the spacer
    fig, axes = plt.subplots(
        len(rows),
        len(widths),
        figsize=(1.6 * sum(widths), 1.6 * len(rows)),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": widths},
    )
    for ax in axes[:, n]:
        ax.axis("off")
    for (stat, i), c in zip(cols, axcols):
        if stat not in ref:
            continue
        gt = ref[stat][i]
        errs = {label: fields[label][stat][i] - gt for label in rows[1:] if stat in fields[label]}
        vd = max((np.abs(x).max() for x in errs.values()), default=0.0)  # per-seed error scale, shared by all methods
        axes[0, c].set_title(f"{STYLE['stat_labels'][stat]}, seed {int(seeds[i])}")
        axes[0, c].imshow(gt, cmap=cmaps[stat], vmin=gt.min(), vmax=gt.max())
        for r, label in enumerate(rows[1:], 1):
            if label in errs:
                axes[r, c].imshow(errs[label], cmap=diff, vmin=-vd, vmax=vd)
        for r in range(len(rows)):
            axes[r, c].set(xticks=[], yticks=[])
        # per-seed colour bars: ground-truth scale under the GT row, signed-error scale under the last row
        fig.colorbar(
            plt.cm.ScalarMappable(plt.Normalize(gt.min(), gt.max()), cmaps[stat]),
            ax=axes[0, c],
            orientation="horizontal",
            location="bottom",
            fraction=0.05,
        )
        fig.colorbar(
            plt.cm.ScalarMappable(plt.Normalize(-vd, vd), diff),
            ax=axes[len(rows) - 1, c],
            orientation="horizontal",
            location="bottom",
            fraction=0.05,
        )
    for r, label in enumerate(rows):
        axes[r, 0].set_ylabel(_letter(r, label))
    return fig


def plot_schedule(run, cfg) -> plt.Figure | None:
    """Mean/std error at each sampler step, one curve per schedule (darkest = fewest steps)."""
    series = {}
    for stat in ("mean", "std"):
        rows = []
        for p in run.artifacts.glob(f"{stat}_error_*.npz"):  # one artifact per schedule -> one curve
            with np.load(p) as d:
                if f"{stat}_sched" in d and f"{stat}_grid" in d:  # both are 0-dim for single-step schedules
                    rows.append((np.atleast_1d(d[f"{stat}_grid"]).copy(), np.atleast_1d(d[f"{stat}_sched"]).copy()))
        series[stat] = sorted(rows, key=lambda r: len(r[0]))
    if not any(series.values()):
        return None

    lo, hi = cfg.get("cmap_range", (0.5, 0.9))  # cmap values at the most-steps and fewest-steps ends
    # `levels` (size of the sampler's noise-level grid) switches x from the raw grid index (0 = sigma_max) to the
    # noise level index counting down from levels-1 -> 0 on an inverted axis, so sampling still runs left -> right
    n = cfg.get("levels")
    fig, axes = plt.subplots(2, 1, figsize=STYLE["figsize"]["single"], sharex=True)
    for ax, stat in zip(axes, ("std", "mean")):  # independently scaled rows, std on top
        cmap = (
            plt.get_cmap(cfg["cmap"])
            if cfg.get("cmap")  # cfg cmap (e.g. cmc.vanimo) overrides the per-stat hue
            else colors.LinearSegmentedColormap.from_list(stat, [STYLE["colors"]["sched"][stat], "black"])
        )
        curves = series[stat]
        for i in reversed(range(len(curves))):  # most steps drawn first, so the darkest (fewest-step) curves sit on top
            grid, values = curves[i]
            ax.plot(
                n - 1 - grid if n else grid,
                values,
                color=cmap(hi - (hi - lo) * i / max(1, len(curves) - 1)),
                linestyle="-" if len(grid) > 1 else "none",
                marker="o" if len(grid) > 1 else None,
                markeredgewidth=0,
            )
        ax.set(
            ylabel=STYLE["stat_labels"][stat], yscale=cfg.get("yscale", "linear")
        )  # autoscale y to the data (don't pin to 0)
        if cfg.get("yscale") != "log":  # fixed-spacing locators only make sense on a linear axis
            ax.yaxis.set_major_locator(
                MultipleLocator(cfg["major"][stat])
                if stat in cfg.get("major", {})
                else MaxNLocator(nbins=4, steps=[1, 2, 5, 10])
            )  # fixed spacing, else round + adapted
            if stat in cfg.get("minor", {}):  # auxiliary gridlines, e.g. one per 1%
                ax.yaxis.set_minor_locator(MultipleLocator(cfg["minor"][stat]))
                ax.grid(True, which="minor", axis="y", linewidth=0.4, alpha=0.15)
    axes[1].set(xlabel=cfg["xlabel"], xlim=(n - 1, 0) if n else (0, None))
    if na := cfg.get("noise_axis"):  # relabel grid indices with their noise levels (Karras grid);
        m = na["steps"]  # index 0 = sigma_max, so the labels run largest -> smallest
        ticks = np.asarray(na.get("ticks", [0, m // 5, 2 * m // 5, 3 * m // 5, 4 * m // 5, m - 1]))
        smin, smax, rho = na["sigma_min"], na["sigma_max"], na.get("rho", 7.0)
        sig = (smax ** (1 / rho) + ticks / (m - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
        axes[1].set_xticks(n - 1 - ticks if n else ticks, [f"{s:.3g}" for s in sig])
    fig.tight_layout()
    return fig


def plot_stat_grid(preds, refs, seeds, stat, cols=2, max_seeds=None) -> plt.Figure:
    """Model / ground-truth / difference field grid, `cols` seeds per row.

    With many seeds the full grid becomes an unwieldy (and memory-hungry) figure, so
    `max_seeds` evenly subsamples across the sorted seed range when set.
    """
    # torchmetrics squeezes numel()==1 tensors after compute(), so a single
    # eval seed arrives 0-dim; restore the leading IC axis before indexing.
    preds, refs = np.atleast_1d(np.asarray(preds)), np.atleast_1d(np.asarray(refs))
    seeds = np.atleast_1d(np.asarray(seeds))
    order = np.argsort(seeds)
    if max_seeds and len(order) > max_seeds:
        order = order[np.linspace(0, len(order) - 1, max_seeds).round().astype(int)]
    field = sns.color_palette(STYLE["cmaps"][stat], as_cmap=True)
    diff_cmap = sns.color_palette(STYLE["cmaps"]["diff"], as_cmap=True)
    n, rows = len(order), math.ceil(len(order) / cols)
    fig, axes = plt.subplots(rows, 3 * cols, figsize=(9 * cols, 3 * rows), squeeze=False)
    for idx, i in enumerate(order):
        r, c = divmod(idx, cols)
        pred, ref = preds[i][0], refs[i][0]
        diff = pred - ref
        lo, hi = min(pred.min(), ref.min()), max(pred.max(), ref.max())
        vd = np.abs(diff).max()
        panels = [
            (pred, field, lo, hi, "Model"),
            (ref, field, lo, hi, "Ground truth"),
            (diff, diff_cmap, -vd, vd, "Difference"),
        ]
        for ax, (img, cmap, vmin, vmax, title) in zip(axes[r, 3 * c : 3 * c + 3], panels):
            fig.colorbar(ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax, fraction=0.046, pad=0.04)
            ax.set(xticks=[], yticks=[], title=title)
        axes[r, 3 * c].set_ylabel(f"seed {int(seeds[i])}")
    for ax in axes.flat[3 * n :]:
        ax.axis("off")
    fig.suptitle(f"{stat} error")
    fig.tight_layout()
    return fig


# --- style & io ---


def _tex_available() -> bool:
    """usetex needs `latex` on PATH plus the newtx fonts the style preamble loads."""
    if not shutil.which("latex"):
        return False
    try:
        return subprocess.run(["kpsewhich", "newtxtext.sty"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _apply_style(path) -> None:
    global STYLE
    STYLE = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    # STOCBENCH_TEX_BIN pins one TeX tree ahead of everything else: matplotlib locates fonts through
    # luatex/kpsewhich, so a second TeX install on PATH (e.g. a cluster module without newtx) can
    # mismatch the latex that built the dvi and fail with spurious "*.tfm not found" errors.
    tex_bin = os.environ.get("STOCBENCH_TEX_BIN")
    if tex_bin and tex_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{tex_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    usetex = STYLE["usetex"]
    if usetex == "auto":
        usetex = _tex_available()
        if not usetex:
            warnings.warn(
                "LaTeX with the newtx fonts not found: rendering figure text with mathtext instead", stacklevel=2
            )
    plt.rcParams.update(STYLE["rc"])
    plt.rcParams["text.usetex"] = bool(usetex)
    preamble = "\n".join(STYLE["latex_preamble"])
    plt.rcParams["text.latex.preamble"] = preamble
    plt.rcParams["pgf.preamble"] = preamble
    plt.rcParams["axes.prop_cycle"] = cycler(color=STYLE["colors"]["cycle"])


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    if plt.rcParams["text.usetex"]:  # the pgf backend measures text with a LaTeX process, so it needs TeX too
        fig.savefig(stem.with_suffix(".pgf"))
    plt.close(fig)


def main() -> None:
    """`stocbench plot` entry."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot metric artifacts from stocbench eval runs.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--style", type=Path, default=CONFIGS / "plotting/style.yaml")
    parser.add_argument("--runs-config", type=Path, default=CONFIGS / "plotting/runs.yaml")
    parser.add_argument("--plots-config", type=Path, default=CONFIGS / "plotting/plots/stocbench/stoc.yaml")
    args = parser.parse_args()
    out = plot_runs(
        args.runs,
        output_dir=args.output_dir,
        style=args.style,
        runs_config=args.runs_config,
        plots_config=args.plots_config,
    )
    print(f"plots written to {out}")
