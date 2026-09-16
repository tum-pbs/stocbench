from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .utils import parse_artifact_name


class BenchmarkResult:
    """Read-only view of an eval run directory (config.yaml + artifacts/*.npz)."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    @property
    def config(self) -> dict:
        path = self.output_dir / "config.yaml"
        return yaml.safe_load(path.read_text()) if path.exists() else {}

    @property
    def results(self) -> list[dict]:
        artifacts = self.output_dir / "artifacts"
        if not artifacts.exists():
            return []
        rows: dict[str, dict] = {}
        for path in artifacts.glob("*.npz"):
            metric, schedule = parse_artifact_name(path)
            with np.load(path) as data:
                value = data[next(k for k in data.files if k.endswith("value"))]
            row = rows.setdefault(
                schedule.label,
                {
                    "sampling_schedule": schedule.label,
                    "schedule_kind": schedule.kind,
                    "inference_steps": schedule.steps,
                    "artifacts": {},
                },
            )
            row[metric] = float(value.item() if value.shape == () else value.mean())
            row["artifacts"][metric] = str(path.relative_to(self.output_dir))
        return sorted(rows.values(), key=lambda row: (row["schedule_kind"], row["inference_steps"]))

    def summary(self) -> str:
        rows = self.results
        names = _metric_names(self.config, rows)
        lines = ["sampling schedule | " + " | ".join(names)]
        lines += [" | ".join([row["sampling_schedule"]] + [_format(row.get(name)) for name in names]) for row in rows]
        return "\n".join(lines)


def write_config(output_dir: Path, config: dict) -> None:
    with (output_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def write_summary(output_dir: Path, config: dict) -> None:
    """Write summary.md: one row per sampling schedule, one column per metric."""
    results = BenchmarkResult(output_dir).results
    names = _metric_names(config, results)
    lines = [
        "# StocBench Evaluation Summary",
        "",
        f"Dataset: {config['dataset']}",
        f"Checkpoint: {config['checkpoint']}",
        "",
        "| sampling schedule | " + " | ".join(names) + " |",
        "|---|" + "|".join(["---:"] * len(names)) + "|",
    ]
    for row in results:
        lines.append("| " + " | ".join([row["sampling_schedule"]] + [_format(row.get(name)) for name in names]) + " |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def save_artifact(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v.detach().cpu().numpy() if hasattr(v, "detach") else v for k, v in arrays.items()})


def _metric_names(config: dict, rows: list[dict]) -> list[str]:
    names = list(config.get("metrics") or [])
    if names:
        return names
    return sorted(
        {
            k
            for row in rows
            for k in row
            if k not in {"sampling_schedule", "schedule_kind", "inference_steps", "artifacts"}
        }
    )


def _format(value) -> str:
    return f"{float(value):.4g}" if value is not None else "·"
