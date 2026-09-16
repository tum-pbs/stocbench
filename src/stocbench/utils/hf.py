from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def _resolve_max_epoch(path: str, hf_cfg) -> str:
    """Resolve `@max` to the highest matching `eNNN` value in the repo."""
    if "@max" not in path:
        return path
    pattern = re.compile(re.escape(path).replace("@max", r"e(\d+)"))
    files = HfApi().list_repo_files(repo_id=hf_cfg.repo_id, revision=hf_cfg.revision, repo_type=hf_cfg.repo_type)
    epochs = [int(m.group(1)) for f in files if (m := pattern.fullmatch(f))]
    if not epochs:
        raise FileNotFoundError(f"no Hugging Face file matches hf:{path}")
    return path.replace("@max", f"e{max(epochs)}")


def resolve_hf_path(path: str | None, hf_cfg) -> str | None:
    """Resolve local paths and `hf:` paths (file or directory prefix)."""
    if path is None or not path.startswith("hf:"):
        return path
    raw = _resolve_max_epoch(path[3:], hf_cfg)
    local_dir = Path(hf_cfg.cache_dir).expanduser()
    snapshot_download(
        repo_id=hf_cfg.repo_id,
        repo_type=hf_cfg.repo_type,
        revision=hf_cfg.revision,
        local_dir=str(local_dir),
        allow_patterns=raw if Path(raw).suffix else f"{raw}/*",
        library_name="stocbench",
    )
    if not (local_dir / raw).exists():
        raise FileNotFoundError(f"no Hugging Face files match hf:{raw}")
    return str(local_dir / raw)
