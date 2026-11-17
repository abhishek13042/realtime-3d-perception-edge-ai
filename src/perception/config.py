"""Configuration loading. All paths and thresholds come from configs/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str = "dataset") -> dict[str, Any]:
    """Load configs/<name>.yaml as a dict."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config("dataset")
    return Path(cfg["data_root"])
