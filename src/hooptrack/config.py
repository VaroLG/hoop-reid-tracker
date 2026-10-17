"""YAML configuration with command-line overrides.

    cfg = load_config("configs/default.yaml", ["tracker.memory_enabled=false"])

Overrides use dotted keys and YAML-parsed values, so ablations never need a
new file:  ``--set reid.type=none tracker.gmc=none``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _set_dotted(cfg: dict, dotted_key: str, value) -> None:
    node = cfg
    *parents, leaf = dotted_key.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))  # "false" -> False, "0.3" -> 0.3
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def save_config(cfg: dict, path: str | Path) -> None:
    """Snapshot the exact config next to the results (reproducibility)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
