"""Configuration helpers for the standalone road DIP experiments."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _expand(value: Any, base_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item, base_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, base_dir) for item in value]
    if not isinstance(value, str):
        return value

    value = os.path.expandvars(os.path.expanduser(value))
    if value.startswith("./") or value.startswith("../"):
        return str((base_dir / value).resolve())
    return value


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config with optional single inheritance via ``base``."""

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    base_ref = config.pop("base", None)
    if base_ref:
        base_path = Path(os.path.expandvars(str(base_ref)))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        config = _merge(load_config(base_path), config)

    config = _expand(config, path.parent)
    config["config_path"] = str(path)
    return config


def apply_overrides(config: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply CLI overrides in ``section.key=value`` form."""

    result = copy.deepcopy(config)
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"Invalid override {expression!r}; expected key=value")
        dotted_key, raw_value = expression.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = result
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    return result


def validate_config(config: Dict[str, Any]) -> None:
    required = {
        "experiment": ["name", "task"],
        "data": ["source_root", "source_list", "support_root", "support_list", "val_root", "val_list"],
        "model": ["type"],
        "train": ["max_iters", "crop_size"],
    }
    for section, keys in required.items():
        if section not in config:
            raise KeyError(f"Missing config section: {section}")
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Missing config field: {section}.{key}")

    model_type = config["model"]["type"]
    if model_type not in {"native_r101", "dinov3_base_rein_hrda"}:
        raise ValueError(f"Unsupported model.type: {model_type}")

    if config["experiment"]["task"] not in {
        "gta2cityscapes",
        "synthia2cityscapes",
        "cityscapes2acdc",
        "cityscapes2muses",
        "cityscapes2mapillary",
    }:
        raise ValueError(f"Unsupported task: {config['experiment']['task']}")

