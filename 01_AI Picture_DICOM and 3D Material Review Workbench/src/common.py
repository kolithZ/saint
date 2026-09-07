"""Shared configuration, manifest, and reproducibility helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("NORMAL", "PNEUMONIA")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    """Load JSON configuration and resolve paths relative to the project root."""
    config_path = Path(path) if path else PROJECT_ROOT / "config.json"
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["data_root"] = (PROJECT_ROOT / config["data_root"]).resolve()
    config["outputs_dir"] = (PROJECT_ROOT / config["outputs_dir"]).resolve()
    config["splits_dir"] = (PROJECT_ROOT / config["splits_dir"]).resolve()
    return config, config_path


def image_paths(data_root: Path, split: str, label: str) -> list[Path]:
    """Return top-level image files only; archive metadata and nested copies are excluded."""
    directory = data_root / split / label
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing data directory: {directory}")
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_SUFFIXES
    )


def set_reproducible_seed(seed: int) -> None:
    """Set all seeds needed by the training pipeline."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def configure_torch_cache() -> Path:
    """Keep downloaded pretrained weights inside the project rather than a user-home cache."""
    cache = PROJECT_ROOT / ".torch"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache))
    return cache
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest is missing: {path}. Run `python src/make_subset.py` first."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def source_path(data_root: Path, row: dict[str, str]) -> Path:
    path = (data_root / row["source_path"]).resolve()
    if data_root not in path.parents:
        raise ValueError(f"Manifest path escapes data root: {row['source_path']}")
    return path
