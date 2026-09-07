"""Configuration, safe data paths and reproducibility shared by all entry points."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("NORMAL", "PNEUMONIA")
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))


def project_path(value: str | Path) -> Path:
    return (PROJECT_ROOT / Path(value).expanduser()).resolve()


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", default="configs/baseline.yaml")
    result.add_argument("--data-root", help="Override external dataset directory")
    return result


def load_config(args) -> dict:
    path = project_path(args.config)
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if args.data_root:
        cfg["data_root"] = args.data_root
    for key in ("data_root", "subset_path", "output_dir", "audit_dir"):
        cfg[key] = str(project_path(cfg[key]))
    for key in ("metadata_csv", "pretrained_weights"):
        if cfg.get(key):
            cfg[key] = str(project_path(cfg[key]))
    for key in ("image_size", "train_samples_per_class", "batch_size", "epochs", "torch_threads"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["num_workers"]) is not int or cfg["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(cfg["random_seed"]) is not int or not 0 <= cfg["random_seed"] < 2**32:
        raise ValueError("random_seed must be an integer in [0, 2**32)")
    for key in ("learning_rate", "backbone_learning_rate"):
        if not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < float("inf"):
            raise ValueError(f"{key} must be finite and positive")
    if cfg["model"] != "resnet18" or cfg["pretrained"] is not True:
        raise ValueError("This experiment requires ImageNet-pretrained ResNet18")
    if type(cfg["freeze_backbone"]) is not bool:
        raise ValueError("freeze_backbone must be a boolean")
    if cfg["device"] not in ("cpu", "cuda", "mps", "auto"):
        raise ValueError("device must be cpu, cuda, mps or auto")
    cfg["config_path"] = str(path)
    return cfg


def image_path(root: Path, relative: str, expected_split: str | None = None,
               label: int | None = None) -> Path:
    """Reject absolute paths, traversal, nested archive copies and label mismatches."""
    parts = Path(relative).parts
    if (Path(relative).is_absolute() or len(parts) != 3 or parts[0] not in SPLITS
            or parts[1] not in LABELS or parts[2].startswith(".")):
        raise ValueError(f"Invalid dataset-relative image path: {relative}")
    if expected_split and parts[0] != expected_split:
        raise ValueError(f"Expected {expected_split} image, got {relative}")
    if label is not None and (label not in (0, 1) or parts[1] != LABELS[label]):
        raise ValueError(f"Label does not match image directory: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Image escapes data root: {relative}")
    if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
        raise FileNotFoundError(f"Image is missing or unsupported: {path}")
    return path


def discover(root: Path, split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    rows = []
    for label, name in enumerate(LABELS):
        directory = root / split / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {directory}")
        files = sorted(p for p in directory.iterdir() if p.is_file()
                       and not p.name.startswith(".") and p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"No images in {directory}")
        for path in files:
            relative = f"{split}/{name}/{path.name}"
            image_path(root, relative, split, label)
            rows.append({"path": relative, "label": label})
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_fingerprint(root: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['path']}\t{row['label']}\t{sha256(image_path(root, row['path']))}\n".encode())
    return digest.hexdigest()


def seed_everything(cfg: dict):
    import torch
    random.seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])
    torch.manual_seed(cfg["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["random_seed"])
    torch.set_num_threads(cfg["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use device: cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; use device: cpu")
    return torch.device(device)


def seed_worker(_worker_id: int):
    import torch
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
