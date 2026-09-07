"""Create deterministic image-path manifests for the 600/100 development subset."""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from common import LABELS, image_paths, load_config, write_csv, write_json


FIELDS = [
    "sample_id",
    "split",
    "label",
    "source_path",
    "byte_sha256",
    "pixel_sha256",
]


def pixel_digest(image: Image.Image) -> str:
    image = ImageOps.exif_transpose(image).convert("L")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:L:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def record(path: Path, data_root: Path, split: str, label: str) -> dict:
    try:
        raw = path.read_bytes()
        with Image.open(path) as image:
            image.load()
            pixel_hash = pixel_digest(image)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise RuntimeError(f"Could not decode {path}: {exc}") from exc
    relative = path.relative_to(data_root).as_posix()
    return {
        "sample_id": "cxr_" + hashlib.sha256(relative.encode()).hexdigest()[:16],
        "split": split,
        "label": label,
        "source_path": relative,
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "pixel_sha256": pixel_hash,
    }


def unique_by_pixel(records: list[dict]) -> list[dict]:
    """Keep one source path per decoded image so it cannot appear twice in a split."""
    unique: dict[str, dict] = {}
    for item in sorted(records, key=lambda row: row["source_path"]):
        unique.setdefault(item["pixel_sha256"], item)
    return list(unique.values())


def validate_disjoint(rows: list[dict]) -> None:
    owners: dict[str, str] = {}
    for row in rows:
        previous = owners.setdefault(row["pixel_sha256"], row["split"])
        if previous != row["split"]:
            raise ValueError(
                f"Decoded-image leakage from {previous} into {row['split']}: {row['source_path']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to JSON config relative to project root.")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    data_root: Path = config["data_root"]
    seed = int(config["seed"])
    subset = config["subset"]
    desired_train = int(subset["train_per_class"])
    desired_val = int(subset["val_per_class"])
    if desired_train < 1 or desired_val < 1:
        raise ValueError("Subset sizes must be positive")

    selected: list[dict] = []
    source_counts: Counter[str] = Counter()
    dropped_due_to_test_overlap: Counter[str] = Counter()
    dropped_within_train: Counter[str] = Counter()
    for label in LABELS:
        # Keep every source test path for evaluation. A source release can contain internal duplicates,
        # but removing them would no longer be an evaluation on its original test directory.
        test_records = [record(path, data_root, "test", label) for path in image_paths(data_root, "test", label)]
        test_hashes = {item["pixel_sha256"] for item in test_records}
        train_records = [
            record(path, data_root, "train", label) for path in image_paths(data_root, "train", label)
        ]
        source_counts[f"train/{label}"] = len(train_records)
        source_counts[f"test/{label}"] = len(test_records)
        unique_train = unique_by_pixel(train_records)
        dropped_within_train[label] = len(train_records) - len(unique_train)
        eligible = [row for row in unique_train if row["pixel_sha256"] not in test_hashes]
        dropped_due_to_test_overlap[label] = len(unique_train) - len(eligible)
        random.Random(f"{seed}:{label}").shuffle(eligible)
        needed = desired_train + desired_val
        if len(eligible) < needed:
            raise ValueError(f"Need {needed} non-overlapping {label} train images; found {len(eligible)}")
        for row in eligible[:desired_train]:
            row["split"] = "train"
            selected.append(row)
        for row in eligible[desired_train:needed]:
            row["split"] = "val"
            selected.append(row)
        selected.extend(test_records)

    validate_disjoint(selected)
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("Sample ID collision; aborting manifest creation")
    splits: Path = config["splits_dir"]
    for split in ("train", "val", "test"):
        rows = sorted((row for row in selected if row["split"] == split), key=lambda row: row["source_path"])
        write_csv(splits / f"{split}.csv", rows, FIELDS)
    all_rows = sorted(selected, key=lambda row: (row["split"], row["label"], row["source_path"]))
    write_csv(splits / "manifest.csv", all_rows, FIELDS)
    summary = {
        "seed": seed,
        "data_root": str(data_root),
        "source_counts": dict(sorted(source_counts.items())),
        "selected_counts": dict(sorted(Counter(f"{row['split']}/{row['label']}" for row in selected).items())),
        "selection_policy": (
            "Deterministic per-class shuffle of unique decoded images from the original train split; "
            "all original test images are retained, and development candidates identical to a test image are excluded."
        ),
        "original_val_used": False,
        "original_val_note": "The provided val directory has only 8 NORMAL and 8 PNEUMONIA images; validation is sampled from original train as specified in DESIGN.md.",
        "duplicates_removed_within_original_train": dict(sorted(dropped_within_train.items())),
        "development_candidates_removed_due_to_test_duplicate": dict(sorted(dropped_due_to_test_overlap.items())),
        "patient_level_split_verified": False,
        "patient_level_split_note": (
            "No authoritative patient/study IDs were supplied with this local release. "
            "Exact decoded-image overlap is prevented, but patient-level independence cannot be verified."
        ),
    }
    write_json(config["outputs_dir"] / "subset_summary.json", summary)
    print("Created deterministic manifests:")
    for key, value in summary["selected_counts"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
