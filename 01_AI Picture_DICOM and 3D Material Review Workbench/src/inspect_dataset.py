"""Audit the configured Chest X-ray dataset without modifying the image source."""

from __future__ import annotations

import argparse
import hashlib
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from common import LABELS, image_paths, load_config, write_csv, write_json


def pixel_digest(image: Image.Image) -> str:
    """Hash decoded grayscale pixels, detecting visually identical re-encodings too."""
    image = ImageOps.exif_transpose(image).convert("L")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:L:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def describe(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "median": 0, "max": 0}
    return {
        "min": int(min(values)),
        "median": float(statistics.median(values)),
        "max": int(max(values)),
    }


def scan(config: dict, sample_limit: int | None = None) -> tuple[list[dict], list[dict]]:
    """Fully decode every selected input image and return records plus read errors."""
    records: list[dict] = []
    errors: list[dict] = []
    data_root: Path = config["data_root"]
    seen = 0
    for split in ("train", "val", "test"):
        for label in LABELS:
            for path in image_paths(data_root, split, label):
                if sample_limit is not None and seen >= sample_limit:
                    return records, errors
                seen += 1
                relative = path.relative_to(data_root).as_posix()
                try:
                    raw = path.read_bytes()
                    with Image.open(path) as image:
                        image.load()
                        records.append(
                            {
                                "source_path": relative,
                                "split": split,
                                "label": label,
                                "width": image.width,
                                "height": image.height,
                                "mode": image.mode,
                                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                                "pixel_sha256": pixel_digest(image),
                            }
                        )
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    errors.append({"source_path": relative, "error": str(exc)})
    return records, errors


def duplicate_groups(records: list[dict], field: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record[field]].append(record)
    result = []
    for digest, members in groups.items():
        if len(members) > 1:
            result.append(
                {
                    "digest": digest,
                    "kind": field,
                    "count": len(members),
                    "splits": sorted({member["split"] for member in members}),
                    "paths": sorted(member["source_path"] for member in members),
                }
            )
    return sorted(result, key=lambda row: (-row["count"], row["digest"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to a JSON config relative to the project root.")
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional smoke-test cap. Omit it for the required full audit.",
    )
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")

    config, _ = load_config(args.config)
    records, errors = scan(config, args.max_images)
    outputs: Path = config["outputs_dir"]
    byte_duplicates = duplicate_groups(records, "byte_sha256")
    pixel_duplicates = duplicate_groups(records, "pixel_sha256")
    cross_split = [
        group
        for group in pixel_duplicates
        if len(group["splits"]) > 1
    ]
    counts = Counter(f"{row['split']}/{row['label']}" for row in records)
    widths = [int(row["width"]) for row in records]
    heights = [int(row["height"]) for row in records]
    canonical_directories = {"train", "val", "test"}
    ignored_directories = sorted(
        child.name
        for child in config["data_root"].iterdir()
        if child.is_dir() and child.name not in canonical_directories
    )
    summary = {
        "data_root": str(config["data_root"]),
        "audit_complete": args.max_images is None,
        "images_read": len(records),
        "read_error_count": len(errors),
        "class_counts": dict(sorted(counts.items())),
        "image_width": describe(widths),
        "image_height": describe(heights),
        "image_modes": dict(sorted(Counter(row["mode"] for row in records).items())),
        "exact_file_duplicate_groups": len(byte_duplicates),
        "decoded_pixel_duplicate_groups": len(pixel_duplicates),
        "cross_split_decoded_pixel_duplicate_groups": len(cross_split),
        "ignored_top_level_directories": ignored_directories,
        "patient_id_available": False,
        "patient_id_note": (
            "The local release contains filenames but no authoritative patient/study ID table. "
            "Filename text is not treated as a verified patient identifier."
        ),
    }
    write_json(outputs / "dataset_summary.json", summary)
    write_json(outputs / "read_errors.json", errors)
    write_json(outputs / "duplicate_images.json", {"byte_duplicates": byte_duplicates, "pixel_duplicates": pixel_duplicates})
    write_csv(
        outputs / "image_inventory.csv",
        records,
        ["source_path", "split", "label", "width", "height", "mode", "byte_sha256", "pixel_sha256"],
    )
    print(
        f"Read {len(records)} images; {len(errors)} unreadable; "
        f"{len(cross_split)} decoded-pixel duplicate groups cross splits."
    )
    print(f"Wrote audit to {outputs}")


if __name__ == "__main__":
    main()
