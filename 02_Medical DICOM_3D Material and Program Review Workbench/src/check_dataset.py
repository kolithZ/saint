"""Audit each original split, decoding images and reporting exact duplicate content."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import random

from PIL import Image

from common import (LABELS, SPLITS, discover, image_path, load_config, parser,
                    read_csv, sha256, write_csv, write_json)
from plots import image_grid


def metadata_audit(root: Path, rows: list[dict], metadata_path: str | None) -> dict:
    if not metadata_path:
        return {"status": "unverified", "reason": "No reliable patient_id/study_id metadata supplied; filenames are not treated as verified IDs.",
                "view_position": "Unavailable; image-only model."}
    metadata = read_csv(Path(metadata_path))
    available = {r["path"] for r in rows}
    seen, records = set(), []
    groups = {field: defaultdict(list) for field in ("patient_id", "study_id")}
    for row in metadata:
        path = row.get("path", "")
        image_path(root, path)
        if path not in available or path in seen:
            raise ValueError(f"Unknown or duplicate metadata image: {path}")
        seen.add(path)
        for field in groups:
            value = row.get(field, "").strip()
            if value:
                groups[field][value].append(path)
        view = row.get("view_position", "").strip().upper()
        if view not in ("", "AP", "PA"):
            raise ValueError(f"Unsupported view_position for {path}: {view}")
        records.append(row)
    overlaps = {field: [{"id": value, "paths": paths} for value, paths in mapping.items()
                       if len({p.split('/')[0] for p in paths}) > 1]
                for field, mapping in groups.items()}
    coverage = {field: sum(bool(r.get(field, "").strip()) for r in records) for field in groups}
    complete = any(n == len(available) for n in coverage.values())
    return {"status": "overlap_detected" if any(overlaps.values()) else "checked" if complete else "partial_unverified",
            "total_images": len(available), "metadata_images": len(seen), "id_coverage": coverage,
            "cross_split_ids": overlaps, "view_position_counts": dict(Counter(r.get("view_position", "").upper() for r in records)),
            "limitation": "Supplied identifiers must be trustworthy; study-level checks alone do not establish patient-level isolation. AP/PA is audited but not a model input."}


def audit(cfg: dict) -> dict:
    root, output = Path(cfg["data_root"]), Path(cfg["audit_dir"])
    rows = [row for split in SPLITS for row in discover(root, split)]
    inventory, errors, good = [], [], []
    hashes, pixel_hashes = defaultdict(list), defaultdict(list)
    for index, row in enumerate(rows):
        path = image_path(root, row["path"])
        try:
            with Image.open(path) as image:
                image.load()
                width, height, mode = image.width, image.height, image.mode
                rgb = image.convert("RGB")
                pixel_hash = hashlib.sha256(f"{width}x{height}:RGB:".encode() + rgb.tobytes()).hexdigest()
            file_hash = sha256(path)
            inventory.append({**row, "split": row["path"].split('/')[0], "width": width,
                              "height": height, "mode": mode, "sha256": file_hash, "pixel_sha256": pixel_hash})
            hashes[file_hash].append(row["path"])
            pixel_hashes[pixel_hash].append(row["path"])
            good.append(row)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            errors.append({"path": row["path"], "error": str(exc)})
        if (index + 1) % 500 == 0:
            print(f"Audited {index + 1}/{len(rows)} images", flush=True)
    stats = []
    for split in SPLITS:
        for label, name in enumerate(LABELS):
            found = sum(r["path"].startswith(split + "/") and r["label"] == label for r in rows)
            valid = sum(r["split"] == split and r["label"] == label for r in inventory)
            stats.append({"split": split, "class": name, "found": found, "readable": valid, "unreadable": found - valid})
    def duplicates(mapping):
        return [{"sha256": digest, "paths": paths, "cross_split": len({p.split('/')[0] for p in paths}) > 1}
                for digest, paths in mapping.items() if len(paths) > 1]
    duplicate_report = {"file_duplicates": duplicates(hashes), "decoded_pixel_duplicates": duplicates(pixel_hashes)}
    patient = metadata_audit(root, rows, cfg.get("metadata_csv"))
    summary = {"data_root": str(root), "found": len(rows), "readable": len(good), "unreadable": len(errors),
               "statistics": stats, "common_sizes": [{"size": size, "count": count} for size, count in
                       Counter(f"{r['width']}x{r['height']}" for r in inventory).most_common(10)],
               "duplicate_groups": {key: len(value) for key, value in duplicate_report.items()},
               "cross_split_duplicate_groups": {key: sum(r["cross_split"] for r in value) for key, value in duplicate_report.items()},
               "patient_isolation": patient, "excluded_rule": "Only unreadable images are excluded; original split directories and labels remain unchanged.",
               "ignored": "Hidden files, __MACOSX, unsupported extensions and nested chest_xray copies are not scanned."}
    write_csv(output / "dataset_stats.csv", stats, ["split", "class", "found", "readable", "unreadable"])
    write_csv(output / "image_inventory.csv", inventory, ["path", "label", "split", "width", "height", "mode", "sha256", "pixel_sha256"])
    write_json(output / "read_errors.json", errors)
    write_json(output / "duplicates.json", duplicate_report)
    write_json(output / "dataset_summary.json", summary)
    rng = random.Random(cfg["random_seed"])
    examples = []
    for label in (0, 1):
        candidates = [r for r in good if r["label"] == label and r["path"].startswith("train/")]
        examples.extend(rng.sample(candidates, min(3, len(candidates))))
    image_grid(root, examples, output / "samples.png", f"Original training samples (seed {cfg['random_seed']})")
    print(f"Audit complete: {len(good)} readable, {len(errors)} unreadable. Report: {output}", flush=True)
    return summary


if __name__ == "__main__":
    audit(load_config(parser(__doc__).parse_args()))
