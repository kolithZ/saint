from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

SPLITS = ("train", "val", "test")
LABELS = {"NORMAL": 0, "PNEUMONIA": 1}
EXTENSIONS = {".jpg", ".jpeg", ".png"}
MANIFEST_FIELDS = ["relative_path", "split", "label", "label_id", "width", "height", "mode", "format", "frame_count", "readable", "file_sha256", "rgb_pixel_sha256", "candidate_group_id", "group_source", "patient_id_verified"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def new_output(path, protected=()):
    path = Path(path).resolve()
    for root in protected:
        root = Path(root).resolve()
        if path == root or root in path.parents or path in root.parents:
            raise ValueError(f"Output overlaps protected input: {root}")
    # Refuse even an existing empty directory; never mix two runs.
    path.mkdir(parents=True, exist_ok=False)
    return path


def sample_path(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or len(rel.parts) != 3 or ".." in rel.parts:
        raise ValueError(f"Invalid relative image path: {relative}")
    if rel.parts[0] not in SPLITS or rel.parts[1] not in LABELS:
        raise ValueError(f"Invalid split/label path: {relative}")
    path = root / rel
    if any(p.is_symlink() for p in [path, path.parent, path.parent.parent]):
        raise ValueError(f"Symlink not allowed: {relative}")
    if root not in path.resolve().parents:
        raise ValueError(f"Image escapes data root: {relative}")
    return path


def counts(rows):
    return {s: {label: sum(r["split"] == s and r["label"] == label for r in rows) for label in LABELS} for s in SPLITS}
