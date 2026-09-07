"""Audit canonical images and copy a deterministic, group-disjoint small subset."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("NORMAL", "PNEUMONIA")
SPLITS = ("train", "val", "test")


def group_proxy(name):
    # Preserve naming namespaces; these are only proxies, not verified patient IDs.
    match = re.fullmatch(r"(person\d+)_(?:bacteria|virus)_\d+(?:_\d+)*\.jpe?g", name, re.I)
    if match:
        return match[1].lower()
    match = re.fullmatch(r"((?:NORMAL2-)?IM-\d+)(?:-\d+)+\.jpe?g", name, re.I)
    if match:
        return match[1].upper()
    raise ValueError(f"Unknown patient grouping pattern: {name}")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def image_info(path):
    with Image.open(path) as im:
        im.load()  # Decode all pixels; truncated/unreadable images fail here.
        gray = im.convert("L")
        width, height = gray.size
        pixel_sha = sha(f"{width}x{height}:L:".encode() + gray.tobytes())
        return dict(width=width, height=height, mode=im.mode, pixel_sha256=pixel_sha)


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def components(rows):
    """Transitive union by filename proxy OR identical decoded grayscale pixels."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in ("patient_proxy", "pixel_sha256"):
            token = (key, row[key])
            if token in seen:
                parent[find(i)] = find(seen[token])
            else:
                seen[token] = i
    result = defaultdict(list)
    for i, row in enumerate(rows):
        result[find(i)].append(row)
    return list(result.values())


def verify_disjoint(rows):
    for key in ("group_id", "patient_proxy", "sha256", "pixel_sha256"):
        owners = {}
        for row in rows:
            previous = owners.setdefault(row[key], row["split"])
            if previous != row["split"]:
                raise ValueError(f"Cross-split leakage: {key}={row[key]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, output, reports = (ROOT / cfg[k] for k in ("source", "output", "reports"))
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}; use a new output path in config.")
    rows, errors = [], []
    for split in SPLITS:
        for label in LABELS:
            folder = source / split / label
            if not folder.is_dir():
                raise SystemExit(f"Missing canonical folder: {folder}")
            # Deliberately do NOT rglob: excludes nested duplicate archive and __MACOSX.
            for path in sorted(folder.iterdir()):
                if path.name.startswith(".") or path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                try:
                    rows.append(dict(source_path=path.relative_to(source).as_posix(),
                                     source_split=split, label=label,
                                     patient_proxy=group_proxy(path.name),
                                     sha256=sha(path.read_bytes()), **image_info(path)))
                except (OSError, ValueError) as exc:
                    errors.append(dict(path=path.relative_to(source).as_posix(), error=str(exc)))
        print(f"Read through {split}: {len(rows)} images", flush=True)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "read_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    if errors:
        raise SystemExit("Unreadable image or unknown group pattern; inspect reports/read_errors.json.")
    pools = {"development": defaultdict(list), "test": defaultdict(list)}
    conflicts, overlaps = [], []
    groups = components(rows)
    for members in groups:
        group_id = sha("\n".join(sorted(r["source_path"] for r in members)).encode())[:16]
        for row in members:
            row["group_id"] = group_id
        if len({r["label"] for r in members}) != 1:
            conflicts.extend(r["source_path"] for r in members)
            continue
        is_test = any(r["source_split"] == "test" for r in members)
        eligible = [r for r in members if (r["source_split"] == "test") == is_test]
        if is_test:
            overlaps.extend(r["source_path"] for r in members if r["source_split"] != "test")
        # One image per connected group prevents repeatedly counting one proxy patient.
        selected = min(eligible, key=lambda r: sha(f'{cfg["seed"]}:{r["source_path"]}'.encode()))
        pools["test" if is_test else "development"][selected["label"]].append(selected)
    chosen = []
    for label in LABELS:
        for pool in pools.values():
            pool[label].sort(key=lambda r: sha(f'{cfg["seed"]}:{r["group_id"]}'.encode()))
        ntrain, nval, ntest = (cfg["images_per_class"][s] for s in SPLITS)
        if min(ntrain, nval, ntest) < 1:
            raise SystemExit("Each split needs a positive per-class count.")
        if len(pools["development"][label]) < ntrain + nval or len(pools["test"][label]) < ntest:
            raise SystemExit(f"Insufficient unique groups for {label}")
        selections = {"train": pools["development"][label][:ntrain],
                      "val": pools["development"][label][ntrain:ntrain + nval],
                      "test": pools["test"][label][:ntest]}
        for split, selected_rows in selections.items():
            for row in selected_rows:
                sample_id = "cxr_" + sha(row["source_path"].encode())[:12]
                chosen.append(dict(row, sample_id=sample_id, split=split,
                                   subset_path=f"{split}/{label}/{sample_id}.jpeg",
                                   view_position="UNKNOWN", view_position_missing=1,
                                   metadata_source="not_provided_in_local_release"))
    verify_disjoint(chosen)
    if len({r["sample_id"] for r in chosen}) != len(chosen):
        raise SystemExit("Sample ID collision")
    output.mkdir(parents=True)
    for row in chosen:
        destination = output / row["subset_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / row["source_path"], destination)
        if sha(destination.read_bytes()) != row["sha256"]:
            raise RuntimeError(f"Copy checksum mismatch: {destination}")
    write_csv(output / "manifest.csv", chosen)
    write_csv(reports / "source_inventory.csv", rows)
    audit = dict(seed=cfg["seed"], config=cfg,
                 source_image_count=len(rows),
                 source_counts=dict(sorted(Counter(f'{r["source_split"]}/{r["label"]}' for r in rows).items())),
                 connected_group_count=len(groups),
                 duplicate_byte_groups=sum(n > 1 for n in Counter(r["sha256"] for r in rows).values()),
                 duplicate_pixel_groups=sum(n > 1 for n in Counter(r["pixel_sha256"] for r in rows).values()),
                 conflicting_label_paths=conflicts,
                 development_paths_excluded_due_to_test_overlap=overlaps,
                 available_unique_groups={p: {label: len(v[label]) for label in LABELS} for p,v in pools.items()},
                 ignored_noncanonical_directories=[p.name for p in sorted(source.iterdir())
                                                   if p.is_dir() and p.name not in SPLITS],
                 subset_counts=dict(sorted(Counter(f'{r["split"]}/{r["label"]}' for r in chosen).items())),
                 subset_images=len(chosen), subset_bytes=sum((output/r["subset_path"]).stat().st_size for r in chosen),
                 cross_split_proxy_and_hash_overlap=0,
                 real_patient_identity_verified=False,
                 auxiliary_metadata_missing_count=len(chosen))
    (reports / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ("source_image_count", "subset_counts", "subset_bytes")}, indent=2))


if __name__ == "__main__":
    main()
