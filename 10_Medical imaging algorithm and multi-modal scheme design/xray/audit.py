from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import platform
import re
from pathlib import Path

import PIL
from PIL import Image

from .common import EXTENSIONS, LABELS, MANIFEST_FIELDS, SPLITS, counts, new_output, sha256, write_csv, write_json


def candidate_group(name):
    stem = Path(name).stem
    match = re.fullmatch(r"(person\d+)_(?:virus|bacteria)_\d+(?:_\d+)*", stem, re.IGNORECASE)
    if match:
        return match[1].lower()
    match = re.fullmatch(r"((?:NORMAL2-)?IM-\d+)-\d+(?:-\d+)*", stem, re.IGNORECASE)
    return match[1].upper() if match else ""


def scan_paths(root):
    root = Path(root).resolve()
    paths, issues = [], []
    for split in SPLITS:
        for label in LABELS:
            folder = root / split / label
            if not folder.is_dir() or folder.is_symlink() or folder.parent.is_symlink():
                raise ValueError(f"Missing or symlink class directory: {folder}")
            for path in sorted(folder.iterdir()):
                relative = path.relative_to(root).as_posix()
                if path.name.startswith("."):
                    issues.append(dict(relative_path=relative, issue="hidden_ignored", detail=""))
                elif path.is_symlink() or not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    issues.append(dict(relative_path=relative, issue="unsupported_entry_ignored", detail="symlink, directory or unsupported format"))
                else:
                    paths.append(path)
    if not paths:
        raise ValueError("No supported images in the six class directories")
    return paths, issues


def inspect(root, out, compare_root=None):
    root = Path(root).resolve()
    paths, issues = scan_paths(root)
    comparison_paths = scan_paths(compare_root)[0] if compare_root else None
    out = new_output(out, [root] + ([compare_root] if compare_root else []))
    rows = []
    for i, path in enumerate(paths):
        relative = path.relative_to(root).as_posix()
        split, label, _ = Path(relative).parts
        group = candidate_group(path.name)
        row = dict.fromkeys(MANIFEST_FIELDS, "")
        row.update(relative_path=relative, split=split, label=label, label_id=LABELS[label], readable=False,
                   candidate_group_id=group, group_source="filename_heuristic" if group else "unmatched", patient_id_verified=False)
        try:
            row["file_sha256"] = sha256(path)
        except OSError as error:
            issues.append(dict(relative_path=relative, issue="file_hash_error", detail=str(error)))
        try:
            with Image.open(path) as im:
                im.load()
                row.update(width=im.width, height=im.height, mode=im.mode, format=im.format,
                           frame_count=getattr(im, "n_frames", 1))
                if row["frame_count"] != 1:
                    raise ValueError("Multi-frame image requires review")
                row["readable"] = True
                if im.mode in ("L", "RGB"):
                    pixel = hashlib.sha256(f"{im.width}x{im.height}:RGB:".encode())
                    pixel.update(im.convert("RGB").tobytes())
                    row["rgb_pixel_sha256"] = pixel.hexdigest()
                else:
                    issues.append(dict(relative_path=relative, issue="unsupported_mode", detail=im.mode))
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            issues.append(dict(relative_path=relative, issue="decode_error", detail=str(error)))
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"Audited {i + 1}/{len(paths)}", flush=True)
    duplicates, duplicate_summary = [], {}
    for field in ("file_sha256", "rgb_pixel_sha256"):
        groups = defaultdict(list)
        for row in rows:
            if row[field]:
                groups[row[field]].append(row)
        groups = {k: v for k, v in groups.items() if len(v) > 1}
        duplicate_summary[field] = {"groups": len(groups), "cross_split_groups": sum(len({r["split"] for r in v}) > 1 for v in groups.values()),
                                    "conflicting_label_groups": sum(len({r["label"] for r in v}) > 1 for v in groups.values())}
        for digest, members in groups.items():
            for row in members:
                duplicates.append(dict(hash_type=field, digest=digest, relative_path=row["relative_path"], split=row["split"], label=row["label"],
                                       cross_split=len({r["split"] for r in members}) > 1, label_conflict=len({r["label"] for r in members}) > 1))
    groups = defaultdict(list)
    for row in rows:
        if row["candidate_group_id"]:
            groups[row["candidate_group_id"]].append(row)
    overlaps = [dict(candidate_group_id=g, relative_path=r["relative_path"], split=r["split"], label=r["label"])
                for g, members in groups.items() if len({r["split"] for r in members}) > 1 for r in members]
    write_csv(out / "manifest.csv", rows, MANIFEST_FIELDS)
    write_csv(out / "issues.csv", issues, ["relative_path", "issue", "detail"])
    write_csv(out / "duplicates.csv", duplicates, ["hash_type", "digest", "relative_path", "split", "label", "cross_split", "label_conflict"])
    write_csv(out / "candidate_group_overlaps.csv", overlaps, ["candidate_group_id", "relative_path", "split", "label"])
    summary = dict(data_root=str(root), scope="Direct JPEG/PNG files in six class directories; hidden entries and symlinks excluded",
                   image_count=len(rows), counts_by_split=counts(rows), unreadable_count=sum(not r["readable"] for r in rows),
                   modes=dict(Counter(r["mode"] for r in rows)), issue_counts=dict(Counter(r["issue"] for r in issues)),
                   dimensions={axis: {"min": min((r[axis] for r in rows if r[axis] != ""), default=None),
                                     "max": max((r[axis] for r in rows if r[axis] != ""), default=None)} for axis in ("width", "height")},
                   duplicates=duplicate_summary, candidate_cross_split_groups=len({r["candidate_group_id"] for r in overlaps}),
                   unmatched_filenames=sum(not r["candidate_group_id"] for r in rows), patient_level_isolation_verified=False,
                   environment={"python": platform.python_version(), "Pillow": PIL.__version__}, manifest_sha256=sha256(out / "manifest.csv"))
    if compare_root:
        compare_root = Path(compare_root).resolve()
        first = {r["relative_path"]: r["file_sha256"] for r in rows}
        second = {p.relative_to(compare_root).as_posix(): sha256(p) for p in comparison_paths}
        different = sorted(k for k in first.keys() & second.keys() if first[k] != second[k])
        write_json(out / "root_comparison.json", dict(data_root=str(root), compare_root=str(compare_root),
                   primary_image_count=len(first), comparison_image_count=len(second), only_primary=sorted(first.keys() - second.keys()),
                   only_comparison=sorted(second.keys() - first.keys()), different_bytes=different,
                   identical_image_paths_and_bytes_in_scope=first == second))
    write_json(out / "summary.json", summary)
    print(summary, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Read-only chest X-ray audit; never modifies images or overwrites reports")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--compare-root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("audit_outputs"))
    args = parser.parse_args()
    try:
        inspect(args.data_root, args.out, args.compare_root)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Audit error: {error}\n")
