from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .common import LABELS, MANIFEST_FIELDS, counts, new_output, read_csv, read_json, sample_path, sha256, write_csv, write_json

POLICY = "quarantine-label-conflicts; exact-deduplicate-test>val>train; heuristic-isolate-test>val>train; no-resplit-v1"


def select_rows(rows):
    """Keep original splits; selection never uses a model score or a random image split."""
    reasons = {}
    for i, row in enumerate(rows):
        if str(row["readable"]).lower() != "true" or row["mode"] not in ("L", "RGB") or str(row["frame_count"]) != "1" or not row["file_sha256"] or not row["rgb_pixel_sha256"]:
            reasons[i] = "unreadable_or_unsupported_image"
    # Connected components handle byte/pixel matches transitively, including conflicts.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for field in ("file_sha256", "rgb_pixel_sha256"):
        seen = {}
        for i, row in enumerate(rows):
            key = row[field]
            if key:
                if key in seen:
                    parent[find(i)] = find(seen[key])
                seen[key] = i
    components = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    priority = {"test": 0, "val": 1, "train": 2}
    for members in components.values():
        if len({rows[i]["label"] for i in members}) > 1:
            for i in members:
                reasons[i] = "exact_duplicate_label_conflict"
        else:
            eligible = sorted((i for i in members if i not in reasons), key=lambda i: (priority[rows[i]["split"]], rows[i]["relative_path"]))
            for i in eligible[1:]:
                reasons[i] = "exact_duplicate_keep:" + rows[eligible[0]]["relative_path"]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if i not in reasons and row["candidate_group_id"]:
            groups[row["candidate_group_id"]].append(i)
    for group, members in groups.items():
        keep_split = min((rows[i]["split"] for i in members), key=priority.get)
        for i in members:
            if rows[i]["split"] != keep_split:
                reasons[i] = f"heuristic_overlap_keep_{keep_split}:{group}"
    records = [dict(row, included=i not in reasons, exclusion_reason=reasons.get(i, ""), final_split=row["split"] if i not in reasons else "") for i, row in enumerate(rows)]
    return [r for i, r in enumerate(rows) if i not in reasons], records


def prepare(audit_dir, out):
    audit_dir = Path(audit_dir).resolve()
    summary = read_json(audit_dir / "summary.json")
    if sha256(audit_dir / "manifest.csv") != summary["manifest_sha256"]:
        raise ValueError("Audit manifest changed since inspection")
    rows = read_csv(audit_dir / "manifest.csv")
    paths = set()
    for row in rows:
        sample_path(summary["data_root"], row["relative_path"])
        if row["relative_path"] in paths or Path(row["relative_path"]).parts[:2] != (row["split"], row["label"]) or int(row["label_id"]) != LABELS[row["label"]]:
            raise ValueError("Inconsistent or duplicate manifest row")
        paths.add(row["relative_path"])
    selected, records = select_rows(rows)
    if {r["label"] for r in selected if r["split"] == "train"} != set(LABELS):
        raise ValueError("Training must retain both classes")
    if any(not any(r["split"] == s for r in selected) for s in ("val", "test")):
        raise ValueError("Validation and test must remain nonempty")
    out = new_output(out, [summary["data_root"], audit_dir])
    write_csv(out / "manifest.csv", selected, MANIFEST_FIELDS)
    write_csv(out / "selection_records.csv", records, MANIFEST_FIELDS + ["included", "exclusion_reason", "final_split"])
    report = dict(policy=POLICY, data_root=summary["data_root"], audit_manifest_sha256=summary["manifest_sha256"],
                  manifest_sha256=sha256(out / "manifest.csv"), original_counts=counts(rows), selected_counts=counts(selected),
                  excluded_count=len(rows) - len(selected), exclusion_counts=dict(Counter(r["exclusion_reason"].split(":")[0] for r in records if not r["included"])),
                  patient_level_isolation_verified=False, evaluation_scope="limited workflow experiment with heuristic filename isolation; not verified patient-independent",
                  test_policy="No reassignment; only exact duplicates, label conflicts and unsupported images may be excluded from original test")
    write_json(out / "split_summary.json", report)
    print(report, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Lock an auditable heuristic-isolated experiment manifest")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.audit, args.out)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Manifest error: {error}\n")
