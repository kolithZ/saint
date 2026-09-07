"""Persist a deterministic balanced subset; existing manifests are reused, never silently replaced."""
from pathlib import Path
import random

from common import data_fingerprint, discover, load_config, parser, sha256, write_csv, write_json
from dataset import load_subset, readable_rows


def prepare(cfg: dict, overwrite=False):
    root, output = Path(cfg["data_root"]), Path(cfg["subset_path"])
    if output.exists() and not overwrite:
        rows = load_subset(root, output, cfg["train_samples_per_class"])
        print(f"Reusing validated manifest: {output} ({len(rows)} images)", flush=True)
        return rows
    valid, errors = readable_rows(root, discover(root, "train"))
    rng, selected = random.Random(cfg["random_seed"]), []
    count = cfg["train_samples_per_class"]
    for label in (0, 1):
        candidates = [r for r in valid if r["label"] == label]
        if len(candidates) < count:
            raise ValueError(f"Class {label} has {len(candidates)} readable images, fewer than requested {count}")
        selected.extend(rng.sample(candidates, count))
    selected.sort(key=lambda row: row["path"])
    write_csv(output, selected, ["path", "label"])
    write_json(output.with_suffix(".json"), {"random_seed": cfg["random_seed"], "samples_per_class": count,
                "manifest_sha256": sha256(output), "data_sha256": data_fingerprint(root, selected),
                "excluded_unreadable": errors, "selection": "Sorted train paths; random.Random(seed); balanced sampling without replacement."})
    print(f"Saved {len(selected)} training paths: {output}", flush=True)
    return selected


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--overwrite", action="store_true", help="Explicitly regenerate the fixed subset")
    args = cli.parse_args()
    prepare(load_config(args), args.overwrite)
