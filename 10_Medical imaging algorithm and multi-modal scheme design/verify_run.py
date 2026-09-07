#!/usr/bin/env python3
"""Verify model locks, train-only scaling and reproducible saved metrics without retraining."""
import argparse
import json
from pathlib import Path

import numpy as np

from xray.common import LABELS, SPLITS, counts, read_csv, read_json, sha256
from xray.evaluation import metrics, predictions
from xray.model import numeric_scores
from xray.splits import select_rows


def verify(run):
    run = Path(run)
    summary = read_json(run / "run_summary.json")
    if summary.get("status") != "complete":
        raise ValueError("Run did not complete; inspect run_summary.json and run.log")
    lock = read_json(run / "model_lock.json")
    for filename, key in [("config.json", "config_sha256"), ("manifest.csv", "manifest_sha256"), ("classifier.npz", "classifier_sha256"), ("backbone.pth", "backbone_sha256")]:
        if sha256(run / filename) != lock[key]:
            raise ValueError(f"Checksum mismatch: {filename}")
    config = read_json(run / "config.json")
    threshold = lock["threshold"]
    if not np.isfinite(threshold) or not 0 <= threshold <= 1 or config["threshold"] != threshold:
        raise ValueError("Invalid threshold or disagreement between config and model lock")
    if config["manifest_sha256"] != lock["manifest_sha256"]:
        raise ValueError("Config and model lock refer to different manifests")
    rows = read_csv(run / "manifest.csv")
    paths = set()
    for row in rows:
        parts = Path(row["relative_path"]).parts
        if (len(parts) != 3 or parts[:2] != (row["split"], row["label"]) or row["split"] not in SPLITS
                or row["label"] not in LABELS or int(row["label_id"]) != LABELS[row["label"]]
                or row["relative_path"] in paths):
            raise ValueError("Invalid labels, path or duplicate manifest row")
        paths.add(row["relative_path"])
    if counts(rows) != config["counts"]:
        raise ValueError("Configured sample counts disagree with manifest")
    selected, _ = select_rows(rows)
    if len(selected) != len(rows):
        raise ValueError("Locked manifest has duplicate, unsupported or overlapping samples")
    with np.load(run / "features.npz", allow_pickle=False) as features, np.load(run / "classifier.npz", allow_pickle=False) as parameters:
        if set(features.files) != set(SPLITS):
            raise ValueError("Feature archive must contain train, val and test")
        for split in SPLITS:
            expected_rows = sum(r["split"] == split for r in rows)
            if not expected_rows or features[split].shape != (expected_rows, 512) or not np.isfinite(features[split]).all():
                raise ValueError(f"Invalid {split} feature shape, sample count or non-finite values")
        for key, shape in (("mean", (512,)), ("scale", (512,)), ("coef", (1, 512)), ("intercept", (1,))):
            if parameters[key].shape != shape or not np.isfinite(parameters[key]).all():
                raise ValueError(f"Invalid classifier parameter: {key}")
        if (parameters["scale"] <= 0).any():
            raise ValueError("Classifier scales must be positive")
        training = features["train"].astype(np.float64)
        np.testing.assert_allclose(parameters["mean"], training.mean(axis=0), rtol=1e-10, atol=1e-10)
        expected_scale = training.std(axis=0)
        expected_scale[expected_scale == 0] = 1
        np.testing.assert_allclose(parameters["scale"], expected_scale, rtol=1e-10, atol=1e-10)
        for split, prefix in [("val", "validation"), ("test", "test")]:
            group = [r for r in rows if r["split"] == split]
            predicted = read_csv(run / f"{prefix}_predictions.csv")
            if [r["relative_path"] for r in group] != [r["relative_path"] for r in predicted]:
                raise ValueError("Prediction row order differs from locked manifest")
            scores = numeric_scores(features[split], parameters)
            np.testing.assert_allclose(scores, [float(r["pneumonia_score"]) for r in predicted], rtol=1e-12, atol=1e-12)
            expected_predictions = predictions(group, scores, threshold)
            for expected, actual in zip(expected_predictions, predicted, strict=True):
                for field in ("relative_path", "split", "true_label", "label_id", "predicted_label", "outcome", "aux_available"):
                    if actual[field] != str(expected[field]):
                        raise ValueError(f"Prediction field {field} disagrees with locked inputs: {expected['relative_path']}")
                if float(actual["threshold"]) != threshold:
                    raise ValueError("CSV prediction threshold disagrees with model lock")
            recomputed = metrics([int(r["label_id"]) for r in group], scores, threshold)
            if recomputed != read_json(run / f"{prefix}_metrics.json"):
                raise ValueError(f"Saved {split} metrics do not match recomputation")
            if summary[prefix] != recomputed:
                raise ValueError(f"Run summary {prefix} metrics do not match recomputation")
    return dict(status="passed", checks=["completed run and consistent summary", "artifact checksums", "manifest labels and counts", "no exact duplicate or candidate-prefix split overlaps", "finite feature and classifier arrays with matching shapes", "scaler equals training-only moments", "all validation and test prediction fields", "recomputed metrics"], patient_level_isolation_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.run), ensure_ascii=False, indent=2))
    except (ValueError, OSError, AssertionError, KeyError) as error:
        parser.exit(2, f"Verification error: {error}\n")


if __name__ == "__main__":
    main()
