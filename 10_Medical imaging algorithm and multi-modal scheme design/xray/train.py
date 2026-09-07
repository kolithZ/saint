from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path
import shutil
import time
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .common import LABELS, SPLITS, counts, new_output, read_csv, read_json, sample_path, sha256, write_csv, write_json
from .evaluation import PREDICTION_FIELDS, export_cases, metrics, predictions
from .model import PREPROCESS, WEIGHTS, extract, load_backbone, numeric_scores
from .splits import select_rows


def run(args):
    started = time.monotonic()
    split_dir = args.manifest_dir.resolve()
    summary = read_json(split_dir / "split_summary.json")
    if sha256(split_dir / "manifest.csv") != summary["manifest_sha256"]:
        raise ValueError("Locked manifest checksum mismatch")
    root = args.data_root.resolve() if args.data_root else Path(summary["data_root"])
    rows = read_csv(split_dir / "manifest.csv")
    seen = set()
    for row in rows:
        sample_path(root, row["relative_path"])
        if row["relative_path"] in seen or Path(row["relative_path"]).parts[:2] != (row["split"], row["label"]) or int(row["label_id"]) != LABELS[row["label"]]:
            raise ValueError("Invalid manifest labels or duplicate paths")
        seen.add(row["relative_path"])
    clean, _ = select_rows(rows)
    if len(clean) != len(rows):
        raise ValueError("Manifest still contains unsupported images or duplicate/group overlaps")
    by_split = {s: [r for r in rows if r["split"] == s] for s in SPLITS}
    if any(not group for group in by_split.values()) or {r["label"] for r in by_split["train"]} != set(LABELS):
        raise ValueError("Need nonempty splits and both training classes")
    if args.batch_size < 1 or args.threads < 1 or args.C <= 0 or not np.isfinite(args.C) or not 0 <= args.threshold <= 1:
        raise ValueError("Invalid batch size, thread count, C or threshold")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")
    out = new_output(args.out, [root, split_dir])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    config = dict(seed=args.seed, device=args.device, threads=args.threads, batch_size=args.batch_size, weights=WEIGHTS,
                  preprocess=PREPROCESS, classifier=dict(C=args.C, class_weight="balanced", solver="lbfgs", max_iter=3000),
                  threshold=args.threshold, threshold_policy="fixed before validation and test; no optimization", aux_available=False,
                  manifest_sha256=summary["manifest_sha256"], data_root=str(root), counts=counts(rows), split_policy=summary["policy"],
                  patient_level_isolation_verified=False, evaluation_scope=summary["evaluation_scope"],
                  environment={k: importlib.metadata.version(k) for k in ("torch", "torchvision", "scikit-learn", "Pillow", "numpy", "scipy")},
                  python=platform.python_version(), platform=platform.platform(), case_rule="3 per outcome; errors furthest from threshold, correct nearest; path tie-break")
    write_json(out / "config.json", config)
    shutil.copyfile(split_dir / "manifest.csv", out / "manifest.csv")
    model = load_backbone(args.weights, args.device)
    # Store the full original checkpoint for portable offline inference.
    if args.weights:
        shutil.copyfile(args.weights, out / "backbone.pth")
    else:
        cached = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
        shutil.copyfile(cached, out / "backbone.pth")
    features = {}
    for split in ("train", "val"):
        print(f"Extracting {split}", flush=True)
        features[split] = extract(model, root, by_split[split], args.batch_size, args.device)
    y = np.array([int(r["label_id"]) for r in by_split["train"]])
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=args.seed))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(features["train"], y)
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    parameters = dict(mean=scaler.mean_, scale=scaler.scale_, coef=classifier.coef_, intercept=classifier.intercept_)
    np.savez_compressed(out / "classifier.npz", **parameters)
    val_scores = numeric_scores(features["val"], parameters)
    if not np.allclose(val_scores, pipeline.predict_proba(features["val"])[:, 1], atol=1e-12):
        raise ValueError("Serialized classifier differs from fitted classifier")
    validation = metrics([int(r["label_id"]) for r in by_split["val"]], val_scores, args.threshold)
    write_json(out / "validation_metrics.json", validation)
    write_csv(out / "validation_predictions.csv", predictions(by_split["val"], val_scores, args.threshold), PREDICTION_FIELDS)
    lock = dict(config_sha256=sha256(out / "config.json"), classifier_sha256=sha256(out / "classifier.npz"),
                backbone_sha256=sha256(out / "backbone.pth"), manifest_sha256=sha256(out / "manifest.csv"), threshold=args.threshold,
                test_used_for_fit_or_selection=False)
    write_json(out / "model_lock.json", lock)
    print("Model and threshold locked. Extracting final test features.", flush=True)
    features["test"] = extract(model, root, by_split["test"], args.batch_size, args.device)
    scores = numeric_scores(features["test"], parameters)
    test_metrics = metrics([int(r["label_id"]) for r in by_split["test"]], scores, args.threshold)
    predicted = predictions(by_split["test"], scores, args.threshold)
    write_csv(out / "test_predictions.csv", predicted, PREDICTION_FIELDS)
    write_json(out / "test_metrics.json", test_metrics)
    np.savez_compressed(out / "features.npz", **features)
    export_cases(root, predicted, out)
    report = dict(status="complete", validation=validation, test=test_metrics, elapsed_seconds=time.monotonic() - started,
                  classifier_iterations=classifier.n_iter_.tolist(), patient_level_isolation_verified=False,
                  evaluation_scope=config["evaluation_scope"], multimodal_experiment=False)
    write_json(out / "run_summary.json", report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Frozen ResNet18 + train-only standardization + balanced logistic regression")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Optional relocated dataset; every image hash must still match")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path, help="Official local resnet18-f37072fd.pth; otherwise torchvision downloads it")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError, ConvergenceWarning) as error:
        parser.exit(2, f"Baseline error: {error}\n")
