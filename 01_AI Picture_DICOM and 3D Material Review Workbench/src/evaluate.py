"""Evaluate the saved ResNet18 checkpoint on the original test image manifest."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import LABELS, load_config, read_manifest, source_path, write_csv, write_json
from metrics import binary_metrics
from modeling import ManifestDataset, build_checkpoint_model, choose_device, evaluation_transform


def case_type(target: int, prediction: int) -> str:
    if target == 1 and prediction == 1:
        return "true_positive"
    if target == 0 and prediction == 0:
        return "true_negative"
    if target == 0:
        return "false_positive"
    return "false_negative"


def export_case_examples(rows: list[dict[str, str]], config: dict, maximum_per_case: int = 2) -> None:
    """Copy a small, traceable review set; originals remain untouched in the external source."""
    destination_root = config["outputs_dir"] / "error_cases"
    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item["case_type"], item["score"], item["sample_id"])):
        if len(selected[row["case_type"]]) < maximum_per_case:
            selected[row["case_type"]].append(row)
    case_rows: list[dict[str, str]] = []
    for kind, examples in selected.items():
        directory = destination_root / kind
        directory.mkdir(parents=True, exist_ok=True)
        for row in examples:
            original = source_path(config["data_root"], row)
            copy_name = f"{row['sample_id']}_{original.name}"
            destination = directory / copy_name
            if not destination.exists():
                shutil.copy2(original, destination)
            case_rows.append(
                {
                    "case_type": kind,
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "prediction": row["prediction"],
                    "score": row["score"],
                    "source_path": row["source_path"],
                    "review_copy": destination.relative_to(config["outputs_dir"]).as_posix(),
                }
            )
    write_csv(
        config["outputs_dir"] / "case_review.csv",
        case_rows,
        ["case_type", "sample_id", "label", "prediction", "score", "source_path", "review_copy"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to JSON config relative to project root.")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    train_config = config["training"]
    manifest = read_manifest(config["splits_dir"] / "manifest.csv")
    test_rows = [row for row in manifest if row["split"] == "test"]
    if {row["label"] for row in test_rows} != set(LABELS):
        raise ValueError("The test manifest must contain both NORMAL and PNEUMONIA cases")
    checkpoint_path = config["outputs_dir"] / "resnet18_frozen_backbone.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint_path}. Run `python src/train.py` first.")

    device = choose_device()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with PyTorch releases before `weights_only`.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("labels") != list(LABELS):
        raise ValueError("Checkpoint labels do not match this project")
    model = build_checkpoint_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    loader = DataLoader(
        ManifestDataset(test_rows, config["data_root"], evaluation_transform(int(train_config["image_size"]))),
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=int(train_config["num_workers"]),
    )
    threshold = float(train_config["threshold"])
    predictions: list[dict[str, str | float]] = []
    labels: list[int] = []
    scores: list[float] = []
    rows_by_id = {row["sample_id"]: row for row in test_rows}
    with torch.inference_mode():
        for images, targets, sample_ids in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            for sample_id, target, score in zip(sample_ids, targets.tolist(), probabilities):
                predicted = int(score >= threshold)
                source = rows_by_id[sample_id]
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "source_path": source["source_path"],
                        "label": LABELS[target],
                        "score": float(score),
                        "prediction": LABELS[predicted],
                        "case_type": case_type(target, predicted),
                    }
                )
                labels.append(target)
                scores.append(float(score))
    metrics = binary_metrics(labels, scores, threshold)
    outputs: Path = config["outputs_dir"]
    write_csv(
        outputs / "predictions.csv",
        predictions,
        ["sample_id", "source_path", "label", "score", "prediction", "case_type"],
    )
    export_case_examples(predictions, config)
    result = {
        "evaluation_split": "original test directory, represented by the deterministic test manifest",
        "test_images": len(test_rows),
        "positive_class": "PNEUMONIA",
        "threshold_policy": "Fixed at 0.5 before validation and test evaluation; no test-set tuning.",
        "metrics": metrics,
        "checkpoint": checkpoint_path.name,
        "pretrained_weights": checkpoint.get("pretrained_weights"),
        "auxiliary_wbc_crp_used": False,
        "case_review_examples_per_type": 2,
    }
    write_json(outputs / "metrics.json", result)
    print(
        f"Test AUROC={metrics['auroc']:.4f}, sensitivity={metrics['sensitivity']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}; saved outputs to {outputs}"
    )


if __name__ == "__main__":
    main()
