"""Evaluate all readable original test images and export TP/TN/FP/FN examples."""
from pathlib import Path
import random

from torch.utils.data import DataLoader

from common import (LABELS, data_fingerprint, discover, load_config, parser, project_path,
                    read_csv, seed_everything, sha256, write_csv, write_json)
from dataset import XRayDataset, readable_rows
from model import load_checkpoint
from plots import evaluation_figures, image_grid
from train import infer


def evaluate(cfg: dict, checkpoint_path: Path, output: Path):
    device = seed_everything(cfg)
    root = Path(cfg["data_root"])
    if (output / "best.pt").exists() and sha256(output / "best.pt") != sha256(checkpoint_path):
        raise ValueError("Output directory belongs to another checkpoint; use a separate --output-dir")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    all_rows = discover(root, "test")
    rows, errors = readable_rows(root, all_rows)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "test_read_errors.json", errors)
    loader = DataLoader(XRayDataset(root, rows, checkpoint["config"]["image_size"]),
                        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], shuffle=False)
    metrics, predictions = infer(model, loader, device)
    metrics.update({"split": "test", "test_found": len(all_rows), "test_used": len(rows), "test_excluded": len(errors),
                    "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
                    "best_epoch": checkpoint["epoch"], "manifest_sha256": checkpoint["manifest_sha256"],
                    "train_data_sha256": checkpoint["train_data_sha256"], "val_data_sha256": checkpoint["val_data_sha256"],
                    "test_data_sha256": data_fingerprint(root, rows), "training_config": checkpoint["config"],
                    "pretrained_weights": checkpoint["pretrained_weights"],
                    "limitation": "Internal dataset evaluation only. Original splits retained; patient isolation requires reliable metadata. Not for clinical use."})
    write_csv(output / "predictions.csv", predictions, ["image_path", "true_label", "predicted_label", "p_normal", "p_pneumonia"])
    write_json(output / "metrics.json", metrics)
    evaluation_figures(metrics, [LABELS.index(r["true_label"]) for r in predictions], [r["p_pneumonia"] for r in predictions], output)
    groups = {"TN": [], "TP": [], "FN": [], "FP": []}
    for prediction in predictions:
        truth = prediction["true_label"] == "PNEUMONIA"
        positive = prediction["predicted_label"] == "PNEUMONIA"
        category = "TP" if truth and positive else "FN" if truth else "FP" if positive else "TN"
        groups[category].append(prediction)
    rng, review = random.Random(checkpoint["config"]["random_seed"]), []
    review_file = output / "case_review.csv"
    previous = {r["image_path"]: r for r in read_csv(review_file)} if review_file.exists() else {}
    review_fields = ["blur", "exposure", "cropping", "acquisition_difference", "review_notes"]
    for category, candidates in groups.items():
        selected = rng.sample(candidates, min(3, len(candidates)))
        images = [{**r, "path": r["image_path"], "label": LABELS.index(r["true_label"])} for r in selected]
        image_grid(root, images, output / "cases" / f"{category}.png", f"{category}: {len(candidates)} available, {len(selected)} shown")
        for prediction in selected:
            annotations = {key: previous.get(prediction["image_path"], {}).get(key, "") for key in review_fields}
            review.append({"category": category, **prediction, **annotations})
    write_csv(output / "case_review.csv", review, ["category", "image_path", "true_label", "predicted_label", "p_normal", "p_pneumonia", "blur", "exposure", "cropping", "acquisition_difference", "review_notes"])
    print(f"Test n={metrics['n']}, Sensitivity={metrics['sensitivity']}, ROC-AUC={metrics['roc_auc']}; outputs: {output}", flush=True)
    return metrics


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--checkpoint", help="Default: <output_dir>/best.pt")
    cli.add_argument("--output-dir", help="Default: config output_dir")
    args = cli.parse_args()
    config = load_config(args)
    output_dir = project_path(args.output_dir) if args.output_dir else Path(config["output_dir"])
    checkpoint_file = project_path(args.checkpoint) if args.checkpoint else Path(config["output_dir"]) / "best.pt"
    evaluate(config, checkpoint_file, output_dir)
