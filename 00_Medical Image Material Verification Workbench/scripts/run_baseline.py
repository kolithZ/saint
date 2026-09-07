"""CPU-only 2D grayscale + L2 logistic regression; no pretrained weights required."""
import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageOps

from prepare_subset import ROOT, image_info, sha, verify_disjoint, write_csv


def preprocess(path, size):
    with Image.open(path) as im:
        im = ImageOps.pad(im.convert("L"), (size, size), method=Image.Resampling.BILINEAR, color=0)
        return np.asarray(im, dtype=np.float64).ravel() / 255.0


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def fit_logistic(x, y, iterations, l2):
    # Weighted binary cross entropy + lambda/2 * ||w||^2, unpenalized bias.
    if l2 <= 0 or iterations < 1:
        raise ValueError("l2 and iterations must be positive")
    counts = np.bincount(y, minlength=2)
    if min(counts) == 0:
        raise ValueError("Training requires both classes")
    weights = len(y) / (2.0 * counts[y])
    design = np.column_stack([x, np.ones(len(x))])
    coef = np.zeros(design.shape[1])
    # Trace upper bound on Hessian spectral norm gives a conservative stable step.
    step = 1.0 / (0.25 * np.mean(weights * np.sum(design**2, axis=1)) + l2)
    history = []
    for i in range(iterations):
        z = design @ coef
        grad = design.T @ (weights * (sigmoid(z) - y)) / len(y)
        grad[:-1] += l2 * coef[:-1]
        coef -= step * grad
        if i % 100 == 0 or i == iterations - 1:
            z = design @ coef
            loss = np.mean(weights * (np.logaddexp(0, z) - y*z)) + l2/2 * np.sum(coef[:-1]**2)
            history.append({"iteration": i+1, "loss": float(loss)})
    if not np.isfinite(coef).all():
        raise ValueError("Non-finite fitted parameters")
    return coef, history, step


def evaluate(y, score, threshold):
    positive, negative = score[y == 1], score[y == 0]
    if not len(positive) or not len(negative):
        raise ValueError("AUC needs both classes")
    # Pairwise definition of AUROC, ties count as 0.5; appropriate for tiny sets.
    diff = positive[:, None] - negative[None, :]
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    return dict(n=len(y), auroc=float(np.mean((diff > 0) + 0.5*(diff == 0))),
                sensitivity=tp/(tp+fn), threshold=threshold,
                counts={"tn":tn, "fp":fp, "fn":fn, "tp":tp})


def contact_sheet(rows, output, source):
    canvas = Image.new("RGB", (1000, 4*280 + 50), "#121a24")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "Dataset labels vs CPU baseline / TP, TN, FP, FN / research only", fill="white")
    for row_index, category in enumerate(("TP", "TN", "FP", "FN")):
        selected = sorted((r for r in rows if r["case_type"] == category), key=lambda r:r["sample_id"])[:4]
        for col in range(4):
            x, y = col*250, 50+row_index*280
            if col >= len(selected):
                draw.text((x+10, y+10), f"{category}: no more cases", fill="white")
                continue
            row = selected[col]
            with Image.open(source/row["subset_path"]) as im:
                thumb = ImageOps.contain(im.convert("RGB"), (238, 210))
                canvas.paste(thumb, (x+(250-thumb.width)//2, y))
            draw.text((x+8, y+216), f'{category} {row["sample_id"]}', fill="white")
            draw.text((x+8, y+233), f'label={row["label"]}', fill="white")
            draw.text((x+8, y+250), f'score={float(row["score"]):.3f}', fill="white")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    subset, reports = ROOT/cfg["output"], ROOT/cfg["reports"]
    settings = cfg["baseline"]
    with (subset/"manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    verify_disjoint(rows)
    if len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("This experiment requires one image per connected group")
    arrays = {}
    for split in ("train", "val", "test"):
        selected = [r for r in rows if r["split"] == split]
        if not selected:
            raise ValueError(f"Empty split: {split}")
        for row in selected:
            path = subset/row["subset_path"]
            if sha(path.read_bytes()) != row["sha256"] or image_info(path)["pixel_sha256"] != row["pixel_sha256"]:
                raise ValueError(f"Manifest checksum mismatch: {path}")
        x = np.array([preprocess(subset/r["subset_path"], settings["image_size"]) for r in selected])
        y = np.array([int(r["label"] == "PNEUMONIA") for r in selected])
        arrays[split] = (x, y, selected)
    # Fit feature statistics using training data only.
    mean = arrays["train"][0].mean(axis=0)
    scale = np.maximum(arrays["train"][0].std(axis=0), 1e-6)
    coef, history, step = fit_logistic((arrays["train"][0]-mean)/scale, arrays["train"][1],
                                     settings["iterations"], settings["l2"])
    result = dict(model="16x16 grayscale + NumPy weighted L2 logistic regression", config=settings,
                  threshold_policy="fixed 0.5 before validation/test; no tuning on test",
                  learning_rate=step, training_loss=history,
                  manifest_sha256=sha((subset/"manifest.csv").read_bytes()),
                  environment={"python":platform.python_version(), "numpy":np.__version__, "pillow":PIL.__version__},
                  auxiliary_metadata_used=False, metrics={})
    predictions = []
    for split in ("train", "val", "test"):
        x, y, selected = arrays[split]
        score = sigmoid(((x-mean)/scale) @ coef[:-1] + coef[-1])
        result["metrics"][split] = evaluate(y, score, settings["threshold"])
        for row, value, target in zip(selected, score, y):
            pred = int(value >= settings["threshold"])
            category = ("TP" if pred else "FN") if target else ("FP" if pred else "TN")
            predictions.append(dict(sample_id=row["sample_id"], split=split, label=row["label"],
                                    subset_path=row["subset_path"], score=float(value),
                                    prediction="PNEUMONIA" if pred else "NORMAL", case_type=category))
    reports.mkdir(parents=True, exist_ok=True)
    (reports/"baseline_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(reports/"predictions.csv", predictions)
    np.savez(reports/"baseline_model.npz", mean=mean, scale=scale, coef=coef)
    contact_sheet([r for r in predictions if r["split"] == "test"], reports/"test_examples.png", subset)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
