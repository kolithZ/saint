from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from PIL import Image, ImageDraw, ImageOps

from .common import sample_path, write_csv
from .model import prepare_image

PREDICTION_FIELDS = ["relative_path", "split", "true_label", "label_id", "pneumonia_score", "threshold", "predicted_label", "outcome", "aux_available"]


def metrics(labels, scores, threshold=0.5):
    y = np.asarray(labels)
    p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected equal one-dimensional binary labels and finite scores in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0, 1]")
    y = y.astype(int)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist() if y.size else [0, 0, 0, 0]
    return dict(n=len(y), threshold=threshold, roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                roc_auc_unavailable_reason=None if len(np.unique(y)) == 2 else "evaluation requires both classes",
                pneumonia_recall=tp / (tp + fn) if tp + fn else None,
                recall_unavailable_reason=None if tp + fn else "no positive labels", TP=tp, FP=fp, TN=tn, FN=fn)


def predictions(rows, scores, threshold):
    result = []
    for row, score in zip(rows, scores, strict=True):
        true = int(row["label_id"])
        pred = int(score >= threshold)
        outcome = ("TP" if true else "FP") if pred else ("FN" if true else "TN")
        result.append(dict(relative_path=row["relative_path"], split=row["split"], true_label=row["label"], label_id=true,
                           pneumonia_score=float(score), threshold=threshold, predicted_label="PNEUMONIA" if pred else "NORMAL",
                           outcome=outcome, aux_available=False))
    return result


def export_cases(root, prediction_rows, out, per_outcome=3):
    # Fixed rule decided before evaluation: errors furthest past threshold; correct
    # examples nearest threshold. Stable path tie-break. No medical relabeling.
    cases = []
    for outcome in ("TP", "TN", "FN", "FP"):
        subset = [r for r in prediction_rows if r["outcome"] == outcome]
        subset.sort(key=lambda r: ((-1 if outcome in ("FN", "FP") else 1) * abs(r["pneumonia_score"] - r["threshold"]), r["relative_path"]))
        for row in subset[:per_outcome]:
            cases.append(dict(row, review_note="Original/preprocessed geometry and brightness for engineering review; dataset label unchanged"))
    write_csv(out / "cases.csv", cases, PREDICTION_FIELDS + ["review_note"])
    if not cases:
        return
    sheet = Image.new("RGB", (740, 278 * len(cases)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(cases):
        path = sample_path(root, row["relative_path"])
        with Image.open(path) as original:
            preview = ImageOps.contain(original.convert("RGB"), (224, 224))
        sheet.paste(preview, (0, i * 278 + 30))
        sheet.paste(prepare_image(path), (242, i * 278 + 30))
        draw.text((0, i * 278 + 6), f"{row['outcome']}   PNEUMONIA score={row['pneumonia_score']:.6f}   original / processed", fill="black")
        draw.text((0, i * 278 + 256), row["relative_path"], fill="black")
    sheet.save(out / "cases.png")
