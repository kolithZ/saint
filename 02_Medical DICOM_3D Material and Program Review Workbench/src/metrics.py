"""PNEUMONIA is positive; ties predict NORMAL, as with two-logit argmax."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def binary_metrics(labels, probabilities) -> dict:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or labels.size == 0 or labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must be nonempty vectors of equal length")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    predicted = (probabilities > 0.5).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    return {
        "n": int(labels.size), "positive_class": "PNEUMONIA", "decision_rule": "p_pneumonia > 0.5",
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "accuracy": (tp + tn) / labels.size,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
