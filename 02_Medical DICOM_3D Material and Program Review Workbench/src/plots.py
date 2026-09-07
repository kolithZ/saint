"""Headless, file-based experiment figures; titles use portable Latin fonts."""
from pathlib import Path

from common import LABELS, image_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve


def image_grid(root: Path, rows: list[dict], output: Path, title: str, columns=3):
    count = max(1, len(rows))
    fig, axes = plt.subplots((count + columns - 1) // columns, columns,
                             figsize=(columns * 4, ((count + columns - 1) // columns) * 3.7), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, rows):
        with Image.open(image_path(root, row["path"])) as image:
            ax.imshow(image.convert("L"), cmap="gray")
        label = LABELS[int(row["label"])]
        caption = f"True: {label}"
        if "predicted_label" in row:
            caption += f" | Pred: {row['predicted_label']}\nP(PNEUMONIA)={float(row['p_pneumonia']):.3f}"
        caption += f"\n{Path(row['path']).name}"
        ax.set_title(caption, fontsize=8, wrap=True)
    if not rows:
        axes.flat[0].text(0.5, 0.5, "No cases in this category", ha="center", va="center")
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def training_curve(history: list[dict], output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [h["epoch"] for h in history]
    for key in ("train_loss", "val_loss"):
        axes[0].plot(epochs, [h[key] for h in history], marker="o", label=key)
    for key in ("val_sensitivity", "val_roc_auc"):
        axes[1].plot(epochs, [h[key] for h in history], marker="o", label=key)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Cross entropy")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Validation metric")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def evaluation_figures(metrics: dict, labels, probabilities, directory: Path):
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=18)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted label", ylabel="True label", title="Test confusion matrix")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        ax.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "ROC-AUC undefined: only one class", ha="center")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="Sensitivity", title="Test ROC", xlim=(0, 1), ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(directory / "roc_curve.png", dpi=140)
    plt.close(fig)
