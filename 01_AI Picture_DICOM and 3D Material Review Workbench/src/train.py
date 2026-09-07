"""Train the frozen-backbone ResNet18 baseline on the deterministic small subset."""

from __future__ import annotations

import argparse
import csv
import platform
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from common import LABELS, load_config, read_manifest, set_reproducible_seed, write_json
from metrics import binary_metrics
from modeling import (
    ManifestDataset,
    build_pretrained_model,
    choose_device,
    evaluation_transform,
    train_transform,
)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[float]]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    with torch.inference_mode():
        for images, targets, _ in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            labels.extend(targets.tolist())
            scores.extend(probabilities.detach().cpu().tolist())
    return labels, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to JSON config relative to project root.")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    train_config = config["training"]
    set_reproducible_seed(int(config["seed"]))

    manifest = read_manifest(config["splits_dir"] / "manifest.csv")
    train_rows = [row for row in manifest if row["split"] == "train"]
    val_rows = [row for row in manifest if row["split"] == "val"]
    if {row["label"] for row in train_rows} != set(LABELS):
        raise ValueError("The training manifest must contain both NORMAL and PNEUMONIA cases")
    if {row["label"] for row in val_rows} != set(LABELS):
        raise ValueError("The validation manifest must contain both NORMAL and PNEUMONIA cases")

    batch_size = int(train_config["batch_size"])
    workers = int(train_config["num_workers"])
    image_size = int(train_config["image_size"])
    train_generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        ManifestDataset(train_rows, config["data_root"], train_transform(image_size)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=train_generator,
    )
    val_loader = DataLoader(
        ManifestDataset(val_rows, config["data_root"], evaluation_transform(image_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    device = choose_device()
    try:
        model, pretrained_weights = build_pretrained_model()
    except Exception as exc:
        raise RuntimeError(
            "Unable to load ImageNet-pretrained ResNet18 weights. This baseline intentionally does not "
            "fall back to random initialization. Check your torchvision installation and weight download access."
        ) from exc
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=float(train_config["learning_rate"]))

    log_rows: list[dict] = []
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        n_examples = 0
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            n_examples += len(targets)
        val_labels, val_scores = predict(model, val_loader, device)
        validation = binary_metrics(val_labels, val_scores, float(train_config["threshold"]))
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / n_examples,
                "val_auroc": validation["auroc"],
                "val_sensitivity": validation["sensitivity"],
                "val_accuracy": validation["accuracy"],
            }
        )
        print(
            f"Epoch {epoch}/{train_config['epochs']}: loss={total_loss / n_examples:.4f}, "
            f"val AUROC={validation['auroc']:.4f}, sensitivity={validation['sensitivity']:.4f}"
        )

    outputs: Path = config["outputs_dir"]
    outputs.mkdir(parents=True, exist_ok=True)
    with (outputs / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    checkpoint = {
        "state_dict": model.state_dict(),
        "labels": list(LABELS),
        "pretrained_weights": pretrained_weights,
        "training_config": train_config,
        "seed": config["seed"],
    }
    torch.save(checkpoint, outputs / "resnet18_frozen_backbone.pt")
    write_json(
        outputs / "training_summary.json",
        {
            "model": "ResNet18 with ImageNet-pretrained frozen backbone and trainable linear classifier",
            "pretrained_weights": pretrained_weights,
            "device": str(device),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "train_images": len(train_rows),
            "validation_images": len(val_rows),
            "auxiliary_wbc_crp_used": False,
            "final_validation": log_rows[-1],
        },
    )
    print(f"Saved checkpoint and logs to {outputs}")


if __name__ == "__main__":
    main()
