"""Train fc or layer4+fc; choose the checkpoint solely by validation cross entropy."""
import platform
from pathlib import Path
import time

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from common import (LABELS, data_fingerprint, discover, load_config, parser, project_path,
                    seed_everything, seed_worker, sha256, write_csv, write_json)
from dataset import XRayDataset, load_subset, readable_rows
from metrics import binary_metrics
from model import build_model, training_mode
from plots import training_curve


@torch.inference_mode()
def infer(model, loader, device):
    model.eval()
    predictions, loss_sum = [], 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for images, labels, paths in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += loss_fn(logits, labels).item()
        probs = logits.softmax(dim=1).cpu().tolist()
        for path, label, prob in zip(paths, labels.cpu().tolist(), probs):
            predicted = int(prob[1] > prob[0])
            predictions.append({"image_path": path, "true_label": LABELS[label],
                                "predicted_label": LABELS[predicted], "p_normal": prob[0], "p_pneumonia": prob[1]})
    if not predictions:
        raise ValueError("Cannot evaluate an empty dataset")
    metrics = binary_metrics([LABELS.index(p["true_label"]) for p in predictions],
                             [p["p_pneumonia"] for p in predictions])
    metrics["loss"] = loss_sum / len(predictions)
    return metrics, predictions


def train(cfg: dict):
    start = time.monotonic()
    device = seed_everything(cfg)
    root, output = Path(cfg["data_root"]), Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("best.pt", "history.csv", "training_summary.json")):
        raise FileExistsError(f"Experiment already exists: {output}. Use --output-dir for a new experiment.")
    train_rows = load_subset(root, Path(cfg["subset_path"]), cfg["train_samples_per_class"])
    original_val = discover(root, "val")
    val_rows, errors = readable_rows(root, original_val)
    write_json(output / "validation_read_errors.json", errors)
    if set(r["label"] for r in val_rows) != {0, 1}:
        raise ValueError("Validation requires at least one readable image in each class")
    manifest_hash = sha256(Path(cfg["subset_path"]))
    train_hash, val_hash = data_fingerprint(root, train_rows), data_fingerprint(root, val_rows)
    write_csv(output / "train_subset.csv", train_rows, ["path", "label"])
    write_json(output / "resolved_config.json", cfg)
    model, weights = build_model(cfg)
    model.to(device)
    groups = [{"params": model.fc.parameters(), "lr": cfg["learning_rate"]}]
    if not cfg["freeze_backbone"]:
        groups.insert(0, {"params": model.layer4.parameters(), "lr": cfg["backbone_learning_rate"]})
    optimizer = torch.optim.Adam(groups)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(cfg["random_seed"])
    loader_options = {"batch_size": cfg["batch_size"], "num_workers": cfg["num_workers"], "worker_init_fn": seed_worker}
    train_loader = DataLoader(XRayDataset(root, train_rows, cfg["image_size"]), shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(XRayDataset(root, val_rows, cfg["image_size"]), shuffle=False, **loader_options)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device={device}; train={len(train_rows)}, val={len(val_rows)}; trainable parameters={trainable:,}", flush=True)
    history, best_loss, best_epoch = [], float("inf"), 0
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.monotonic()
        training_mode(model, cfg["freeze_backbone"])
        loss_sum, correct = 0.0, 0
        for step, (images, labels, _) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            if step % 10 == 0:
                print(f"Epoch {epoch}/{cfg['epochs']}, batch {step}/{len(train_loader)}", flush=True)
        val_metrics, _ = infer(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_rows), "train_accuracy": correct / len(train_rows),
               "val_loss": val_metrics["loss"], "val_sensitivity": val_metrics["sensitivity"],
               "val_roc_auc": val_metrics["roc_auc"], "seconds": time.monotonic() - epoch_start}
        history.append(row)
        write_csv(output / "history.csv", history, list(row))
        print(f"Epoch {epoch}: train_loss={row['train_loss']:.4f}, val_loss={row['val_loss']:.4f}, val_auc={row['val_roc_auc']:.4f}", flush=True)
        if val_metrics["loss"] < best_loss:
            best_loss, best_epoch = val_metrics["loss"], epoch
            checkpoint = {"format_version": 1, "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "labels": list(LABELS), "config": cfg, "epoch": epoch,
                          "selection": "minimum validation cross entropy; earliest epoch wins ties",
                          "validation_metrics": val_metrics, "manifest_sha256": manifest_hash,
                          "train_data_sha256": train_hash, "val_data_sha256": val_hash, "pretrained_weights": weights}
            temporary = output / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best.pt")
    summary = {"variant": "baseline" if cfg["freeze_backbone"] else "layer4_finetune", "device": str(device),
               "train_images": len(train_rows), "val_found": len(original_val), "val_used": len(val_rows),
               "excluded_validation": errors, "epochs": cfg["epochs"], "best_epoch": best_epoch,
               "best_val_loss": best_loss, "trainable_parameters": trainable,
               "total_parameters": sum(p.numel() for p in model.parameters()), "seconds": time.monotonic() - start,
               "manifest_sha256": manifest_hash, "train_data_sha256": train_hash, "val_data_sha256": val_hash,
               "pretrained_weights": weights, "checkpoint_sha256": sha256(output / "best.pt"),
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "torchvision": str(torchvision.__version__), "platform": platform.platform()},
               "selection": "Minimum val loss; test is never read by the training script."}
    training_curve(history, output / "training_curve.png")
    write_json(output / "training_summary.json", summary)
    print(f"Training complete. Best epoch={best_epoch}; saved {output / 'best.pt'}", flush=True)
    return summary


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--output-dir", help="Use a new directory for another training run")
    args = cli.parse_args()
    config = load_config(args)
    if args.output_dir:
        config["output_dir"] = str(project_path(args.output_dir))
    train(config)
