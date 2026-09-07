"""ResNet18 with explicit frozen BatchNorm behavior and safe checkpoint loading."""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from common import LABELS, PROJECT_ROOT, sha256


def configure_trainable(model, freeze_backbone: bool):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if not freeze_backbone:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True


def training_mode(model, freeze_backbone: bool):
    # requires_grad=False alone would still allow BatchNorm running statistics to change.
    model.eval()
    model.fc.train()
    if not freeze_backbone:
        model.layer4.train()


def build_model(cfg: dict):
    weights = ResNet18_Weights.IMAGENET1K_V1
    if cfg.get("pretrained_weights"):
        weights_path = Path(cfg["pretrained_weights"])
    else:
        torch.hub.set_dir(str(PROJECT_ROOT / ".torch"))
        weights_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    expected_prefix = Path(weights.url).stem.split("-")[-1]
    if weights_path.exists():
        if not sha256(weights_path).startswith(expected_prefix):
            raise ValueError(f"Pretrained weights checksum mismatch: {weights_path}")
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    elif cfg.get("pretrained_weights"):
        raise FileNotFoundError(f"Pretrained weights missing: {weights_path}")
    else:
        try:
            model = resnet18(weights=weights, progress=True)
        except Exception as exc:
            raise RuntimeError("Cannot load ImageNet weights. Enable network access, or set pretrained_weights to the official resnet18-f37072fd.pth file. Random-weight fallback is not used.") from exc
    model.fc = nn.Linear(model.fc.in_features, 2)
    configure_trainable(model, cfg["freeze_backbone"])
    return model, {"name": weights.name, "url": weights.url, "sha256": sha256(weights_path)}


def load_checkpoint(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("labels") != list(LABELS):
        raise ValueError("Unsupported checkpoint format or label order")
    model = resnet18(weights=None, num_classes=2)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
