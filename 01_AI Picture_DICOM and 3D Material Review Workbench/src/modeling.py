"""PyTorch data loading and the frozen ImageNet-pretrained ResNet18 baseline."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from common import LABELS, configure_torch_cache, source_path


class ManifestDataset(Dataset):
    """Read image paths from a manifest, keeping raw images outside the project folder."""

    def __init__(self, rows: list[dict[str, str]], data_root: Path, transform) -> None:
        self.rows = rows
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = source_path(self.data_root, row)
        with Image.open(path) as image:
            image.load()
            tensor = self.transform(image.convert("RGB"))
        return tensor, LABELS.index(row["label"]), row["sample_id"]


def train_transform(image_size: int):
    """Lightweight augmentation suitable for a small chest X-ray proof of concept."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def evaluation_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_pretrained_model() -> tuple[nn.Module, str]:
    """Load ImageNet-pretrained ResNet18 and expose only its linear classifier to training."""
    torch.hub.set_dir(str(configure_torch_cache()))
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model, weights.name


def build_checkpoint_model() -> nn.Module:
    """Create the ResNet18 topology before loading a saved state dictionary."""
    return resnet18(weights=None, num_classes=len(LABELS))


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
