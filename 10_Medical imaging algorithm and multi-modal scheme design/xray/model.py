from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as TF

from .common import sample_path, sha256

PREPROCESS = {"size": 224, "interpolation": "bilinear", "padding": "center_black", "grayscale": "Pillow L replicated to RGB",
              "orientation": "EXIF transpose", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "crop": False}
WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"


def prepare_image(path, config=None):
    config = PREPROCESS if config is None else config
    with Image.open(path) as original:
        original.load()
        if original.mode not in ("L", "RGB") or getattr(original, "n_frames", 1) != 1:
            raise ValueError(f"Unsupported mode or frame count: {path}")
        im = ImageOps.exif_transpose(original).convert("L")
        size = config["size"]
        ratio = size / max(im.size)
        dims = tuple(max(1, min(size, round(value * ratio))) for value in im.size)
        im = im.resize(dims, Image.Resampling.BILINEAR)
        canvas = Image.new("L", (size, size), 0)
        canvas.paste(im, ((size - dims[0]) // 2, (size - dims[1]) // 2))
        return canvas.convert("RGB")


def image_tensor(path, config=None):
    config = PREPROCESS if config is None else config
    return TF.normalize(TF.to_tensor(prepare_image(path, config)), config["mean"], config["std"])


def load_backbone(weights_path=None, device="cpu", expected_sha256=None):
    if weights_path:
        weights_path = Path(weights_path)
        digest = sha256(weights_path)
        if expected_sha256:
            if digest != expected_sha256:
                raise ValueError("Backbone checksum mismatch")
        elif not digest.startswith("f37072fd"):
            raise ValueError("Expected official ResNet18 IMAGENET1K_V1 checkpoint (f37072fd)")
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    else:
        # Explicit version; default torchvision geometry is deliberately not applied.
        state = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
        model = resnet18(weights=None)
        model.load_state_dict(state)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


class ImageDataset(Dataset):
    def __init__(self, root, rows, config=None):
        self.root, self.rows, self.config = root, rows, config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = sample_path(self.root, row["relative_path"])
        # Verify immediately before feature extraction, including each test image.
        if sha256(path) != row["file_sha256"]:
            raise ValueError(f"Image changed since audit: {row['relative_path']}")
        return image_tensor(path, self.config)


def extract(model, root, rows, batch_size=32, device="cpu", config=None):
    if not rows:
        return np.empty((0, 512), dtype=np.float32)
    loader = DataLoader(ImageDataset(root, rows, config), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            chunks.append(model(batch.to(device)).cpu().numpy())
            if (index + 1) % 10 == 0 or index + 1 == len(loader):
                print(f"Features {min((index + 1) * batch_size, len(rows))}/{len(rows)}", flush=True)
    features = np.concatenate(chunks)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite extracted features")
    return features


def numeric_scores(features, parameters):
    logits = ((features - parameters["mean"]) / parameters["scale"]) @ parameters["coef"].ravel() + parameters["intercept"].item()
    # Stable sigmoid without unsafe pickle/joblib model loading.
    result = np.empty_like(logits, dtype=np.float64)
    mask = logits >= 0
    result[mask] = 1 / (1 + np.exp(-logits[mask]))
    exp = np.exp(logits[~mask])
    result[~mask] = exp / (1 + exp)
    return result
