#!/usr/bin/env python3
"""Offline single-image inference from a completed local run, without pickle."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from xray.common import read_json, sha256
from xray.model import image_tensor, load_backbone, numeric_scores


def predict(run_dir, image, device="cpu"):
    run_dir = Path(run_dir)
    lock = read_json(run_dir / "model_lock.json")
    for name, key in (("config.json", "config_sha256"), ("classifier.npz", "classifier_sha256"), ("backbone.pth", "backbone_sha256")):
        if sha256(run_dir / name) != lock[key]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    config = read_json(run_dir / "config.json")
    torch.set_num_threads(config["threads"])
    model = load_backbone(run_dir / "backbone.pth", device, lock["backbone_sha256"])
    with torch.inference_mode():
        features = model(image_tensor(image, config["preprocess"]).unsqueeze(0).to(device)).cpu().numpy()
    with np.load(run_dir / "classifier.npz", allow_pickle=False) as parameters:
        score = float(numeric_scores(features, parameters)[0])
    return dict(image=str(image), pneumonia_score=score, threshold=lock["threshold"],
                predicted_label="PNEUMONIA" if score >= lock["threshold"] else "NORMAL", aux_available=False,
                use="research workflow only; score is not a calibrated clinical probability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    try:
        print(json.dumps(predict(args.run, args.image, args.device), ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Prediction error: {error}\n")


if __name__ == "__main__":
    main()
