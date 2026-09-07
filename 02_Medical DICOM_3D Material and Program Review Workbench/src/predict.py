"""Run image-only inference on one image using the saved preprocessing configuration."""
import json
from pathlib import Path

from PIL import Image
import torch

from common import LABELS, load_config, parser, project_path, seed_everything, write_json
from dataset import preprocessing
from model import load_checkpoint


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--image", required=True)
    cli.add_argument("--checkpoint")
    cli.add_argument("--output", help="Optional JSON file")
    args = cli.parse_args()
    cfg = load_config(args)
    device = seed_everything(cfg)
    checkpoint_path = project_path(args.checkpoint) if args.checkpoint else Path(cfg["output_dir"]) / "best.pt"
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    path = project_path(args.image)
    with Image.open(path) as image:
        tensor = preprocessing(checkpoint["config"]["image_size"])(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(1)[0].cpu().tolist()
    result = {"image_path": str(path), "predicted_label": LABELS[int(probabilities[1] > probabilities[0])],
              "p_normal": probabilities[0], "p_pneumonia": probabilities[1],
              "usage": "Algorithm research only; not a clinical diagnosis."}
    if args.output:
        write_json(project_path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
