"""Read the original images in place; never re-partition val or test."""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common import image_path, read_csv

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocessing(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def readable_rows(root: Path, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    valid, errors = [], []
    for row in rows:
        try:
            with Image.open(image_path(root, row["path"])) as image:
                image.load()
                image.convert("RGB")
            valid.append(row)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            errors.append({"path": row["path"], "error": str(exc)})
    return valid, errors


def load_subset(root: Path, path: Path, samples_per_class: int) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run src/prepare_train_subset.py first")
    rows = read_csv(path)
    seen, counts = set(), [0, 0]
    for row in rows:
        try:
            row["label"] = int(row["label"])
            image_path(root, row["path"], "train", row["label"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid training manifest row: {row}") from exc
        if row["path"] in seen:
            raise ValueError(f"Duplicate training manifest entry: {row['path']}")
        seen.add(row["path"])
        counts[row["label"]] += 1
    if counts != [samples_per_class, samples_per_class]:
        raise ValueError(f"Manifest counts {counts} do not match configured {samples_per_class}/class")
    _, errors = readable_rows(root, rows)
    if errors:
        raise ValueError(f"Training manifest contains unreadable images: {errors[:3]}")
    return rows


class XRayDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], image_size: int):
        self.root, self.rows = root, rows
        self.transform = preprocessing(image_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(image_path(self.root, row["path"], label=int(row["label"]))) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), row["path"]
