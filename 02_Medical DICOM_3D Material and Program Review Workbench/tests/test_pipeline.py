"""Behavioral regression checks for split isolation, metrics and frozen training."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18

from check_dataset import metadata_audit
from common import discover, image_path, write_csv
from dataset import XRayDataset, load_subset, readable_rows
from metrics import binary_metrics
from model import configure_trainable, training_mode
from prepare_train_subset import prepare
from compare import compare


class DatasetChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "data"
        for split in ("train", "val", "test"):
            for label in ("NORMAL", "PNEUMONIA"):
                directory = self.root / split / label
                directory.mkdir(parents=True)
                for i in range(4):
                    Image.new("L", (24, 32), 40 + 20 * i).save(directory / f"sample{i}.png")
        self.manifest = Path(self.temp.name) / "subset.csv"

    def tearDown(self):
        self.temp.cleanup()

    def config(self):
        return {"data_root": str(self.root), "subset_path": str(self.manifest),
                "train_samples_per_class": 2, "random_seed": 42}

    def test_fixed_subset_reused_and_balanced_despite_unreadable_image(self):
        (self.root / "train/NORMAL/bad.jpeg").write_text("corrupt")
        rows = prepare(self.config())
        original = self.manifest.read_bytes()
        self.assertEqual([sum(r["label"] == label for r in rows) for label in (0, 1)], [2, 2])
        self.assertTrue(all(r["path"].startswith("train/") for r in rows))
        self.assertFalse(any("bad.jpeg" in r["path"] for r in rows))
        changed_seed = {**self.config(), "random_seed": 99}
        self.assertEqual(prepare(changed_seed), rows)
        self.assertEqual(self.manifest.read_bytes(), original)
        prepare(self.config(), overwrite=True)
        self.assertEqual(self.manifest.read_bytes(), original)

    def test_manifest_rejects_leakage_wrong_labels_duplicates_and_traversal(self):
        cases = [
            [{"path": "test/NORMAL/sample0.png", "label": 0}],
            [{"path": "train/NORMAL/sample0.png", "label": 1}],
            [{"path": "../NORMAL/sample0.png", "label": 0}],
            [{"path": str(self.root / "train/NORMAL/sample0.png"), "label": 0}],
            [{"path": "train/NORMAL/sample0.png", "label": 0}] * 2,
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                write_csv(self.manifest, rows, ["path", "label"])
                with self.assertRaises(ValueError):
                    load_subset(self.root, self.manifest, 1)

    def test_nested_archive_hidden_files_and_escaping_symlinks(self):
        hidden = self.root / "train/NORMAL/._sample.png"
        hidden.write_bytes(b"metadata")
        nested = self.root / "chest_xray/train/NORMAL"
        nested.mkdir(parents=True)
        Image.new("L", (2, 2)).save(nested / "copy.png")
        self.assertEqual(len(discover(self.root, "train")), 8)
        outside = Path(self.temp.name) / "outside.png"
        Image.new("L", (2, 2)).save(outside)
        (self.root / "train/NORMAL/escape.png").symlink_to(outside)
        with self.assertRaises(ValueError):
            image_path(self.root, "train/NORMAL/escape.png")

    def test_grayscale_preprocessing_and_read_error(self):
        rows = discover(self.root, "test")
        tensor, label, path = XRayDataset(self.root, rows, 224)[0]
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(tensor).all())
        self.assertEqual(label, 0)
        (self.root / path).write_bytes(b"broken image")
        good, errors = readable_rows(self.root, rows)
        self.assertEqual((len(good), len(errors)), (7, 1))

    def test_patient_overlap_and_missing_metadata_are_distinct(self):
        rows = [r for split in ("train", "val", "test") for r in discover(self.root, split)]
        self.assertEqual(metadata_audit(self.root, rows, None)["status"], "unverified")
        metadata = Path(self.temp.name) / "metadata.csv"
        write_csv(metadata, [{"path": "train/NORMAL/sample0.png", "patient_id": "anon1"},
                             {"path": "test/NORMAL/sample0.png", "patient_id": "anon1"}], ["path", "patient_id"])
        result = metadata_audit(self.root, rows, str(metadata))
        self.assertEqual(result["status"], "overlap_detected")
        self.assertEqual(len(result["cross_split_ids"]["patient_id"]), 1)


class MetricChecks(unittest.TestCase):
    def test_known_confusion_matrix_auc_and_tie(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.3, 0.9])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertEqual(metrics["roc_auc"], 0.75)
        self.assertEqual(binary_metrics([0, 1], [0.5, 0.5])["confusion_matrix"], [[1, 0], [1, 0]])

    def test_undefined_metrics_and_invalid_probabilities(self):
        result = binary_metrics([0, 0], [0.1, 0.2])
        self.assertIsNone(result["sensitivity"])
        self.assertIsNone(result["roc_auc"])
        for labels, probabilities in (([], []), ([1], [np.nan]), ([0], [1.1]), ([0, 1], [0.1]), ([3], [0.5])):
            with self.assertRaises(ValueError):
                binary_metrics(labels, probabilities)


class ModelChecks(unittest.TestCase):
    def test_frozen_parameters_and_batchnorm_buffers_remain_unchanged(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = resnet18(weights=None, num_classes=2)
        configure_trainable(model, True)
        training_mode(model, True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.1)
        loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 1]))
        loss.backward()
        optimizer.step()
        for name, value in model.state_dict().items():
            if not name.startswith("fc."):
                self.assertTrue(torch.equal(before[name], value), name)
        self.assertFalse(torch.equal(before["fc.weight"], model.fc.weight))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 1026)

    def test_improved_updates_layer4_but_keeps_earlier_statistics_frozen(self):
        torch.set_num_threads(2)
        model = resnet18(weights=None, num_classes=2)
        configure_trainable(model, False)
        training_mode(model, False)
        bn1 = model.bn1.running_mean.clone()
        bn4 = model.layer4[0].bn1.running_mean.clone()
        model(torch.randn(2, 3, 64, 64)).sum().backward()
        self.assertIsNone(model.layer3[0].conv1.weight.grad)
        self.assertIsNotNone(model.layer4[0].conv1.weight.grad)
        self.assertTrue(torch.equal(bn1, model.bn1.running_mean))
        self.assertFalse(torch.equal(bn4, model.layer4[0].bn1.running_mean))


class ComparisonChecks(unittest.TestCase):
    def test_comparison_rejects_different_test_content_or_seed(self):
        cfg = {key: 1 for key in ("model", "pretrained", "random_seed", "image_size", "batch_size", "epochs", "learning_rate", "device", "num_workers", "torch_threads")}
        cfg["freeze_backbone"] = True
        baseline = {**{key: "same" for key in ("manifest_sha256", "train_data_sha256", "val_data_sha256", "test_data_sha256")},
                    "test_found": 2, "test_used": 2, "pretrained_weights": {"sha256": "same"}, "training_config": cfg,
                    **{key: 1 for key in ("best_epoch", "n", "sensitivity", "roc_auc", "specificity", "accuracy", "tp", "tn", "fp", "fn")}}
        improved = copy.deepcopy(baseline)
        improved["training_config"]["freeze_backbone"] = False
        self.assertEqual(compare(baseline, improved)["improved_minus_baseline"]["roc_auc"], 0)
        improved["test_data_sha256"] = "changed"
        with self.assertRaises(ValueError):
            compare(baseline, improved)
        improved["test_data_sha256"] = "same"
        improved["training_config"]["random_seed"] = 2
        with self.assertRaises(ValueError):
            compare(baseline, improved)


if __name__ == "__main__":
    unittest.main()
