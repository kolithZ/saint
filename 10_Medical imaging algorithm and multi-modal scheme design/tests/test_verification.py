"""Interrupted/corrupted artifact regressions using small synthetic numeric fixtures."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from xray.common import MANIFEST_FIELDS, counts, read_csv, read_json, sha256, write_csv, write_json
from xray.evaluation import PREDICTION_FIELDS, metrics, predictions
from xray.model import numeric_scores
from verify_run import verify


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.rows = []
        for split in ("train", "val", "test"):
            for label_id, label in enumerate(("NORMAL", "PNEUMONIA")):
                key = f"{split}-{label_id}"
                row = dict.fromkeys(MANIFEST_FIELDS, "")
                row.update(relative_path=f"{split}/{label}/{key}.png", split=split, label=label, label_id=label_id,
                           width=224, height=224, mode="L", frame_count=1, readable=True, file_sha256=key,
                           rgb_pixel_sha256=key, patient_id_verified=False)
                self.rows.append(row)
        write_csv(self.run / "manifest.csv", self.rows, MANIFEST_FIELDS)
        x = np.stack([np.full(512, -1.), np.full(512, 1.)])
        parameters = dict(mean=x.mean(axis=0), scale=x.std(axis=0), coef=np.full((1, 512), 1 / 512), intercept=np.array([0.]))
        np.savez(self.run / "classifier.npz", **parameters)
        np.savez(self.run / "features.npz", train=x, val=x, test=x)
        (self.run / "backbone.pth").write_bytes(b"checksum fixture only; never loaded as a model")
        write_json(self.run / "config.json", dict(threshold=0.5, counts=counts(self.rows), manifest_sha256=sha256(self.run / "manifest.csv")))
        write_json(self.run / "model_lock.json", dict(
            **{key: sha256(self.run / name) for name, key in [("config.json", "config_sha256"), ("manifest.csv", "manifest_sha256"), ("classifier.npz", "classifier_sha256"), ("backbone.pth", "backbone_sha256")]},
            threshold=0.5, test_used_for_fit_or_selection=False))
        summary = dict(status="complete")
        for split, prefix in [("val", "validation"), ("test", "test")]:
            scores = numeric_scores(x, parameters)
            group = [r for r in self.rows if r["split"] == split]
            write_csv(self.run / f"{prefix}_predictions.csv", predictions(group, scores, 0.5), PREDICTION_FIELDS)
            result = metrics([0, 1], scores, 0.5)
            write_json(self.run / f"{prefix}_metrics.json", result)
            summary[prefix] = result
        write_json(self.run / "run_summary.json", summary)

    def test_complete_numeric_run_passes(self):
        self.assertEqual(verify(self.run)["status"], "passed")

    def test_interrupted_run_without_completion_marker_rejected(self):
        (self.run / "run_summary.json").unlink()
        with self.assertRaises((ValueError, OSError)):
            verify(self.run)

    def test_incomplete_status_and_inconsistent_summary_rejected(self):
        original = read_json(self.run / "run_summary.json")
        for mutation in (dict(original, status="running"), dict(original, test=dict(original["test"], TP=999))):
            with self.subTest(summary=mutation):
                write_json(self.run / "run_summary.json", mutation)
                with self.assertRaises(ValueError):
                    verify(self.run)

    def test_prediction_metadata_corruption_rejected(self):
        path = self.run / "test_predictions.csv"
        original = read_csv(path)
        for field, bad in [("predicted_label", "PNEUMONIA"), ("true_label", "PNEUMONIA"), ("label_id", "1"),
                           ("outcome", "TP"), ("split", "train"), ("threshold", "0.9"), ("aux_available", "True")]:
            with self.subTest(field=field):
                changed = [dict(r) for r in original]
                changed[0][field] = bad
                write_csv(path, changed, PREDICTION_FIELDS)
                with self.assertRaises(ValueError):
                    verify(self.run)

    def test_feature_rows_and_nonfinite_values_rejected(self):
        x = np.stack([np.full(512, -1.), np.full(512, 1.)])
        for train in (np.concatenate([x, x]), np.full((2, 512), np.nan)):
            with self.subTest(shape=train.shape):
                np.savez(self.run / "features.npz", train=train, val=x, test=x)
                with self.assertRaises(ValueError):
                    verify(self.run)

    def test_lock_and_config_threshold_must_agree(self):
        lock = read_json(self.run / "model_lock.json")
        lock["threshold"] = 0.9
        write_json(self.run / "model_lock.json", lock)
        with self.assertRaises(ValueError):
            verify(self.run)


if __name__ == "__main__":
    unittest.main()
