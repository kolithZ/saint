"""Engineering regression tests; synthetic fixtures are not model performance evidence."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from xray.audit import candidate_group, inspect
from xray.common import LABELS, SPLITS, new_output, read_csv, sample_path, sha256
from xray.evaluation import metrics
from xray.model import ImageDataset, prepare_image, numeric_scores
from xray.splits import prepare, select_rows


class PipelineTests(unittest.TestCase):
    def test_candidate_groups_preserve_namespaces(self):
        self.assertEqual(candidate_group("person1_virus_6.jpeg"), "person1")
        self.assertEqual(candidate_group("person1_bacteria_7.jpeg"), "person1")
        self.assertEqual(candidate_group("IM-0011-0001-0002.jpeg"), "IM-0011")
        self.assertEqual(candidate_group("NORMAL2-IM-0011-0001.jpeg"), "NORMAL2-IM-0011")
        self.assertEqual(candidate_group("unknown.png"), "")
        self.assertEqual(candidate_group("person1_virus_6_extra.jpeg"), "")

    def test_audit_real_decoding_duplicates_scope_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / "data"
            for i, split in enumerate(SPLITS):
                for j, label in enumerate(LABELS):
                    folder = data / split / label
                    folder.mkdir(parents=True)
                    Image.new("L", (10 + i, 12 + j), 10 + i * 30 + j).save(folder / f"sample-{i}-{j}.png")
            normal = data / "train/NORMAL"
            Image.new("L", (14, 15), 20).save(normal / "IM-0001-0001.png")
            Image.new("L", (14, 15), 20).save(data / "test/NORMAL/IM-0001-0002.png")
            Image.new("L", (14, 15), 21).save(data / "val/NORMAL/IM-0001-0003.png")
            (normal / "broken.jpeg").write_bytes(b"not an image")
            (normal / ".hidden.png").write_bytes(b"hidden")
            (normal / "nested").mkdir()
            (normal / "link.png").symlink_to(normal / "IM-0001-0001.png")
            Image.new("I;16", (12, 12), 4000).save(normal / "high-depth.png")
            Image.new("RGB", (8, 8), "red").save(normal / "multiframe.png", save_all=True, append_images=[Image.new("RGB", (8, 8), "blue")], duration=20)
            before = sha256(normal / "IM-0001-0001.png")
            with contextlib.redirect_stdout(io.StringIO()):
                report = inspect(data, base / "audit", data)
            self.assertEqual(report["unreadable_count"], 2)
            self.assertEqual(report["duplicates"]["rgb_pixel_sha256"]["cross_split_groups"], 1)
            self.assertEqual(report["candidate_cross_split_groups"], 1)
            self.assertFalse(report["patient_level_isolation_verified"])
            self.assertEqual(before, sha256(normal / "IM-0001-0001.png"))
            with self.assertRaises(FileExistsError):
                inspect(data, base / "audit")
            with contextlib.redirect_stdout(io.StringIO()):
                prepared = prepare(base / "audit", base / "splits")
            selected = read_csv(base / "splits/manifest.csv")
            self.assertTrue(any(r["relative_path"] == "test/NORMAL/IM-0001-0002.png" for r in selected))
            self.assertFalse(any(r["relative_path"].endswith(("broken.jpeg", "high-depth.png", "multiframe.png")) for r in selected))
            self.assertEqual(len([r for r in selected if r["candidate_group_id"] == "IM-0001"]), 1)
            self.assertFalse(prepared["patient_level_isolation_verified"])

    def test_transitive_conflicts_quarantine_all_members(self):
        def row(path, label, file_hash, pixel_hash):
            return dict(relative_path=path, split=path.split("/")[0], label=label, mode="L", frame_count="1", readable="True", file_sha256=file_hash, rgb_pixel_sha256=pixel_hash, candidate_group_id="")
        rows = [row("train/NORMAL/a.png", "NORMAL", "a", "pixel1"), row("val/NORMAL/b.png", "NORMAL", "a", "pixel2"), row("test/PNEUMONIA/c.png", "PNEUMONIA", "c", "pixel2")]
        selected, records = select_rows(rows)
        self.assertEqual(selected, [])
        self.assertTrue(all(r["exclusion_reason"] == "exact_duplicate_label_conflict" for r in records))

    def test_no_patient_claim_and_deterministic_selection(self):
        rows = [dict(relative_path=f"{split}/NORMAL/{index}.png", split=split, label="NORMAL", label_id=0, mode="L", frame_count=1, readable=True, file_sha256=str(index), rgb_pixel_sha256=str(index), candidate_group_id="IM-1", patient_id_verified=False) for index, split in enumerate(SPLITS)]
        a, _ = select_rows(rows)
        b, _ = select_rows(list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertEqual(a[0]["split"], "test")
        self.assertFalse(a[0]["patient_id_verified"])

    def test_geometry_keeps_full_field_and_rejects_high_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            source = Image.new("L", (400, 100), 128)
            source.paste(255, (0, 0, 20, 100))
            source.paste(255, (380, 0, 400, 100))
            source.save(path)
            result = np.array(prepare_image(path))
            self.assertEqual(result.shape, (224, 224, 3))
            self.assertTrue((result[:84] == 0).all())
            self.assertTrue((result[140:] == 0).all())
            self.assertTrue((result[100, 0] == 255).all())
            self.assertTrue((result[100, -1] == 255).all())
            self.assertTrue(np.array_equal(result[:, :, 0], result[:, :, 1]))
            Image.new("I;16", (20, 20), 5000).save(path)
            with self.assertRaises(ValueError):
                prepare_image(path)

    def test_exif_orientation_is_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "orientation.jpg"
            im = Image.new("L", (200, 100), 200)
            exif = Image.Exif()
            exif[274] = 6
            im.save(path, exif=exif)
            result = np.array(prepare_image(path))
            self.assertTrue((result[:, :56] == 0).all())
            self.assertGreater(result[0, 112, 0], 0)

    def test_metrics_and_undefined_cases(self):
        result = metrics([0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9])
        self.assertEqual(result["roc_auc"], 0.75)
        self.assertEqual(result["pneumonia_recall"], 0.5)
        self.assertEqual([result[k] for k in ("TP", "FP", "TN", "FN")], [1, 1, 1, 1])
        with self.assertRaises(ValueError):
            metrics([0.5], [0.2])
        empty = metrics([], [])
        self.assertIsNone(empty["roc_auc"])
        self.assertIsNone(empty["pneumonia_recall"])
        self.assertIsNone(metrics([0], [0.2])["pneumonia_recall"])
        self.assertEqual(metrics([1], [0.5])["TP"], 1)
        with self.assertRaises(ValueError):
            metrics([1], [float("nan")])

    def test_output_and_sample_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            for path in [data, data / "reports", Path(tmp)]:
                with self.assertRaises(ValueError):
                    new_output(path, [data])
            for rel in ["../escape.png", "/tmp/image.png", "train/UNKNOWN/a.png", "train/NORMAL/../a.png"]:
                with self.assertRaises(ValueError):
                    sample_path(data, rel)

    def test_stale_image_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train/NORMAL/image.png"
            path.parent.mkdir(parents=True)
            Image.new("L", (20, 20), 1).save(path)
            dataset = ImageDataset(tmp, [dict(relative_path="train/NORMAL/image.png", file_sha256=sha256(path))])
            Image.new("L", (20, 20), 2).save(path)
            with self.assertRaisesRegex(ValueError, "changed since audit"):
                dataset[0]

    def test_numeric_classifier_matches_sklearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        train_x = np.array([[-2, 0], [-1, 1], [1, 2], [2, 3]], dtype=float)
        scaler = StandardScaler().fit(train_x)
        classifier = LogisticRegression(class_weight="balanced").fit(scaler.transform(train_x), [0, 0, 1, 1])
        params = dict(mean=scaler.mean_, scale=scaler.scale_, coef=classifier.coef_, intercept=classifier.intercept_)
        heldout = np.array([[-100, 100], [100, -100], [0, 0]])
        np.testing.assert_allclose(numeric_scores(heldout, params), classifier.predict_proba(scaler.transform(heldout))[:, 1], atol=1e-12)
        # Heldout transformations do not alter training-only fitted moments.
        np.testing.assert_array_equal(scaler.mean_, train_x.mean(axis=0))


if __name__ == "__main__":
    unittest.main()
