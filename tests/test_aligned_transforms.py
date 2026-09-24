"""Small CPU checks for the aligned Seen-10 training transforms."""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from csgo_seen10.action_normalization import ActionNormalization, fit_seen_train_stats
from csgo_seen10.augmentations import augment_image


_MANIFEST_HASH = "a" * 64


def _train_rows() -> list[dict]:
    # Four varying dimensions plus a constant one exercise OFT's zero mask.
    return [
        {
            "split": "seen_train",
            "sample_id": f"de_mirage/sample_{index:04d}",
            "target_pose": [index / 20, index / 30, 0.25, index / 40, index / 50],
        }
        for index in range(20)
    ]


class ActionNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stats = fit_seen_train_stats(_train_rows(), _MANIFEST_HASH, expected_count=20)
        self.transform = ActionNormalization("bounds_q99", self.stats)

    def test_train_fit_provenance_and_tamper_rejection(self) -> None:
        self.assertEqual(self.stats["source_split"], "seen_train")
        self.assertEqual(self.stats["sample_count"], 20)
        self.assertEqual(self.stats["mask"], [True] * 5)
        self.assertEqual(self.stats["q01"][2], 0.25)
        self.assertNotEqual(self.stats["ordered_sample_ids_sha256"], "" * 64)

        rows = _train_rows()
        rows[0]["split"] = "seen_validation"
        with self.assertRaisesRegex(ValueError, "seen_train"):
            fit_seen_train_stats(rows, _MANIFEST_HASH, expected_count=20)
        with self.assertRaisesRegex(ValueError, "Expected all"):
            fit_seen_train_stats(_train_rows()[:-1], _MANIFEST_HASH, expected_count=20)
        altered = copy.deepcopy(self.stats)
        altered["q01"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "hash"):
            ActionNormalization("bounds_q99", altered)

    def test_round_trip_clip_and_unbounded_prediction(self) -> None:
        interior = torch.tensor([[0.5, 1 / 3, 0.25, 0.25, 0.2]], dtype=torch.float64)
        qnorm = self.transform.normalize(interior)
        self.assertEqual(tuple(qnorm.shape), (1, 5))
        self.assertEqual(qnorm[0, 2].item(), 0.0)
        restored = self.transform.inverse(qnorm)
        torch.testing.assert_close(restored[..., [0, 1, 3, 4]], interior[..., [0, 1, 3, 4]], rtol=0, atol=1e-9)
        self.assertAlmostEqual(restored[0, 2].item(), 0.25 + 0.5e-8, places=12)

        tail = torch.tensor([[100., 100., 0.25, 100., 100.]], dtype=torch.float64)
        target = self.transform.normalize(tail)
        self.assertTrue(torch.all(target <= 1))
        self.assertTrue(torch.all(target >= -1))
        self.assertEqual(target[0, 0].item(), 1.0)
        self.assertLess(self.transform.inverse(target)[0, 0].item(), 100)

        prediction = torch.tensor([[2., 2., 0., 2., 2.]], dtype=torch.float64)
        self.assertGreater(self.transform.inverse(prediction)[0, 0].item(), self.stats["q99"][0])
        self.assertAlmostEqual(self.transform.inverse(prediction)[0, 2].item(), 0.25 + 0.5e-8, places=12)

    def test_serialization_and_identity(self) -> None:
        recovered = ActionNormalization.from_dict(self.transform.to_dict())
        values = torch.tensor([[1., 2., 3., 4., 5.]])
        torch.testing.assert_close(recovered.normalize(values), self.transform.normalize(values))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action_normalization.json"
            self.transform.save(path)
            loaded = ActionNormalization.load(path)
            torch.testing.assert_close(loaded.normalize(values), self.transform.normalize(values))
        none = ActionNormalization()
        self.assertIs(none.normalize(values), values)
        self.assertIs(none.inverse(values), values)


class PhotometricTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.mgrid[:16, :24]
        self.image = Image.fromarray(
            np.stack(((x * 11) % 256, (y * 17) % 256, ((x + y) * 7) % 256), axis=-1).astype(np.uint8),
            mode="RGB",
        )

    def test_deterministic_independent_views_and_geometry(self) -> None:
        args = dict(policy="oft_photometric_only", seed=42, epoch=3, sample_id="de_mirage/sample_0010")
        state_before = random.getstate()
        fpv_a = augment_image(self.image, view="fpv", **args)
        fpv_b = augment_image(self.image, view="fpv", **args)
        radar = augment_image(self.image, view="radar", **args)
        self.assertEqual(random.getstate(), state_before)
        self.assertEqual(fpv_a.size, self.image.size)
        self.assertEqual(fpv_a.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(fpv_a), np.asarray(fpv_b))
        self.assertFalse(np.array_equal(np.asarray(fpv_a), np.asarray(radar)))
        self.assertFalse(np.array_equal(np.asarray(fpv_a), np.asarray(self.image)))

        unaugmented = augment_image(self.image, policy="none", sample_id="id", view="fpv")
        np.testing.assert_array_equal(np.asarray(unaugmented), np.asarray(self.image))


if __name__ == "__main__":
    unittest.main()
