"""Unit tests for classification and segmentation evaluation metrics."""

import unittest

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from src.utils.metrics import (
    calculate_classification_metrics,
    calculate_dice_coefficient,
    calculate_iou,
    calculate_segmentation_metrics,
)


class TestMetrics(unittest.TestCase):
    """Test suite for metrics computation."""

    def test_classification_metrics_perfect(self):
        """Test classification metrics on 100% agreement."""
        y_true = [0, 1, 2, 3, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 0, 1, 2, 3]
        class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

        res = calculate_classification_metrics(y_true, y_pred, class_names=class_names)
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["macro_f1"], 1.0)
        self.assertEqual(res["macro_precision"], 1.0)
        self.assertEqual(res["macro_recall"], 1.0)

    def test_classification_metrics_imperfect(self):
        """Test classification metrics with misclassifications."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        res = calculate_classification_metrics(y_true, y_pred)
        self.assertEqual(res["accuracy"], 0.5)
        self.assertEqual(res["macro_f1"], 0.5)

    def test_dice_coefficient_perfect_overlap(self):
        """Test Dice coefficient when pred and target match exactly."""
        if NUMPY_AVAILABLE:
            mask = np.zeros((10, 10), dtype=np.float32)
            mask[2:5, 2:5] = 1.0
        else:
            mask = [[1.0 if (2 <= r < 5 and 2 <= c < 5) else 0.0 for c in range(10)] for r in range(10)]

        dice = calculate_dice_coefficient(mask, mask)
        self.assertAlmostEqual(dice, 1.0, places=4)

    def test_dice_coefficient_disjoint(self):
        """Test Dice coefficient when masks are completely disjoint."""
        if NUMPY_AVAILABLE:
            m1 = np.zeros((10, 10), dtype=np.float32)
            m2 = np.zeros((10, 10), dtype=np.float32)
            m1[0:2, 0:2] = 1.0
            m2[8:10, 8:10] = 1.0
        else:
            m1 = [[1.0 if (r < 2 and c < 2) else 0.0 for c in range(10)] for r in range(10)]
            m2 = [[1.0 if (r >= 8 and c >= 8) else 0.0 for c in range(10)] for r in range(10)]

        dice = calculate_dice_coefficient(m1, m2)
        self.assertAlmostEqual(dice, 0.0, places=4)

    def test_iou_calculation(self):
        """Test Intersection over Union calculation."""
        if NUMPY_AVAILABLE:
            m1 = np.zeros((10, 10), dtype=np.float32)
            m2 = np.zeros((10, 10), dtype=np.float32)
            m1[0:2, 0:2] = 1.0
            m2[0:2, 1:3] = 1.0
        else:
            m1 = [[1.0 if (r < 2 and c < 2) else 0.0 for c in range(10)] for r in range(10)]
            m2 = [[1.0 if (r < 2 and 1 <= c < 3) else 0.0 for c in range(10)] for r in range(10)]

        # intersection = 2, union = 6 -> iou = 2/6 = 0.3333
        iou = calculate_iou(m1, m2)
        self.assertAlmostEqual(iou, 2.0 / 6.0, places=3)

    def test_segmentation_metrics_dictionary(self):
        """Test comprehensive segmentation metrics dictionary output."""
        if NUMPY_AVAILABLE:
            pred = np.zeros((20, 20), dtype=np.float32)
            true = np.zeros((20, 20), dtype=np.float32)
            pred[5:15, 5:15] = 0.8
            true[5:15, 5:15] = 1.0
        else:
            pred = [[0.8 if (5 <= r < 15 and 5 <= c < 15) else 0.0 for c in range(20)] for r in range(20)]
            true = [[1.0 if (5 <= r < 15 and 5 <= c < 15) else 0.0 for c in range(20)] for r in range(20)]

        metrics = calculate_segmentation_metrics(pred, true, threshold=0.5)
        self.assertIn("dice_coefficient", metrics)
        self.assertIn("iou", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("pixel_accuracy", metrics)
        self.assertAlmostEqual(metrics["dice_coefficient"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
