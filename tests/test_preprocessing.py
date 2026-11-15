"""Unit tests for data preprocessing and dataset integrity verification."""

import os
import shutil
import tempfile
import unittest

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

from src.data.preprocessing import (
    preprocess_mask,
    verify_dataset_integrity,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.utils.visualization import create_segmentation_overlay


class TestPreprocessing(unittest.TestCase):
    """Test suite for image/mask preprocessing and verification."""

    def test_supported_extensions(self):
        """Verify supported image extension constant."""
        self.assertIn(".jpg", SUPPORTED_IMAGE_EXTENSIONS)
        self.assertIn(".png", SUPPORTED_IMAGE_EXTENSIONS)
        self.assertIn(".jpeg", SUPPORTED_IMAGE_EXTENSIONS)

    @unittest.skipUnless(NUMPY_AVAILABLE and PIL_AVAILABLE, "NumPy and PIL required for mask processing")
    def test_preprocess_mask_binarization_and_shape(self):
        """Verify that preprocess_mask strictly outputs binary 0.0 and 1.0 values."""
        raw_mask = np.zeros((100, 100), dtype=np.uint8)
        raw_mask[20:60, 20:60] = 255  # White square on black background

        processed = preprocess_mask(raw_mask, target_size=(224, 224))

        self.assertEqual(processed.shape, (224, 224))
        unique_vals = set(np.unique(processed))
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))
        self.assertEqual(processed.dtype, np.float32)

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy required for overlay testing")
    def test_segmentation_overlay_shape_and_range(self):
        """Verify that segmentation overlay generates normalized 3-channel image."""
        img = np.ones((50, 50, 3), dtype=np.float32) * 0.5
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:30, 10:30] = 1.0

        overlay = create_segmentation_overlay(img, mask)
        self.assertEqual(overlay.shape, (50, 50, 3))
        self.assertLessEqual(overlay.max(), 1.0)
        self.assertGreaterEqual(overlay.min(), 0.0)

    def test_verify_dataset_integrity_nonexistent_dir(self):
        """Verify dataset integrity function handles missing root path gracefully."""
        res = verify_dataset_integrity("nonexistent_path_test_1234")
        self.assertFalse(res["dataset_found"])
        self.assertTrue(len(res["errors"]) > 0)

    @unittest.skipUnless(PIL_AVAILABLE, "PIL required for creating synthetic test images")
    def test_verify_dataset_integrity_on_synthetic_tree(self):
        """Test dataset verification routine on a synthetic temporary directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create synthetic classification folders
            train_cls = os.path.join(temp_dir, "classification_task", "train")
            for c in ["glioma", "meningioma", "no_tumor", "pituitary"]:
                c_dir = os.path.join(train_cls, c)
                os.makedirs(c_dir, exist_ok=True)
                dummy_img = Image.new("RGB", (32, 32), color="gray")
                dummy_img.save(os.path.join(c_dir, "sample_1.jpg"))

            # Create synthetic segmentation folders
            seg_img_dir = os.path.join(temp_dir, "segmentation_task", "train", "images")
            seg_msk_dir = os.path.join(temp_dir, "segmentation_task", "train", "masks")
            os.makedirs(seg_img_dir, exist_ok=True)
            os.makedirs(seg_msk_dir, exist_ok=True)

            img1 = Image.new("RGB", (32, 32), color="black")
            msk1 = Image.new("L", (32, 32), color="white")
            img1.save(os.path.join(seg_img_dir, "case_01.png"))
            msk1.save(os.path.join(seg_msk_dir, "case_01.png"))

            summary = verify_dataset_integrity(temp_dir)
            self.assertTrue(summary["dataset_found"])
            self.assertTrue(summary["classification"]["is_valid"])
            self.assertEqual(summary["classification"]["train_counts"]["glioma"], 1)
            self.assertEqual(summary["segmentation"]["train_pairs"], 1)
            self.assertTrue(summary["segmentation"]["is_valid"])

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
