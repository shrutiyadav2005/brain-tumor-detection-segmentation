"""Unit tests for U-Net Brain Tumor Segmenter."""

import unittest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestSegmenter(unittest.TestCase):
    """Test suite for segmentation model architecture and inference."""

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed in test environment")
    def test_segmenter_output_dimensions(self):
        """Verify U-Net output spatial dimensions [B, 1, 224, 224]."""
        from models.segmenter import BrainTumorSegmenter

        model = BrainTumorSegmenter(encoder_name="resnet34", encoder_weights=None)
        model.eval()

        dummy_batch = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy_batch)

        self.assertEqual(logits.shape, (2, 1, 224, 224))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed in test environment")
    def test_segmenter_predict_mask(self):
        """Verify binary mask output and tumor percentage calculation."""
        from models.segmenter import BrainTumorSegmenter

        model = BrainTumorSegmenter(encoder_name="resnet34", encoder_weights=None)
        dummy_tensor = torch.randn(1, 3, 224, 224)

        prob_map, binary_mask, tumor_pct = model.predict_mask(dummy_tensor, threshold=0.5)

        self.assertEqual(prob_map.shape, (224, 224))
        self.assertEqual(binary_mask.shape, (224, 224))
        self.assertTrue(set(binary_mask.flatten()).issubset({0, 1}))
        self.assertGreaterEqual(tumor_pct, 0.0)
        self.assertLessEqual(tumor_pct, 100.0)


if __name__ == "__main__":
    unittest.main()
