"""Unit tests for ResNet50 Brain Tumor Classifier."""

import unittest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestClassifier(unittest.TestCase):
    """Test suite for classifier architecture and inference."""

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed in test environment")
    def test_classifier_output_dimensions(self):
        """Verify model output shape on batch input [B, 3, 224, 224]."""
        from models.classifier import BrainTumorClassifier

        model = BrainTumorClassifier(num_classes=4, pretrained=False)
        model.eval()

        dummy_batch = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            outputs = model(dummy_batch)

        self.assertEqual(outputs.shape, (2, 4))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed in test environment")
    def test_classifier_predict_function(self):
        """Verify single image prediction pipeline and probability distribution."""
        from models.classifier import BrainTumorClassifier

        model = BrainTumorClassifier(num_classes=4, pretrained=False)
        dummy_img = torch.randn(1, 3, 224, 224)

        pred_idx, pred_name, conf, probs = model.predict(dummy_img)

        self.assertIsInstance(pred_idx, int)
        self.assertIn(pred_idx, [0, 1, 2, 3])
        self.assertIsInstance(pred_name, str)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed in test environment")
    def test_gradcam_target_layer_hookable(self):
        """Verify that classifier exposes layer4[-1] for Grad-CAM."""
        from models.classifier import BrainTumorClassifier

        model = BrainTumorClassifier(num_classes=4, pretrained=False)
        target_layer = model.get_target_layer_for_cam()
        self.assertIsNotNone(target_layer)

    def test_classifier_class_constants(self):
        """Verify standard 4 tumor categories."""
        from models.classifier import DEFAULT_CLASS_NAMES
        self.assertEqual(DEFAULT_CLASS_NAMES, ["glioma", "meningioma", "no_tumor", "pituitary"])


if __name__ == "__main__":
    unittest.main()
