# Model Training & Evaluation Guide

This guide provides step-by-step instructions for configuring, training, evaluating, and monitoring the deep learning models across different hardware environments.

---

## 1. Prerequisites & Environment Setup

### 1.1 Create Virtual Environment

```bash
# 1. Create a fresh virtual environment
python3 -m venv .venv

# 2. Activate environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows (cmd):
.venv\Scripts\activate.bat
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Training the ResNet-50 Classifier

### 2.1 Default Training
Train the ResNet-50 4-class classifier using settings defined in `config/config.yaml`:

```bash
python -m src.training.train_classifier
```

### 2.2 CLI Overrides
You can override training hyperparameters directly from the command line:

```bash
python -m src.training.train_classifier \
    --config config/config.yaml \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.0003
```

### 2.3 Artifacts Generated
Upon training completion or early stopping, the script saves:
- `checkpoints/best_classifier.pth`: Checkpoint with highest validation accuracy / lowest validation loss.
- `results/classification/training_history.json`: Epoch-by-epoch loss and accuracy log.
- `results/classification/training_curves.png`: High-resolution plot of learning curves.

---

## 3. Training the U-Net Tumor Segmenter

### 3.1 Default Training
Train the U-Net segmentation network with ResNet-34 encoder:

```bash
python -m src.training.train_segmenter
```

### 3.2 CLI Overrides

```bash
python -m src.training.train_segmenter \
    --config config/config.yaml \
    --epochs 25 \
    --batch-size 16 \
    --lr 0.0005
```

### 3.3 Artifacts Generated
- `checkpoints/best_segmenter.pth`: Weights achieving highest validation Dice Similarity Coefficient.
- `results/segmentation/training_history.json`: Epoch log of loss, Dice score, and IoU.
- `results/segmentation/training_curves.png`: Training vs validation Dice and loss curves.

---

## 4. Independent Model Evaluation

Run evaluations on the dedicated test sets without re-training:

### 4.1 Evaluate Classifier

```bash
python -m src.evaluation.evaluate_classifier \
    --checkpoint checkpoints/best_classifier.pth \
    --test-dir data/brisc2025/classification_task/test
```
Outputs generated in `results/classification/`:
- `test_metrics.json`: Accuracy, Macro F1, Precision, Recall, and per-class metrics.
- `confusion_matrix.png`: Heatmap plot with annotated cell counts.

### 4.2 Evaluate Segmenter

```bash
python -m src.evaluation.evaluate_segmenter \
    --checkpoint checkpoints/best_segmenter.pth \
    --test-images data/brisc2025/segmentation_task/test/images \
    --test-masks data/brisc2025/segmentation_task/test/masks
```
Outputs generated in `results/segmentation/`:
- `test_metrics.json`: Mean Dice, Mean IoU, Precision, Recall, and Pixel Accuracy.
- `sample_predictions/`: Side-by-side 4-panel visual comparisons (Original, Ground Truth, Predicted Mask, Overlay).

---

## 5. Hardware-Specific Execution

### 5.1 MacBook Apple Silicon (M1 / M2 / M3 / M4)
The system automatically detects and uses Apple Silicon GPU acceleration via Apple Metal Performance Shaders (`mps`). No extra flags are needed.

### 5.2 Google Colab (NVIDIA T4 / V100 / A100)
1. In Colab, select **Runtime > Change runtime type > GPU (T4 or A100)**.
2. Clone repository or upload files.
3. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
4. Verify CUDA acceleration:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

---

## 6. Troubleshooting Common Issues

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **CUDA Out of Memory (OOM)** | Batch size too large for GPU VRAM. | Reduce `--batch-size` (e.g. from 32 to 16 for classification; from 16 to 8 for segmentation). |
| **Missing Image-Mask Pairs** | Non-matching filenames or file formats. | Run `python -m src.data.preprocessing --verify` to identify mismatched stems. |
| **Loss is NaN / Diverging** | Learning rate too high. | Lower `--lr` to `0.0001` or `0.00005`. |
| **Mask Resizing Artifacts** | Mask resized using Bilinear mode. | Verify `preprocess_mask` is used (which enforces `Image.Resampling.NEAREST`). |
