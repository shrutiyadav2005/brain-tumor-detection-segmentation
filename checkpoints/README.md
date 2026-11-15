# Model Checkpoints

This directory stores trained weights for the classification and segmentation models.

## Expected Files

- `best_classifier.pth`: State dictionary of the best ResNet50 model evaluated on the classification validation set.
- `best_segmenter.pth`: State dictionary of the best U-Net (ResNet34 encoder) model evaluated on the segmentation validation set.

## Saving & Loading Convention

Checkpoints are PyTorch state dictionaries saved via:

```python
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "config": config,
    },
    checkpoint_path,
)
```

The inference pipeline and Streamlit application also support bare state dictionaries saved directly via `torch.save(model.state_dict(), checkpoint_path)` for backward compatibility with the original notebooks.

## Git Tracking

Trained weight binaries (`*.pth`, `*.pt`, `*.ckpt`) are excluded from Git via `.gitignore` to keep the repository lightweight. When cloning this repository, place pre-trained weights in this folder or run the training scripts to generate them.
