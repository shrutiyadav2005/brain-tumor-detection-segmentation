# Dataset Setup Guide (BRISC2025)

This project uses the **BRISC2025 (Brain MRI Segmentation and Classification 2025)** benchmark dataset.

---

## 1. Directory Structure

Place the unzipped dataset inside `data/brisc2025/` following this layout:

```text
data/
└── brisc2025/
    ├── classification_task/
    │   ├── train/
    │   │   ├── glioma/
    │   │   ├── meningioma/
    │   │   ├── no_tumor/
    │   │   └── pituitary/
    │   └── test/
    │       ├── glioma/
    │       ├── meningioma/
    │       ├── no_tumor/
    │       └── pituitary/
    │
    └── segmentation_task/
        ├── train/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
```

> **Note**: If your `segmentation_task` only has a `train/` directory, our dataset loader will automatically and deterministically partition it into stratified training, validation, and testing splits according to `config/config.yaml`.

---

## 2. Dataset Download Instructions

1. **Official Competition / Repository Access**:
   - Download the BRISC2025 archive from your competition portal, Google Drive link, or Kaggle mirror.
2. **Extraction**:
   - Extract the archive directly into the `data/` directory:
     ```bash
     mkdir -p data/brisc2025
     unzip brisc2025.zip -d data/brisc2025/
     ```
3. **Verify Integrity**:
   - You can run the preprocessing verification script to inspect pairings and check for missing or corrupted images:
     ```bash
     python -m src.data.preprocessing --verify
     ```

---

## 3. Class Definitions (Classification Task)

| Class Name | Index | Description |
| :--- | :---: | :--- |
| **Glioma** | 0 | Glial cell brain tumor arising in the spine or brain. |
| **Meningioma** | 1 | Tumor that arises from the meninges surrounding brain & spinal cord. |
| **No Tumor** | 2 | Normal healthy brain MRI scan with no neoplastic lesions. |
| **Pituitary** | 3 | Neoplasm that develops in the pituitary gland at the base of the brain. |

---

## 4. Mask Specifications (Segmentation Task)

- **Modality**: Pixel-level spatial annotation of neoplastic regions.
- **Values**: Binary target where `0` denotes background / healthy brain tissue and `1` (or `255` in raw PNG/JPG) denotes active tumor tissue.
- **Interpolation Notice**: During preprocessing and resizing, **nearest-neighbor interpolation** is strictly enforced to prevent partial-volume artifacts and preserve binary ground-truth boundaries.
