# BRISC2025 Dataset Guide & Preprocessing Specifications

This guide outlines the specifications, folder structures, preprocessing procedures, and validation protocols for the **BRISC2025 (Brain MRI Segmentation and Classification 2025)** dataset.

---

## 1. Dataset Overview

The BRISC2025 dataset provides high-resolution magnetic resonance imaging (MRI) scans acquired in axial projections, paired with clinical diagnostic labels and pixel-level tumor annotations.

### 1.1 Classification Task
The classification component contains brain MRI scans partitioned across four clinically distinct categories:

| Class Name | Directory Name | Clinical Description |
| :--- | :--- | :--- |
| **Glioma** | `glioma` | Primary intra-axial brain tumors originating from glial cells (astrocytomas, oligodendrogliomas, glioblastomas). Characterized by infiltrative growth and heterogeneous signal intensity. |
| **Meningioma** | `meningioma` | Predominantly benign, extra-axial neoplasms arising from the arachnoid cap cells of the meninges. Typically exhibit well-circumscribed borders and dural attachment. |
| **No Tumor** | `no_tumor` | Scans from non-pathological subjects with intact ventricular morphology, symmetric hemispheres, and no neoplastic tissue. |
| **Pituitary** | `pituitary` | Sellar and parasellar tumors originating from the anterior pituitary gland, often causing compression of the optic chiasm. |

### 1.2 Segmentation Task
The segmentation component provides ground-truth spatial boundaries for neoplastic lesions:
- **Image Modality**: 2D axial MRI slices (RGB / Grayscale converted to 3-channel RGB).
- **Mask Format**: Single-channel grayscale images where:
  - Value `0` represents background tissue and non-lesion cerebral parenchyma.
  - Value `255` (or `1`) denotes active tumor parenchyma.

---

## 2. Directory Layout & Organization

Ensure your extracted dataset conforms to this exact directory layout:

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
        │   │   ├── case_001.png
        │   │   └── ...
        │   └── masks/
        │       ├── case_001.png
        │       └── ...
        └── test/
            ├── images/
            └── masks/
```

> **Note on Segmentation Partitions**: If your dataset download contains only a `segmentation_task/train/` folder without an explicit `test/` folder, our dataset module (`src.data.segmentation_dataset`) automatically applies a deterministic stratified split (70% train, 15% validation, 15% test) using a fixed seed (`seed: 42`) to guarantee reproducibility.

---

## 3. Critical Preprocessing Rules

### 3.1 Mask Interpolation (Nearest-Neighbor vs Bilinear)
> [!IMPORTANT]
> **Nearest-Neighbor Interpolation is Mandatory for Ground-Truth Masks.**
> Standard image resizing (such as bilinear or bicubic interpolation) computes a weighted average of neighboring pixels. When applied to a binary mask containing values $\{0, 1\}$, bilinear interpolation introduces synthetic floating-point values (e.g. $0.34, 0.72$) at tumor borders.
> 
> To preserve true binary target semantics, our preprocessing pipeline strictly enforces:
> ```python
> mask_resized = mask_img.resize((224, 224), resample=Image.Resampling.NEAREST)
> binary_mask = (np.array(mask_resized, dtype=np.float32) > 0).astype(np.float32)
> ```

### 3.2 Pretrained Normalization for MRI Scans
For ResNet-50 and ResNet-34 encoders pretrained on ImageNet, images must be normalized using the canonical ImageNet channel-wise mean and standard deviation:
$$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{Std} = [0.229, 0.224, 0.225]$$

---

## 4. Dataset Integrity Verification

Before initiating model training, verify file matching, check for corrupted images, and detect orphaned masks using our integrity verification utility:

```bash
python -m src.data.preprocessing --verify --path data/brisc2025
```

The script reports:
- Number of valid samples in each classification category.
- Total count of verified pairwise matched image-mask pairs.
- Specific filenames of any orphaned images (missing mask) or orphaned masks (missing image).
