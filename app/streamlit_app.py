"""Streamlit Application: Brain Tumor Detection, Segmentation & Explainable Diagnosis Dashboard."""

import io
import json
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import streamlit as st

# Page setup (must be first Streamlit call)
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Detection & Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for a medical-grade user interface
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e222a;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2d3340;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .disclaimer-box {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 0.9rem;
    }
    .info-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# MODEL CACHING & INFERENCE ENGINE
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_models_cached(
    classifier_path: str,
    segmenter_path: str,
    device_name: str = "auto",
):
    """Load and cache PyTorch models."""
    try:
        import torch
        from models.classifier import BrainTumorClassifier
        from models.segmenter import BrainTumorSegmenter
        from src.utils.seed import get_device

        device = get_device(device_name)

        classifier = None
        segmenter = None
        errors = []

        # 1. Load Classifier
        if os.path.exists(classifier_path):
            try:
                clf = BrainTumorClassifier(num_classes=4, pretrained=False)
                clf.load_checkpoint(classifier_path, device=str(device))
                clf = clf.to(device)
                clf.eval()
                classifier = clf
            except Exception as e:
                errors.append(f"Classifier loading error: {str(e)}")
        else:
            errors.append(f"Classifier checkpoint not found at: {classifier_path}")

        # 2. Load Segmenter
        if os.path.exists(segmenter_path):
            try:
                seg = BrainTumorSegmenter(encoder_name="resnet34", encoder_weights=None)
                seg.load_checkpoint(segmenter_path, device=str(device))
                seg = seg.to(device)
                seg.eval()
                segmenter = seg
            except Exception as e:
                errors.append(f"Segmenter loading error: {str(e)}")
        else:
            errors.append(f"Segmenter checkpoint not found at: {segmenter_path}")

        return classifier, segmenter, device, errors

    except ImportError:
        return None, None, "cpu", ["PyTorch is not installed in the current environment."]


# ==============================================================================
# UI COMPONENTS & SIDEBAR
# ==============================================================================

def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.sidebar.title("NeuroScan AI")
    st.sidebar.caption("Deep Learning MRI Analysis System")
    st.sidebar.markdown("---")

    st.sidebar.subheader("⚙️ Configuration")
    device_opt = st.sidebar.selectbox("Inference Device", ["auto", "cuda", "mps", "cpu"], index=0)

    clf_path = st.sidebar.text_input(
        "Classifier Checkpoint",
        value="checkpoints/best_classifier.pth",
        help="Path to trained ResNet50 classifier weights",
    )
    seg_path = st.sidebar.text_input(
        "Segmenter Checkpoint",
        value="checkpoints/best_segmenter.pth",
        help="Path to trained U-Net segmentation weights",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Analysis Parameters")
    threshold = st.sidebar.slider("Segmentation Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    overlay_alpha = st.sidebar.slider("Mask Overlay Opacity", min_value=0.1, max_value=0.8, value=0.45, step=0.05)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Supported Categories:**\n"
        "- 🧬 Glioma\n"
        "- 🛡️ Meningioma\n"
        "- ✅ No Tumor (Healthy)\n"
        "- 📍 Pituitary"
    )

    return clf_path, seg_path, device_opt, threshold, overlay_alpha


def create_synthetic_demo_mri() -> Image.Image:
    """Generate a clean synthetic brain MRI scan slice for zero-dependency demos."""
    h, w = 224, 224
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2

    # Elliptical skull and parenchyma
    mask_outer = ((x - center_x) / 80) ** 2 + ((y - center_y) / 95) ** 2 <= 1.0
    mask_inner = ((x - center_x) / 72) ** 2 + ((y - center_y) / 85) ** 2 <= 1.0

    brain = np.zeros((h, w), dtype=np.float32)
    brain[mask_outer] = 0.35
    brain[mask_inner] = 0.65

    # Ventricles
    ventricle_left = ((x - (center_x - 12)) / 7) ** 2 + ((y - center_y) / 25) ** 2 <= 1.0
    ventricle_right = ((x - (center_x + 12)) / 7) ** 2 + ((y - center_y) / 25) ** 2 <= 1.0
    brain[ventricle_left | ventricle_right] = 0.1

    # Simulated hyperintense tumor region in right frontal lobe
    tumor = ((x - (center_x + 35)) / 18) ** 2 + ((y - (center_y - 25)) / 16) ** 2 <= 1.0
    brain[tumor] = 0.95

    # Add slight gaussian noise
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.03, (h, w))
    brain_noisy = np.clip(brain + noise, 0.0, 1.0)

    uint_img = (brain_noisy * 255).astype(np.uint8)
    return Image.fromarray(uint_img).convert("RGB")


# ==============================================================================
# MAIN APPLICATION LOGIC
# ==============================================================================

def main():
    clf_path, seg_path, device_opt, threshold, overlay_alpha = render_sidebar()

    # Header section
    st.title("🧠 Brain Tumor Detection, Segmentation & Explainability System")
    st.markdown(
        "A multi-modal deep learning platform combining **ResNet50 Classification**, "
        "**U-Net Semantic Segmentation**, and **Grad-CAM Explainable AI** for T1-weighted brain MRI scans."
    )

    # Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ ACADEMIC & RESEARCH DISCLAIMER:</strong> This application is a scientific research prototype
            developed for academic evaluation and benchmarking. It is <strong>NOT</strong> an FDA-approved medical
            device or diagnostic instrument and must never be utilized as a substitute for professional clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File upload section
    st.subheader("📂 1. Upload Brain MRI Scan")
    col_up1, col_up2 = st.columns([3, 1])

    with col_up1:
        uploaded_file = st.file_uploader(
            "Select a brain MRI image file (JPEG, PNG, DICOM-converted)",
            type=["jpg", "jpeg", "png"],
            help="Upload an axial T1 or T2 brain MRI slice.",
        )

    with col_up2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        use_sample = st.button("🧪 Load Synthetic Demo Scan", help="Demonstrate system with an internal brain MRI pattern")

    pil_image = None
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.success(f"Loaded image: `{uploaded_file.name}` ({pil_image.size[0]}×{pil_image.size[1]} px)")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    elif use_sample:
        pil_image = create_synthetic_demo_mri()
        st.info("Loaded synthetic benchmark Brain MRI slice (Simulated Neoplasm).")

    if pil_image is None:
        st.markdown("---")
        st.info("👈 Please upload a brain MRI scan or click **'Load Synthetic Demo Scan'** to run the diagnosis pipeline.")

        # Display architectural capabilities preview
        st.subheader("📐 System Workflow Overview")
        st.markdown(
            """
            | Pipeline Stage | Deep Learning Architecture | Clinical Objective |
            | :--- | :--- | :--- |
            | **1. Classification** | **ResNet-50** (ImageNet Weights) | Categorize scan into Glioma, Meningioma, Pituitary, or No Tumor. |
            | **2. Segmentation** | **U-Net** (ResNet-34 Encoder) | Delineate exact pixel-level spatial boundaries of neoplastic lesions. |
            | **3. Explainability** | **Grad-CAM** (Layer 4 Backprop) | Reveal convolutional visual attention maps explaining classification decisions. |
            """
        )
        return

    # Load models
    classifier, segmenter, device, model_errors = load_models_cached(
        clf_path, seg_path, device_name=device_opt
    )

    # Check model loading status
    models_ready = classifier is not None and segmenter is not None

    if not models_ready:
        st.warning("⚠️ **Trained Model Weights Notice**")
        st.markdown(
            f"""
            One or more trained model checkpoints were not located at the specified paths:
            - **Classifier**: `{clf_path}` (Status: {'Found' if classifier is not None else 'Missing'})
            - **Segmenter**: `{seg_path}` (Status: {'Found' if segmenter is not None else 'Missing'})

            **How to train and produce the models:**
            ```bash
            # 1. Place BRISC2025 dataset in data/brisc2025/
            # 2. Train the ResNet50 Classifier:
            python -m src.training.train_classifier --epochs 15

            # 3. Train the U-Net Tumor Segmenter:
            python -m src.training.train_segmenter --epochs 20
            ```
            """
        )
        if model_errors:
            with st.expander("Diagnostic details"):
                for err in model_errors:
                    st.write(f"- {err}")

    # Run Analysis if models are available
    st.markdown("---")
    st.subheader("🔬 2. Diagnostic Analysis Results")

    from src.data.preprocessing import preprocess_image_for_inference
    from src.utils.visualization import create_segmentation_overlay

    input_tensor, _ = preprocess_image_for_inference(pil_image, image_size=(224, 224))
    img_resized_np = np.array(pil_image.resize((224, 224)), dtype=np.float32) / 255.0

    if models_ready:
        input_tensor = input_tensor.to(device)

        # 1. Run Classification
        pred_idx, pred_class, confidence, prob_dict = classifier.predict(input_tensor)

        # 2. Run Segmentation
        prob_mask, binary_mask, tumor_area_pct = segmenter.predict_mask(input_tensor, threshold=threshold)

        # 3. Run Grad-CAM
        from src.explainability.gradcam import GradCAMExplainer
        explainer = GradCAMExplainer(classifier)
        cam_heatmap, cam_overlay = explainer.explain(
            input_tensor=input_tensor,
            original_image=img_resized_np,
            target_class_idx=pred_idx,
            alpha=0.5,
        )
    else:
        # Fallback simulation mode for demonstration when weights are not yet trained
        st.info("ℹ️ Running in **Demonstration Mode** (Model checkpoints pending training).")
        pred_class = "Glioma (Pending Checkpoint)"
        confidence = 0.88
        prob_dict = {"glioma": 0.88, "meningioma": 0.06, "no_tumor": 0.02, "pituitary": 0.04}
        binary_mask = (img_resized_np[..., 0] > 0.85).astype(np.uint8)
        tumor_area_pct = round(float(np.sum(binary_mask) / binary_mask.size * 100.0), 2)
        cam_heatmap = binary_mask.astype(np.float32)
        cam_overlay = create_segmentation_overlay(img_resized_np, binary_mask, color=(0.9, 0.2, 0.2))

    # Metric Banner
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Predicted Category", pred_class.capitalize())
    with col_m2:
        st.metric("Diagnostic Confidence", f"{confidence * 100:.1f}%")
    with col_m3:
        st.metric("Neoplasm Area Coverage", f"{tumor_area_pct}%")
    with col_m4:
        st.metric("Active Hardware", str(device).upper())

    # Tabbed Detailed Views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Spatial Localization & Overlay",
        "📊 Probability Distribution",
        "💡 Explainable AI (Grad-CAM)",
        "📋 Diagnostic Summary & Export",
    ])

    # Tab 1: Spatial Localization
    with tab1:
        st.markdown("#### Spatial Lesion Delineation")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("1. Original Scan")
            st.image(pil_image, caption="Uploaded Patient MRI", use_container_width=True)

        with c2:
            st.subheader("2. Predicted Mask")
            st.image(binary_mask * 255, caption=f"U-Net Binary Mask (Threshold: {threshold})", use_container_width=True)

        with c3:
            st.subheader("3. Anatomical Overlay")
            overlay = create_segmentation_overlay(img_resized_np, binary_mask, alpha=overlay_alpha)
            st.image(overlay, caption="Tumor Boundary Overlay (Red)", use_container_width=True)

    # Tab 2: Probabilities
    with tab2:
        st.markdown("#### Multi-Class Softmax Probability Distribution")
        for cls_name, prob in prob_dict.items():
            col_bar_lbl, col_bar_val = st.columns([1, 4])
            with col_bar_lbl:
                st.write(f"**{cls_name.capitalize()}**")
            with col_bar_val:
                st.progress(min(max(float(prob), 0.0), 1.0), text=f"{prob*100:.2f}%")

    # Tab 3: Grad-CAM Explainability
    with tab3:
        st.markdown("#### Convolutional Class Activation Mapping (Grad-CAM)")
        gc1, gc2 = st.columns(2)
        with gc1:
            st.image(cam_heatmap, caption="Grad-CAM 2D Activation Heatmap", use_container_width=True)
        with gc2:
            st.image(cam_overlay, caption="Grad-CAM Saliency Overlay on MRI", use_container_width=True)

        st.markdown("---")
        notes = (
            GradCAMExplainer.get_clinical_interpretation_notes()
            if "GradCAMExplainer" in locals()
            else {}
        )
        with st.expander("ℹ️ Understanding Grad-CAM vs U-Net Segmentation", expanded=True):
            st.markdown(
                f"""
                - **What Grad-CAM Visualizes**: {notes.get('what_it_is', 'Visualizes convolutional attention driving classification.')}
                - **Mechanism**: {notes.get('how_it_works', 'Computes gradient-weighted feature maps at the final layer.')}
                - **Key Clinical Distinction**: {notes.get('clinical_limitations', 'Grad-CAM shows classification influence, NOT anatomical resection borders.')}
                """
            )

    # Tab 4: Export Report
    with tab4:
        st.markdown("#### Structured Diagnostic Report")
        report_data = {
            "predicted_class": pred_class,
            "confidence": float(round(confidence, 4)),
            "probabilities": prob_dict,
            "tumor_area_percentage": float(tumor_area_pct),
            "segmentation_threshold": float(threshold),
            "inference_device": str(device),
            "disclaimer": "Academic and Research Prototype Only. Not for clinical use.",
        }

        st.json(report_data)

        report_json = json.dumps(report_data, indent=4)
        st.download_button(
            label="💾 Download Diagnostic JSON Report",
            data=report_json,
            file_name="diagnostic_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
