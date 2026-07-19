# Image-Based Disease Detection — Pneumonia from Chest X-Rays

Detecting pneumonia in pediatric chest X-rays using deep learning, as a proof-of-concept for AI-assisted radiology screening.

## Problem Statement

Reading chest X-rays for pneumonia takes time and radiologist expertise that isn't always available quickly, especially in high-volume or resource-limited settings. This project builds a CNN-based classifier that flags likely pneumonia cases from a raw chest X-ray image, as a decision-support proof-of-concept.

## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle / Mendeley Data (Kermany et al.)
- **Size:** 5,863 X-ray images (JPEG), pre-split into train / val / test
- **Classes:** `NORMAL`, `PNEUMONIA`
- **Subjects:** pediatric patients aged 1–5 years, Guangzhou Women and Children's Medical Center

## Approach

1. **Data pipeline** — resized to 224x224, rescaled, augmented (rotation, zoom, shift, flip) on the training set only
2. **Class imbalance** — training set is ~74% PNEUMONIA / 26% NORMAL, handled with computed class weights rather than discarding data
3. **Model** — transfer learning with **DenseNet121** pretrained on ImageNet (the same backbone family used in CheXNet, a well-known pneumonia-detection paper), trained in two phases:
   - Phase 1: frozen base, train the classification head
   - Phase 2: unfreeze the top ~30 layers and fine-tune at a low learning rate
4. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve
5. **Explainability** — Grad-CAM heatmaps to visually confirm the model is focusing on the lungs, not image artifacts

## Why Recall Matters Most Here

In a medical screening context, a **false negative** (telling a sick patient they're healthy) is far more costly than a false positive. Evaluation focuses on recall for the PNEUMONIA class as the primary metric of clinical relevance, alongside the standard metric set.

## Tech Stack

- Python, TensorFlow / Keras
- DenseNet121 (transfer learning)
- scikit-learn (metrics)
- Matplotlib, Seaborn
- Grad-CAM for model explainability

## How to Run

Built for **Google Colab** (GPU runtime recommended — Runtime → Change runtime type → GPU).

1. Open `Pneumonia_Xray_Detection.ipynb` in Colab
2. Run all cells top to bottom — the dataset downloads automatically via `kagglehub` (you'll be prompted to authenticate with a Kaggle account on first run)

```bash
pip install kagglehub tensorflow scikit-learn matplotlib seaborn
```

## Disclaimer

This is a **proof-of-concept for learning purposes**, not a diagnostic tool. Real clinical deployment would require a much larger and more diverse dataset, validation against radiologist consensus, and regulatory approval.

## Possible Extensions

- Compare against other backbones (ResNet50, EfficientNetB0)
- K-fold cross-validation for a more robust performance estimate
- Multi-class extension: bacterial vs. viral vs. normal pneumonia
- Deploy behind a simple Streamlit/Gradio demo app

## Author

Nirmal — B.Tech CSE (Data Science), D Y Patil International University
