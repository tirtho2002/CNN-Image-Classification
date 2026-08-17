# Fashion-MNIST Image Classification with CNN

A Convolutional Neural Network (CNN) built in **PyTorch** to classify clothing images from the **Fashion-MNIST** dataset into 10 categories. The model is trained, evaluated on a held-out test set, and further stress-tested on custom real-world photos to study its generalization limits.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project implements an end-to-end image classification pipeline:

- Loads and preprocesses the Fashion-MNIST dataset
- Trains a custom 2-block CNN from scratch (no pretrained weights)
- Tracks training/validation loss and accuracy across epochs
- Evaluates on the test set with a confusion matrix and classification report
- Tests the trained model on **custom real-world clothing photos** to analyze the train/real-world domain gap

| Metric | Score |
|---|---|
| Training Accuracy | **96.61%** |
| Validation Accuracy | **92.47%** |
| Test Accuracy | **91.62%** |

---

## 🗂️ Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 grayscale, 28×28 images across 10 clothing classes.

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Split used: **48,000 train / 12,000 validation / 10,000 test**.

<p align="center">
  <img src="imgs/class_distribution.png" width="600"><br>
  <em>Class distribution across train/validation/test splits</em>
</p>

<p align="center">
  <img src="imgs/sample_image.png" width="220"><br>
  <em>Sample preprocessed training image</em>
</p>

---

## 🏗️ Model Architecture

```
Input (1×28×28)
   │
Conv2D(1→32, 3×3, pad=1) → ReLU → MaxPool(2×2)     → 32×14×14
   │
Conv2D(32→64, 3×3, pad=1) → ReLU → MaxPool(2×2)    → 64×7×7
   │
Flatten                                             → 3136
   │
Linear(3136→128) → ReLU
   │
Linear(128→10)                                      → 10 class logits
```

**Training config:**
- Loss: Cross-Entropy Loss
- Optimizer: Adam (`lr=0.001`)
- Batch size: 64
- Epochs: 10

---

## 📊 Results

### Training curves
Training loss decreases steadily; validation loss plateaus after epoch ~5–6, indicating mild overfitting in later epochs.

<p align="center">
  <img src="imgs/loss_curve.png" width="420">
  <img src="imgs/accuracy_curve.png" width="420">
</p>

### Confusion matrix

<p align="center">
  <img src="imgs/confusion_matrix.png" width="500">
</p>

### Per-class performance (test set)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| T-shirt/top | 0.85 | 0.86 | 0.86 |
| Trouser | 0.99 | 0.99 | 0.99 |
| Pullover | 0.84 | 0.91 | 0.87 |
| Dress | 0.91 | 0.93 | 0.92 |
| Coat | 0.89 | 0.85 | 0.87 |
| Sandal | 0.98 | 0.98 | 0.98 |
| Shirt | 0.79 | 0.72 | 0.75 |
| Sneaker | 0.95 | 0.97 | 0.96 |
| Bag | 0.98 | 0.98 | 0.98 |
| Ankle boot | 0.97 | 0.96 | 0.97 |

<p align="center">
  <img src="imgs/per_class_metrics.png" width="500">
  <img src="imgs/per_class_accuracy.png" width="420">
</p>

**Key finding:** most confusion happens between visually similar upper-body garments (Shirt, T-shirt, Pullover, Coat), while shape-distinctive classes (Trouser, Sandal, Bag) are classified almost perfectly.

### Qualitative predictions on test images

<p align="center">
  <img src="imgs/test_predictions_grid.png" width="600">
</p>

---

## 🌍 Real-World Generalization Test

The trained model was also tested on 10 custom photographs of real clothing items. Despite strong benchmark accuracy, it performed poorly on these out-of-distribution images (frequently predicting "Bag"), highlighting a **domain-shift gap** between clean benchmark data and real-world photos (background clutter, lighting, framing, resolution loss).

<p align="center">
  <img src="imgs/custom_predictions.png" width="700">
</p>

See `report.pdf` for the full analysis of this generalization gap.

---

## 📁 Repository Structure

```
.
├── README.md
├── 220150_CNN.ipynb        # Full notebook: data loading, model, training, evaluation
├── report.pdf              # Detailed lab report (theory, results, analysis)
├── report.tex              # LaTeX source of the report
└── imgs/                   # All images used in this README
    ├── sample_image.png
    ├── class_distribution.png
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── confusion_matrix.png
    ├── per_class_metrics.png
    ├── per_class_accuracy.png
    ├── test_predictions_grid.png
    └── custom_predictions.png
```

### 🖼️ Where each image goes

Create a folder named **`imgs/`** in the root of your repo (same level as this README) and place the images with these **exact filenames** — the README above already links to them by these names, so nothing else needs to change once they're in place:

| File to save as | Which image it is |
|---|---|
| `imgs/sample_image.png` | The single preprocessed grayscale training image (Dress example) |
| `imgs/class_distribution.png` | The bar chart of train/val/test class counts |
| `imgs/loss_curve.png` | Training vs. validation **loss** curve |
| `imgs/accuracy_curve.png` | Training vs. validation **accuracy** curve |
| `imgs/confusion_matrix.png` | The 10×10 confusion matrix heatmap |
| `imgs/per_class_metrics.png` | Grouped bar chart of Precision/Recall/F1 per class |
| `imgs/per_class_accuracy.png` | Bar chart of per-class accuracy vs. overall test accuracy |
| `imgs/test_predictions_grid.png` | The 12-image grid of True vs. Predicted labels from the test set |
| `imgs/custom_predictions.png` | The grid of 10 custom real-world photos with predictions |

All 9 of these images are ready-made and attached alongside this README — just drop them straight into your repo's `imgs/` folder with the filenames above.

---

## 🚀 Getting Started

### Requirements
```bash
pip install torch torchvision matplotlib numpy pillow scikit-learn
```

### Run
```bash
jupyter notebook 220150_CNN.ipynb
```
The Fashion-MNIST dataset is downloaded automatically via `torchvision.datasets.FashionMNIST` on first run.

---

## 🔮 Future Improvements

- Add dropout / batch normalization to reduce overfitting
- Data augmentation (flips, rotations, translations) for better generalization
- Fine-tune on real-world photos to close the domain gap
- Experiment with deeper architectures (ResNet-style blocks)

---

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research
- [PyTorch](https://pytorch.org/)
