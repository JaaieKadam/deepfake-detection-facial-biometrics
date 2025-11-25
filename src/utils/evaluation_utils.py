"""
Evaluation utilities for DeepFake Detection Pipeline.
Includes confusion matrix, ROC curve, and sample prediction visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from tensorflow.keras.preprocessing import image

from .config import OUTPUT_DIR


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    output_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📁 Saved confusion matrix → {output_path}")


# ---------------------------------------------------------
# ROC CURVE
# ---------------------------------------------------------

def plot_roc_curve(y_true, y_pred, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    output_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📁 Saved ROC curve → {output_path}")


# ---------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------

def print_classification_report(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    print("\n📊 Classification Report\n")
    print(classification_report(y_true, y_pred_binary))


# ---------------------------------------------------------
# SAMPLE PREDICTIONS (for README)
# ---------------------------------------------------------

def visualize_predictions(model, images, labels, model_name, num_samples=6):
    """
    Visualize model predictions for README / documentation use.
    """
    idxs = np.random.choice(len(images), num_samples, replace=False)
    plt.figure(figsize=(12, 6))

    for i, idx in enumerate(idxs):
        img = images[idx]
        label = labels[idx]

        pred = model.predict(np.expand_dims(img, 0))[0][0]
        pred_label = "FAKE" if pred > 0.5 else "REAL"

        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"True: {label} | Pred: {pred_label}\nScore: {pred:.3f}")

    output_path = os.path.join(OUTPUT_DIR, f"sample_predictions_{model_name}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"📁 Saved sample prediction grid → {output_path}")
