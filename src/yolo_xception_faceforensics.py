"""
YOLO + Xception DeepFake Detection Pipeline (Template)
------------------------------------------------------
Optional experiment:
Uses YOLO for face detection, followed by Xception for deepfake classification.

This script is a template only — implement YOLO detection if desired.
"""

import os
from utils.config import (
    REAL_FRAMES_DIR,
    FAKE_FRAMES_DIR,
    XCEPTION,
    OUTPUT_DIR
)
from utils.dataset_utils import load_dataset, split_dataset
from utils.training_utils import train_model
from utils.evaluation_utils import (
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_report
)

# ---------------------------------------------------------
# YOLO Detection Placeholder
# ---------------------------------------------------------

def detect_faces_yolo(image):
    """
    TODO:
    - Load YOLO model (e.g., YOLOv5/YOLOv8)
    - Detect faces
    - Crop faces and return

    Since YOLO wasn't part of the original thesis,
    this remains optional for future experimentation.
    """
    raise NotImplementedError("YOLO face detector not implemented.")


# ---------------------------------------------------------
# TRAINING PIPELINE
# ---------------------------------------------------------

print("⚠ YOLO-based face extraction is not implemented; this is a future-work template.")
print("📥 Loading dataset...")

X, y = load_dataset(REAL_FRAMES_DIR, FAKE_FRAMES_DIR)
print(f"➡ Loaded {len(X)} images")

X_train, X_val, y_train, y_val = split_dataset(X, y)

model, history = train_model(
    model_name=XCEPTION,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

y_pred = model.predict(X_val)

print_classification_report(y_val, y_pred)
plot_confusion_matrix(y_val, y_pred, model_name=XCEPTION)
plot_roc_curve(y_val, y_pred, model_name=XCEPTION)

print("🎉 YOLO Template Experiment Complete!")
