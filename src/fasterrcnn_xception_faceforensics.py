"""
Faster R-CNN + Xception DeepFake Detection Pipeline (Template)
--------------------------------------------------------------
This script outlines the workflow used in the thesis for
face extraction using the Faster R-CNN object detector.

NOTE:
- Faster R-CNN face extraction requires a PyTorch-based model
- Dataset must be annotated (bounding boxes)
- This script is a realistic engineering template with TODO steps
- Xception is used as the classification backbone
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
# PLACEHOLDER: Faster R-CNN Detector
# ---------------------------------------------------------

def detect_faces_fasterrcnn(image):
    """
    TODO:
    - Load PyTorch Faster R-CNN model
    - Run detection
    - Return cropped face(s)

    This placeholder function allows the script to exist as
    part of the framework without requiring a full PyTorch setup.

    Thesis reference:
    Face extraction was performed using Faster R-CNN for comparison
    against the MTCNN-based pipeline.

    Returns
    -------
    None (template)
    """
    raise NotImplementedError(
        "Faster R-CNN face extraction must be implemented separately."
    )


# ---------------------------------------------------------
# TRAINING PIPELINE (Classification only)
# ---------------------------------------------------------

print("⚠ Note: This script uses MTCNN-extracted faces as input for the Xception model.")
print("Faster R-CNN detection is left as a TODO template.")

print("📥 Loading dataset...")

X, y = load_dataset(REAL_FRAMES_DIR, FAKE_FRAMES_DIR)
print(f"➡ Loaded {len(X)} images")

X_train, X_val, y_train, y_val = split_dataset(X, y)
print(f"➡ Train: {len(X_train)} | Val: {len(X_val)}")

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


