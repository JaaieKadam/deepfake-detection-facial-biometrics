"""
MTCNN + ResNet50 DeepFake Detection Pipeline
--------------------------------------------
This script:
1. Loads preprocessed face images (real & fake)
2. Splits into train/validation sets
3. Trains a ResNet50-based deepfake classifier
4. Saves the best model and evaluation metrics
"""

import os
from utils.config import (
    REAL_FRAMES_DIR,
    FAKE_FRAMES_DIR,
    RESNET50,
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
# LOAD DATA
# ---------------------------------------------------------

print("📥 Loading dataset from pre-extracted MTCNN face crops...")

X, y = load_dataset(REAL_FRAMES_DIR, FAKE_FRAMES_DIR)
print(f"➡ Loaded {len(X)} face images.")

# ---------------------------------------------------------
# TRAIN/VAL SPLIT
# ---------------------------------------------------------

X_train, X_val, y_train, y_val = split_dataset(X, y)
print(f"➡ Training samples: {len(X_train)}")
print(f"➡ Validation samples: {len(X_val)}")

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

model, history = train_model(
    model_name=RESNET50,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)

# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------

print("\n🔍 Evaluating model on validation set...")

y_pred = model.predict(X_val)

# Classification report
print_classification_report(y_val, y_pred)

# Confusion matrix
plot_confusion_matrix(y_val, y_pred, model_name=RESNET50)

# ROC curve
plot_roc_curve(y_val, y_pred, model_name=RESNET50)

print("\n🎉 ResNet50 training complete!")
print(f"📁 Outputs saved to: {OUTPUT_DIR}")
