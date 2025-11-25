"""
Global configuration file for DeepFake Detection Pipeline.
Centralized place for dataset paths, hyperparameters, and constants.
"""

import os

# ---------------------------------------------------------
# Dataset Paths (Update these according to your local setup)
# ---------------------------------------------------------

# Example structure (FaceForensics++ compressed c23 extracted frames):
# dataset/
#   real/
#       video_1/
#           frame_001.jpg
#           frame_002.jpg
#   fake/
#       video_2/
#           frame_001.jpg
#           frame_002.jpg

BASE_DATASET_DIR = "/path/to/FaceForensicsPP_frames"  # <- UPDATE THIS

REAL_FRAMES_DIR = os.path.join(BASE_DATASET_DIR, "real")
FAKE_FRAMES_DIR = os.path.join(BASE_DATASET_DIR, "fake")

# Output directory for saved models, logs, plots
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Face Extraction
# ---------------------------------------------------------

FRAME_EXTRACTION_RATE = 30  # extract 1 frame per second from 30fps videos

FACE_SIZE = (224, 224)  # standardized size for all models

# ---------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2

# ---------------------------------------------------------
# Model Names
# ---------------------------------------------------------

RESNET50 = "resnet50"
EFFICIENTNET_B0 = "efficientnet_b0"
XCEPTION = "xception"

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

SEED = 42
