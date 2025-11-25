"""
Utility functions for loading cropped face images into the training pipeline.
Handles train/validation splitting and label encoding.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from .config import FACE_SIZE, VAL_SPLIT, SEED


def load_images_from_directory(directory):
    """
    Loads all face images from a directory and returns as a NumPy array.
    """
    images = []
    paths = sorted(os.listdir(directory))

    for img_name in paths:
        img_path = os.path.join(directory, img_name)
        img = image.load_img(img_path, target_size=FACE_SIZE)
        img = image.img_to_array(img) / 255.0
        images.append(img)

    return np.array(images)


def load_dataset(real_dir, fake_dir):
    """
    Loads real and fake face crops, assigns labels, and returns X, y arrays.
    """
    real_images = load_images_from_directory(real_dir)
    fake_images = load_images_from_directory(fake_dir)

    X = np.concatenate([real_images, fake_images], axis=0)
    y = np.array([0] * len(real_images) + [1] * len(fake_images))  # 0=real, 1=fake

    return X, y


def split_dataset(X, y, val_split=VAL_SPLIT):
    """
    Splits dataset into train and validation sets.
    """
    return train_test_split(
        X,
        y,
        test_size=val_split,
        random_state=SEED,
        shuffle=True,
        stratify=y,
    )
