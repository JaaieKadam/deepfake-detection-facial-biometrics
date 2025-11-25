"""
Training utilities for DeepFake Detection Pipeline.
Includes model constructors and unified training wrapper.
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50,
    EfficientNetB0,
    Xception
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

from .config import (
    OUTPUT_DIR,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    RESNET50,
    EFFICIENTNET_B0,
    XCEPTION
)

# Ensure directories exist
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)


# ---------------------------------------------------------
# MODEL BUILDERS
# ---------------------------------------------------------

def build_backbone(model_name, input_shape=(224, 224, 3)):
    """
    Returns model backbone based on selected architecture.
    """
    if model_name == RESNET50:
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == EFFICIENTNET_B0:
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == XCEPTION:
        base = Xception(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model backbone: {model_name}")

    base.trainable = False  # Transfer learning base
    return base


def build_model(model_name, input_shape=(224, 224, 3), dropout_rate=0.3):
    """
    Builds full classification model with selected backbone.
    """
    base = build_backbone(model_name, input_shape)

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation="sigmoid")(x)  # binary classification

    model = Model(inputs=base.input, outputs=output)

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


# ---------------------------------------------------------
# TRAINING WRAPPER
# ---------------------------------------------------------

def train_model(model_name, X_train, y_train, X_val, y_val):
    """
    Unified training function for all architectures.
    """
    print(f"\n🔧 Building model: {model_name}\n")

    model = build_model(model_name)

    # Callbacks
    checkpoint_path = os.path.join(OUTPUT_DIR, "models", f"{model_name}_best.h5")
    log_path = os.path.join(OUTPUT_DIR, "logs", f"{model_name}_training.csv")

    callbacks = [
        ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, verbose=1
        ),
        CSVLogger(log_path)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
    )

    print(f"\n✅ Training completed. Best model saved: {checkpoint_path}\n")

    return model, history
