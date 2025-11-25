"""
Video DeepFake Prediction Script
--------------------------------
Given a video file:
1. Extract frames
2. Detect faces using MTCNN
3. Run classification model on each face
4. Aggregate predictions across video
5. Output REAL or FAKE decision
"""

import os
import cv2
import numpy as np
from statistics import mean
from tensorflow.keras.models import load_model

from utils.face_extraction_mtcnn import extract_frames, extract_faces_from_frames
from utils.config import FACE_SIZE

def classify_frame(model, face_img):
    """
    Takes a cropped face image (numpy array)
    and outputs predicted probability.
    """
    img = cv2.resize(face_img, FACE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return pred


def predict_video(video_path, model_path):
    """
    Full pipeline for video inference.
    """
    temp_frames = "temp_frames"
    temp_faces = "temp_faces"

    os.makedirs(temp_frames, exist_ok=True)
    os.makedirs(temp_faces, exist_ok=True)

    # 1. Extract frames
    print("🎥 Extracting frames...")
    extract_frames(video_path, temp_frames)

    # 2. Extract faces
    print("🙂 Detecting faces...")
    extract_faces_from_frames(temp_frames, temp_faces)

    face_files = os.listdir(temp_faces)
    if len(face_files) == 0:
        print("⚠ No faces detected.")
        return None

    # 3. Load model
    print("📥 Loading model...")
    model = load_model(model_path)

    predictions = []

    # 4. Classify each face
    print("🔍 Classifying frames...")
    for face_file in face_files:
        face_path = os.path.join(temp_faces, face_file)
        face_img = cv2.imread(face_path)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        pred = classify_frame(model, face_img)
        predictions.append(pred)

    # 5. Aggregate
    avg_score = mean(predictions)
    label = "FAKE" if avg_score > 0.5 else "REAL"

    print(f"\n🎉 FINAL RESULT: {label}")
    print(f"📊 Average Fake Probability: {avg_score:.4f}")

    # Cleanup
    # shutil.rmtree(temp_frames)
    # shutil.rmtree(temp_faces)

    return label, avg_score
