"""
Face extraction using MTCNN for DeepFake detection pipeline.
Given a directory of videos, extracts frames and crops faces for training.
"""

import os
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from pathlib import Path

from .config import FRAME_EXTRACTION_RATE, FACE_SIZE


def extract_frames(video_path, output_dir, frame_rate=FRAME_EXTRACTION_RATE):
    """
    Extract 1 frame per second from a 30fps video.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_interval = max(int(fps / frame_rate), 1)

    frame_id = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_id += 1

    cap.release()


def extract_faces_from_frames(frames_dir, output_dir):
    """
    Detect faces from extracted frames using MTCNN.
    Saves cropped, normalized face images.
    """
    os.makedirs(output_dir, exist_ok=True)
    detector = MTCNN()

    frame_paths = list(Path(frames_dir).glob("*.jpg"))

    for frame_path in frame_paths:
        img = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)

        detections = detector.detect_faces(img)
        if len(detections) == 0:
            continue

        x, y, w, h = detections[0]["box"]
        x, y = max(x, 0), max(y, 0)

        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, FACE_SIZE)

        output_path = os.path.join(output_dir, f"face_{frame_path.stem}.jpg")
        Image.fromarray(face).save(output_path)


def process_video(video_path, output_dir):
    """
    Full pipeline:
    1. extract frames
    2. extract faces
    """
    temp_frames_dir = os.path.join(output_dir, "frames")
    faces_dir = os.path.join(output_dir, "faces")

    extract_frames(video_path, temp_frames_dir)
    extract_faces_from_frames(temp_frames_dir, faces_dir)

    # Optionally clear frame directory to save space:
    # shutil.rmtree(temp_frames_dir)

    return faces_dir
