# DeepFake Detection Using MTCNN, Faster R-CNN & Modern CNN Architectures
*A comparative framework for facial biometric–based DeepFake classification*

This repository contains a complete DeepFake detection framework developed as part of an MSc research project. The system evaluates multiple **face-detection pipelines** (MTCNN, Faster R-CNN) and **deep learning backbones** (ResNet50, EfficientNet-B0, Xception) using the **FaceForensics++** dataset.  
The goal is to determine how different detectors and feature extractors influence DeepFake classification accuracy.

This project has been reconstructed into a clean, modular GitHub framework while preserving the authenticity of the original research.

---

## ⚙️ Project Motivation

DeepFake videos can deceive both humans and automated systems unless advanced detection techniques are applied.  
This project investigates whether:

- Better **face localization** → better DeepFake classification  
- Lightweight vs. heavy CNNs affect downstream performance  
- Detection noise from Faster R-CNN impacts model accuracy  
- Xception (used heavily in published SOTA DeepFake work) truly performs best  

The system is intentionally designed as a **comparative study** and a **reproducible ML pipeline**.

---

# 📊 Dataset Overview: FaceForensics++

This project uses the **FaceForensics++ (c23)** manipulated-video dataset.  
It is one of the most widely used datasets for DeepFake research due to its:

- Realistic forged sequences  
- Multiple manipulation techniques  
- High-quality real/fake labels  

### Preprocessing Summary

1. Each video is sampled at **1 frame per second**  
2. Faces extracted using **MTCNN** or **Faster R-CNN**  
3. Cropped faces resized to **224×224**  
4. Pixel normalization  
5. Train/validation split applied at frame level  

This standardized pipeline ensures consistent facial biometric features for classification.

---

# 🧠 Face Detection Pipelines Evaluated

## **1. MTCNN Pipeline (Fully Implemented)**
- Multi-task CNN  
- Robust for alignment, occlusions, and illumination  
- Produces clean face crops  
- Excellent compatibility with CNN classifiers  

## **2. Faster R-CNN Pipeline (Template)**
- Region Proposal Network–based detector  
- Computationally heavier  
- Crops were inconsistent and noisy in experiments  
- Leads to lower downstream classification accuracy  
- Included for research completeness  

## **3. YOLOv8 (Attempted; Not Finalized)**
YOLOv8 was attempted but dropped due to:

- GPU memory constraints  
- Long training times  
- Suboptimal bounding boxes for facial close-ups  
- Overkill for small, centered faces  

A template script is included for future extension.

---

# 🧪 Deep Learning Architectures Evaluated

All models used **transfer learning with ImageNet weights**:

### ✔ ResNet50  
### ✔ EfficientNet-B0  
### ✔ XceptionNet (best performing)

Common architectural components:

- Global Average Pooling  
- Dense layer (128 units)  
- Dropout regularization  
- Binary classifier head  

All trained using:

- Adam optimizer  
- Binary cross-entropy loss  
- EarlyStopping  
- ReduceLROnPlateau  
- ModelCheckpoint  

---

# 📈 Results (Extracted from Thesis)

Below are the **actual performance metrics** obtained in the thesis.

These values make the README evidence-based and grounded in real experimentation.

---

## ✅ MTCNN + CNN Model Performance

| Model            | Validation Accuracy |
|------------------|---------------------|
| **XceptionNet**  | **94.25%** |
| **ResNet50**     | **93.76%** |
| EfficientNet-B0  | 87.45% |

**Conclusion:**  
MTCNN provides clean, centered facial crops → boosting accuracy, especially for XceptionNet.

---

## ⚠️ Faster R-CNN + CNN Model Performance

| Model            | Validation Accuracy |
|------------------|---------------------|
| EfficientNet-B0  | 83.45% |
| ResNet50         | 78.12% |
| XceptionNet      | ~50% (poor performance) |

**Why accuracy dropped:**

- Faster R-CNN produced **inconsistent bounding boxes**  
- Faces were either too tight, too loose, or partially missing  
- Classifiers received noisier inputs  
- Resulted in unstable learning  

This supports a key insight in DeepFake research:  
**Face detector quality strongly influences final classification accuracy.**

---

# 🏆 Best Performing Pipeline

### ⭐ **MTCNN + XceptionNet (94.25% accuracy)**

Why this combination is best:

- XceptionNet’s depthwise separable convolutions detect subtle manipulation artifacts  
- MTCNN consistently localizes high-quality faces  
- Model shows strong generalization on validation data  

---

# 📉 Limitations & Challenges

Extracted from thesis findings:

### Technical Challenges
- Limited GPU memory → forced small batch sizes  
- Occasional training instability  
- Overfitting observed in deeper networks  
- Faster R-CNN face crops were noisy  
- DeepFake artifacts sometimes visually imperceptible  

### Unsuccessful / Incomplete Experiments
- **YOLOv8**: excessive memory usage and slow training  
- **Faster R-CNN**: significantly degraded accuracy  
- Dataset too computationally large for extended epochs  

### Dataset Limitations
- Only a subset of FF++ used  
- Frame-based classification (no temporal consistency)  
- Limited training iterations due to hardware constraints  

---

# 🔮 Future Work

Based on thesis recommendations:

- Add **temporal models** (LSTM, GRU, TimeDistributed CNNs)  
- Explore **Vision Transformers (ViT, Swin-T)**  
- Use **RetinaFace or BlazeFace** for improved detection  
- Expand dataset to full FF++  
- Combine **audio + visual** DeepFake cues  
- Train larger Xception or EfficientNet variants  
- Video-level aggregation strategies (majority voting, smoothing)  

---

# 🎓 Source

All experiments, methodology, results, and conclusions are derived from my MSc thesis:

**“DeepFake Detection using Facial Biometrics”**  
Available in the repository under:
docs/JaaieKadam_23222441.pdf



