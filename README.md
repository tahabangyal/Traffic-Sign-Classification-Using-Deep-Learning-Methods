# 🚦 Traffic Sign-Classification Using Deep Learning Methods

An end-to-end computer vision and deep learning system comparing classical image processing, handcrafted feature descriptors with machine learning baselines, and transfer learning with explainable AI (Grad-CAM) on the **BelgiumTSC** benchmark dataset.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-EfficientNetB0-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Classical%20ML-SVM%20%7C%20PCA-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BelgiumTSC%20(62%20Classes)-blue.svg)]()
[![Explainability](https://img.shields.io/badge/XAI-Grad--CAM-brightgreen.svg)]()

---

## 📌 Project Overview

Traffic sign recognition is an essential component for intelligent transportation systems and autonomous driving. Rather than treating the task solely as an uninterpretable black box, this project implements a complete three-stage vision pipeline across 62 real-world European traffic sign categories:

* **Module 1 — Low-Level Image Processing & Distortions:** Simulating real-world driving distortions (rotations, scale changes), Gaussian noise modeling, edge/boundary analysis, and Non-Local Means (NLMeans) filtering.
* **Module 2 — Handcrafted Descriptors & Classical ML:** Building fused feature representations (HOG + ORB), compressing the feature subspace with Principal Component Analysis (PCA), and classifying with Support Vector Machines (SVM).
* **Module 3 — Deep Transfer Learning & Explainability (XAI):** Fine-tuning EfficientNetB0 with extensive data augmentation, achieving over 95% accuracy, and visualizing attention patterns via Grad-CAM.

---

## 📊 Benchmark & Comparative Analysis

| Approach | Feature Representation | Classifier / Model | Validation Accuracy | Characteristics & Error Observations |
| :--- | :--- | :--- | :---: | :--- |
| **Classical ML Baseline** | Concatenated HOG + ORB pooled vectors reduced via PCA | Support Vector Machine (RBF Kernel) | **~70% – 85%** | Fast training baseline; exhibits confusion on fine intra-class variations (e.g., speed limits 30, 50, 70) and sensitivity to rotation. |
| **Deep Transfer Learning** | Hierarchical convolutional features (ImageNet pre-trained) | **EfficientNetB0** + Custom Classification Head | **> 95%** | High robustness against motion blur, perspective shifts, and lighting extremes; verified via Grad-CAM heatmaps. |

---

## 🏗️ Technical Architecture & Pipeline

### Module 1: Image Processing & Distortion Modeling
* **Geometric Transforms:** $\pm30^{\circ}$ rotations, horizontal flipping, and multi-scale sizing to simulate oblique vehicle camera angles and distance variations.
* **Noise Modeling & Restoration:** Injected synthetic Gaussian sensor noise and restored high-frequency details using **Non-Local Means (NLMeans)** denoising.
* **Boundary Analysis:** Evaluated **Canny**, **Sobel**, and **Laplacian** operators to confirm gradient preservation on sharp sign contours.

### Module 2: Handcrafted Feature Engineering (Classical Pipeline)
Input Image ──► Resizing ──► [HOG Gradients] ──────────┐
                       └──► [ORB Keypoints Mean-Pool]──┴──► [Feature Fusion] ──► [PCA Reduction] ──► [RBF-SVM Classifier]

* **Histogram of Oriented Gradients (HOG):** Encodes localized edge directions and structural geometry[cite: 3].
* **Oriented FAST and Rotated BRIEF (ORB):** Extracts efficient, rotation-compensated binary keypoint descriptors[cite: 3].
* **Dimensionality Reduction:** Principal Component Analysis (PCA) removes redundant dimensions and compresses concatenated representations prior to SVM training[cite: 3].

### Module 3: Deep Transfer Learning & Grad-CAM Interpretability
* **Base Network:** **EfficientNetB0** pre-trained on ImageNet.
* **Classifier Head:**
  * Global Average Pooling 2D
  * Dense (256 units, ReLU activation) + Dropout
  * Output Dense layer (Softmax, 62 classes)
* **Two-Stage Training Routine:**
  1. *Warmup Phase:* Base network frozen to train custom classification dense layers.
  2. *Fine-Tuning:* Unfroze the top 30 layers and trained with a reduced learning rate to tailor low/mid-level representations to traffic signs.
* **Data Augmentation:** Real-time shift transformations, horizontal flipping, brightness adjustments, random rotations, and zoom.
* **Explainable AI (Grad-CAM):** Gradient-weighted Class Activation Mapping computes heatmaps from final convolutional feature activations to verify that predictions stem from central symbols rather than background road artifacts.

---

## 🗂️ Dataset: BelgiumTSC

The **Belgium Traffic Sign Classification (BelgiumTSC)** benchmark features 62 classes of European traffic signs captured under real driving conditions:
* **Real-World Variations:** Multi-camera vehicle setup introduces variations in sunlight/shadow, motion blur, partial occlusions, and background clutter.
* **Structure:** Cropped sign regions partitioned into official training and testing subsets.

---

## 💻 Project Structure

```text
├── project.py                         # Interactive pipeline and training notebook
└── README.md                          # Repository documentation
