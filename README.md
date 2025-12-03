# Facial Emotion Recognition using Custom CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

**Accuracy Achieved:** 64.70% (Validation)  
**Dataset:** FER2013 (Grayscale, 48x48 pixels)

## 📌 Overview
This project implements a deep learning model capable of classifying human facial expressions into 7 categories: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.

While many approaches use heavy Transfer Learning models, this project demonstrates that a **custom-built, lightweight CNN** can achieve human-level performance on this specific low-resolution dataset.

## 🏗️ Architecture & Approach
I built a custom Convolutional Neural Network (CNN) optimized for the small 48x48 input size.

* **Input:** 48x48 Grayscale images.
* **Backbone:** 4 blocks of Conv2D + BatchNorm + ReLU + MaxPooling.
* **Regularization:** Used `Dropout` layers to prevent overfitting.
* **Head:** Replaced the traditional `Flatten` layer with `GlobalAveragePooling2D` to drastically reduce parameters (~130k total parameters) and improve generalization.

## 🔧 Key Techniques
1.  **Data Augmentation:** Applied random rotations (15°), zooms, and shifts to robustness.
2.  **Class Weights:** Computed and applied class weights to handle the severe class imbalance in FER2013.
3.  **Callbacks:** Implemented `EarlyStopping` and `ReduceLROnPlateau` to fine-tune the learning rate dynamically during training.

## 🚀 Results
* **Final Accuracy:** **64.70%**
* The model balances efficiency with performance, proving that custom architectures can outperform pre-trained models (like EfficientNet) on small, grayscale images.

