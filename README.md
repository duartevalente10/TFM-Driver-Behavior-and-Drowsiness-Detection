# Master's Dissertation - Driver Behavior and Drowsiness Detection

This repository contains the code and machine learning pipelines developed as part of my Master's Thesis at ISEL - Instituto Superior de Engenharia de Lisboa. The project addresses two major causes of road accidents: **aggressive driving** and **driver drowsiness**.

## 📌 Project Overview

Road accidents remain a global concern, with driver behavior and fatigue playing major roles. This project implements two complementary solutions:

1. **Driver Profile Classification**: Detects aggressive, risky, and non-aggressive driving styles using telematics data.
2. **Drowsiness Detection**: Predicts drowsy states using Heart Rate Variability (HRV) features derived from ECG data.

---

## 🚗 Driver Profile Classification

**Goal:** Build upon the i-DREAMS project and prior research to classify driving behavior using ML.

### Key Features:
- Data cleaning, normalization, and dimensionality reduction
- Two-stage K-Means clustering to segment trip types
- Supervised classifiers (XGBoost, SVM) to label driving styles
- Instance balancing using ADASYN

📂 Code: [`DriverProfile/`](./DriverProfile)

---

## 😴 Drowsiness Detection

**Goal:** Predict drowsy states using physiological signals.

### Approach:
- Extract HRV features from ECG
- Align physiological signals with Karolinska Sleepiness Scale (KSS)
- Compare traditional ML models (Random Forest, etc.)
- Implement LSTM for time-series classification

📂 Code: [`DrowsinessDetection/`](./DrowsinessDetection)

---

## 📊 Technologies Used

- Python
- Scikit-learn
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib / Seaborn

---

## 📁 Folder Structure

```plaintext
.
├── DriverProfile/          # Driving behavior classification pipeline
├── DrowsinessDetection/    # Drowsiness detection models
└── README.md
