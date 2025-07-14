# 🧠 Breast Cancer Detection using Neural Networks

This project leverages artificial neural networks (ANNs) to classify whether a tumor is **malignant** or **benign**, based on features extracted from breast cancer cell images. It aims to assist in early diagnosis and improve patient outcomes using machine learning.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [How to Use](#how-to-use)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 🧠 Overview

Breast cancer is one of the most common and deadly cancers affecting women worldwide. Early and accurate diagnosis is critical for effective treatment. This project implements a neural network classifier trained on real diagnostic data to predict the presence of breast cancer.

---

## 📊 Dataset

We use the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)).

It contains **569** instances with **30 features** computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

### Target Labels:
- **M** = Malignant
- **B** = Benign

---

## 🔬 Features

Some key features:
- Radius (mean of distances from center to points on the perimeter)
- Texture (standard deviation of gray-scale values)
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal dimension

(All features include mean, standard error, and worst values.)

---

## 🧱 Model Architecture

A simple feedforward neural network using Keras:
- Input layer: 30 neurons (one for each feature)
- Hidden layers: 2 fully connected layers with ReLU activation
- Output layer: 1 neuron with sigmoid activation

Loss: `binary_crossentropy`  
Optimizer: `Adam`  
Metric: `accuracy`

---

## 🛠 Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook / Streamlit (optional for deployment)

---

## 🚀 How to Use

1. **Clone the repository**
```bash
git clone https://github.com/your-username/breast-cancer-detection-nn.git
cd breast-cancer-detection-nn
