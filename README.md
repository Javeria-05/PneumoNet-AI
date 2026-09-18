# 🫁 PneumoNet-AI

### AI-Powered Pneumonia Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

PneumoNet-AI is a Deep Learning-based diagnostic support tool that classifies Chest X-Ray images as **NORMAL** or **PNEUMONIA**, using Transfer Learning on a MobileNetV2 backbone. It ships with an interactive Streamlit dashboard for real-time predictions.

---

## 🎯 Overview

Pneumonia remains one of the leading causes of hospitalization worldwide, and early detection through chest imaging plays a critical role in patient outcomes. PneumoNet-AI demonstrates how deep learning and computer vision can support faster, data-driven screening for public health applications.

---

## ✨ Key Features

- 🔬 **Transfer Learning** — Fine-tuned MobileNetV2 (pre-trained on ImageNet) for medical image classification
- 📊 **Balanced Training** — Class-weighted loss to handle dataset imbalance (NORMAL vs PNEUMONIA)
- 🖥️ **Interactive Dashboard** — Upload an X-ray and get instant predictions with confidence scores
- 📈 **Full Evaluation Suite** — Precision, recall, F1-score, and confusion matrix on unseen test data

---

## 🖼️ Demo

<img width="1068" height="633" alt="image" src="https://github.com/user-attachments/assets/cdb1f765-2330-43b8-81a3-891a01da6c1b" />
<img width="983" height="608" alt="image" src="https://github.com/user-attachments/assets/d1f89348-e856-466d-bc9c-a5cb7b7e2779" />



---

## 📂 Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle dataset containing 5,856 labeled chest X-ray images, split into training, validation, and test sets.

| Split | NORMAL | PNEUMONIA |
|---|---|---|
| Train | 1,341 | 3,875 |
| Test | 234 | 390 |

---

## 🧠 Model Architecture
Input (224x224x3)
↓
MobileNetV2 (frozen, ImageNet weights)
↓
GlobalAveragePooling2D
↓
Dense(128, ReLU)
↓
Dropout(0.3)
↓
Dense(1, Sigmoid)


**Training config:** 10 epochs, Adam optimizer (lr=0.0001), binary cross-entropy loss, class-weighted to handle imbalance.

---

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy | **89%** |
| Pneumonia Recall | **96%** |
| Normal Recall | **76%** |
| F1-Score (weighted) | **88%** |

<!-- ![Confusion Matrix](assets/confusion_matrix.png) -->

---

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `OpenCV` · `Streamlit` · `Scikit-learn` · `Pandas` · `NumPy` · `Google Colab`

---

## 📁 Project Structure

PneumoNet-AI/
├── app.py                   # Streamlit dashboard (with Grad-CAM)
├── pneumonia_model.keras    # Trained model
├── pneumonia_training.ipynb # Training notebook
├── requirements.txt         # Dependencies
├── LICENSE                  # MIT License
└── README.md


---

## 🚀 Usage

```bash
git clone https://github.com/Javeria-05/PneumoNet-AI.git
cd PneumoNet-AI
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Improvements

- Fine-tune deeper MobileNetV2 layers for higher accuracy
- Expand dataset with more diverse, multi-source X-rays
---

## ⚠️ Disclaimer

This tool is developed for **research and educational purposes only**. It is not a certified medical device and should not be used as a substitute for professional medical diagnosis.

---

## 👤 Author

Javeria Irum
