<div align="center">

# 🩸 Blood Cancer Detection Model

### Detecting Malignant Blood Cells with Deep Learning — 97% Accuracy

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-EfficientNetB0-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=for-the-badge)](https://github.com/ADARSH685-BOT/Blood-Cancer-Detection-Model)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

---

*An end-to-end deep learning pipeline that classifies blood cell microscopy images as **Benign** or **Malignant** using transfer learning with EfficientNetB0.*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Disclaimer](#-disclaimer)

---

## 🔬 Overview

Blood cancers such as **AML (Acute Myeloid Leukemia)**, **ALL (Acute Lymphoblastic Leukemia)**, and **CLL (Chronic Lymphocytic Leukemia)** are often diagnosed late due to the manual, time-intensive nature of blood smear analysis.

This project applies **Computer Vision** and **Transfer Learning** to automate the detection of malignant blood cells from microscopy images — offering a fast, reproducible, and highly accurate classification pipeline that can assist pathologists and medical researchers.

> 🎯 **Goal:** Classify blood cell images as `Benign` or `Malignant` with high precision and recall.

---

## 🎬 Demo

```
Input Image (Blood Smear)
        │
        ▼
┌───────────────────────────┐
│     EfficientNetB0        │
│   (Transfer Learning)     │
│                           │
│  • Data Augmentation      │
│  • Global Avg Pooling     │
│  • Dropout (0.2)          │
│  • Softmax Output         │
└───────────────────────────┘
        │
        ▼
  ┌─────────────┐
  │  BENIGN ✅  │  or  │  MALIGNANT ❌  │
  └─────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Transfer Learning** | EfficientNetB0 pretrained on ImageNet — fine-tuned for medical imaging |
| 🔄 **Data Augmentation** | Random flip, rotation, zoom, contrast & brightness to prevent overfitting |
| 📊 **Full Evaluation Suite** | Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC-AUC Curve |
| ⚡ **Smart Callbacks** | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint for optimal training |
| 🎯 **Fine-Tuning** | Two-phase training — frozen base → unfrozen fine-tune for maximum accuracy |
| 🔮 **Inference Ready** | Built-in prediction function for classifying new blood sample images |
| 📁 **Portfolio Ready** | Clean, well-documented code structured for GitHub and research review |

---

## 📈 Model Performance

<div align="center">

| Metric | Score |
|---|---|
| ✅ **Accuracy** | **97%** |
| 🎯 **Precision** | High |
| 🔁 **Recall** | High |
| 📐 **F1 Score** | High |
| 📉 **Loss** | Minimal |

</div>

> The model was trained with an **80 / 10 / 10** train-validation-test split and evaluated on a held-out test set to ensure unbiased performance reporting.

---

## 🛠️ Tech Stack

```
Language      →  Python 3.10+
Framework     →  TensorFlow 2.x / Keras
Architecture  →  EfficientNetB0 (ImageNet pretrained)
Environment   →  Google Colab (GPU accelerated)
Libraries     →  NumPy, Matplotlib, Seaborn, Scikit-learn, Pillow
```

---

## 🗂️ Dataset

The dataset consists of labeled blood cell microscopy images organized into two classes:

```
blood_cancer_dataset/
├── Benign/          ← Normal / healthy blood cells (~804 images)
│   ├── WBC-Benign-001.jpg
│   ├── WBC-Benign-002.jpg
│   └── ...
└── Malignant/       ← Cancerous blood cells (~804 images)
    ├── WBC-Malignant-001.jpg
    ├── WBC-Malignant-002.jpg
    └── ...
```

**Malignant cell characteristics the model learned to detect:**
- Irregular, enlarged nuclei
- Prominent nucleoli (visible red dots under microscope)
- High nucleus-to-cytoplasm (N:C) ratio
- Abnormal cell clustering and mitotic figures

**Data split:**
- 🟦 Training — 80%
- 🟨 Validation — 10%
- 🟥 Test — 10%

---

## 📁 Project Structure

```
Blood-Cancer-Detection-Model/
│
├── Untitled4.ipynb          ← Main training & evaluation notebook
├── README.md                ← Project documentation (you are here)
│
├── best_model.keras         ← Saved best model checkpoint (after training)
├── training_history.pkl     ← Pickled training history for analysis
│
└── plots/                   ← Generated evaluation plots
    ├── accuracy_curve.png
    ├── loss_curve.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## ⚙️ How It Works

### Phase 1 — Feature Extraction (Frozen Base)
```python
# Load EfficientNetB0 pretrained on ImageNet
base_model = EfficientNetB0(include_top=False, weights='imagenet')
base_model.trainable = False   # Freeze pretrained layers

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.2)(x)
output = Dense(2, activation='softmax')(x)
```

### Phase 2 — Fine-Tuning (Unfrozen Top Layers)
```python
# Unfreeze top layers and retrain with a very low learning rate
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5), ...)
```

### Training Callbacks
```python
callbacks = [
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
]
```

### Prediction on New Images
```python
def predict(image_path, model, class_names):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ADARSH685-BOT/Blood-Cancer-Detection-Model.git
cd Blood-Cancer-Detection-Model
```

### 2. Install dependencies
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow
```

### 3. Prepare the dataset
Place your dataset in the following structure:
```
/content/blood_cancer_dataset/
    ├── Benign/
    └── Malignant/
```

### 4. Run the notebook
Open `Untitled4.ipynb` in **Google Colab** or **Jupyter Notebook** and run all cells top to bottom.

### 5. Predict on a new image
```python
predicted_class, confidence = predict('your_blood_sample.jpg', model, class_names)
print(f"Prediction: {predicted_class} ({confidence}% confidence)")
```

---

## 📊 Results

After training for up to 30 epochs with early stopping:

- ✅ **Test Accuracy: 97%**
- 📉 Validation loss converged smoothly — no significant overfitting
- 📈 ROC-AUC score near 1.0 — excellent discriminative ability
- 🔁 High recall on malignant class — critical for medical applications

---

## ⚠️ Disclaimer

> This project is developed strictly for **educational and research purposes**.
> It is **not** a certified medical device and must **not** be used as a substitute for professional medical diagnosis.
> Always consult a qualified medical professional for clinical decisions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ and Python by [ADARSH685-BOT](https://github.com/ADARSH685-BOT)

⭐ If this project helped you, please consider giving it a star!

</div>
