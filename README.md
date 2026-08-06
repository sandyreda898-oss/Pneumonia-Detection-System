# Pneumonia-Detection-System
AI-powered Pneumonia Detection System using Deep Learning and Streamlit for Chest X-ray image classification.

#  Pneumonia Detection System using EfficientNetB0

An AI-powered web application for detecting **Pneumonia** from **Chest X-ray images** using **Transfer Learning with EfficientNetB0** and an interactive **Streamlit** interface.

The application enables users to upload a chest X-ray image and instantly receive a prediction indicating whether the case is **Normal** or **Pneumonia**, along with the model's confidence score.

---

# 📌 Project Overview

This project presents an end-to-end Deep Learning pipeline for automated pneumonia detection from chest X-ray images.

The workflow includes:

- Dataset cleaning
- Corrupted image removal
- Image preprocessing
- Dataset balancing and visualization
- Transfer Learning
- Fine-Tuning
- Model evaluation
- Interactive Streamlit deployment

The goal is to assist in the early detection of pneumonia through medical image classification.

---

# 🧠 Model Architecture

The model is built using **EfficientNetB0** pretrained on **ImageNet**.

### Training Strategy

- Transfer Learning
- Frozen base model
- Global Average Pooling
- Dropout (0.3)
- Dense Softmax Output Layer
- Fine-Tuning of the last 30 layers

Input Image Size:

```
224 × 224
```

Output Classes:

- NORMAL
- PNEUMONIA

---

# 📂 Dataset

Dataset:

**Chest X-Ray Images (Pneumonia)**

Source:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

# 🧹 Data Preparation

Before training, the dataset was carefully cleaned by:

- Removing non-image files
- Detecting corrupted images
- Inspecting image dimensions
- Checking image color modes
- Visualizing sample images
- Verifying class distribution

The original dataset was then merged and split into:

- 70% Training
- 15% Validation
- 15% Testing

using a fixed random seed for reproducibility.

---

# ⚙️ Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- Streamlit
- NumPy
- Pillow
- Matplotlib
- Pandas

---

# 📁 Project Structure

```
Pneumonia-Detection-System
│
├── app.py
├── chest_xray_classifier.keras
├── requirements.txt
├── README.md
│
├── assets/
├── sample_images/
├── screenshots/

```

---

# 🚀 Running the Application

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 📥 Download the Trained Model

The trained model is hosted separately because it exceeds GitHub's upload size limit.

Download it here:

**Google Drive**

https://drive.google.com/file/d/1xlfBfya-qwLXYnTLkcC6mSj4GI52yeZD/view?usp=sharing

Place the downloaded file inside the project folder:

```
Pneumonia-Detection-System/
│
└── chest_xray_classifier.keras
```

---

# 📸 Application Preview

## 🏠 Home Page

<p align="center">
  <img src="screenshots/Home.jpg" alt="Home Page" width="900"/>
</p>

---

## ✅ Normal Prediction

<p align="center">
  <img src="screenshots/normal_prediction.jpg" alt="Normal Prediction" width="900"/>
</p>

---

## 🦠 Pneumonia Prediction

<p align="center">
  <img src="screenshots/pneumonia_prediction.jpg" alt="Pneumonia Prediction" width="900"/>
</p>
# ✨ Features

- Upload Chest X-ray images
- AI-powered prediction
- Confidence score
- Modern Streamlit dashboard
- Image preview
- Fast inference
- Responsive interface

---

# 🔮 Future Improvements

- Multi-class lung disease classification
- Grad-CAM explainability
- PDF medical report generation
- Cloud deployment
- REST API integration
- Patient history management

---

# 👩‍💻 Author

**Sandy Reda**

Faculty of Artificial Intelligence

Deep Learning | Computer Vision | Data Science

---

# ⭐ Support

If you like this project, don't forget to give it a ⭐ on GitHub.
