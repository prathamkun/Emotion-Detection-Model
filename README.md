# Emotion Detection using OpenCV + Deep Learning (End-to-End)

This is an **end-to-end Emotion Detection project** built using **OpenCV + TensorFlow/Keras**.
It trains a CNN model on the **FER-2013 dataset** and performs **real-time emotion prediction** using webcam.

✅ Trained Model Output: `models/emotion_model.h5`  
✅ Labels Saved: `models/label_map.json`  
✅ Training Graph: `models/training_plot.png`

---

## 🚀 Features

- Train a CNN model on FER-2013 dataset
- Real-time emotion detection using webcam (OpenCV)
- Shows emotion + confidence score
- Saves best model automatically during training
- Training loss/accuracy graphs saved after training

---

## 🧠 Emotions Supported

- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

---

## 📂 Project Structure

```
Emotion-Detection-Model/
│── app/
│ ├── webcam.py
│── data/
│ ├── train/
│ └── test/
│── models/
│ ├── emotion_model.h5
│ ├── label_map.json
│ └── training_plot.png
│── train.py
│── requirements.txt
│── README.md
│── .gitignore
```
---

> ⚠️ `data/` folder is ignored in GitHub because it is large.

---

## ⚙️ Setup & Installation (Mac / Linux)

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd Emotion-Detection-Model
```

---

## ✅ Tech Stack

Python

OpenCV

TensorFlow / Keras

NumPy

Matplotlib

---

## 📌 Future Improvements (Next Steps)

FastAPI backend for prediction API

Web UI for uploading image/webcam

Deploy full app (Render/Railway)

Better model using Transfer Learning (MobileNet/EfficientNet)
