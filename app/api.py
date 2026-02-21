from flask import render_template, redirect, url_for, session
from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import json

from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import json

api_bp = Blueprint("api", __name__)

MODEL_PATH = "models/emotion_model.h5"
LABELS_PATH = "models/label_map.json"
IMG_SIZE = 48

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

label_map = {int(k): v for k, v in label_map.items()}

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


@api_bp.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face)[0]
        emotion_index = np.argmax(preds)

        results.append({
            "emotion": label_map[emotion_index],
            "confidence": float(preds[emotion_index])
        })

    return jsonify(results)



