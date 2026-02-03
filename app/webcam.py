import cv2
import numpy as np
import json
import tensorflow as tf

MODEL_PATH = "models/emotion_model.h5"
LABELS_PATH = "models/label_map.json"
IMG_SIZE = 48


model = tf.keras.models.load_model(MODEL_PATH)


with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)     

        preds = model.predict(face, verbose=0)[0]
        emotion_index = np.argmax(preds)
        emotion_label = label_map[emotion_index]
        confidence = preds[emotion_index]

       
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{emotion_label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
