import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import os

# Load trained model
model = load_model('model/sign_model.h5')

# Get labels from folder names
label_map = sorted(os.listdir('dataset'))
num_classes = len(label_map)

# Setup TTS (optional)
engine = pyttsx3.init()

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

IMG_SIZE = 64
prediction = ""
text_buffer = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box of hand
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Add padding to the crop
            offset = 20
            x1, y1 = max(0, x_min - offset), max(0, y_min - offset)
            x2, y2 = min(w, x_max + offset), min(h, y_max + offset)

            hand_img = frame[y1:y2, x1:x2]
            if hand_img.size == 0:
                continue

            try:
                hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            except:
                continue

            input_img = np.expand_dims(hand_img / 255.0, axis=0)
            pred = model.predict(input_img)
            class_idx = np.argmax(pred)
            prediction = label_map[class_idx]

            # Display prediction
            cv2.putText(frame, f'Prediction: {prediction}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Instructions
    cv2.putText(frame, "[S]peak  [C]lear  [Q]uit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Display typed text
    if prediction and prediction not in ['nothing']:
        if prediction == 'space':
            text_buffer += ' '
        elif prediction == 'delete':
            text_buffer = text_buffer[:-1]
        else:
            text_buffer += prediction
        prediction = ""  # prevent duplicates

    cv2.putText(frame, f'Text: {text_buffer}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Speak the sentence
        engine.say(text_buffer)
        engine.runAndWait()
    elif key == ord('c'):  # Clear text
        text_buffer = ""
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
