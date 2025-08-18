import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load your trained model
model = load_model('age_classifier_model_2.h5')

# Your age group labels
labels = ['Child', 'Teen', 'Young Adult', 'Middle-aged', 'Senior']

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define target input size for your model
img_size = (128, 128)  # Change this if your model expects a different size

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting webcam... press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop and preprocess the face
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, img_size)
        face_array = np.expand_dims(face, axis=0)
        face_array = preprocess_input(face_array)  # Required for MobileNetV2

        # Predict age group
        prediction = model.predict(face_array)
        predicted_label = labels[np.argmax(prediction)]

        # Show prediction
        cv2.putText(frame, predicted_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show the frame
    cv2.imshow('Age Group Classification', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
