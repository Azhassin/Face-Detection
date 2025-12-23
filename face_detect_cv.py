import os

# 1. Suppress TensorFlow Info/Warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2. Now import the heavy libraries
import cv2
from deepface import DeepFace


# Load the Haar Cascade for face detection (Fast)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Emotion Color Map (BGR format for OpenCV)
emotion_colors = {
    'angry': (0, 0, 255),       # Red
    'disgust': (0, 100, 0),     # Dark Green
    'fear': (128, 0, 128),      # Purple
    'happy': (0, 255, 0),       # Bright Green
    'sad': (255, 0, 0),         # Blue
    'surprise': (0, 255, 255),  # Yellow
    'neutral': (255, 255, 255)  # White
}

cap = cv2.VideoCapture(0)

print("System Online. Initializing Neural Analysis...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Detect Faces (Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 2. Extract the Face ROI (Region of Interest)
        face_roi = frame[y:y+h, x:x+w]

        # 3. Analyze Attributes using DeepFace
        # We wrap this in a try-block because DeepFace can be picky about image quality
        try:
            # enforce_detection=False because we already detected the face with OpenCV
            results = DeepFace.analyze(face_roi, actions=['age', 'gender', 'emotion'], enforce_detection=False, silent=True)
            
            # DeepFace returns a list of results, we take the first one
            result = results[0]
            
            # Extract Data
            age = result['age']
            gender = result['dominant_gender']
            emotion = result['dominant_emotion']
            
            # Determine Color based on Emotion
            box_color = emotion_colors.get(emotion, (255, 255, 255))
            
            # Create Label Text
            label = f"{gender}, {age}, {emotion}"

        except Exception as e:
            # Fallback if analysis fails
            box_color = (200, 200, 200)
            label = "Analyzing..."

        # 4. Draw Dynamic Box and Text
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    cv2.imshow('JARVIS - Advanced Face Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()