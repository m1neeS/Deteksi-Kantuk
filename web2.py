import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import pygame
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import tempfile

# Initialize Pygame and the mixer
pygame.init()
pygame.mixer.init()

# Load the alarm sound
alarm_sound = pygame.mixer.Sound('test1.mp3')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Haarcascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Parameters for drowsiness detection
drowsy_threshold = 0.5  # Threshold for eye closure prediction
drowsy_counter = 0  # Counter for consecutive frames with closed eyes
drowsy_frames_threshold = 7  # Number of consecutive frames to trigger drowsiness detection

# Queue to store previous predictions
predictions_queue = deque(maxlen=drowsy_frames_threshold)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Function to preprocess eye image
def preprocess_eye(eye):
    eye_resized = cv2.resize(eye, (150, 150))
    eye_normalized = eye_resized / 255.0
    eye_reshaped = np.expand_dims(eye_normalized, axis=0).astype(np.float32)
    return eye_reshaped

def is_mouth_open(landmarks, threshold=0.05):
    # Get the coordinates of specific mouth landmarks
    upper_lip = landmarks[13].y
    lower_lip = landmarks[14].y
    distance = lower_lip - upper_lip
    return distance > threshold

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.drowsy_counter = 0
        self.predictions_queue = deque(maxlen=drowsy_frames_threshold)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Process face landmarks
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        mouth_open = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mouth_open = is_mouth_open(face_landmarks.landmark)
        
        eyes_closed = len(eyes) == 0
        if eyes_closed:
            self.drowsy_counter += 1
            if self.drowsy_counter >= drowsy_frames_threshold:
                cv2.putText(img, "Drowsy Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if not pygame.mixer.get_busy():  # Check if the alarm is not already playing
                    alarm_sound.play()
        else:
            self.drowsy_counter = 0
            for (x, y, w, h) in eyes:
                eye = img[y:y+h, x:x+w]
                eye_preprocessed = preprocess_eye(eye)
                
                interpreter.set_tensor(input_details[0]['index'], eye_preprocessed)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
                self.predictions_queue.append(prediction)
                
                # Calculate the average prediction
                average_prediction = np.mean(self.predictions_queue) if len(self.predictions_queue) > 0 else 0
                
                # Display open/closed eye status and accuracy
                if average_prediction >= drowsy_threshold:
                    status_text = f"Open Eyes: {average_prediction * 100:.2f}%"
                    cv2.putText(img, status_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    status_text = f"Closed Eyes: {(1 - average_prediction) * 100:.2f}%"
                    cv2.putText(img, status_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display mouth open/closed status
        if mouth_open:
            cv2.putText(img, "Mouth Open", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "Mouth Closed", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Check if average prediction indicates drowsiness
        if len(self.predictions_queue) == drowsy_frames_threshold:
            average_prediction = np.mean(self.predictions_queue)
            if average_prediction < drowsy_threshold and eyes_closed and not mouth_open:
                drowsiness_accuracy = (1 - average_prediction) * 100
                cv2.putText(img, f"Drowsy Detected! ({drowsiness_accuracy:.2f}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if not pygame.mixer.get_busy():  # Check if the alarm is not already playing
                    alarm_sound.play()
            elif not eyes_closed and mouth_open:
                yawn_accuracy = average_prediction * 100
                cv2.putText(img, f"Yawn Detected! ({yawn_accuracy:.2f}%)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif not eyes_closed and not mouth_open:
                not_drowsy_accuracy = average_prediction * 100
                cv2.putText(img, f"Not Drowsy! ({not_drowsy_accuracy:.2f}%)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw rectangle around eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw rectangle around face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return img

# Add CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f0f0;
        }
        .title {
            color: #2c3e50;
            font-size: 2em;
            font-weight: bold;
            text-align: center;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #2c3e50;
            color: white;
            text-align: center;
            padding: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Add header
st.markdown('<div class="title">Real-Time Drowsiness Detection</div>', unsafe_allow_html=True)

# Add a sidebar for file uploads
st.sidebar.title("Upload Options")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4"])

# Handle image upload
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    if uploaded_file.type.startswith("image"):
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        st.image(img, caption='Uploaded Image (HSV)', use_column_width=True)

        # Process the uploaded image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in eyes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        st.image(img, caption='Processed Image (HSV)', use_column_width=True)
    elif uploaded_file.type == "video/mp4":
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Display the uploaded video
        st.video(tfile.name)

# Video stream with drowsiness detection
webrtc_streamer(key="drowsiness-detection", video_transformer_factory=VideoTransformer)

# Add a footer
st.markdown('<div class="footer">MUHAMAD SURHES ANGGRHESTA</div>', unsafe_allow_html=True)
