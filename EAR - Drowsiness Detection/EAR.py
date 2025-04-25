import cv2
import torch
from ultralytics import YOLO
import numpy as np
from mediapipe import solutions
import time
import winsound  # For sound alert

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_landmarks, h, w):
    p1 = np.array([eye_landmarks[1].x * w, eye_landmarks[1].y * h])
    p2 = np.array([eye_landmarks[5].x * w, eye_landmarks[5].y * h])
    p3 = np.array([eye_landmarks[2].x * w, eye_landmarks[2].y * h])
    p4 = np.array([eye_landmarks[4].x * w, eye_landmarks[4].y * h])
    p5 = np.array([eye_landmarks[0].x * w, eye_landmarks[0].y * h])
    p6 = np.array([eye_landmarks[3].x * w, eye_landmarks[3].y * h])

    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# EAR threshold and time duration for drowsiness detection
EAR_THRESHOLD = 1.50
DURATION_THRESHOLD = 5  # in seconds

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

try:
    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Error opening webcam")

    # Initialize MediaPipe Face Mesh
    face_mesh = solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    start_time = None  # Track the time EAR crosses the threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Perform face detection using the YOLOv8 model
        results = model.predict(frame, classes=0)  # class 0 is person

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mp = face_mesh.process(rgb_frame)

        # Annotate the frame with bounding boxes
        annotated_frame = results[0].plot()

        # Draw facial landmarks for eyes only from MediaPipe and calculate EAR
        ear = None
        if results_mp.multi_face_landmarks:
            for face_landmarks in results_mp.multi_face_landmarks:
                h, w, _ = frame.shape

                # Left eye landmarks (indices for EAR calculation)
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                left_eye_landmarks = [face_landmarks.landmark[idx] for idx in left_eye_indices]
                left_ear = calculate_ear(left_eye_landmarks, h, w)

                # Right eye landmarks (indices for EAR calculation)
                right_eye_indices = [362, 385, 387, 263, 373, 374]
                right_eye_landmarks = [face_landmarks.landmark[idx] for idx in right_eye_indices]
                right_ear = calculate_ear(right_eye_landmarks, h, w)

                # Average EAR
                ear = (left_ear + right_ear) / 2.0

                # Draw eyes landmarks
                for idx in left_eye_indices + right_eye_indices:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)

        # Display EAR value on the frame
        if ear is not None:
            cv2.putText(annotated_frame, f"EAR: {ear:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Check for drowsiness
            if ear > EAR_THRESHOLD:
                if start_time is None:  # Start timer if threshold is exceeded
                    start_time = time.time()
                elif time.time() - start_time > DURATION_THRESHOLD:  # Alert if time exceeds threshold
                    cv2.putText(annotated_frame, "DROWSINESS ALERT!", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms
            else:
                start_time = None  # Reset timer if EAR drops below the threshold

        # Display the resulting frame
        cv2.imshow('Face and Eye Detection', annotated_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

except Exception as err:
    print(str(err))
