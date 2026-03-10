import cv2
import numpy as np
import os

def detect_edges(frame):
    """Perform Canny Edge Detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def detect_motion(frame, background_subtractor):
    """Detect motion using background subtraction."""
    fg_mask = background_subtractor.apply(frame)
    # Perform some noise reduction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

def detect_faces(frame, face_cascade):
    """Detect faces using Haar Cascades."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

def detect_objects_dnn(frame, net, classes):
    """Detect objects using a DNN model (e.g., SSD MobileNet)."""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{classes[idx]}: {confidence*100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

from .face_recognition import FaceRecognizer
from .ocr import detect_text_areas
from .segmentation import remove_background_mog2, chroma_key, replace_background

def run_vision_loop(source=0, mode='edge'):
    """
    Run a video processing loop.
    Modes: 'edge', 'motion', 'face', 'recognize', 'object', 'ocr', 'rembg', 'chroma'
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Initialize tools based on mode
    back_sub = cv2.createBackgroundSubtractorMOG2() if mode in ['motion', 'rembg'] else None
    
    # For chroma key, we might want a replacement background
    replacement_bg = None
    if mode == 'chroma':
        # Create a simple blue background if none provided
        replacement_bg = np.zeros((480, 640, 3), np.uint8)
        replacement_bg[:] = (255, 0, 0) # Blue
        # Green screen range (HSV)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

    face_cascade = None
    if mode == 'face':
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)

    recognizer = None
    if mode == 'recognize':
        recognizer = FaceRecognizer()
        if os.path.exists('training_data'):
            recognizer.train('training_data')
        else:
            print("Warning: No 'training_data' directory found.")

    print(f"Running in {mode} mode. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = frame
        if mode == 'edge':
            processed = detect_edges(frame)
        elif mode == 'motion':
            processed = detect_motion(frame, back_sub)
        elif mode == 'face':
            processed = detect_faces(frame, face_cascade)
        elif mode == 'recognize':
            processed = recognizer.recognize(frame)
        elif mode == 'ocr':
            processed = detect_text_areas(frame)
        elif mode == 'rembg':
            processed, _ = remove_background_mog2(frame, back_sub)
        elif mode == 'chroma':
            processed = chroma_key(frame, lower_green, upper_green, replacement_bg)

        cv2.imshow('FFTrix Vision', processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
