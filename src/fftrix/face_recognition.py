import cv2
import os
import numpy as np

class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        self.labels_map = {}

    def train(self, data_path):
        """
        Train the recognizer with images from data_path.
        data_path should have subdirectories for each person.
        """
        faces = []
        labels = []
        label_id = 0
        
        for root, dirs, files in os.walk(data_path):
            for directory in dirs:
                person_path = os.path.join(root, directory)
                self.labels_map[label_id] = directory
                for file in os.listdir(person_path):
                    if file.endswith(('jpg', 'jpeg', 'png')):
                        path = os.path.join(person_path, file)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if img is None: continue
                        
                        # Detect face in the training image
                        detected_faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
                        for (x, y, w, h) in detected_faces:
                            faces.append(img[y:y+h, x:x+w])
                            labels.append(label_id)
                label_id += 1
        
        if faces:
            self.recognizer.train(faces, np.array(labels))
            print(f"Trained on {len(faces)} faces for {label_id} people.")
        else:
            print("No training data found.")

    def recognize(self, frame):
        """Recognize faces in a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in detected_faces:
            roi_gray = gray[y:y+h, x:x+w]
            label_id, confidence = self.recognizer.predict(roi_gray)
            
            # Lower confidence score means better match for LBPH
            name = self.labels_map.get(label_id, "Unknown") if confidence < 100 else "Unknown"
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame
