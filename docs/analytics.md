# Analytics Engine

The FFTrix Analytics Engine is the "Source of Truth" for how video data is processed, analyzed, and transformed into actionable events. It is built strictly on **Classical Computer Vision (CV)** principles, ensuring low-latency processing without the overhead of deep learning frameworks.

## 🧠 Core Processing Pipeline

Every camera stream in FFTrix is managed by a `CameraNode`. The analytics occur in an iterative loop:

1.  **Ingestion**: VidGear fetches frames from the RTSP source.
2.  **Analysis**: The frame is passed to the `AnalyticsPipeline`.
3.  **Result**: The pipeline returns a processed frame (with overlays) and a boolean `triggered` flag.
4.  **Action**: If triggered and "Armed," the system logs an event, saves a snapshot, and fires notifications.

---

## 🛠️ Detection Algorithms

FFTrix uses specialized algorithms for different monitoring scenarios:

### 1. Motion Detection (`MOG2`)
- **Algorithm**: Mixture of Gaussians (MOG2) Background Subtraction.
- **How it works**: It builds a statistical model of the background. Any significant group of pixels that deviates from this model is identified as a foreground object (contour).
- **Filtering**: Small contours (noise) are ignored. Only contours exceeding a specific area threshold are flagged.

### 2. Pedestrian Detection (`HOG + SVM`)
- **Algorithm**: Histogram of Oriented Gradients (HOG) combined with a Linear Support Vector Machine (SVM).
- **How it works**: It analyzes the distribution of intensity gradients (edges) in the frame. The SVM is pre-trained on human shapes and patterns.

### 3. Face Detection (`Haar Cascades`)
- **Algorithm**: Haar Feature-based Cascade Classifiers.
- **How it works**: A machine-learning based approach where a cascade function is trained from a lot of positive (faces) and negative (non-faces) images. It uses simple "Haar features" (like the intensity difference between the eye region and the bridge of the nose) to find face-like structures.

### 4. License Plate Recognition (`LPR`)
- **Step 1: Localization**: Uses morphological operations (rect kernels, dilation, and contour ratio filtering) to find plate-shaped rectangles.
- **Step 2: Recognition**: Crops the candidate region and uses **Tesseract OCR** to extract the alphanumeric characters.

---

## 🛡️ Privacy and Safety

### Privacy Zones (Gaussian Blur)
FFTrix allows you to define "Privacy Zones" on any camera. These are processed using a **Gaussian Blur** directly on the raw frame.
- **Critical Security**: The blur is applied *before* the frame is written to disk or sent to the dashboard. The original un-blurred data is never stored.

### Inclusion/Exclusion Zones
- **Logic**: You can draw rectangles to define where analytics should be active.
- **Filtering**: A trigger is only valid if the center-point of the detected object falls within a defined Inclusion Zone.

---

## 🏗️ Storage and Logging

- **Event Logs**: Stored in the SQLite `events` table with a timestamp, source ID, and detection type.
- **Snapshots**: JPEG images are saved at `85%` quality to the `snapshots/` directory whenever an armed trigger occurs.
- **Watermarking**: Real-time text and image overlays (transparent PNGs) are burned into the frame during processing.
