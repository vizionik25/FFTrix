# AI Analytics & Zoning

FFTrix features an enterprise-grade analytics engine that combines traditional computer vision with modern AI detection techniques.

## 🤖 Analytics Modes

### 1. Motion Intrusion
Uses a high-performance **MOG2 (Mixture of Gaussians)** background subtractor. 
- **Surveillance Focus:** Optimized for static backgrounds (security cameras).
- **Feedback:** Moving objects are highlighted with a green alpha-overlay.
- **Trigger:** Activates when the non-zero pixel count exceeds a specific threshold within a monitored zone.

### 2. Personnel Detection
Utilizes **HOG (Histogram of Oriented Gradients)** with a pre-trained SVM people detector.
- **Precision:** Specifically tuned to recognize the human form.
- **Feedback:** Displays a red "PERSON" bounding box around detected subjects.

### 3. Facial Detection
Implements multi-scale cascade classifiers for rapid face localization.
- **Feedback:** Displays a cyan "FACE DETECTED" bounding box.

### 4. License Plate Recognition (LPR)
Uses a two-stage pipeline with OpenCV morphology for plate localization and Tesseract OCR for text extraction.
- **Precision:** Filters candidates by contour area and aspect ratio before running OCR on the ROI.
- **Feedback:** Displays a green bounding box featuring the OCR text and confidence percentage.

---

## 📐 Smart Zoning

Smart Zoning allows you to define specific areas of interest (AOIs) within a video feed.

### How it works:
1. **Isolated Logic:** AI triggers (Motion, Person, Face) are only evaluated if the detection occurs *inside* a defined zone.
2. **Visual Feedback:** All active zones are rendered as cyan rectangles on the video feed.
3. **Multiple Zones:** A single camera can have multiple independent security zones.

### Configuring Zones in the UI:
1. Select a camera in the Grid (it will highlight in blue).
2. In the Sidebar, click the **"Edit Location" (Crop icon)**.
3. Click the **top-left** corner of your desired zone on the video feed.
4. Click the **bottom-right** corner to complete the zone.
5. The zone is instantly saved to the database and deployed to the analytics engine.

---

## 🚩 Event Flagging

When a trigger is tripped *inside* a zone:
1. The system creates a **High-Priority Flag** in the database.
2. The event is instantly pushed to the **Incident Timeline** in the dashboard.
3. If configured, the **Alert Dispatcher** (`alerts.py`) sends immediate Email and/or Webhook notifications (e.g., Slack, Discord) respecting a customizable per-camera cooldown period.
4. If 24/7 recording is active, these flags serve as index points for forensic review.
