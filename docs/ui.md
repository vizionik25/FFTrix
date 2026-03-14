# Dashboard & UI

FFTrix features a modern, web-based dashboard built with **NiceGUI**. This interface is used for managing cameras, defining zones, viewing live streams, and reviewing recorded events.

## 🔑 Authentication

### First-Use Dialog
On your first login with the default `admin:admin` credentials, FFTrix will force a "Security Setup" dialog. You **must** create a new username and password to proceed.

---

## 📸 Camera Management

### Adding a Camera
Navigate to the **Cameras** tab and click "Add Camera".
- **Camera ID**: A unique, URL-safe slug (e.g., `back-yard`).
- **Source**: The RTSP or HTTP stream URL.
- **Mode**: Choose the "Classic CV" analytics mode.

### Analytics Modes:
- `Motion`: Uses MOG2 background subtraction.
- `Object`: Pedestrian detection using HOG + SVM.
- `Face`: Uses Haar Cascades.
- `LPR`: License Plate Recognition.
- `Edge`: Visualizes gradients (diagnostic).
- `None`: Raw stream viewing only.

---

## 📐 Zone Management

Detection zones allow you to define regions of interest, preventing false alerts from background movement.

### Inactive/Inclusion Zones:
1. Open a camera's settings.
2. Use the **Zone Editor** to draw rectangular inclusion zones.
3. If zones are set, alerts will *only* trigger when activity is detected *within* those rectangles.

### Privacy Masking:
Privacy zones permanently blur specific regions of the frame *at the source* before they are recorded or viewed.

---

## 📽️ Playback & Evidence

The **Playback** tab allows you to browse historical footage.
- **Segment Library**: Recorded segments are stored in 15-minute blocks.
- **Clip Export**: Select a start and end time to generate a single, concatenated MP4 clip using `ClipExporter`.

---

## 🕹️ PTZ Control

For ONVIF-compatible cameras:
- **Movement**: 8-way directional controls (Up, Down, Left, Right, Diagonals).
- **Zoom**: Continuous zoom in/out.
- **Presets**: Access and trigger pre-configured camera positions.
