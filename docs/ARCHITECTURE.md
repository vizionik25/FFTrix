# System Architecture

FFTrix is designed using a modular, multi-threaded architecture to ensure high-performance video processing and stability.

## 🏗 Component Overview

### 1. The Core Engine (`src/fftrix/engine.py`)
The engine is responsible for stream lifecycle management. 
- **CameraNode:** A specialized class that encapsulates a single video source. Each node runs in a dedicated Python `threading.Thread`.
- **Device Management:** Integration with ONVIF Auto-Discovery (`discovery.py`) for automatic camera provisioning, and PTZ controllers (`ptz.py`) for Pan/Tilt/Zoom continuous movement.
- **Concurrency:** By isolating cameras into separate threads, FFTrix prevents a single lagging RTSP stream from blocking the global application state or the User Interface.
- **Resilience:** Automatic error handling for signal loss, providing a "NO SIGNAL" placeholder until the stream recovers.

### 2. Analytics Pipeline (`src/fftrix/analytics.py` & `src/fftrix/lpr.py`)
A decoupled AI layer that processes frames independently for each camera.
- **Pipelines:** Supports Motion Detection, Facial Detection, Person Detection, and License Plate Recognition (LPR).
- **OSD (On-Screen Display):** Renders timestamps, bounding boxes, and security zones directly onto the frame before it is sent to the UI or DVR.

### 3. Command Center UI (`src/fftrix/dashboard.py`)
Built on **NiceGUI**, the dashboard provides a modern, reactive web interface.
- **Asynchronous Updates:** Uses a high-speed timer loop to fetch processed frames from background threads and stream them to the browser via WebSockets as base64-encoded JPEGs.
- **State Management:** Dynamically reconfigures nodes (zoning, modes, watermarking) in real-time without restarting streams.

### 4. Persistence Layer (`src/fftrix/database.py`)
A local **SQLite3** database ensures that the system is production-ready.
- **Configuration:** Stores camera URLs, names, security zones, and watermark settings.
- **Logging:** A permanent audit log of all system events and AI-triggered incidents.

### 5. Notification & Retention Services (`src/fftrix/alerts.py`, `src/fftrix/retention.py`, `src/fftrix/clipper.py`)
Background daemon processes and services that handle system-wide tasks.
- **Alert Dispatcher:** Monitors the event log and fires Email (SMTP) and Webhook (Slack/Discord) notifications based on customizable cooldown periods.
- **Retention Manager:** A nightly thread that automatically cleans up old DVR recordings based on per-camera retention policies.
- **Clip Exporter:** A utility to stitch and extract specific time-ranged MP4 clips from the raw DVR footage.

## 🔄 Data Flow

1. **Source:** `VidGear` connects to RTSP/HTTP/Local video (often auto-discovered via WS-Discovery).
2. **Capture:** `CameraNode` thread reads raw frames.
3. **Analyze:** `AnalyticsPipeline` applies AI filters (Motion, Face, Person, LPR) and checks for zone intrusions.
4. **Trigger:** Intrusions trigger database events, which may fire Email/Webhook alerts.
5. **DVR:** `WriteGear` encodes the processed frame to disk if recording is active. `ClipExporter` can later extract time-ranged clips.
6. **Display:** The UI loop fetches the latest frame and pushes it to the browser.
