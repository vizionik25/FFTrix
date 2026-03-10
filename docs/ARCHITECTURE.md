# System Architecture

FFTrix is designed using a modular, multi-threaded architecture to ensure high-performance video processing and stability.

## 🏗 Component Overview

### 1. The Core Engine (`src/fftrix/engine.py`)
The engine is responsible for stream lifecycle management. 
- **CameraNode:** A specialized class that encapsulates a single video source. Each node runs in a dedicated Python `threading.Thread`.
- **Concurrency:** By isolating cameras into separate threads, FFTrix prevents a single lagging RTSP stream from blocking the global application state or the User Interface.
- **Resilience:** Automatic error handling for signal loss, providing a "NO SIGNAL" placeholder until the stream recovers.

### 2. Analytics Pipeline (`src/fftrix/analytics.py`)
A decoupled AI layer that processes frames independently for each camera.
- **Pipelines:** Supports Motion Detection, Facial Detection, and Person Detection.
- **OSD (On-Screen Display):** Renders timestamps, bounding boxes, and security zones directly onto the frame before it is sent to the UI or DVR.

### 3. Command Center UI (`src/fftrix/dashboard.py`)
Built on **NiceGUI**, the dashboard provides a modern, reactive web interface.
- **Asynchronous Updates:** Uses a high-speed timer loop to fetch processed frames from background threads and stream them to the browser via WebSockets as base64-encoded JPEGs.
- **State Management:** Dynamically reconfigures nodes (zoning, modes, watermarking) in real-time without restarting streams.

### 4. Persistence Layer (`src/fftrix/database.py`)
A local **SQLite3** database ensures that the system is production-ready.
- **Configuration:** Stores camera URLs, names, security zones, and watermark settings.
- **Logging:** A permanent audit log of all system events and AI-triggered incidents.

## 🔄 Data Flow

1. **Source:** `VidGear` connects to RTSP/HTTP/Local video.
2. **Capture:** `CameraNode` thread reads raw frames.
3. **Analyze:** `AnalyticsPipeline` applies AI filters and checks for zone intrusions.
4. **DVR:** `WriteGear` encodes the processed frame to disk if recording is active.
5. **Display:** The UI loop fetches the latest frame and pushes it to the browser.
