# FFTrix Enterprise Security & Streaming Server

FFTrix is a professional-grade, multi-threaded NVR (Network Video Recorder), AI Surveillance, and Streaming Production server built entirely in Python. It combines high-performance video handling with real-time AI analytics, auto-discovery, PTZ control, and a modern, secure web-based Command Center.

## 🚀 Key Features

- **Multi-Threaded Node Architecture:** Every camera stream runs in its own isolated background thread for maximum stability and low latency.
- **AI Analytics Suite:** Real-time Motion Intrusion, Facial Detection, Human Personnel tracking, and **License Plate Recognition (LPR)**.
- **Auto-Discovery & PTZ:** WS-Discovery for automatic ONVIF camera configuration and zero-friction Pan/Tilt/Zoom continuous control.
- **Smart Zoning & Alerts:** Draw custom security boundaries directly on the video feed to trigger **Email and Webhook** notifications on high-value intrusions.
- **Enterprise DVR:** Automated 24/7 continuous recording with intelligent event flagging, **time-ranged Clip Exports**, and **Auto-Retention cleanup policies**.
- **Professional Watermarking:** Customizable static or dynamic (floating) alpha-blended text and image overlays for branding and anti-theft.
- **Secure Remote Access:** Built-in Zero-Trust tunneling (via Tailscale Funnel & Angie) and session-based authentication for worldwide secure monitoring.
- **Dynamic Surveillance Matrix:** Scalable 1x1 to 4x4 hardware-accelerated monitoring grid.

## 🛠 Tech Stack

- **UI Framework:** [NiceGUI](https://nicegui.io/) (Pure Python Web UI)
- **Streaming Engine:** [VidGear](https://abhitronix.github.io/vidgear/) (High-performance FFmpeg wrapper)
- **Computer Vision:** [OpenCV](https://opencv.org/) and **Tesseract OCR** (for LPR)
- **Device Management:** ONVIF protocol over SOAP
- **Database:** SQLite3
- **Automation:** [uv](https://github.com/astral-sh/uv) (Package & Environment Management)

## 🚦 Quick Start

### Prerequisites
- Python 3.14+
- FFmpeg installed on your system path.
- Tesseract OCR (Optional, for OCR mode).

### Installation

**Using uv (Recommended):**
```bash
uv add fftrix
```

**Using pip:**
```bash
pip install fftrix
```

### Running the Server
```bash
# Launch the Dashboard (Local Network)
fftrix serve

# Launch with Secure Remote Tunnel (Public Internet)
fftrix serve --remote

# Force CLI Mode (No UI)
fftrix serve --cli-mode --mode motion --source 0
```

**Default Credentials:** 
- **Username:** `admin`
- **Password:** `admin`
- *Change these immediately using:* `fftrix user add admin`

## 🛠 Development

To install for development:
```bash
git clone https://github.com/your-username/fftrix.git
cd fftrix
uv sync
```

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- [System Architecture](docs/ARCHITECTURE.md)
- [AI Analytics & Zoning](docs/ANALYTICS.md)
- [Watermarking Engine](docs/WATERMARKING.md)
- [Storage & DVR](docs/STORAGE.md)
- [Security & Remote Access](docs/REMOTE_ACCESS.md)

## 📄 License
MIT License - see [LICENSE](LICENSE) for details.
