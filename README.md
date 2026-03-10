# FFTrix Enterprise Security & Streaming Server

FFTrix is a professional-grade, multi-threaded NVR (Network Video Recorder), AI Surveillance, and Streaming Production server built entirely in Python. It combines high-performance video handling with real-time AI analytics and a modern, secure web-based Command Center.

## 🚀 Key Features

- **Multi-Threaded Node Architecture:** Every camera stream runs in its own isolated background thread for maximum stability and low latency.
- **AI Analytics Suite:** Real-time Motion Intrusion, Facial Detection, and Human Personnel tracking.
- **Smart Zoning:** Draw custom security boundaries directly on the video feed to isolate monitoring to specific high-value areas.
- **Enterprise DVR:** Automated 24/7 continuous recording with intelligent event flagging for rapid incident retrieval.
- **Professional Watermarking:** Customizable static or dynamic (floating) alpha-blended text and image overlays for branding and anti-theft.
- **Secure Remote Access:** Built-in Zero-Trust tunneling (via pyngrok) and session-based authentication for worldwide secure monitoring.
- **Dynamic Surveillance Matrix:** Scalable 1x1 to 4x4 hardware-accelerated monitoring grid.

## 🛠 Tech Stack

- **UI Framework:** [NiceGUI](https://nicegui.io/) (Pure Python Web UI)
- **Streaming Engine:** [VidGear](https://abhitronix.github.io/vidgear/) (High-performance FFmpeg wrapper)
- **Computer Vision:** [OpenCV](https://opencv.org/)
- **Database:** SQLite3
- **Automation:** [uv](https://github.com/astral-sh/uv) (Package & Environment Management)

## 🚦 Quick Start

### Prerequisites
- Python 3.14+
- FFmpeg installed on your system path.
- Tesseract OCR (Optional, for OCR mode).

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/fftrix.git
cd fftrix

# Install dependencies using uv
uv sync
```

### Running the Server
```bash
# Launch the Dashboard (Local Network)
uv run fftrix

# Launch with Secure Remote Tunnel (Public Internet)
uv run fftrix --remote

# Force CLI Mode (No UI)
uv run fftrix --cli --mode motion --source 0
```

**Default Credentials:** 
- **Username:** `admin`
- **Password:** `admin`

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- [System Architecture](docs/ARCHITECTURE.md)
- [AI Analytics & Zoning](docs/ANALYTICS.md)
- [Watermarking Engine](docs/WATERMARKING.md)
- [Storage & DVR](docs/STORAGE.md)
- [Security & Remote Access](docs/REMOTE_ACCESS.md)

## 📄 License
MIT License - see [LICENSE](LICENSE) for details.
