# FFTrix

FFTrix is a high-performance video surveillance and analysis platform utilizing classical Computer Vision (CV) and pattern recognition techniques for real-time monitoring and event detection.

## Features

- **Real-time Monitoring**: Multi-camera streaming with low-latency processing via `vidgear`.
- **Classical CV Analytics**:
  - **Motion Detection**: Background subtraction using the MOG2 (Mixture of Gaussians) algorithm.
  - **Person Detection**: Histogram of Oriented Gradients (HOG) with Linear SVM.
  - **Face Detection**: Local Binary Patterns (LBP) and Haar Feature-based Cascade Classifiers.
  - **LPR (License Plate Recognition)**: Morphology-based plate localization combined with Tesseract OCR.
  - **Zone Monitoring**: User-defined exclusion and inclusion zones for targeted event triggering.
- **24/7 Recording**: Continuous background recording with automatic retention management.
- **Interactive Dashboard**: Modern web-based interface built with NiceGUI for camera management, live viewing, and event playback.
- **Automated Alerts**: Event-driven notifications via Email (SMTP) and JSON Webhooks.
- **PTZ & Discovery**: ONVIF-compatible device discovery and Pan-Tilt-Zoom control.
- **Privacy Controls**: Dynamic privacy masking with Gaussian blurring.

## Architecture

FFTrix is built on a "Classic CV" architecture, prioritizing deterministic behavior and low resource overhead:
- **Vision Engine**: Pure OpenCV-based pipeline (no GPU/Deep Learning dependencies).
- **OCR Engine**: Tesseract (via `pytesseract`) for text and plate recognition.
- **Database**: SQLite for configuration, user management, and event logging.
- **GUI**: NiceGUI (FastAPI + Vue.js + Quasar) for a responsive management interface.

## Installation

FFTrix can be installed directly from PyPI:

```bash
pip install fftrix
```

Or using `uv` as a tool:

```bash
uv tool install fftrix
```

### Prerequisites

- **Python 3.14** or higher.
- **Tesseract OCR**: Required for LPR and OCR features. Install it via your system package manager (e.g., `apt install tesseract-ocr` or `brew install tesseract`).
- **Docker** (for secure deployment): ***NOTE: Although you can replicate the deployment process without Docker, it will not be supported by me or anyone else from my team.***
- **ngrok**: Can be implemented for remote access. While not officially supported, you can contact the developer for guidance.

## Quick Start

Launch the FFTrix surveillance server:

```bash
fftrix serve --ui 
```

The dashboard will be available at `http://localhost:8080`.

### Default Credentials
- **Username**: `admin`
- **Password**: `admin`
*(Note: A password change is required upon first login)*

## CLI Usage

FFTrix includes a Click-based CLI for server control and user management.

```bash
# Start headless server (no local browser launch)
fftrix serve --no-ui  # ***NOTE: The UI is REQUIRED for configuration of watermarking and zone configuration but once configured the server can be run headless***

# Provision a new operator
fftrix user add <username> --role [admin|viewer]

# Emergency reset (wipes all users, restores admin:admin)
fftrix user reset-admin
```

## Configuration

Configure the notification engine via environment variables:

| Variable | Description |
|----------|-------------|
| `FFTRIX_SMTP_HOST` | SMTP server address |
| `FFTRIX_SMTP_PORT` | SMTP port (e.g., 587) |
| `FFTRIX_SMTP_USER` | SMTP username |
| `FFTRIX_SMTP_PASSWORD`| SMTP password |
| `FFTRIX_ALERT_FROM` | Sender email address |

## Tech Stack

- **Framework**: [NiceGUI](https://nicegui.io/)
- **CLI**: [Click](https://click.palletsprojects.com/)
- **Core Vision**: [OpenCV](https://opencv.org/)
- **OCR**: [Tesseract](https://github.com/tesseract-ocr/tesseract)
- **Networking**: [VidGear](https://abelovic.github.io/vidgear/), [ONVIF-Zeep](https://github.com/Foscam-Fcl/onvif-zeep)

## Development

If you want to contribute to FFTrix start by forking the repository then checking the requirements listed in the CONTRIBUTING.md file.

```bash
# Clone the repository
git clone https://github.com/yourusername/fftrix.git
cd fftrix

# Sync dependencies using uv
uv sync
```

## License

MIT License
