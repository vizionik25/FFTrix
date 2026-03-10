# Storage & DVR

FFTrix uses a dual-layer storage strategy to manage system metadata and high-volume video recordings.

## 🗄 1. System Database (`fftrix_system.db`)
A local SQLite3 database acts as the system's brain. It stores persistent data that remains available across application restarts.

### Schema Overview:
- **`cameras`:** Stores stream URLs, AI modes, security zone coordinates, and watermark settings.
- **`events`:** A permanent audit log of all system actions and AI-triggered alerts.
- **`users`:** Securely stores hashed credentials for remote access.

---

## 🎥 2. Digital Video Recorder (DVR)
FFTrix features a high-performance DVR subsystem powered by `WriteGear`.

### Recording Modes:
1. **24/7 Continuous Recording:**
   - When enabled, the system continuously writes the processed video stream to disk.
   - Files are stored in the `recordings/[camera_id]/` directory.
   - Filenames are timestamped for easy chronological retrieval.
2. **Manual Recording:**
   - Can be triggered per-camera directly from the UI.
3. **Trigger-Based Flagging:**
   - While in 24/7 mode, the system does not need to start/stop recording for alerts. Instead, it creates a **Flag** in the database.
   - These flags point to specific timestamps in your continuous footage, allowing you to find incidents instantly.

### Encoding Details:
- **Codec:** H.264 (via `libx264`) for maximum compatibility.
- **Optimization:** Encoding is handled in background threads to ensure the live monitoring grid remains fluid.
- **Location:** All footage is stored in the project's `recordings/` directory by default.

---

## 📂 File Structure
```text
/home/nik/fftrix/
├── fftrix_system.db       # System configuration & event log
└── recordings/
    ├── C01/               # Footage from Camera 01
    │   ├── cont_171000.mp4
    │   └── cont_172000.mp4
    └── C02/               # Footage from Camera 02
        └── rec_171500.mp4
```

---

## 🧹 Maintenance
- **Manual Deletion:** Old footage can be managed via the standard OS file explorer.
- **Database Backups:** The `fftrix_system.db` file can be backed up simply by copying it while the server is offline.
