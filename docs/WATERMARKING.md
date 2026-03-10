# Professional Watermarking Engine

FFTrix includes a versatile watermarking subsystem designed for broadcast branding, anti-theft protection, and forensic verification.

## 🛠 Features

### 1. Alpha-Blended Text
Overlay customizable text (e.g., "Property of FFTrix") onto any stream. 
- **Real-time:** Text updates are applied instantly to the video feed.
- **Transparency:** Adjustable opacity (0.1 to 1.0) using the UI slider.

### 2. Image/Logo Overlays
Load PNG or JPEG files directly as watermarks.
- **Alpha Channel Support:** Transparent PNGs are correctly blended using the image's native alpha channel combined with the global transparency setting.
- **Path-Based:** Simply provide the local path to the image file in the configuration panel.

### 3. Dynamic Floating Mode
Prevents "screen scraping" and burn-in by moving the watermark along a calculated bouncing path.
- **Bouncing Logic:** The watermark moves across the viewport and reverses direction upon hitting the boundaries.
- **Anti-Theft:** Makes it difficult for unauthorized parties to remove the watermark using simple static masks.

---

## ⚙️ Configuration

To configure watermarking for a camera:

1. Select the target camera in the **Surveillance Grid**.
2. Locate the **Watermark Engine** section in the sidebar.
3. **Branding Text:** Type the text you want to appear.
4. **Logo Path:** Enter the full system path to your PNG logo.
5. **Opacity Slider:** Adjust the visibility of the watermark.
6. **Behavior:** Toggle between **Static (Manual)** and **Floating (Bouncing)**.

---

## 💡 Pro Tips

- **Forensic Use:** Use the watermarking engine to bake the camera's location or ID directly into the recorded footage for legal compliance.
- **Production Branding:** Combine a floating logo with a static "LIVE" text overlay for professional streaming aesthetics.
- **Low Latency:** The watermarking engine is hardware-optimized and adds negligible overhead to the processing pipeline.
