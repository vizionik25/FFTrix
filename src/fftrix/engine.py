import time
import threading
from vidgear.gears import VideoGear, WriteGear
import cv2
import os
import numpy as np
from .database import RECORDINGS_DIR

class CameraNode:
    def __init__(self, cam_id, source, name, analytics_pipeline, db, record_247=False):
        self.id = cam_id
        self.source = source
        self.name = name
        self.stream = None
        self.writer = None
        self.is_running = False
        self.is_recording = False # Manual/Trigger recording
        self.record_247 = record_247 # Continuous recording
        self.analytics = analytics_pipeline
        self.db = db
        self.processed_frame = np.zeros((480, 640, 3), np.uint8)
        self.thread = None
        self.fps = 0
        self.trigger_active = False
        self.last_trigger_time = 0
        
    def start(self):
        if self.is_running: return
        options = {"THREADED_QUEUE_MODE": True}
        if isinstance(self.source, str) and self.source.startswith(('rtsp', 'http')):
            options["rtsp_transport"] = "tcp"
            
        try:
            processed_src = int(self.source) if str(self.source).isdigit() else self.source
            self.stream = VideoGear(source=processed_src, logging=False, **options).start()
            self.is_running = True
            
            # Start 24/7 writer if enabled
            if self.record_247:
                self._start_writer(prefix="cont")
                
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            self.db.log_event("SYSTEM", self.name, "Stream connected")
        except Exception as e:
            self.db.log_event("ERROR", self.name, f"Startup failed: {e}")

    def _start_writer(self, prefix="rec"):
        output_dir = RECORDINGS_DIR / self.id
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = str(output_dir / f"{prefix}_{int(time.time())}.mp4")
        self.writer = WriteGear(output=filename, logging=False, **{"-vcodec": "libx264", "-crf": "28"})
        self.is_recording = True

    def _update(self):
        while self.is_running:
            if not self.stream: break
            frame = self.stream.read()
            if frame is None: continue
            
            # AI Processing
            try:
                proc, triggered = self.analytics.process(frame, self.id)
                self.processed_frame = proc
                
                # Flag Logic (Intelligent Cooldown)
                if triggered:
                    now = time.time()
                    if now - self.last_trigger_time > 10: # Only flag every 10 seconds per camera
                        self.db.log_event("ALERT", self.name, "Trigger Event Flagged", is_flag=1)
                        self.last_trigger_time = now
                
                # Persistence Logic
                if self.is_recording and self.writer:
                    self.writer.write(self.processed_frame)
                    
            except Exception as e:
                 print(f"Node {self.id} Error: {e}")
            time.sleep(0.01)

    def stop(self):
        self.is_running = False
        if self.writer:
            self.writer.close()
            self.writer = None
        if self.stream:
            self.stream.stop()
