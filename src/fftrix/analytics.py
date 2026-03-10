import cv2
import numpy as np
import time
import os

class WatermarkEngine:
    def __init__(self):
        self.configs = {} # cam_id -> config dict
        self.cached_images = {} # path -> image
        self.float_offsets = {} # cam_id -> [x, y, dx, dy]

    def set_config(self, cam_id, config):
        self.configs[cam_id] = config
        # Initialize floating state: [x, y, velocity_x, velocity_y]
        if config.get('mode') == 'floating':
            self.float_offsets[cam_id] = [100, 100, 2, 2]

    def _draw_text(self, frame, text, pos, alpha):
        if not text: return frame
        overlay = frame.copy()
        cv2.putText(overlay, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def _draw_image(self, frame, img_path, pos, alpha):
        if not img_path or not os.path.exists(img_path): return frame
        
        if img_path not in self.cached_images:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None: return frame
            self.cached_images[img_path] = img
        
        watermark = self.cached_images[img_path]
        h_frame, w_frame = frame.shape[:2]
        h_wm, w_wm = watermark.shape[:2]
        x, y = pos
        
        # Clip coordinates
        x, y = max(0, min(x, w_frame - w_wm)), max(0, min(y, h_frame - h_wm))
        
        roi = frame[y:y+h_wm, x:x+w_wm]
        
        if watermark.shape[2] == 4: # PNG with Alpha
            wm_alpha = (watermark[:, :, 3] / 255.0) * alpha
            for c in range(3):
                roi[:, :, c] = (wm_alpha * watermark[:, :, c] + (1 - wm_alpha) * roi[:, :, c])
        else: # Simple blend
            cv2.addWeighted(watermark, alpha, roi, 1 - alpha, 0, roi)
            
        frame[y:y+h_wm, x:x+w_wm] = roi
        return frame

    def apply(self, frame, cam_id):
        config = self.configs.get(cam_id)
        if not config: return frame
        
        h, w = frame.shape[:2]
        alpha = config.get('transparency', 0.5)
        
        # Determine Position
        if config.get('mode') == 'floating':
            st = self.float_offsets.get(cam_id, [100, 100, 2, 2])
            st[0] += st[2]
            st[1] += st[3]
            # Bounce logic
            if st[0] <= 0 or st[0] >= w - 200: st[2] *= -1
            if st[1] <= 50 or st[1] >= h - 50: st[3] *= -1
            pos = (st[0], st[1])
            self.float_offsets[cam_id] = st
        else:
            pos = (config.get('x', 50), config.get('y', 50))

        # Apply Text
        frame = self._draw_text(frame, config.get('text', ''), pos, alpha)
        
        # Apply Image
        img_path = config.get('image_path')
        if img_path:
            frame = self._draw_image(frame, img_path, (pos[0], pos[1] + 40), alpha)
            
        return frame

class AnalyticsPipeline:
    def __init__(self):
        self.modes = {}
        self.back_subs = {}
        self.zones = {} 
        self.watermarker = WatermarkEngine()
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def set_config(self, cam_id, mode=None, zones=None, watermark=None):
        if mode: self.modes[cam_id] = mode
        if zones is not None: self.zones[cam_id] = zones
        if watermark is not None: self.watermarker.set_config(cam_id, watermark)
        
        if self.modes.get(cam_id) == 'motion' and cam_id not in self.back_subs:
            self.back_subs[cam_id] = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    def is_in_zone(self, cam_id, x, y, w, h):
        cam_zones = self.zones.get(cam_id, [])
        if not cam_zones: return True 
        cx, cy = x + w//2, y + h//2
        for (zx, zy, zw, zh) in cam_zones:
            if zx <= cx <= zx + zw and zy <= cy <= zy + zh: return True
        return False

    def process(self, frame, cam_id):
        mode = self.modes.get(cam_id, 'none')
        trigger_tripped = False
        processed = frame.copy()
        h, w = processed.shape[:2]
        
        # 1. AI Processing
        if mode == 'motion':
            fg_mask = self.back_subs[cam_id].apply(frame)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 1000: continue
                (x, y, w_det, h_det) = cv2.boundingRect(cnt)
                if self.is_in_zone(cam_id, x, y, w_det, h_det):
                    trigger_tripped = True
                    cv2.rectangle(processed, (x, y), (x + w_det, y + h_det), (0, 255, 0), 2)
        elif mode == 'object':
            (rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
            for (x, y, w_det, h_det) in rects:
                if self.is_in_zone(cam_id, x, y, w_det, h_det):
                    trigger_tripped = True
                    cv2.rectangle(processed, (x, y), (x + w_det, y + h_det), (0, 0, 255), 2)
        elif mode == 'face':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w_det, h_det) in faces:
                if self.is_in_zone(cam_id, x, y, w_det, h_det):
                    trigger_tripped = True
                    cv2.rectangle(processed, (x, y), (x + w_det, y + h_det), (255, 0, 0), 2)

        # 2. Apply Security Watermark (Anti-Theft)
        processed = self.watermarker.apply(processed, cam_id)

        # 3. Draw Zones & OSD
        for (zx, zy, zw, zh) in self.zones.get(cam_id, []):
            cv2.rectangle(processed, (zx, zy), (zx+zw, zy+zh), (255, 255, 0), 1)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(processed, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(processed, f"CAM {cam_id} | {mode.upper()} | {timestamp}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if trigger_tripped:
            cv2.rectangle(processed, (0, 0), (w, h), (0, 0, 255), 4)

        return processed, trigger_tripped
