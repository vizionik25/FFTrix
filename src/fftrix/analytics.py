import cv2
import numpy as np
import time
import os
import pytesseract
from pathlib import Path

class WatermarkEngine:
    def __init__(self):
        self.configs = {} 
        self.cached_images = {} 
        self.float_offsets = {} 

    def set_config(self, cam_id, config):
        self.configs[cam_id] = config
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
        x, y = max(0, min(x, w_frame - w_wm)), max(0, min(y, h_frame - h_wm))
        roi = frame[y:y+h_wm, x:x+w_wm]
        
        if watermark.shape[2] == 4:
            wm_alpha = (watermark[:, :, 3] / 255.0) * alpha
            for c in range(3):
                roi[:, :, c] = (wm_alpha * watermark[:, :, c] + (1 - wm_alpha) * roi[:, :, c])
        else:
            cv2.addWeighted(watermark, alpha, roi, 1 - alpha, 0, roi)
        frame[y:y+h_wm, x:x+w_wm] = roi
        return frame

    def apply(self, frame, cam_id):
        config = self.configs.get(cam_id)
        if not config: return frame
        h, w = frame.shape[:2]
        alpha = config.get('transparency', 0.5)
        if config.get('mode') == 'floating':
            st = self.float_offsets.get(cam_id, [100, 100, 2, 2])
            st[0] += st[2]
            st[1] += st[3]
            if st[0] <= 0 or st[0] >= w - 200: st[2] *= -1
            if st[1] <= 50 or st[1] >= h - 50: st[3] *= -1
            pos = (st[0], st[1])
            self.float_offsets[cam_id] = st
        else:
            pos = (config.get('x', 50), config.get('y', 50))
        frame = self._draw_text(frame, config.get('text', ''), pos, alpha)
        if config.get('image_path'):
            frame = self._draw_image(frame, config['image_path'], (pos[0], pos[1] + 40), alpha)
        return frame

class AnalyticsPipeline:
    def __init__(self, hog=None, face_cascade=None, ocr_engine=None,
                 db=None, alert_manager=None, snapshots_dir: Path | None = None):
        self.modes = {}
        self.back_subs = {}
        self.zones = {}
        self.privacy_zones: dict[str, list] = {}
        self.arm_schedules: dict[str, list] = {}
        self.watermarker = WatermarkEngine()
        self.hog = hog if hog else cv2.HOGDescriptor()
        if not hog: self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.face_cascade = face_cascade if face_cascade else cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.ocr_engine = ocr_engine if ocr_engine else pytesseract
        self.db = db
        self.alert_manager = alert_manager
        self.snapshots_dir = snapshots_dir

    def set_config(self, cam_id, mode=None, zones=None, watermark=None, privacy_zones=None, arm_schedule=None):
        if mode: self.modes[cam_id] = mode
        if zones is not None: self.zones[cam_id] = zones
        if privacy_zones is not None: self.privacy_zones[cam_id] = privacy_zones
        if arm_schedule is not None: self.arm_schedules[cam_id] = arm_schedule
        if watermark is not None: self.watermarker.set_config(cam_id, watermark)
        if self.modes.get(cam_id) == 'motion' and cam_id not in self.back_subs:
            self.back_subs[cam_id] = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    def is_armed(self, cam_id: str) -> bool:
        """Return True if the camera should process alerts right now.

        If no schedule is set (empty list), the camera is always armed.
        Each schedule entry: {"days": [0..6], "start": "HH:MM", "end": "HH:MM"}
        0=Monday, 6=Sunday.
        """
        schedule = self.arm_schedules.get(cam_id, [])
        if not schedule:
            return True  # no schedule = always armed
        import time as _time
        now = _time.localtime()
        today = now.tm_wday  # 0=Mon, 6=Sun
        now_str = f"{now.tm_hour:02d}:{now.tm_min:02d}"
        for entry in schedule:
            days = entry.get('days', list(range(7)))
            start = entry.get('start', '00:00')
            end = entry.get('end', '23:59')
            if today in days and start <= now_str <= end:
                return True
        return False


    def is_in_zone(self, cam_id, x, y, w, h):
        cam_zones = self.zones.get(cam_id, [])
        if not cam_zones: return True
        cx, cy = x + w//2, y + h//2
        for (zx, zy, zw, zh) in cam_zones:
            if zx <= cx <= zx + zw and zy <= cy <= zy + zh: return True
        return False

    def apply_privacy_blur(self, frame: np.ndarray, cam_id: str) -> np.ndarray:
        """Blur all privacy zones on the given frame before display/recording."""
        for (x1, y1, x2, y2) in self.privacy_zones.get(cam_id, []):
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)
        return frame

    def _save_snapshot(self, frame: np.ndarray, cam_id: str) -> str | None:
        """Write a JPEG snapshot to disk and return the path, or None on failure."""
        if self.snapshots_dir is None:
            return None
        try:
            snap_dir = Path(self.snapshots_dir) / cam_id
            snap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = snap_dir / f"{ts}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return str(path)
        except Exception as exc:
            print(f"Snapshot save error: {exc}")
            return None

    def process(self, frame, cam_id):
        try:
            mode = self.modes.get(cam_id, 'none')
            trigger_tripped = False
            processed = frame.copy()
            h, w = processed.shape[:2]

            if mode == 'motion' and cam_id in self.back_subs:
                fg_mask = self.back_subs[cam_id].apply(frame)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) < 1000: continue
                    (x, y, w_det, h_det) = cv2.boundingRect(cnt)
                    if self.is_in_zone(cam_id, x, y, w_det, h_det):
                        trigger_tripped = True
                        cv2.rectangle(processed, (x, y), (x + w_det, y + h_det), (0, 255, 0), 2)
            elif mode == 'object':
                (rects, _) = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
                for (x, y, wd, hd) in rects:
                    if self.is_in_zone(cam_id, x, y, wd, hd):
                        trigger_tripped = True
                        cv2.rectangle(processed, (x, y), (x + wd, y + hd), (0, 0, 255), 2)
            elif mode == 'face':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, wd, hd) in faces:
                    if self.is_in_zone(cam_id, x, y, wd, hd):
                        trigger_tripped = True
                        cv2.rectangle(processed, (x, y), (x + wd, y + hd), (255, 0, 0), 2)
            elif mode == 'edge':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                processed = cv2.cvtColor(cv2.Canny(gray, 100, 200), cv2.COLOR_GRAY2BGR)
            elif mode == 'ocr':
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    d = self.ocr_engine.image_to_data(gray, output_type=pytesseract.Output.DICT)
                    for i in range(len(d['text'])):
                        if int(d['conf'][i]) > 60:
                            (x, y, wd, hd) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                            cv2.rectangle(processed, (x, y), (x + wd, y + hd), (0, 255, 0), 2)
                except: pass
            elif mode == 'lpr':
                try:
                    from .lpr import LicensePlateReader
                    lpr = getattr(self, '_lpr', None)
                    if lpr is None:
                        self._lpr = LicensePlateReader(ocr_engine=self.ocr_engine)
                        lpr = self._lpr
                    detections = lpr.process(processed)
                    if detections:
                        trigger_tripped = True
                        processed = lpr.annotate(processed, detections)
                        # Log each plate text in the event details
                        plate_texts = ', '.join(d.text for d in detections)
                        if self.db and self.is_armed(cam_id):
                            snap = self._save_snapshot(processed, cam_id)
                            self.db.log_event('LPR', cam_id, f'Plates: {plate_texts}',
                                              is_flag=1, snapshot_path=snap)
                        if self.alert_manager and self.is_armed(cam_id):
                            self.alert_manager.fire(cam_id, 'LPR',
                                                    details=f'Plates: {plate_texts}')
                except Exception as _lpr_exc:
                    pass


            # Apply privacy blur before watermark and recording
            processed = self.apply_privacy_blur(processed, cam_id)
            processed = self.watermarker.apply(processed, cam_id)
            for (zx, zy, zw, zh) in self.zones.get(cam_id, []):
                cv2.rectangle(processed, (zx, zy), (zx+zw, zy+zh), (255, 255, 0), 1)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.rectangle(processed, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(processed, f"CAM {cam_id} | {mode.upper()} | {timestamp}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if trigger_tripped:
                cv2.rectangle(processed, (0, 0), (w, h), (0, 0, 255), 4)
                if self.is_armed(cam_id):
                    # Save snapshot and fire alerts only when armed
                    snapshot_path = self._save_snapshot(processed, cam_id)
                    if self.db:
                        self.db.log_event(mode.upper(), cam_id, "Trigger detected",
                                          is_flag=1, snapshot_path=snapshot_path)
                    if self.alert_manager:
                        self.alert_manager.fire(cam_id, mode.upper(),
                                                details="Trigger detected",
                                                snapshot_path=snapshot_path)
            return processed, trigger_tripped
        except Exception as e:
            print(f"Analytics Pipeline Error: {e}")
            return frame, False
