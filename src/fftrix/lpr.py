"""FFTrix License Plate Reader — OpenCV + OCR based LPR module."""

from __future__ import annotations

import re
import logging
from typing import NamedTuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ICAO plate character set for basic validation
_PLATE_PATTERN = re.compile(r'^[A-Z0-9]{2,10}$')


class PlateDetection(NamedTuple):
    """Single plate detection result."""
    text: str
    confidence: float  # 0.0 – 1.0
    bbox: tuple[int, int, int, int]  # x, y, w, h


class LicensePlateReader:
    """Detect and read license plates from a single video frame.

    Uses a two-stage pipeline:
      1. OpenCV morphology + contour filtering to locate plate candidate ROIs
      2. Tesseract OCR (via pytesseract) to extract the text

    Example::

        lpr = LicensePlateReader()
        plates = lpr.process(frame)
        for p in plates:
            print(p.text, p.confidence, p.bbox)
    """

    def __init__(self, ocr_engine=None, min_area: int = 1500,
                 aspect_min: float = 1.5, aspect_max: float = 6.0):
        """
        Args:
            ocr_engine: pytesseract module (injected for testing).
            min_area: Minimum contour area to consider as a plate candidate.
            aspect_min / aspect_max: Allowed w/h ratio for a plate bounding box.
        """
        import pytesseract as _tess
        self._ocr = ocr_engine if ocr_engine is not None else _tess
        self.min_area = min_area
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max

        # Structuring elements for plate segmentation
        self._rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        self._sq_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def _candidate_rois(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return bounding boxes of plate-shaped regions."""
        # Contrast stretch + gradient
        eq = cv2.equalizeHist(gray)
        grad = cv2.morphologyEx(eq, cv2.MORPH_GRADIENT, self._sq_kern)
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Close horizontally to merge plate characters
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self._rect_kern)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self._sq_kern)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            ratio = w / h
            if self.aspect_min <= ratio <= self.aspect_max:
                candidates.append((x, y, w, h))
        return candidates

    def _ocr_roi(self, roi: np.ndarray) -> tuple[str, float]:
        """Run OCR on a cropped ROI; return (text, confidence)."""
        # Upscale small ROIs for better OCR
        h, w = roi.shape[:2]
        if w < 100:
            roi = cv2.resize(roi, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        gray = roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        try:
            data = self._ocr.image_to_data(
                bw, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                output_type=self._ocr.Output.DICT,
            )
            texts, confs = [], []
            for t, c in zip(data['text'], data['conf']):
                c = int(c)
                if c > 40 and t.strip():
                    texts.append(t.strip().upper())
                    confs.append(c)
            joined = ''.join(texts)
            avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
            return joined, avg_conf
        except Exception as exc:
            log.debug("OCR error: %s", exc)
            return '', 0.0

    def process(self, frame: np.ndarray) -> list[PlateDetection]:
        """Detect license plates in *frame*.

        Returns a list of :class:`PlateDetection` (may be empty).
        """
        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        candidates = self._candidate_rois(gray)
        detections: list[PlateDetection] = []
        for (x, y, w, h) in candidates:
            roi = frame[y:y + h, x:x + w]
            text, conf = self._ocr_roi(roi)
            cleaned = re.sub(r'\s+', '', text)
            if _PLATE_PATTERN.match(cleaned):
                detections.append(PlateDetection(text=cleaned, confidence=conf,
                                                  bbox=(x, y, w, h)))
        return detections

    def annotate(self, frame: np.ndarray,
                 detections: list[PlateDetection]) -> np.ndarray:
        """Draw bounding boxes and plate text onto *frame* in-place."""
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 100), 2)
            label = f"{det.text} ({det.confidence:.0%})"
            cv2.putText(frame, label, (x, max(y - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        return frame
