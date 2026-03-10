import cv2
import numpy as np

def detect_text_areas(frame):
    """Detect areas likely containing text using morphological operations."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # Edge detection
    grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    grad = np.absolute(grad)
    (min_val, max_val) = (np.min(grad), np.max(grad))
    if max_val > 0:
        grad = (255 * (grad - min_val) / (max_val - min_val)).astype("uint8")
    else:
        grad = grad.astype("uint8")
    
    # Blur and threshold
    grad = cv2.GaussianBlur(grad, (9, 9), 0)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Morphological closing to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "Text Region", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
    return frame

# For true OCR, integration with a model like Tesseract is required.
# Example usage:
# import pytesseract
# text = pytesseract.image_to_string(roi)
