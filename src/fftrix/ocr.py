import cv2
import numpy as np
import pytesseract

def perform_ocr(frame):
    """
    Perform character recognition on a frame.
    1. Pre-processes the frame for better OCR accuracy.
    2. Uses pytesseract to extract text.
    3. Draws bounding boxes and text overlays.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get black text on white background (or vice versa)
    # Using adaptive thresholding for varying lighting conditions
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # OCR with pytesseract
    # config='--psm 6' assumes a single uniform block of text
    try:
        # Get OCR data including bounding boxes
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            # Only process if confidence is > 60 and text is not empty
            if int(data['conf'][i]) > 60:
                text = data['text'][i].strip()
                if text:
                    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except Exception as e:
        print(f"OCR Error: {e}")
        cv2.putText(frame, "OCR Error: Ensure Tesseract is installed on your OS", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    return frame

def detect_text_areas(frame):
    """
    Legacy method renamed for backward compatibility.
    Now calls the improved perform_ocr.
    """
    return perform_ocr(frame)
