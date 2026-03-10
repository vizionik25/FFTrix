import cv2
import numpy as np

def remove_background_mog2(frame, back_sub):
    """
    Remove background using MOG2 background subtractor.
    Useful for videos with moving foreground objects.
    """
    fg_mask = back_sub.apply(frame)
    # Refine mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    
    # Apply mask to frame
    res = cv2.bitwise_and(frame, frame, mask=fg_mask)
    return res, fg_mask

def extract_foreground_grabcut(frame, rect=None):
    """
    Extract foreground using the GrabCut algorithm.
    rect: (x, y, w, h) defining the foreground area.
    """
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    if rect is None:
        # Default to a centered rectangle
        h, w = frame.shape[:2]
        rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    
    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify mask: 0 and 2 are background, 1 and 3 are foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    res = frame * mask2[:, :, np.newaxis]
    return res, mask2

def replace_background(foreground_frame, mask, background_image):
    """
    Replace the background of a foreground frame with a new image.
    """
    # Resize background to match foreground
    bg_resized = cv2.resize(background_image, (foreground_frame.shape[1], foreground_frame.shape[0]))
    
    # Invert mask for background
    bg_mask = cv2.bitwise_not(mask * 255)
    
    # Extract background part
    bg_part = cv2.bitwise_and(bg_resized, bg_resized, mask=bg_mask)
    # Extract foreground part
    fg_part = cv2.bitwise_and(foreground_frame, foreground_frame, mask=mask * 255)
    
    # Combine
    return cv2.add(fg_part, bg_part)

def chroma_key(frame, lower_color, upper_color, replacement_bg):
    """
    Perform chroma keying (e.g., green screen removal).
    lower_color, upper_color: HSV range for the key color.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Refine mask
    mask = cv2.medianBlur(mask, 5)
    inv_mask = cv2.bitwise_not(mask)
    
    # Resize background
    bg_resized = cv2.resize(replacement_bg, (frame.shape[1], frame.shape[0]))
    
    # Combine
    fg = cv2.bitwise_and(frame, frame, mask=inv_mask)
    bg = cv2.bitwise_and(bg_resized, bg_resized, mask=mask)
    
    return cv2.add(fg, bg)
