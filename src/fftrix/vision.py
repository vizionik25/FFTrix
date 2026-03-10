import cv2
import numpy as np
import os
from vidgear.gears import CamGear, VideoGear, WriteGear, NetGear

def detect_edges(frame):
    """Perform Canny Edge Detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def detect_motion(frame, background_subtractor):
    """Detect motion using background subtraction."""
    fg_mask = background_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

def detect_faces(frame, face_cascade):
    """Detect faces using Haar Cascades."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

from .face_recognition import FaceRecognizer
from .ocr import perform_ocr
from .segmentation import remove_background_mog2, chroma_key

def run_vision_loop(source=0, mode='edge', record_path=None, stream_address=None, stabilize=False):
    """
    Enhanced Video processing loop using VidGear.
    Modes: 'edge', 'motion', 'face', 'recognize', 'ocr', 'rembg', 'chroma'
    """
    # 1. High-Performance Streaming & 4. Stabilization
    options = {}
    if stabilize:
        options["stabilize"] = True
    
    # Add optimizations for IP Cameras / Network Streams
    if isinstance(source, str) and (source.startswith('rtsp') or source.startswith('http')):
        options.update({
            "THREADED_QUEUE_MODE": True, # Prevents blocking
            "rtsp_transport": "tcp",     # More stable than UDP for many IPCams
            "REORDER_QUEUE_MODE": True   # Ensures frames are in order
        })
    
    # VideoGear automatically chooses between CamGear and other gears
    stream = VideoGear(source=source, **options).start()
    
    # 2. High-Performance Recording (WriteGear)
    writer = None
    if record_path:
        # Compression Mode (FFmpeg) is default if output extension is common
        writer = WriteGear(output=record_path)

    # 3. Network Streaming (NetGear)
    server = None
    if stream_address:
        # Splitting address for NetGear (e.g. "127.0.0.1:5555")
        try:
            addr, port = stream_address.split(':')
            server = NetGear(address=addr, port=port, protocol="tcp", pattern=0)
        except ValueError:
            print("Invalid stream address format. Use IP:PORT")

    # Initialization
    back_sub = cv2.createBackgroundSubtractorMOG2() if mode in ['motion', 'rembg'] else None
    face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')) if mode == 'face' else None
    
    recognizer = None
    if mode == 'recognize':
        recognizer = FaceRecognizer()
        if os.path.exists('training_data'): recognizer.train('training_data')

    if mode == 'chroma':
        replacement_bg = np.zeros((480, 640, 3), np.uint8)
        replacement_bg[:] = (255, 0, 0)
        lower_green, upper_green = np.array([35, 100, 100]), np.array([85, 255, 255])

    print(f"Running {mode} (Stabilize={stabilize}, Record={record_path is not None}, Net={stream_address is not None})")

    try:
        while True:
            frame = stream.read()
            if frame is None: break

            processed = frame.copy()
            if mode == 'edge': processed = detect_edges(frame)
            elif mode == 'motion': processed = detect_motion(frame, back_sub)
            elif mode == 'face': processed = detect_faces(processed, face_cascade)
            elif mode == 'recognize': processed = recognizer.recognize(processed)
            elif mode == 'ocr': processed = perform_ocr(processed)
            elif mode == 'rembg': processed, _ = remove_background_mog2(frame, back_sub)
            elif mode == 'chroma': processed = chroma_key(frame, lower_green, upper_green, replacement_bg)

            # Record if enabled
            if writer: writer.write(processed)
            
            # Send over network if enabled
            if server: server.send(processed)

            cv2.imshow('FFTrix Vision (VidGear)', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        # Proper VidGear cleanup
        stream.stop()
        if writer: writer.close()
        if server: server.close()
        cv2.destroyAllWindows()
