import cv2
import numpy as np
import base64
import os
import time
from nicegui import ui, app
from .vision import detect_edges, detect_motion, detect_faces
from .ocr import perform_ocr
from .segmentation import remove_background_mog2, chroma_key
from .face_recognition import FaceRecognizer
from vidgear.gears import VideoGear, WriteGear

class Dashboard:
    def __init__(self):
        # Viewport Settings (Locked Size)
        self.VIEWPORT_WIDTH = 1280
        self.VIEWPORT_HEIGHT = 720
        
        # Stream State
        self.stream = None
        self.writer = None
        self.is_running = False
        self.is_recording = False
        self.mode = 'edge'
        self.source = '0'
        
        # App Settings
        self.theme_dark = True
        self.show_fps = True
        self.auto_start = False
        self.recording_dir = os.path.join(os.getcwd(), 'recordings')
        
        # Device & Encoding Settings
        self.stabilize = False
        self.rtsp_transport = 'tcp'
        self.threaded_queue = True
        self.reorder_queue = True
        self.output_filename = 'capture.mp4'
        self.ffmpeg_custom_flags = "-vcodec libx264 -crf 23 -preset fast"
        
        # Vision Tools
        self.back_sub = None
        self.face_cascade = None
        self.recognizer = None
        
        # UI References
        self.image_display = None
        self.status_label = None
        self.fps_label = None
        self.rec_status_label = None
        self.last_frame_time = 0

    def parse_ffmpeg_params(self):
        """Convert string flags to a VidGear-compatible dictionary."""
        params = {}
        parts = self.ffmpeg_custom_flags.split()
        for i in range(0, len(parts)-1, 2):
            if parts[i].startswith('-'):
                params[parts[i]] = parts[i+1]
        return params

    def initialize_vision_tools(self):
        """Pre-load necessary CV models."""
        self.back_sub = cv2.createBackgroundSubtractorMOG2() if self.mode in ['motion', 'rembg'] else None
        self.face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')) if self.mode == 'face' else None
        
        if self.mode == 'recognize':
            self.recognizer = FaceRecognizer()
            if os.path.exists('training_data'):
                self.recognizer.train('training_data')

    def start_stream(self):
        """Initialize the VidGear stream."""
        if self.stream: self.stop_all()
            
        options = {
            "stabilize": self.stabilize,
            "THREADED_QUEUE_MODE": self.threaded_queue,
            "REORDER_QUEUE_MODE": self.reorder_queue
        }
        
        if isinstance(self.source, str) and self.source.startswith(('rtsp', 'http')):
            options["rtsp_transport"] = self.rtsp_transport
        
        processed_source = int(self.source) if self.source.isdigit() else self.source
        
        try:
            self.stream = VideoGear(source=processed_source, **options).start()
            self.initialize_vision_tools()
            self.is_running = True
            ui.notify(f"Stream Active: {self.mode}", type='info')
            if self.status_label:
                self.status_label.set_text(f"LIVE: {self.mode}")
                self.status_label.classes('text-green-400', remove='text-red-500')
        except Exception as e:
            ui.notify(f"Connection Failed: {e}", type='negative')

    def toggle_recording(self):
        """Start or stop the WriteGear recording process."""
        if not self.is_running:
            ui.notify("Start stream before recording", type='warning')
            return

        if not self.is_recording:
            try:
                if not os.path.exists(self.recording_dir):
                    os.makedirs(self.recording_dir)
                
                full_path = os.path.join(self.recording_dir, self.output_filename)
                params = self.parse_ffmpeg_params()
                
                # Initialize WriteGear with custom parameters
                self.writer = WriteGear(output=full_path, **params)
                self.is_recording = True
                ui.notify(f"Recording to {self.output_filename}", type='positive')
                if self.rec_status_label:
                    self.rec_status_label.set_text("RECORDING")
                    self.rec_status_label.classes('text-red-500 animate-pulse')
            except Exception as e:
                ui.notify(f"Recording Error: {e}", type='negative')
        else:
            self.stop_recording()

    def stop_recording(self):
        """Stop writing to file."""
        self.is_recording = False
        if self.writer:
            self.writer.close()
            self.writer = None
        if self.rec_status_label:
            self.rec_status_label.set_text("IDLE")
            self.rec_status_label.classes('text-slate-400', remove='text-red-500 animate-pulse')
        ui.notify("Recording saved", type='info')

    def stop_all(self):
        """Full cleanup of stream and recording."""
        self.stop_recording()
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream = None
        if self.status_label:
            self.status_label.set_text("OFFLINE")
            self.status_label.classes('text-red-500', remove='text-green-400')
        if self.image_display:
            self.image_display.set_source('https://via.placeholder.com/1280x720?text=SIGNAL+LOST')

    async def update_frame(self):
        """Process and fit frame into the static viewport, and record if active."""
        if not self.is_running or self.stream is None:
            return

        frame = self.stream.read()
        if frame is None: return

        # FPS Tracking
        curr_time = time.time()
        fps = 1 / (curr_time - self.last_frame_time) if self.last_frame_time > 0 else 0
        self.last_frame_time = curr_time
        if self.fps_label and self.show_fps:
            self.fps_label.set_text(f"FPS: {fps:.1f}")

        # Core Vision Logic
        processed = frame.copy()
        try:
            if self.mode == 'edge': processed = detect_edges(frame)
            elif self.mode == 'motion': processed = detect_motion(frame, self.back_sub)
            elif self.mode == 'face': processed = detect_faces(processed, self.face_cascade)
            elif self.mode == 'recognize' and self.recognizer: processed = self.recognizer.recognize(processed)
            elif self.mode == 'ocr': processed = perform_ocr(processed)
            elif self.mode == 'rembg': processed, _ = remove_background_mog2(frame, self.back_sub)
        except Exception as e:
            print(f"Loop Error: {e}")

        # Recording
        if self.is_recording and self.writer:
            self.writer.write(processed)

        # UI Scaling & Update
        _, buffer = cv2.imencode('.jpg', processed)
        encoded = base64.b64encode(buffer).decode('utf-8')
        if self.image_display:
            self.image_display.set_source(f'data:image/jpeg;base64,{encoded}')

    def build_ui(self):
        """Advanced UI with locked viewport and encoding settings."""
        ui.query('body').style('background-color: #0f172a;')
        
        with ui.header().classes('bg-slate-800 p-4 justify-between items-center shadow-lg'):
            with ui.row().classes('items-center'):
                ui.icon('videocam', size='md').classes('text-blue-400')
                ui.label('FFTrix Advanced Suite').classes('text-2xl font-black tracking-tight')
            
            with ui.row().classes('items-center gap-6'):
                self.fps_label = ui.label('FPS: 0.0').classes('font-mono text-blue-300')
                self.rec_status_label = ui.label('IDLE').classes('font-bold text-slate-400')
                self.status_label = ui.label('OFFLINE').classes('font-bold text-red-500')
                ui.button(icon='power_settings_new', on_click=self.stop_all).props('flat round').classes('text-white')

        with ui.tabs().classes('w-full bg-slate-800 text-white shadow-md') as tabs:
            view_tab = ui.tab('Stream View', icon='visibility')
            app_tab = ui.tab('App Settings', icon='tune')
            device_tab = ui.tab('Hardware & Encoding', icon='memory')

        with ui.tab_panels(tabs, value=view_tab).classes('w-full bg-transparent p-0'):
            
            # --- LIVE MONITOR PANEL ---
            with ui.tab_panel(view_tab).classes('p-4'):
                with ui.row().classes('w-full justify-center'):
                    # Static Locked Viewport
                    with ui.card().style(f'width: {self.VIEWPORT_WIDTH}px; height: {self.VIEWPORT_HEIGHT}px; background-color: black; padding: 0; overflow: hidden;'):
                        self.image_display = ui.interactive_image().style('width: 100%; height: 100%; object-fit: contain;')
                        self.image_display.set_source('https://via.placeholder.com/1280x720?text=SYSTEM+READY')

                # Quick Dashboard Controls
                with ui.row().classes('w-full justify-center mt-4 gap-4'):
                    with ui.card().classes('bg-slate-800 p-3 px-8 flex-row items-center gap-6'):
                        ui.input(label='Source', value=self.source, on_change=lambda e: setattr(self, 'source', e.value)).props('dark dense outlined')
                        ui.select({
                            'edge': 'Edge Detect', 'motion': 'Motion', 'face': 'Faces', 
                            'recognize': 'Recognition', 'ocr': 'True OCR', 'rembg': 'Isolation'
                        }, label='Mode', value=self.mode, on_change=lambda e: setattr(self, 'mode', e.value)).props('dark dense outlined').classes('w-40')
                        
                        ui.button('START STREAM', on_click=self.start_stream).classes('bg-blue-700 font-bold px-6')
                        ui.button('RECORD', on_click=self.toggle_recording).bind_classes({
                            'bg-red-700': 'is_recording',
                            'bg-slate-700': 'not is_recording'
                        }, target=self).classes('font-bold px-6')

            # --- APP SETTINGS PANEL ---
            with ui.tab_panel(app_tab).classes('p-8'):
                with ui.column().classes('max-w-2xl mx-auto bg-slate-800 p-8 rounded-xl shadow-2xl'):
                    ui.label('Application Preferences').classes('text-2xl font-bold mb-4 text-blue-400')
                    ui.switch('Show Live FPS Counter', value=self.show_fps, on_change=lambda e: setattr(self, 'show_fps', e.value)).props('dark')
                    ui.switch('Launch stream on application start', value=self.auto_start, on_change=lambda e: setattr(self, 'auto_start', e.value)).props('dark')
                    
                    ui.separator().classes('my-4')
                    ui.label('Recording Storage Path').classes('text-sm text-slate-400')
                    ui.input(value=self.recording_dir, on_change=lambda e: setattr(self, 'recording_dir', e.value)).props('dark filled dense')
                    
                    ui.label('Default Output Filename').classes('text-sm text-slate-400 mt-4')
                    ui.input(value=self.output_filename, on_change=lambda e: setattr(self, 'output_filename', e.value)).props('dark filled dense')

            # --- DEVICE & ENCODING SETTINGS PANEL ---
            with ui.tab_panel(device_tab).classes('p-8'):
                with ui.column().classes('max-w-2xl mx-auto bg-slate-800 p-8 rounded-xl shadow-2xl'):
                    ui.label('Stream Hardware Optimization').classes('text-2xl font-bold mb-4 text-blue-400')
                    with ui.row().classes('gap-8'):
                        ui.switch('Video Stabilization', value=self.stabilize, on_change=lambda e: setattr(self, 'stabilize', e.value)).props('dark')
                        ui.switch('Low Latency (Threaded)', value=self.threaded_queue, on_change=lambda e: setattr(self, 'threaded_queue', e.value)).props('dark')
                    
                    ui.label('RTSP Network Transport').classes('mt-4 text-sm text-slate-400')
                    ui.radio(['tcp', 'udp'], value=self.rtsp_transport, on_change=lambda e: setattr(self, 'rtsp_transport', e.value)).props('dark inline')
                    
                    ui.separator().classes('my-6')
                    ui.label('Custom FFmpeg Parameters (Advanced Encoding)').classes('text-xl font-bold mb-2 text-red-400')
                    ui.label('Directly passed to WriteGear. Ensure valid FFmpeg flag pairs.').classes('text-xs text-slate-500 mb-2')
                    ui.textarea(value=self.ffmpeg_custom_flags, on_change=lambda e: setattr(self, 'ffmpeg_custom_flags', e.value)).props('dark filled').classes('w-full font-mono')
                    
                    ui.label('Encoding Buffer Delay').classes('text-sm text-slate-400 mt-4')
                    ui.slider(min=1, max=100, value=30).props('dark')

        # Async frame update loop
        ui.timer(0.01, self.update_frame)

def run_dashboard():
    dashboard = Dashboard()
    @ui.page('/')
    def main_page():
        dashboard.build_ui()
    ui.run(title="FFTrix Pro Monitor", port=8080, reload=False, show=False, dark=True)

if __name__ in {"__main__", "__mp_main__"}:
    run_dashboard()
