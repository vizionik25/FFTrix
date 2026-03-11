import cv2
import numpy as np
import base64
import os
import time
import json
from nicegui import ui, app
from fastapi.responses import RedirectResponse
from .database import Database, RECORDINGS_DIR
from .engine import CameraNode
from .analytics import AnalyticsPipeline

class Dashboard:
    def __init__(self, db=None, analytics=None):
        self.db = db if db else Database()
        self.analytics = analytics if analytics else AnalyticsPipeline()
        self.cameras = {} 
        self.grid_columns = 2
        self.selected_cam_id = None
        self.is_drawing_zone = False
        self.temp_zone_start = None
        
        # Safe mount
        if not os.environ.get('FFTRIX_TEST_MODE'):
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            app.add_static_files('/recordings', str(RECORDINGS_DIR))
        
        for c in self.db.get_cameras():
            self._init_camera(c['id'], c['name'], c['url'], c['mode'], 
                              c['zones'], c['record_247'], c['watermark'], start=False)

    def _init_camera(self, cam_id, name, url, mode, zones, record_247, watermark, start=True):
        self.analytics.set_config(cam_id, mode, zones, watermark)
        node = CameraNode(cam_id, url, name, self.analytics, self.db, record_247=record_247)
        self.cameras[cam_id] = node
        if start: node.start()

    def handle_login(self, username, password):
        if self.db.verify_user(username, password):
            app.storage.user.update({'authenticated': True, 'username': username})
            ui.navigate.to('/')
            return True
        return False

    def handle_logout(self):
        app.storage.user.update({'authenticated': False})
        ui.navigate.to('/login')

    def handle_viewport_click(self, e):
        if not self.selected_cam_id or not self.is_drawing_zone: return
        x, y = int(e.image_x), int(e.image_y)
        if self.temp_zone_start is None:
            self.temp_zone_start = (x, y)
        else:
            x1, y1 = self.temp_zone_start
            new_zone = [min(x, x1), min(y, y1), abs(x - x1), abs(y - y1)]
            current_zones = self.analytics.zones.get(self.selected_cam_id, [])
            current_zones.append(new_zone)
            self.analytics.zones[self.selected_cam_id] = current_zones
            self.db.update_camera_config(self.selected_cam_id, zones=current_zones)
            self.temp_zone_start = None
            self.is_drawing_zone = False
            self._refresh_grid()

    def update_watermark(self, cam_id, key, value):
        config = self.analytics.watermarker.configs.get(cam_id, {
            'text': '', 'image_path': '', 'mode': 'static', 'x': 50, 'y': 50, 'transparency': 0.5
        })
        config[key] = value
        self.analytics.watermarker.set_config(cam_id, config)
        self.db.update_camera_config(cam_id, watermark=config)

    def add_camera_ui(self, name, url, mode):
        if not name or not url: return
        existing_ids = [int(cid[1:]) for cid in self.cameras.keys() if cid.startswith('C') and cid[1:].isdigit()]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        cam_id = f"C{next_id:02d}"
        self.db.add_camera(cam_id, name, url, mode)
        self._init_camera(cam_id, name, url, mode, [], False, {}, start=True)
        self._refresh_grid()

    def delete_camera(self, cam_id):
        if cam_id in self.cameras:
            self.cameras[cam_id].stop()
            del self.cameras[cam_id]
            self.db.remove_camera(cam_id)
            self.selected_cam_id = None
            self._refresh_grid()

    @ui.page('/login')
    def login_page(self):
        if app.storage.user.get('authenticated', False): return RedirectResponse('/')
        with ui.card():
            u = ui.input('ID')
            p = ui.input('PW', password=True)
            ui.button('GO', on_click=lambda: self.handle_login(u.value, p.value))

    @ui.page('/')
    def main_page(self):
        if not app.storage.user.get('authenticated', False): return RedirectResponse('/login')
        self.build_ui()

    def build_ui(self):
        with ui.header():
            ui.label('FFTrix')
            ui.button(on_click=self.handle_logout)
        with ui.row():
            self._build_sidebar()
            self._build_grid()
            self._build_timeline()
        ui.timer(0.05, self._update_ui_loop)

    def _build_sidebar(self):
        with ui.column(): ui.label('Sidebar')

    def _build_grid(self):
        self.grid_container = ui.grid(columns=self.grid_columns)
        self._refresh_grid()

    def _refresh_grid(self):
        if hasattr(self, 'grid_container') and self.grid_container:
            self.grid_container.clear()
            with self.grid_container:
                for cam_id, node in self.cameras.items():
                    with ui.card():
                        img = ui.interactive_image().on('click', self.handle_viewport_click)
                        node.ui_image = img

    def _build_timeline(self):
        self.timeline_container = ui.column()
        self._refresh_timeline()

    def _refresh_timeline(self):
        if hasattr(self, 'timeline_container') and self.timeline_container:
            self.timeline_container.clear()
            for f in self.db.get_events(10, flags_only=True):
                ui.label(f"{f['source']} @ {f['timestamp']}")

    async def _update_ui_loop(self):
        for cam_id, node in self.cameras.items():
            if hasattr(node, 'ui_image') and getattr(node, 'processed_frame', None) is not None:
                _, buf = cv2.imencode('.jpg', node.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                node.ui_image.set_source(f'data:image/jpeg;base64,{base64.b64encode(buf).decode("utf-8")}')

def run_dashboard(remote=False):
    if remote:
        try:
            from pyngrok import ngrok
            ngrok.connect(8080)
        except: pass
    dashboard = Dashboard()
    ui.run(title="FFTrix", port=8080, host='0.0.0.0', storage_secret='secret')
