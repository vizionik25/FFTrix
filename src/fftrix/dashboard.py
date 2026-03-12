import cv2
import numpy as np
import base64
import os
import time
import json
import warnings
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
        self._timeline_tick = 0

        self._setup_routes()

        # Safe mount
        if not os.environ.get('FFTRIX_TEST_MODE'):
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            app.add_static_files('/recordings', str(RECORDINGS_DIR))

        for c in self.db.get_cameras():
            self._init_camera(c['id'], c['name'], c['url'], c['mode'],
                              c['zones'], c['record_247'], c['watermark'], start=False)

    def _setup_routes(self):
        @ui.page('/login')
        def login_page():
            if app.storage.user.get('authenticated', False): return RedirectResponse('/')
            with ui.card().classes('absolute-center'):
                ui.label('FFTrix').classes('text-h5 text-bold')
                u = ui.input('Username').props('outlined')
                p = ui.input('Password', password=True, password_toggle_button=True).props('outlined')
                ui.button('Login', on_click=lambda: self.handle_login(u.value, p.value)).classes('full-width')

        @ui.page('/')
        def main_page():
            if not app.storage.user.get('authenticated', False): return RedirectResponse('/login')
            self.build_ui()

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
        ui.notify('Invalid credentials', type='negative')
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

    def build_ui(self):
        with ui.header().classes('items-center justify-between'):
            ui.label('🎥 FFTrix').classes('text-h6 text-bold')
            ui.button('Logout', on_click=self.handle_logout, icon='logout').props('flat color=white')
        with ui.row().classes('w-full no-wrap'):
            self._build_sidebar()
            self._build_grid()
            self._build_timeline()
        ui.timer(0.05, self._update_ui_loop)

    def _build_sidebar(self):
        with ui.column().classes('q-pa-md bg-grey-9 rounded shadow-2').style('min-width:260px;max-width:280px'):
            ui.label('Add Camera').classes('text-subtitle1 text-bold')
            name_in = ui.input('Name').props('outlined dense')
            url_in = ui.input('Source URL / Index').props('outlined dense')
            mode_sel = ui.select(
                options=['none', 'motion', 'object', 'face', 'edge', 'ocr'],
                value='none', label='AI Mode'
            ).props('outlined dense')
            ui.button('Add', icon='add',
                      on_click=lambda: self.add_camera_ui(name_in.value, url_in.value, mode_sel.value)
                      ).classes('full-width')

            ui.separator()
            ui.label('Cameras').classes('text-subtitle1 text-bold')
            self.camera_list_container = ui.column().classes('w-full')
            self._refresh_camera_list()

            ui.separator()
            ui.label('Selected Camera').classes('text-subtitle1 text-bold')
            self.controls_container = ui.column().classes('w-full')
            self._refresh_controls()

    def _refresh_camera_list(self):
        if not hasattr(self, 'camera_list_container') or self.camera_list_container is None:
            return
        self.camera_list_container.clear()
        with self.camera_list_container:
            if not self.cameras:
                ui.label('No cameras added.').classes('text-caption text-grey')
            for cam_id, node in self.cameras.items():
                with ui.row().classes('items-center full-width'):
                    ui.button(f'{node.name} ({cam_id})',
                              on_click=lambda cid=cam_id: self._select_camera(cid)
                              ).props('flat dense').classes('flex-grow text-left')
                    ui.button(icon='delete',
                              on_click=lambda cid=cam_id: self.delete_camera(cid)
                              ).props('flat dense color=negative')

    def _select_camera(self, cam_id):
        self.selected_cam_id = cam_id
        self._refresh_controls()

    def _refresh_controls(self):
        if not hasattr(self, 'controls_container') or self.controls_container is None:
            return
        self.controls_container.clear()
        if not self.selected_cam_id or self.selected_cam_id not in self.cameras:
            with self.controls_container:
                ui.label('Select a camera first.').classes('text-caption text-grey')
            return
        cam_id = self.selected_cam_id
        cam_cfg = next((c for c in self.db.get_cameras() if c['id'] == cam_id), {})
        with self.controls_container:
            # 24/7 recording toggle
            rec = ui.switch('24/7 Record',
                            value=cam_cfg.get('record_247', False),
                            on_change=lambda e: self.db.update_camera_config(cam_id, record_247=e.value))
            # Zone draw / clear
            with ui.row():
                ui.button('Draw Zone', icon='crop',
                          on_click=lambda: setattr(self, 'is_drawing_zone', True)).props('dense')
                ui.button('Clear Zones', icon='clear',
                          on_click=lambda: [
                              self.analytics.zones.update({cam_id: []}),
                              self.db.update_camera_config(cam_id, zones=[]),
                              self._refresh_grid()
                          ]).props('dense color=warning')
            # Watermark
            wm_cfg = cam_cfg.get('watermark', {})
            ui.input('Watermark Text', value=wm_cfg.get('text', ''),
                     on_change=lambda e: self.update_watermark(cam_id, 'text', e.value)
                     ).props('outlined dense')
            ui.select(options=['static', 'floating'], value=wm_cfg.get('mode', 'static'),
                      label='WM Mode',
                      on_change=lambda e: self.update_watermark(cam_id, 'mode', e.value)
                      ).props('outlined dense')
            ui.slider(min=0.0, max=1.0, step=0.05, value=wm_cfg.get('transparency', 0.5),
                      on_change=lambda e: self.update_watermark(cam_id, 'transparency', e.value))

    def _build_grid(self):
        self.grid_container = ui.grid(columns=self.grid_columns).classes('flex-grow')
        self._refresh_grid()

    def _refresh_grid(self):
        if not hasattr(self, 'grid_container') or self.grid_container is None:
            return
        self.grid_container.clear()
        self._refresh_camera_list()
        self._refresh_controls()
        with self.grid_container:
            for cam_id, node in self.cameras.items():
                with ui.card():
                    ui.label(f'{node.name} — {cam_id}').classes('text-caption')
                    img = ui.interactive_image().on('click', self.handle_viewport_click)
                    node.ui_image = img

    def _build_timeline(self):
        self.timeline_container = ui.column().classes('q-pa-md bg-grey-9 rounded shadow-2').style('min-width:240px;max-width:260px')
        self._refresh_timeline()

    def _refresh_timeline(self):
        if not hasattr(self, 'timeline_container') or self.timeline_container is None:
            return
        self.timeline_container.clear()
        events = self.db.get_events(20, flags_only=True)
        with self.timeline_container:
            with ui.row().classes('items-center'):
                ui.label('⚑ Flagged Events').classes('text-subtitle1 text-bold flex-grow')
                ui.badge(str(len(events)), color='red')
            if not events:
                ui.label('No events flagged.').classes('text-caption text-grey')
            for ev in events:
                with ui.card().classes('w-full q-mb-xs'):
                    ui.label(f"[{ev['type']}] {ev['source']}").classes('text-caption text-bold')
                    ui.label(ev['details']).classes('text-caption')
                    ui.label(ev['timestamp']).classes('text-caption text-grey')

    async def _update_ui_loop(self):
        self._timeline_tick += 1
        # Refresh timeline every ~2 s (timer fires at 0.05s, so every 40 ticks)
        if self._timeline_tick % 40 == 0:
            self._refresh_timeline()

        for cam_id, node in self.cameras.items():
            if hasattr(node, 'ui_image') and getattr(node, 'processed_frame', None) is not None:
                _, buf = cv2.imencode('.jpg', node.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                node.ui_image.set_source(f'data:image/jpeg;base64,{base64.b64encode(buf).decode("utf-8")}')


def run_dashboard(remote=False):
    secret = os.environ.get("FFTRIX_SECRET_KEY")
    if not secret:
        warnings.warn(
            "FFTRIX_SECRET_KEY not set — using insecure default! "
            "Set this environment variable before deploying.",
            UserWarning,
            stacklevel=2,
        )
        secret = "insecure-default-change-me"

    port = int(os.environ.get("FFTRIX_PORT", 8080))
    host = os.environ.get("FFTRIX_HOST", "0.0.0.0")

    if remote:
        try:
            from pyngrok import ngrok
            ngrok.connect(port)
        except Exception:
            pass

    dashboard = Dashboard()
    # reload=False is critical when running as an installed package to avoid RuntimeError
    ui.run(title="FFTrix", port=port, host=host, storage_secret=secret, reload=False, show=False, dark=True)
