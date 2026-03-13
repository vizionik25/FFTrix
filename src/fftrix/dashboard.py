import cv2
import numpy as np
import base64
import os
import time
import json
import threading
import warnings
import subprocess
import shutil
from pathlib import Path
from nicegui import ui, app
from fastapi.responses import RedirectResponse
from .database import Database, RECORDINGS_DIR, SNAPSHOTS_DIR, STATIC_DIR
from .engine import CameraNode
from .analytics import AnalyticsPipeline
from .alerts import AlertManager
from .retention import RetentionManager
from .discovery import ONVIFDiscovery, DiscoveredDevice
from .clipper import ClipExporter

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
        self._discovered: list[DiscoveredDevice] = []
        self._discovery_scanning = False

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
            role = self.db.get_user_role(username)
            app.storage.user.update({'authenticated': True, 'username': username, 'role': role})
            if self.db.is_default_credentials():
                # Keep them on the login page until they change defaults
                self._show_first_use_dialog(username)
                return True
            ui.navigate.to('/')
            return True
        ui.notify('Invalid credentials', type='negative')
        return False

    def is_admin(self) -> bool:
        """Return True if the currently logged-in user has the admin role."""
        return app.storage.user.get('role', 'admin') == 'admin'

    def handle_logout(self):
        app.storage.user.update({'authenticated': False})
        ui.navigate.to('/login')

    def _show_first_use_dialog(self, old_username: str = 'admin'):
        """Show a non-dismissable dialog forcing the user to change default credentials."""
        dialog = ui.dialog().props('persistent')  # persistent = cannot close by clicking outside
        with dialog, ui.card().classes('q-pa-lg').style('min-width:380px'):
            ui.label('🔒 Security Setup Required').classes('text-h6 text-bold text-negative')
            ui.label(
                'You are using the default admin credentials. '
                'You must set a unique username and password before continuing.'
            ).classes('text-body2 text-grey-6 q-mb-md')

            new_user_in = ui.input('New Username').props('outlined dense')
            new_pass_in = ui.input('New Password', password=True, password_toggle_button=True
                                   ).props('outlined dense')
            confirm_pass_in = ui.input('Confirm Password', password=True, password_toggle_button=True
                                       ).props('outlined dense')
            status_label = ui.label('').classes('text-caption text-negative')

            def _save():
                uname = new_user_in.value.strip()
                pw = new_pass_in.value
                cpw = confirm_pass_in.value
                if not uname:
                    status_label.set_text('Username cannot be empty.')
                    return
                if uname == 'admin' and pw == 'admin':
                    status_label.set_text('You must choose different credentials.')
                    return
                if len(pw) < 8:
                    status_label.set_text('Password must be at least 8 characters.')
                    return
                if pw != cpw:
                    status_label.set_text('Passwords do not match.')
                    return
                self.db.change_user_password(old_username, uname, pw)
                app.storage.user.update({'username': uname})
                dialog.close()
                ui.notify('✅ Credentials updated — you\'re all set!', type='positive')
                ui.navigate.to('/')

            ui.button('Save & Continue', icon='lock', on_click=_save
                      ).classes('full-width bg-positive text-white q-mt-sm')
        dialog.open()


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
            ui.label('FFTrix NVR').classes('text-h6 text-bold')
            with ui.row().classes('items-center gap-2'):
                role = app.storage.user.get('role', 'admin')
                user_name = app.storage.user.get('username', '')
                ui.badge(role, color='teal' if role == 'admin' else 'grey').props('outline')
                ui.label(user_name).classes('text-caption text-white')
            ui.button('Logout', on_click=self.handle_logout, icon='logout').props('flat color=white')
        with ui.row().classes('w-full no-wrap'):
            self._build_sidebar()
            with ui.column().classes('flex-grow'):
                self._build_grid()
                self._build_health_widget()
            self._build_timeline()
        ui.timer(0.05, self._update_ui_loop)

    def _build_sidebar(self):
        with ui.column().classes('q-pa-md bg-grey-9 rounded shadow-2').style('min-width:260px;max-width:280px'):
            if self.is_admin():
                ui.label('Add Camera').classes('text-subtitle1 text-bold')
                name_in = ui.input('Name').props('outlined dense')
                url_in = ui.input('Source URL / Index').props('outlined dense')
                mode_sel = ui.select(
                    options=['none', 'motion', 'object', 'face', 'edge', 'ocr', 'lpr'],
                    value='none', label='AI Mode'
                ).props('outlined dense')
                ui.button('Add', icon='add',
                          on_click=lambda: self.add_camera_ui(name_in.value, url_in.value, mode_sel.value)
                          ).classes('full-width')
                ui.separator()
            else:
                with ui.row().classes('items-center q-mb-sm'):
                    ui.icon('visibility', color='teal').classes('q-mr-xs')
                    ui.label('Viewer mode — read only').classes('text-caption text-teal')

            ui.label('Cameras').classes('text-subtitle1 text-bold')
            self.camera_list_container = ui.column().classes('w-full')
            self._refresh_camera_list()

            ui.separator()
            ui.label('Selected Camera').classes('text-subtitle1 text-bold')
            self.controls_container = ui.column().classes('w-full')
            self._refresh_controls()

            if self.is_admin():
                ui.separator()
                self._build_discovery_panel()


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
            if self.is_admin():
                # 24/7 recording toggle
                ui.switch('24/7 Record',
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

                # ---- PTZ Controls ----
                xaddr = cam_cfg.get('xaddr', '')
                if xaddr:
                    ui.separator()
                    ui.label('PTZ').classes('text-caption text-bold')
                    self._build_ptz_panel(cam_id, xaddr)

                # ---- Privacy Zones ----
                ui.separator()
                ui.label('Privacy Zones').classes('text-caption text-bold')
                self._build_privacy_panel(cam_id, cam_cfg)

                # ---- Arm Schedule ----
                ui.separator()
                ui.label('Arm Schedule').classes('text-caption text-bold')
                self._build_arm_schedule_panel(cam_id, cam_cfg)

    def _build_ptz_panel(self, cam_id: str, xaddr: str):
        """Render a D-pad + zoom control strip for a PTZ camera."""
        from .ptz import PTZController
        ptz = PTZController(xaddr=xaddr)

        # D-pad grid: up / stop / zoom-in / left / right / zoom-out / down
        with ui.grid(columns=3).classes('gap-0'):
            ui.label('')
            ui.button(icon='arrow_upward',    on_click=lambda: ptz.move('up')).props('dense flat')
            ui.button(icon='zoom_in',         on_click=lambda: ptz.zoom('in')).props('dense flat')
            ui.button(icon='arrow_back',      on_click=lambda: ptz.move('left')).props('dense flat')
            ui.button(icon='stop',            on_click=lambda: ptz.stop()).props('dense flat color=warning')
            ui.button(icon='arrow_forward',   on_click=lambda: ptz.move('right')).props('dense flat')
            ui.label('')
            ui.button(icon='arrow_downward',  on_click=lambda: ptz.move('down')).props('dense flat')
            ui.button(icon='zoom_out',        on_click=lambda: ptz.zoom('out')).props('dense flat')

    def _build_privacy_panel(self, cam_id: str, cam_cfg: dict):
        """Pixel-region privacy zone list with add/remove."""
        zones = list(cam_cfg.get('privacy_zones') or [])
        x1_in = ui.input('x1', value='0').props('outlined dense').style('max-width:60px')
        y1_in = ui.input('y1', value='0').props('outlined dense').style('max-width:60px')
        x2_in = ui.input('x2', value='100').props('outlined dense').style('max-width:60px')
        y2_in = ui.input('y2', value='100').props('outlined dense').style('max-width:60px')

        def _add_zone():
            try:
                zone = [int(x1_in.value), int(y1_in.value), int(x2_in.value), int(y2_in.value)]
                zones.append(zone)
                self.analytics.set_config(cam_id, privacy_zones=zones)
                self.db.update_camera_config(cam_id, privacy_zones=zones)
                ui.notify('Privacy zone added', type='positive')
            except ValueError:
                ui.notify('Enter integer pixel values', type='warning')

        with ui.row():
            x1_in; y1_in; x2_in; y2_in
            ui.button('Add', icon='add', on_click=_add_zone).props('dense')

        if zones:
            for i, z in enumerate(zones):
                with ui.row().classes('items-center'):
                    ui.label(f'Zone {i+1}: {z}').classes('text-caption')
                    ui.button(icon='delete', on_click=lambda _, idx=i: [
                        zones.pop(idx),
                        self.analytics.set_config(cam_id, privacy_zones=zones),
                        self.db.update_camera_config(cam_id, privacy_zones=zones),
                        self._refresh_controls()
                    ]).props('dense flat color=negative')

    def _build_arm_schedule_panel(self, cam_id: str, cam_cfg: dict):
        """Arm schedule: add time-window rules per day-of-week."""
        schedule = list(cam_cfg.get('arm_schedule') or [])
        days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        start_in = ui.input('Start HH:MM', value='08:00').props('outlined dense').style('max-width:90px')
        end_in   = ui.input('End HH:MM',   value='18:00').props('outlined dense').style('max-width:90px')
        day_checks = []
        with ui.row().classes('flex-wrap'):
            for i, d in enumerate(days_of_week):
                cb = ui.checkbox(d, value=(i < 5))  # Mon-Fri default
                day_checks.append(cb)

        def _add_rule():
            selected_days = [i for i, cb in enumerate(day_checks) if cb.value]
            rule = {'days': selected_days, 'start': start_in.value, 'end': end_in.value}
            schedule.append(rule)
            self.analytics.set_config(cam_id, arm_schedule=schedule)
            self.db.update_camera_config(cam_id, arm_schedule=schedule)
            self._refresh_controls()

        ui.button('Add Rule', icon='schedule', on_click=_add_rule).props('dense')

        for i, rule in enumerate(schedule):
            day_labels = [days_of_week[d] for d in rule.get('days', [])]
            desc = f"{', '.join(day_labels)} {rule.get('start')}–{rule.get('end')}"
            with ui.row().classes('items-center'):
                ui.label(desc).classes('text-caption')
                ui.button(icon='delete', on_click=lambda _, idx=i: [
                    schedule.pop(idx),
                    self.analytics.set_config(cam_id, arm_schedule=schedule),
                    self.db.update_camera_config(cam_id, arm_schedule=schedule),
                    self._refresh_controls()
                ]).props('dense flat color=negative')


    def _build_grid(self):
        self.grid_container = ui.grid(columns=self.grid_columns).classes('flex-grow')
        self._refresh_grid()

    def _build_discovery_panel(self):
        """Render the ONVIF auto-discovery section in the sidebar."""
        ui.label('ONVIF Discovery').classes('text-subtitle1 text-bold')
        self._disc_user_in = ui.input('Username (opt.)').props('outlined dense')
        self._disc_pass_in = ui.input('Password (opt.)', password=True).props('outlined dense')
        self._disc_timeout_sel = ui.select(
            options=[3, 5, 10, 15], value=5, label='Timeout (s)'
        ).props('outlined dense')
        with ui.row().classes('full-width'):
            self._disc_scan_btn = ui.button(
                'Scan Network', icon='radar',
                on_click=self._run_discovery_scan
            ).classes('flex-grow')
            ui.button(icon='clear', on_click=self._clear_discovery).props('flat dense')
        self.discovery_list_container = ui.column().classes('w-full')
        self._refresh_discovery_list()

    def _build_recordings_panel(self):
        """Browse and export recordings for the selected camera (admin only)."""
        if not self.is_admin():
            return
        ui.separator()
        ui.label('Recordings').classes('text-subtitle1 text-bold')
        if not self.selected_cam_id or self.selected_cam_id not in self.cameras:
            ui.label('Select a camera first.').classes('text-caption text-grey')
            return
        cam_id = self.selected_cam_id
        exporter = ClipExporter(recordings_root=RECORDINGS_DIR)
        segments = exporter.list_segments(cam_id)
        if not segments:
            ui.label('No recordings found.').classes('text-caption text-grey')
            return
        for seg in segments[-10:][::-1]:  # show latest 10
            size_mb = round(seg['size'] / 1_048_576, 2)
            import datetime
            ts = datetime.datetime.fromtimestamp(seg['mtime']).strftime('%Y-%m-%d %H:%M')
            with ui.row().classes('items-center full-width'):
                ui.label(f"{ts}  {size_mb} MB").classes('text-caption flex-grow')
                ui.button(icon='download', on_click=lambda s=seg: [
                    exporter.export_async(cam_id, start_ts=s['mtime'] - 1, end_ts=s['mtime'] + 1,
                                         callback=lambda p: ui.notify(
                                             f'Exported: {Path(p).name}' if p else 'Export failed',
                                             type='positive' if p else 'negative'))
                ]).props('flat dense')
                ui.button(icon='delete', on_click=lambda s=seg: [
                    exporter.delete_segment(s['path']),
                    self._refresh_controls()
                ]).props('flat dense color=negative')

    def _build_health_widget(self):
        """Row of per-camera health stat cards shown below the video grid."""
        self._health_container = ui.row().classes('q-pa-sm gap-2 flex-wrap')
        ui.timer(2.0, self._refresh_health)

    def _refresh_health(self):
        if not hasattr(self, '_health_container'):
            return
        self._health_container.clear()
        with self._health_container:
            for cam_id, node in self.cameras.items():
                h = node.get_health()
                color = 'teal' if h['running'] else 'red'
                with ui.card().classes('q-pa-xs'):
                    with ui.row().classes('items-center gap-1'):
                        ui.icon('circle', color=color, size='xs')
                        ui.label(h['name']).classes('text-caption text-bold')
                    ui.label(f"{h['fps']} fps").classes('text-caption')
                    ui.label(f"Up {h['uptime_s']}s").classes('text-caption text-grey')
                    ui.label(f"{h['frames_processed']} frames").classes('text-caption text-grey')
                    if h['dropped_frames']:
                        ui.label(f"{h['dropped_frames']} dropped").classes('text-caption text-warning')


    def _run_discovery_scan(self):
        """Launch a WS-Discovery scan in a background thread."""
        if self._discovery_scanning:
            return
        self._discovery_scanning = True
        if hasattr(self, '_disc_scan_btn'):
            self._disc_scan_btn.props('loading')

        username = getattr(self, '_disc_user_in', None)
        password = getattr(self, '_disc_pass_in', None)
        timeout = getattr(self, '_disc_timeout_sel', None)
        u = username.value if username else ''
        p = password.value if password else ''
        t = timeout.value if timeout else 5

        def _scan_worker():
            discovery = ONVIFDiscovery()
            results = discovery.scan(timeout=t, username=u, password=p)
            # Cache all results in DB
            for dev in results:
                self.db.upsert_discovered_device(dev.to_dict())
            self._discovered = results
            self._discovery_scanning = False
            if hasattr(self, '_disc_scan_btn'):
                self._disc_scan_btn.props(remove='loading')
            self._refresh_discovery_list()

        threading.Thread(target=_scan_worker, daemon=True).start()

    def _clear_discovery(self):
        """Clear scan results from memory and DB cache."""
        self._discovered = []
        self.db.clear_discovered_devices()
        self._refresh_discovery_list()

    def _refresh_discovery_list(self):
        """Redraw the discovery results list."""
        if not hasattr(self, 'discovery_list_container') or self.discovery_list_container is None:
            return
        self.discovery_list_container.clear()
        # Show cached DB results if no live results yet
        devices = self._discovered or [
            DiscoveredDevice.from_dict(d) for d in self.db.get_discovered_devices()
        ]
        with self.discovery_list_container:
            if not devices:
                ui.label('No devices found. Run a scan.').classes('text-caption text-grey')
                return
            for dev in devices:
                with ui.card().classes('w-full q-mb-xs'):
                    with ui.row().classes('items-center'):
                        icon = 'videocam' if not dev.requires_auth else 'lock'
                        ui.icon(icon).classes('text-primary')
                        with ui.column().classes('flex-grow'):
                            ui.label(dev.name).classes('text-caption text-bold')
                            ui.label(dev.ip).classes('text-caption text-grey')
                            if dev.requires_auth:
                                ui.label('⚠ Requires credentials').classes('text-caption text-warning')
                        ui.button(icon='add', on_click=lambda d=dev: self.add_from_discovery(d)
                                  ).props('flat dense color=positive')

    def add_from_discovery(self, device: DiscoveredDevice):
        """Pre-fill add_camera_ui with the best RTSP URI from a discovered device."""
        rtsp = device.rtsp_uris[0] if device.rtsp_uris else device.xaddr
        self.add_camera_ui(device.name, rtsp, 'motion')

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
                    snap = ev.get('snapshot_path')
                    if snap and Path(snap).exists():
                        ui.image(snap).classes('w-full rounded').style('max-height:120px;object-fit:cover')

    async def _update_ui_loop(self):
        self._timeline_tick += 1
        # Refresh timeline every ~2 s (timer fires at 0.05s, so every 40 ticks)
        if self._timeline_tick % 40 == 0:
            self._refresh_timeline()

        for cam_id, node in self.cameras.items():
            if hasattr(node, 'ui_image') and getattr(node, 'processed_frame', None) is not None:
                _, buf = cv2.imencode('.jpg', node.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                node.ui_image.set_source(f'data:image/jpeg;base64,{base64.b64encode(buf).decode("utf-8")}')


def _write_pwa_manifest(port: int) -> str:
    """Write manifest.json to STATIC_DIR and return its public URL."""
    import json as _json
    manifest = {
        "name": "FFTrix NVR",
        "short_name": "FFTrix",
        "description": "Intelligent Network Video Recorder",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#121212",
        "theme_color": "#009688",
        "orientation": "any",
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
        "screenshots": [],
    }
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = STATIC_DIR / 'manifest.json'
    manifest_path.write_text(_json.dumps(manifest, indent=2))
    return '/static/manifest.json'


def _add_pwa_head_tags():
    """Inject PWA and mobile-optimised <head> tags into the NiceGUI page."""
    ui.add_head_html('<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">', shared=True)
    ui.add_head_html('<meta name="mobile-web-app-capable" content="yes">', shared=True)
    ui.add_head_html('<meta name="apple-mobile-web-app-capable" content="yes">', shared=True)
    ui.add_head_html('<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">', shared=True)
    ui.add_head_html('<meta name="apple-mobile-web-app-title" content="FFTrix">', shared=True)
    ui.add_head_html('<meta name="theme-color" content="#009688">', shared=True)
    ui.add_head_html('<link rel="manifest" href="/static/manifest.json">', shared=True)
    ui.add_head_html(
        '<style>'
        'body{-webkit-text-size-adjust:100%;}'
        '@media(max-width:768px){'
        '.q-drawer{width:100%!important;}'
        '.nicegui-content{flex-direction:column!important;}'
        '}'
        '</style>', shared=True
    )


def _generate_angie_config(port: int):
    """Generate a secure Angie (NGINX) configuration template for ingress."""
    config = f"""# Angie (NGINX) Configuration for FFTrix Ingress
server {{
    listen 80;
    server_name _;

    # Increase body size for potential large file uploads/API requests
    client_max_body_size 50M;

    location / {{
        proxy_pass http://127.0.0.1:{port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (Critical for NiceGUI/FastAPI)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Security headers
        add_header X-Frame-Options "SAMEORIGIN";
        add_header X-XSS-Protection "1; mode=block";
        add_header X-Content-Type-Options "nosniff";
    }}
}}
"""
    from .database import FFTRIX_HOME
    conf_path = FFTRIX_HOME / "angie.conf"
    if not conf_path.exists():
        conf_path.write_text(config)
        print(f"✅ Generated {conf_path} for Angie ingress.")


def _setup_remote_access(port: int):
    """Securely expose the dashboard via Tailscale Funnel and Angie."""
    # 1. Generate Angie config if it doesn't exist
    _generate_angie_config(port)

    # 2. Attempt to start Tailscale Funnel (Secure Egress/Tunnel)
    if shutil.which("tailscale"):
        try:
            # Optional: Automatic authentication if TS_AUTHKEY is provided
            auth_key = os.environ.get("TS_AUTHKEY")
            if auth_key:
                subprocess.run(["tailscale", "up", "--authkey", auth_key], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # We run it in the background. 
            # Note: In production, users should manage this via systemd or similar.
            subprocess.Popen(["tailscale", "funnel", str(port)],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            print(f"🚀 Tailscale Funnel active on port {port}")
        except Exception as e:
            warnings.warn(f"Tailscale Funnel failed: {e}")
    else:
        warnings.warn("Tailscale CLI not found. Please install Tailscale for secure remote access.")

    # 3. Inform user about Angie
    if shutil.which("angie"):
        from .database import FFTRIX_HOME
        print(f"🛡️ Angie detected. Recommend running: 'sudo angie -c {FFTRIX_HOME}/angie.conf'")


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
        _setup_remote_access(port)

    db = Database()
    alert_manager = AlertManager()
    analytics = AnalyticsPipeline(db=db, alert_manager=alert_manager, snapshots_dir=SNAPSHOTS_DIR)
    dashboard = Dashboard(db=db, analytics=analytics)

    # Write PWA manifest and inject head tags
    _write_pwa_manifest(port)
    app.add_static_files('/static', str(STATIC_DIR))
    _add_pwa_head_tags()

    # Start nightly retention cleanup
    retention = RetentionManager(db=db, recordings_root=RECORDINGS_DIR)
    retention.start()

    # reload=False is critical when running as an installed package to avoid RuntimeError
    ui.run(title="FFTrix NVR", port=port, host=host, storage_secret=secret,
           reload=False, show=False, dark=True)
