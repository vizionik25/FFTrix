import cv2
import numpy as np
import base64
import os
import time
import json
from nicegui import ui, app
from fastapi.responses import RedirectResponse
from .database import Database
from .engine import CameraNode
from .analytics import AnalyticsPipeline

class Dashboard:
    def __init__(self):
        self.db = Database()
        self.analytics = AnalyticsPipeline()
        self.cameras = {} 
        self.recording_dir = os.path.join(os.getcwd(), 'recordings')
        self.grid_columns = 2
        
        # UI State
        self.selected_cam_id = None
        self.is_drawing_zone = False
        self.temp_zone_start = None
        
        # Serve recordings statically if needed
        app.add_static_files('/recordings', self.recording_dir)
        
        # Load existing configuration from Database
        for c in self.db.get_cameras():
            self._init_camera(c['id'], c['name'], c['url'], c['mode'], 
                              c['zones'], c['record_247'], c['watermark'], start=False)

    def _init_camera(self, cam_id, name, url, mode, zones, record_247, watermark, start=True):
        """Standard camera node initialization."""
        self.analytics.set_config(cam_id, mode, zones, watermark)
        node = CameraNode(cam_id, url, name, self.analytics, self.db, record_247=record_247)
        self.cameras[cam_id] = node
        if start: node.start()

    # --- AUTHENTICATION LOGIC ---
    def handle_login(self, username, password):
        if self.db.verify_user(username, password):
            app.storage.user.update({'authenticated': True, 'username': username})
            ui.navigate.to('/')
        else:
            ui.notify('Invalid credentials', type='negative')

    def handle_logout(self):
        app.storage.user.update({'authenticated': False})
        ui.navigate.to('/login')

    # --- ZONING & WATERMARKS ---
    def handle_viewport_click(self, e):
        """Zoning logic for security boundaries."""
        if not self.selected_cam_id or not self.is_drawing_zone: return
        x, y = int(e.image_x), int(e.image_y)
        if self.temp_zone_start is None:
            self.temp_zone_start = (x, y)
            ui.notify(f"Point A set at {x},{y}")
        else:
            x1, y1 = self.temp_zone_start
            new_zone = [min(x, x1), min(y, y1), abs(x - x1), abs(y - y1)]
            current_zones = self.analytics.zones.get(self.selected_cam_id, [])
            current_zones.append(new_zone)
            self.analytics.zones[self.selected_cam_id] = current_zones
            self.db.update_camera_config(self.selected_cam_id, zones=current_zones)
            self.temp_zone_start = None
            self.is_drawing_zone = False
            ui.notify("Security Zone Deployed", type='positive')

    def update_watermark(self, cam_id, key, value):
        """Real-time watermark configuration sync."""
        config = self.analytics.watermarker.configs.get(cam_id, {
            'text': '', 'image_path': '', 'mode': 'static', 'x': 50, 'y': 50, 'transparency': 0.5
        })
        config[key] = value
        self.analytics.watermarker.set_config(cam_id, config)
        self.db.update_camera_config(cam_id, watermark=config)

    def add_camera_ui(self, name, url, mode):
        if not name or not url: return
        cam_id = f"C{len(self.cameras)+1:02d}"
        self.db.add_camera(cam_id, name, url, mode)
        self._init_camera(cam_id, name, url, mode, [], False, {}, start=True)
        ui.notify(f"Node {name} provisioned", type='positive')
        self.build_grid.refresh()

    def delete_camera(self, cam_id):
        if cam_id in self.cameras:
            self.cameras[cam_id].stop()
            del self.cameras[cam_id]
            self.db.remove_camera(cam_id)
            ui.notify(f"Camera {cam_id} decommissioned", type='info')
            self.selected_cam_id = None
            self.build_grid.refresh()

    @ui.refreshable
    def build_grid(self):
        """Fixed-size Surveillance Matrix (1280x720 locked)."""
        if not self.cameras:
            with ui.column().classes('w-full h-full items-center justify-center'):
                ui.icon('security', size='100px').classes('text-slate-700')
                ui.label("ENTERPRISE CORE OFFLINE").classes('text-2xl text-slate-600 font-black')
            return

        with ui.grid(columns=self.grid_columns).classes('w-full h-full gap-1 p-1 bg-black'):
            for cam_id, node in self.cameras.items():
                border = 'border-2 border-blue-500' if cam_id == self.selected_cam_id else 'border border-slate-800'
                with ui.card().classes(f'p-0 bg-slate-900 relative rounded-none {border} overflow-hidden cursor-pointer').on('click', lambda c=cam_id: setattr(self, 'selected_cam_id', c) or self.build_sidebar.refresh()):
                    img = ui.interactive_image().style('width: 100%; height: 100%; object-fit: contain;')
                    img.on('click', self.handle_viewport_click)
                    
                    # Mini-HUD
                    with ui.row().classes('absolute bottom-0 left-0 w-full bg-black/60 p-1 justify-between items-center'):
                        ui.label(node.name).classes('text-[10px] font-bold truncate ml-1 text-slate-300')
                        with ui.row().classes('gap-2 mr-1'):
                            if node.record_247: ui.icon('fiber_smart_record', size='14px').classes('text-red-500')
                            ui.button(icon='edit_location', on_click=lambda c=cam_id: setattr(self, 'selected_cam_id', c) or setattr(self, 'is_drawing_zone', True)).props('flat dense size=xs color=cyan')
                    
                    node.ui_image = img

    @ui.refreshable
    def build_timeline(self):
        """Flagged Trigger Timeline."""
        flags = self.db.get_events(20, flags_only=True)
        if not flags:
            ui.label("System Secure. No Incidents.").classes('text-slate-500 text-xs italic')
        for f in flags:
            with ui.row().classes('w-full items-center bg-red-950/20 p-2 rounded mb-2 border-l-4 border-red-600'):
                ui.icon('flag', size='xs').classes('text-red-500')
                with ui.column().classes('gap-0'):
                    ui.label(f"{f['source']} @ {f['timestamp']}").classes('text-[10px] font-black text-red-200')
                    ui.label(f['details']).classes('text-[9px] text-slate-400')

    async def update_ui_loop(self):
        """High-speed MJPEG streaming loop."""
        for cam_id, node in self.cameras.items():
            if hasattr(node, 'ui_image') and getattr(node, 'processed_frame', None) is not None:
                _, buffer = cv2.imencode('.jpg', node.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                encoded = base64.b64encode(buffer).decode('utf-8')
                node.ui_image.set_source(f'data:image/jpeg;base64,{encoded}')
        
        if int(time.time() * 10) % 30 == 0:
            self.build_timeline.refresh()

    @ui.refreshable
    def build_sidebar(self):
        with ui.column().classes('w-80 bg-slate-950 h-full border-r border-slate-800 p-4 gap-4 overflow-y-auto flex-shrink-0'):
            with ui.expansion('PROVISIONING', icon='add_circle').classes('w-full bg-slate-900/50 rounded'):
                name = ui.input('Name').props('dark dense')
                url = ui.input('URL').props('dark dense')
                mode = ui.select({'none':'Raw', 'motion':'Motion', 'face':'Faces', 'object':'AI Personnel'}, value='none').props('dark dense')
                ui.button('DEPLOY NODE', on_click=lambda: self.add_camera_ui(name.value, url.value, mode.value)).classes('w-full mt-2 bg-blue-900')

            ui.separator().classes('border-slate-800')

            if self.selected_cam_id and self.selected_cam_id in self.cameras:
                cam = self.cameras[self.selected_cam_id]
                wm = self.analytics.watermarker.configs.get(self.selected_cam_id, {})
                
                with ui.column().classes('w-full gap-2 p-3 bg-slate-900 rounded border border-blue-900/50 shadow-inner'):
                    ui.label(f'TARGET: {cam.name}').classes('text-xs font-black text-blue-400 tracking-widest')
                    
                    ui.switch('24/7 Continuous DVR', value=cam.record_247, on_change=lambda e: self.db.update_camera_config(self.selected_cam_id, record_247=e.value)).props('dark size=sm')
                    
                    with ui.row().classes('w-full gap-2'):
                        ui.button('DRAW ZONE', icon='crop_free', on_click=lambda: setattr(self, 'is_drawing_zone', True)).props('outline size=sm color=cyan').classes('flex-grow')
                        ui.button('CLEAR ZONES', icon='layers_clear', on_click=lambda: self.db.update_camera_config(self.selected_cam_id, zones=[]) or self.analytics.zones.__setitem__(self.selected_cam_id, [])).props('outline size=sm color=red')
                    
                    ui.separator().classes('my-2 border-slate-800')
                    
                    ui.label('WATERMARK ENGINE').classes('text-[10px] font-bold text-slate-500 tracking-widest')
                    ui.input('Branding Text', value=wm.get('text',''), on_change=lambda e: self.update_watermark(self.selected_cam_id, 'text', e.value)).props('dark dense filled').classes('w-full')
                    ui.input('Logo Path (PNG)', value=wm.get('image_path',''), on_change=lambda e: self.update_watermark(self.selected_cam_id, 'image_path', e.value)).props('dark dense filled').classes('w-full')
                    
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label('Opacity').classes('text-[10px] text-slate-400')
                        ui.slider(min=0.1, max=1.0, step=0.1, value=wm.get('transparency', 0.5), on_change=lambda e: self.update_watermark(self.selected_cam_id, 'transparency', e.value)).props('dark dense').classes('w-32')
                    
                    ui.select({'static':'Static (Manual)', 'floating':'Floating (Bouncing)'}, label="Behavior", value=wm.get('mode','static'), 
                              on_change=lambda e: self.update_watermark(self.selected_cam_id, 'mode', e.value)).props('dark dense outlined').classes('w-full mt-2')
                    
                    ui.button('DELETE NODE', icon='delete', on_click=lambda: self.delete_camera(self.selected_cam_id)).props('flat size=sm color=negative').classes('w-full mt-4')

            ui.separator().classes('border-slate-800')
            ui.label('MATRIX CONFIG').classes('text-[10px] font-bold text-slate-500 tracking-widest')
            ui.slider(min=1, max=4, value=2, on_change=lambda e: setattr(self, 'grid_columns', e.value) or self.build_grid.refresh()).props('dark color=blue markers snap')
            with ui.row().classes('w-full justify-between px-2 text-[10px] text-slate-500 font-mono'):
                ui.label('1x1')
                ui.label('4x4')

    @ui.page('/login')
    def login_page(self):
        if app.storage.user.get('authenticated', False):
            return RedirectResponse('/')
            
        ui.query('body').style('background-color: #020617; display: flex; justify-content: center; align-items: center; height: 100vh;')
        
        with ui.card().classes('w-96 p-8 bg-slate-900 border border-slate-800 shadow-2xl items-center rounded-xl'):
            ui.icon('security', size='xl').classes('text-blue-600 mb-2')
            ui.label('FFTRIX SECURE GATEWAY').classes('text-xl font-black tracking-widest text-slate-100 mb-8 text-center')
            
            user = ui.input('Operator ID').props('dark filled').classes('w-full mb-4')
            pw = ui.input('Passcode', password=True).props('dark filled').classes('w-full mb-8').on('keydown.enter', lambda: self.handle_login(user.value, pw.value))
            
            ui.button('AUTHORIZE', on_click=lambda: self.handle_login(user.value, pw.value)).classes('w-full bg-blue-700 hover:bg-blue-600 font-bold py-3 text-lg')
            ui.label('RESTRICTED ACCESS').classes('text-[10px] text-red-900/50 mt-8 tracking-[0.4em] font-black')

    @ui.page('/')
    def main_page(self):
        if not app.storage.user.get('authenticated', False):
            return RedirectResponse('/login')
            
        ui.query('body').style('background-color: #020617; color: #f8fafc; overflow: hidden; margin: 0; padding: 0;')
        
        with ui.header().classes('bg-slate-950 border-b border-slate-800 py-3 px-6 justify-between items-center shadow-2xl z-50'):
            with ui.row().classes('items-center gap-4'):
                ui.icon('camera_outdoor', size='md').classes('text-blue-500')
                ui.label('FFTrix PRO').classes('text-xl font-black tracking-[0.2em]')
                ui.label('ENTERPRISE EDITION').classes('text-[9px] bg-blue-900/30 text-blue-400 px-2 py-1 rounded font-bold tracking-widest ml-2')
            
            with ui.row().classes('gap-6 items-center'):
                ui.button('ENGAGE ALL', on_click=lambda: [n.start() for n in self.cameras.values()]).props('outline size=sm color=green')
                ui.button('HALT ALL', on_click=lambda: [n.stop() for n in self.cameras.values()]).props('outline size=sm color=red')
                ui.separator().props('vertical dark').classes('h-6 mx-2')
                ui.icon('account_circle', size='sm').classes('text-slate-500')
                ui.label(f"{app.storage.user.get('username', 'OPERATOR').upper()}").classes('text-xs font-mono text-slate-300')
                ui.button(icon='logout', on_click=self.handle_logout).props('flat size=sm color=slate-500 hover:text-white')

        with ui.row().classes('w-full h-[calc(100vh-64px)] no-wrap'):
            self.build_sidebar()
            
            with ui.column().classes('flex-grow bg-black overflow-hidden relative shadow-inner'):
                self.build_grid()

            with ui.column().classes('w-72 bg-slate-950 h-full border-l border-slate-800 p-4 overflow-y-auto flex-shrink-0'):
                ui.label('INCIDENT TIMELINE').classes('text-[10px] font-black text-red-500 tracking-[0.3em] mb-4')
                self.build_timeline()

        ui.timer(0.05, self.update_ui_loop)

def run_dashboard(remote=False):
    if remote:
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(8080)
            print(f"\n{'='*70}")
            print(f"🌍 SECURE REMOTE ACCESS DEPLOYED")
            print(f"🔗 Public URL: {public_url.public_url}")
            print(f"🔒 Gateway requires operator authentication.")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"Failed to deploy secure tunnel: {e}")

    dashboard = Dashboard()
    ui.run(title="FFTrix Enterprise", port=8080, host='0.0.0.0', storage_secret='secure_surveillance_secret_key_2024', dark=True)

if __name__ in {"__main__", "__mp_main__"}:
    run_dashboard()
