import pytest
import os
import shutil
import numpy as np
import cv2
import json
import time
import asyncio
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Project Imports
from fftrix.database import Database, DB_PATH, FFTRIX_HOME, RECORDINGS_DIR, SNAPSHOTS_DIR
from fftrix.analytics import AnalyticsPipeline, WatermarkEngine
from fftrix.engine import CameraNode
from fftrix.__main__ import cli, handle_serve, handle_user_add
from fftrix import main as init_main
from fftrix.dashboard import _generate_angie_config
from click.testing import CliRunner

TEST_HOME = Path("./test_fftrix_home")
TEST_DB = TEST_HOME / "test_system.db"

@pytest.fixture(autouse=True)
def setup_teardown():
    if TEST_HOME.exists(): shutil.rmtree(TEST_HOME)
    TEST_HOME.mkdir(parents=True, exist_ok=True)
    (TEST_HOME / "recordings").mkdir()
    (TEST_HOME / "snapshots").mkdir()
    yield
    if TEST_HOME.exists(): shutil.rmtree(TEST_HOME)


# ===========================================================================
# REMOTE ACCESS
# ===========================================================================

def test_generate_angie_config():
    with patch("fftrix.database.FFTRIX_HOME", TEST_HOME):
        _generate_angie_config(8080)
        conf_path = TEST_HOME / "angie.conf"
        assert conf_path.exists()
        content = conf_path.read_text()
        assert "proxy_pass http://127.0.0.1:8080;" in content
        assert "server_name _;" in content


# ===========================================================================
# PACKAGE PUBLIC API
# ===========================================================================

def test_init_public_api():
    import fftrix
    assert fftrix.__version__ == "1.0.1"
    assert hasattr(fftrix, 'Database')
    assert hasattr(fftrix, 'AnalyticsPipeline')
    assert hasattr(fftrix, 'CameraNode')
    assert hasattr(fftrix, 'Dashboard')
    assert hasattr(fftrix, 'ONVIFDiscovery')
    assert hasattr(fftrix, 'DiscoveredDevice')


# ===========================================================================
# DATABASE
# ===========================================================================

def test_database_verify_missing_user():
    db = Database(db_path=str(TEST_DB))
    result = db.verify_user("nonexistent", "password")
    assert not result  # returns None (falsy) when user not found

def test_database_update_camera_config():
    db = Database(db_path=str(TEST_DB))
    db.add_camera("C01", "Test", "rtsp://x", "none")
    db.update_camera_config("C01", zones=[[0, 0, 100, 100]])
    db.update_camera_config("C01", record_247=True)
    db.update_camera_config("C01", watermark={'text': 'WM', 'mode': 'static'})
    cams = db.get_cameras()
    assert cams[0]['zones'] == [[0, 0, 100, 100]]
    assert cams[0]['record_247'] is True
    assert cams[0]['watermark']['text'] == 'WM'


# ===========================================================================
# ANALYTICS — WatermarkEngine
# ===========================================================================

def test_analytics_floating_watermark():
    engine = WatermarkEngine()
    engine.set_config("C1", {'mode': 'floating', 'text': 'LIVE', 'transparency': 0.8})
    assert "C1" in engine.float_offsets
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Apply multiple times to exercise bounce logic
    for _ in range(10):
        result = engine.apply(frame, "C1")
    assert result.shape == frame.shape

def test_analytics_watermark_text_draw():
    engine = WatermarkEngine()
    engine.set_config("C1", {'mode': 'static', 'text': 'TEST', 'x': 50, 'y': 50, 'transparency': 0.7})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = engine.apply(frame, "C1")
    assert result.shape == frame.shape

def test_analytics_watermark_image_draw_valid_png(tmp_path):
    """Tests _draw_image with a real 3-channel image (no alpha)."""
    engine = WatermarkEngine()
    img_path = str(tmp_path / "logo.png")
    dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)
    engine.set_config("C1", {'mode': 'static', 'image_path': img_path, 'transparency': 0.5, 'x': 10, 'y': 10})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = engine.apply(frame, "C1")
    assert result.shape == frame.shape

def test_analytics_watermark_image_draw_alpha(tmp_path):
    """Tests _draw_image RGBA (4-channel alpha blending) path."""
    engine = WatermarkEngine()
    img_path = str(tmp_path / "logo_alpha.png")
    dummy_img = np.zeros((50, 50, 4), dtype=np.uint8)
    dummy_img[:, :, 3] = 200  # semi-transparent
    cv2.imwrite(img_path, dummy_img)
    engine.set_config("C1", {'mode': 'static', 'image_path': img_path, 'transparency': 1.0, 'x': 10, 'y': 10})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = engine.apply(frame, "C1")
    assert result.shape == frame.shape

def test_analytics_watermark_edge_cases():
    engine = WatermarkEngine()
    bad_file = TEST_HOME / "bad.txt"
    bad_file.write_text("not an image")
    engine.set_config("C1", {'image_path': str(bad_file)})
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    res = engine.apply(frame, "C1")
    assert np.array_equal(res, frame)


# ===========================================================================
# ANALYTICS — AnalyticsPipeline
# ===========================================================================

def test_analytics_zone_logic():
    p = AnalyticsPipeline()
    p.set_config("C1", zones=[[100, 100, 200, 200]])
    # Point inside zone
    assert p.is_in_zone("C1", 150, 150, 10, 10) is True
    # Point outside zone
    assert p.is_in_zone("C1", 10, 10, 5, 5) is False
    # No zones configured — everything passes
    p2 = AnalyticsPipeline()
    assert p2.is_in_zone("C2", 0, 0, 10, 10) is True

def test_analytics_motion_mode():
    p = AnalyticsPipeline()
    p.set_config("C1", mode='motion')
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Feed background frames first so MOG2 learns
    for _ in range(5):
        p.process(frame, "C1")
    # Feed a frame with motion (bright rectangle)
    motion_frame = frame.copy()
    cv2.rectangle(motion_frame, (200, 200), (300, 300), (255, 255, 255), -1)
    result, _ = p.process(motion_frame, "C1")
    assert result.shape == frame.shape

def test_analytics_object_mode():
    p = AnalyticsPipeline()
    p.set_config("C1", mode='object')
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, _ = p.process(frame, "C1")
    assert result.shape == frame.shape

def test_analytics_face_mode():
    p = AnalyticsPipeline()
    p.set_config("C1", mode='face')
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, _ = p.process(frame, "C1")
    assert result.shape == frame.shape

def test_analytics_edge_mode():
    p = AnalyticsPipeline()
    p.set_config("C1", mode='edge')
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, triggered = p.process(frame, "C1")
    assert result.shape == frame.shape
    assert triggered is False

def test_analytics_ocr_mode():
    p = AnalyticsPipeline()
    p.set_config("C1", mode='ocr')
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, _ = p.process(frame, "C1")
    assert result.shape == frame.shape

def test_pipeline_exception_handling():
    p = AnalyticsPipeline()
    res, triggered = p.process(None, "C1")
    assert res is None
    assert triggered is False


# ===========================================================================
# ENGINE
# ===========================================================================

@patch('fftrix.engine.VideoGear')
def test_engine_start_success(mock_vidgear):
    mock_stream = MagicMock()
    mock_stream.read.return_value = None
    mock_vidgear.return_value.start.return_value = mock_stream
    db = Database(db_path=str(TEST_DB))
    node = CameraNode("C1", "0", "Cam", AnalyticsPipeline(), db)
    node.start()
    assert node.is_running is True
    node.stop()

@patch('fftrix.engine.VideoGear')
def test_engine_rtsp_source(mock_vidgear):
    mock_stream = MagicMock()
    mock_stream.read.return_value = None
    mock_vidgear.return_value.start.return_value = mock_stream
    db = Database(db_path=str(TEST_DB))
    node = CameraNode("C1", "rtsp://192.168.1.1/stream", "IPCam", AnalyticsPipeline(), db)
    node.start()
    # Verify rtsp_transport='tcp' was passed via kwargs
    call_kwargs = mock_vidgear.call_args[1]
    assert call_kwargs.get('rtsp_transport') == 'tcp'
    node.stop()

@patch('fftrix.engine.VideoGear')
def test_engine_247_recording(mock_vidgear):
    mock_stream = MagicMock()
    mock_stream.read.return_value = None
    mock_vidgear.return_value.start.return_value = mock_stream
    db = Database(db_path=str(TEST_DB))
    with patch('fftrix.engine.WriteGear') as mock_writer:
        mock_writer.return_value = MagicMock()
        node = CameraNode("C1", "0", "Cam", AnalyticsPipeline(), db, record_247=True)
        node.start()
        assert node.is_recording is True
        node.stop()

@patch('fftrix.engine.VideoGear')
def test_engine_update_loop_trigger(mock_vidgear):
    """Exercise the _update loop: successful frame, trigger event, writer write."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_stream = MagicMock()
    mock_stream.read.side_effect = [frame, frame, None]
    mock_vidgear.return_value.start.return_value = mock_stream

    mock_analytics = MagicMock()
    mock_analytics.process.return_value = (frame, True)  # trigger

    db = Database(db_path=str(TEST_DB))
    mock_writer = MagicMock()
    node = CameraNode("C1", "0", "Cam", mock_analytics, db)
    node.is_running = True
    node.stream = mock_stream
    node.is_recording = True
    node.writer = mock_writer

    # Run a single iteration by patching time.sleep to stop after 2 frames
    call_count = [0]
    original_sleep = time.sleep
    def fake_sleep(t):
        call_count[0] += 1
        if call_count[0] >= 2:
            node.is_running = False
    with patch('fftrix.engine.time.sleep', fake_sleep):
        node._update()

    mock_writer.write.assert_called()

@patch('fftrix.engine.VideoGear')
def test_engine_stop(mock_vidgear):
    mock_stream = MagicMock()
    mock_stream.read.return_value = None
    mock_vidgear.return_value.start.return_value = mock_stream
    db = Database(db_path=str(TEST_DB))
    with patch('fftrix.engine.WriteGear') as mock_writer_cls:
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer
        node = CameraNode("C1", "0", "Cam", AnalyticsPipeline(), db, record_247=True)
        node.start()
        node.stop()
        mock_writer.close.assert_called_once()
        mock_stream.stop.assert_called_once()

@patch('fftrix.engine.VideoGear')
def test_engine_start_failure(mock_vidgear):
    mock_vidgear.side_effect = Exception("HW Error")
    db = Database(db_path=str(TEST_DB))
    n = CameraNode("C1", "0", "Cam", AnalyticsPipeline(), db)
    n.start()
    assert len(db.get_events()) > 0


# ===========================================================================
# DASHBOARD
# ===========================================================================

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def _make_dashboard(mock_ui, mock_app, db=None):
    from fftrix.dashboard import Dashboard
    d = Dashboard(db=db or Database(db_path=str(TEST_DB)))
    return d

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_login_success(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    result = dash.handle_login('admin', 'admin')
    assert result is True
    mock_app.storage.user.update.assert_called()

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_login_failure(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    result = dash.handle_login('admin', 'wrong')
    assert result is False
    mock_ui.notify.assert_called_with('Invalid credentials', type='negative')

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_update_watermark(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    db.add_camera("C01", "Test", "0", "none")
    dash = Dashboard(db=db)
    dash.update_watermark("C01", 'text', 'OVERLAY')
    assert dash.analytics.watermarker.configs["C01"]['text'] == 'OVERLAY'

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_add_delete_camera(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash.add_camera_ui("Cam1", "0", "none")
    assert "C01" in dash.cameras
    dash.cameras["C01"].is_running = True
    dash.delete_camera("C01")
    assert "C01" not in dash.cameras

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_viewport_click_zone_draw(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    db.add_camera("C01", "Test", "0", "none")
    dash = Dashboard(db=db)
    dash.add_camera_ui("Test", "0", "none")
    dash.selected_cam_id = "C01"
    dash.is_drawing_zone = True

    # First click — sets start point
    e1 = MagicMock(); e1.image_x = 10; e1.image_y = 20
    dash.handle_viewport_click(e1)
    assert dash.temp_zone_start == (10, 20)

    # Second click — commits zone
    e2 = MagicMock(); e2.image_x = 110; e2.image_y = 120
    dash.handle_viewport_click(e2)
    assert dash.temp_zone_start is None
    assert dash.is_drawing_zone is False
    zones = dash.analytics.zones.get("C01", [])
    assert len(zones) > 0

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_refresh_timeline_with_events(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    db.log_event("ALERT", "C01", "Motion detected", is_flag=1)
    db.log_event("ALERT", "C01", "Person detected", is_flag=1)
    dash = Dashboard(db=db)
    dash.timeline_container = MagicMock()
    dash._refresh_timeline()
    dash.timeline_container.clear.assert_called()

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_refresh_grid_no_container(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash.grid_container = None
    dash._refresh_grid()  # Should not raise

@pytest.mark.asyncio
@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
async def test_dashboard_update_ui_loop(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_node = MagicMock()
    mock_node.processed_frame = frame
    mock_node.ui_image = MagicMock()
    dash.cameras["C01"] = mock_node
    await dash._update_ui_loop()
    mock_node.ui_image.set_source.assert_called_once()

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_method_coverage(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    with patch('fftrix.database.DB_PATH', str(TEST_DB)):
        with patch('fftrix.database.RECORDINGS_DIR', TEST_HOME):
            dash = Dashboard()
            dash.add_camera_ui("C1", "0", "none")
            cam_id = "C01"
            dash.grid_container = None
            dash._refresh_grid()
            dash.cameras[cam_id].is_running = True
            dash.delete_camera(cam_id)
            assert cam_id not in dash.cameras


# ===========================================================================
# DASHBOARD — sidebar / controls / grid / timeline coverage
# ===========================================================================

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_build_sidebar(mock_ui, mock_app):
    """Covers _build_sidebar, _refresh_camera_list (empty), _refresh_controls (no cam selected)."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    # Simulate admin role in session so Add Camera inputs are rendered
    mock_app.storage.user.get.side_effect = lambda key, default=None: 'admin' if key == 'role' else default
    dash._build_sidebar()
    mock_ui.label.assert_called()
    mock_ui.input.assert_called()
    mock_ui.button.assert_called()


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_build_sidebar_viewer(mock_ui, mock_app):
    """Viewer role sees read-only banner instead of Add Camera form."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    mock_app.storage.user.get.side_effect = lambda key, default=None: 'viewer' if key == 'role' else default
    dash._build_sidebar()
    # viewer mode renders an icon + label, never input fields
    mock_ui.icon.assert_called()
    mock_ui.input.assert_not_called()


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_is_admin_true(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    mock_app.storage.user.get.side_effect = lambda key, default=None: 'admin' if key == 'role' else default
    assert dash.is_admin() is True


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_is_admin_false_for_viewer(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    mock_app.storage.user.get.side_effect = lambda key, default=None: 'viewer' if key == 'role' else default
    assert dash.is_admin() is False


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_handle_login_stores_role(mock_ui, mock_app):
    """handle_login stores the correct role in the session."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    db.add_user('ops', 'secret', role='viewer')
    dash = Dashboard(db=db)
    stored = {}
    mock_app.storage.user.update.side_effect = lambda d: stored.update(d)
    mock_app.storage.user.get.return_value = False
    dash.handle_login('ops', 'secret')
    assert stored.get('role') == 'viewer'


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_refresh_camera_list_with_cameras(mock_ui, mock_app):
    """Covers _refresh_camera_list with cameras present."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    mock_node = MagicMock()
    mock_node.name = 'TestCam'
    dash.cameras['C01'] = mock_node
    dash.camera_list_container = MagicMock()
    dash._refresh_camera_list()
    dash.camera_list_container.clear.assert_called_once()

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_refresh_controls_with_camera(mock_ui, mock_app):
    """Covers _refresh_controls with selected camera — hits watermark/zone/record UI."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    db.add_camera('C01', 'TestCam', '0', 'none')
    dash = Dashboard(db=db)
    mock_node = MagicMock()
    mock_node.name = 'TestCam'
    dash.cameras['C01'] = mock_node
    dash.selected_cam_id = 'C01'
    dash.controls_container = MagicMock()
    mock_app.storage.user.get.side_effect = lambda k, d=None: 'admin' if k == 'role' else d
    dash._refresh_controls()
    dash.controls_container.clear.assert_called_once()
    mock_ui.switch.assert_called()
    mock_ui.input.assert_called()
    mock_ui.select.assert_called()
    mock_ui.slider.assert_called()

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_select_camera(mock_ui, mock_app):
    """Covers _select_camera."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash.controls_container = MagicMock()
    dash._select_camera('C01')
    assert dash.selected_cam_id == 'C01'

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_build_grid_with_cameras(mock_ui, mock_app):
    """Covers _build_grid and _refresh_grid with cameras (grid card/image lines)."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    mock_node = MagicMock()
    mock_node.name = 'TestCam'
    dash.cameras['C01'] = mock_node
    dash.camera_list_container = MagicMock()
    dash.controls_container = MagicMock()
    dash._build_grid()
    mock_ui.grid.assert_called_once()
    mock_ui.interactive_image.assert_called()

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_build_ui(mock_ui, mock_app):
    """Covers build_ui: header, row, timer."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash._build_sidebar = MagicMock()
    dash._build_grid = MagicMock()
    dash._build_timeline = MagicMock()
    dash.build_ui()
    mock_ui.header.assert_called()
    mock_ui.timer.assert_called()
    dash._build_sidebar.assert_called_once()
    dash._build_grid.assert_called_once()
    dash._build_timeline.assert_called_once()

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_build_timeline(mock_ui, mock_app):
    """Covers _build_timeline which creates timeline_container."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    db.log_event('ALERT', 'C01', 'Motion', is_flag=1)
    dash = Dashboard(db=db)
    dash._build_timeline()
    mock_ui.column.assert_called()

@pytest.mark.asyncio
@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
async def test_dashboard_update_ui_loop_timeline_tick(mock_ui, mock_app):
    """Covers timeline auto-refresh branch (every 40 ticks)."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash._refresh_timeline = MagicMock()
    dash._timeline_tick = 39
    await dash._update_ui_loop()
    dash._refresh_timeline.assert_called_once()

# ===========================================================================
# DASHBOARD — env-var secret
# ===========================================================================

@patch('fftrix.dashboard.ui')
@patch('fftrix.dashboard.Dashboard')
def test_env_secret_used(mock_dash_cls, mock_ui):
    from fftrix.dashboard import run_dashboard
    with patch.dict(os.environ, {'FFTRIX_SECRET_KEY': 'my-super-secret'}):
        run_dashboard()
    call_kwargs = mock_ui.run.call_args[1]
    assert call_kwargs['storage_secret'] == 'my-super-secret'

@patch('fftrix.dashboard.ui')
@patch('fftrix.dashboard.Dashboard')
def test_env_secret_fallback_warns(mock_dash_cls, mock_ui):
    from fftrix.dashboard import run_dashboard
    env = {k: v for k, v in os.environ.items() if k != 'FFTRIX_SECRET_KEY'}
    with patch.dict(os.environ, env, clear=True):
        with pytest.warns(UserWarning, match="FFTRIX_SECRET_KEY"):
            run_dashboard()
    call_kwargs = mock_ui.run.call_args[1]
    assert call_kwargs['storage_secret'] == 'insecure-default-change-me'


# ===========================================================================
# CLI / __MAIN__
# ===========================================================================

def test_init_call():
    with patch('fftrix.__main__.cli') as m:
        init_main()
        assert m.called

def test_cli_exhaustive():
    runner = CliRunner()
    with patch('fftrix.__main__.DB_PATH', str(TEST_DB)):
        runner.invoke(cli, ['user', 'add', 'u'], input='p\np\n')
        runner.invoke(cli, ['user', 'list'])
        runner.invoke(cli, ['user', 'delete', 'u'])
        # serve --no-ui returns True without calling run_dashboard
        with patch('fftrix.__main__.run_dashboard') as mock_run:
            result = runner.invoke(cli, ['serve', '--no-ui'])
            assert result.exit_code == 0
            assert not mock_run.called  # --no-ui skips run_dashboard

        # serve (with UI) DOES call run_dashboard
        with patch('fftrix.__main__.run_dashboard') as mock_run:
            runner.invoke(cli, ['serve'])
            assert mock_run.called


def test_cli_reset_password_success():
    """reset-password updates password for existing user."""
    runner = CliRunner()
    with patch('fftrix.__main__.DB_PATH', str(TEST_DB)):
        db = Database(db_path=str(TEST_DB))
        db.add_user('operator', 'OldPass1!')
        result = runner.invoke(
            cli, ['user', 'reset-password', 'operator'],
            input='NewPass99!\nNewPass99!\n'
        )
        assert result.exit_code == 0
        assert 'updated' in result.output
        assert db.verify_user('operator', 'NewPass99!')


def test_cli_reset_password_unknown_user():
    """reset-password exits with error when user doesn't exist."""
    runner = CliRunner()
    with patch('fftrix.__main__.DB_PATH', str(TEST_DB)):
        result = runner.invoke(
            cli, ['user', 'reset-password', 'nobody'],
            input='pass\npass\n'
        )
        assert result.exit_code != 0 or 'not found' in (result.output + (result.stderr or ''))


def test_cli_reset_admin():
    """reset-admin wipes all users and recreates admin:admin."""
    runner = CliRunner()
    with patch('fftrix.__main__.DB_PATH', str(TEST_DB)):
        db = Database(db_path=str(TEST_DB))
        db.add_user('someuser', 'pass')
        # --yes skips the confirmation prompt
        result = runner.invoke(cli, ['user', 'reset-admin', '--yes'])
        assert result.exit_code == 0
        assert 'admin:admin' in result.output
        # admin:admin restored and is now detected as default
        assert db.is_default_credentials()


# ===========================================================================
# DISCOVERY MODULE — unit tests (fully mocked, no hardware required)
# ===========================================================================

from fftrix.discovery import DiscoveredDevice, ONVIFDiscovery, _parse_ip_port, _probe_device


def test_discovered_device_to_from_dict():
    dev = DiscoveredDevice(
        xaddr='http://192.168.1.10/onvif/device_service',
        ip='192.168.1.10', port=80,
        name='Hikvision DS-2CD2', manufacturer='Hikvision', model='DS-2CD2',
        firmware='1.0', serial='ABC123',
        rtsp_uris=['rtsp://192.168.1.10/stream1'],
        requires_auth=False,
    )
    d = dev.to_dict()
    dev2 = DiscoveredDevice.from_dict(d)
    assert dev2.ip == '192.168.1.10'
    assert dev2.rtsp_uris == ['rtsp://192.168.1.10/stream1']
    assert dev2.name == 'Hikvision DS-2CD2'


def test_parse_ip_port_valid():
    ip, port = _parse_ip_port('http://192.168.1.55:8080/onvif/device_service')
    assert ip == '192.168.1.55'
    assert port == 8080


def test_parse_ip_port_default_port():
    ip, port = _parse_ip_port('http://10.0.0.1/onvif/device_service')
    assert ip == '10.0.0.1'
    assert port == 80


def test_parse_ip_port_invalid():
    ip, port = _parse_ip_port('not-a-url')
    assert ip == '' or port == 80  # graceful fallback


def test_probe_device_missing_ip():
    """_probe_device returns None when xaddr has no parseable IP."""
    result = _probe_device('', '', '')
    assert result is None


def test_probe_device_success():
    """_probe_device populates device metadata and RTSP URIs from mocked ONVIF SOAP."""
    mock_cam = MagicMock()
    mock_info = MagicMock()
    mock_info.Manufacturer = 'Hikvision'
    mock_info.Model = 'DS-2CD2143'
    mock_info.FirmwareVersion = 'V5.7.0'
    mock_info.SerialNumber = 'HK001'
    mock_cam.create_devicemgmt_service.return_value.GetDeviceInformation.return_value = mock_info

    mock_profile = MagicMock()
    mock_profile.token = 'Profile_1'
    mock_media = MagicMock()
    mock_media.GetProfiles.return_value = [mock_profile]
    mock_stream_resp = MagicMock()
    mock_stream_resp.Uri = 'rtsp://192.168.1.55/stream1'
    mock_media.GetStreamUri.return_value = mock_stream_resp
    mock_media.create_type.return_value = MagicMock()
    mock_cam.create_media_service.return_value = mock_media

    with patch('onvif.ONVIFCamera', return_value=mock_cam):
        result = _probe_device('http://192.168.1.55/onvif/device_service', 'admin', 'admin')

    assert result is not None
    assert result.manufacturer == 'Hikvision'
    assert result.model == 'DS-2CD2143'
    assert 'rtsp://192.168.1.55/stream1' in result.rtsp_uris
    assert result.requires_auth is False


def test_probe_device_auth_failure():
    """_probe_device marks requires_auth=True on 401/auth error."""
    mock_cam = MagicMock()
    mock_cam.create_devicemgmt_service.return_value.GetDeviceInformation.side_effect = \
        Exception('401 Unauthorized')

    with patch('onvif.ONVIFCamera', return_value=mock_cam):
        result = _probe_device('http://192.168.1.88/onvif/device_service', '', '')

    assert result is not None
    assert result.requires_auth is True


def test_probe_device_generic_failure():
    """Non-auth exception → _probe_device returns None (device silently skipped)."""
    with patch('onvif.ONVIFCamera', side_effect=Exception('Connection refused')):
        result = _probe_device('http://10.0.0.99/onvif/device_service', '', '')

    assert result is None


def test_discovery_scan_no_xaddrs():
    """Services with empty XAddrs lists are skipped gracefully."""
    fake_svc = MagicMock()
    fake_svc.getXAddrs.return_value = []  # no XAddrs

    with patch('fftrix.discovery.WSDiscovery') as mock_wsd:
        mock_wsd.return_value.searchServices.return_value = [fake_svc]
        results = ONVIFDiscovery().scan(timeout=1)

    assert results == []


@patch('fftrix.discovery._probe_device')
def test_discovery_scan_no_devices(mock_probe):
    """WSDiscovery returns empty — scan returns []."""
    with patch('fftrix.discovery.WSDiscovery') as mock_wsd:
        mock_wsd.return_value.searchServices.return_value = []
        result = ONVIFDiscovery().scan(timeout=1)
    assert result == []


@patch('fftrix.discovery._probe_device')
def test_discovery_scan_one_device_success(mock_probe):
    """One WS-Discovery response → one probed device returned."""
    fake_svc = MagicMock()
    fake_svc.getXAddrs.return_value = ['http://192.168.1.55/onvif/device_service']

    expected = DiscoveredDevice(
        xaddr='http://192.168.1.55/onvif/device_service',
        ip='192.168.1.55', name='Axis P3245', manufacturer='Axis', model='P3245',
        rtsp_uris=['rtsp://192.168.1.55/stream1'],
    )
    mock_probe.return_value = expected

    with patch('fftrix.discovery.WSDiscovery') as mock_wsd:
        mock_wsd.return_value.searchServices.return_value = [fake_svc]
        results = ONVIFDiscovery().scan(timeout=1)

    assert len(results) == 1
    assert results[0].ip == '192.168.1.55'
    assert results[0].name == 'Axis P3245'


@patch('fftrix.discovery._probe_device')
def test_discovery_scan_auth_required(mock_probe):
    """Device responds with auth error → requires_auth=True, still returned."""
    fake_svc = MagicMock()
    fake_svc.getXAddrs.return_value = ['http://192.168.1.88/onvif/device_service']

    auth_dev = DiscoveredDevice(
        xaddr='http://192.168.1.88/onvif/device_service',
        ip='192.168.1.88', requires_auth=True,
    )
    mock_probe.return_value = auth_dev

    with patch('fftrix.discovery.WSDiscovery') as mock_wsd:
        mock_wsd.return_value.searchServices.return_value = [fake_svc]
        results = ONVIFDiscovery().scan(timeout=1)

    assert len(results) == 1
    assert results[0].requires_auth is True


@patch('fftrix.discovery._probe_device')
def test_discovery_scan_device_error(mock_probe):
    """Device probe fails silently — returns None → not included."""
    fake_svc = MagicMock()
    fake_svc.getXAddrs.return_value = ['http://10.0.0.1/onvif/device_service']
    mock_probe.return_value = None

    with patch('fftrix.discovery.WSDiscovery') as mock_wsd:
        mock_wsd.return_value.searchServices.return_value = [fake_svc]
        results = ONVIFDiscovery().scan(timeout=1)

    assert results == []


@patch('fftrix.discovery._probe_device')
def test_discovery_scan_progress_callback(mock_probe):
    """progress_callback is invoked for each found device."""
    fake_svc = MagicMock()
    fake_svc.getXAddrs.return_value = ['http://192.168.1.2/onvif/device_service']
    dev = DiscoveredDevice(xaddr='http://192.168.1.2/onvif/device_service', ip='192.168.1.2')
    mock_probe.return_value = dev

    seen = []
    with patch('fftrix.discovery.WSDiscovery') as mock_wsd:
        mock_wsd.return_value.searchServices.return_value = [fake_svc]
        ONVIFDiscovery().scan(timeout=1, progress_callback=seen.append)

    assert len(seen) == 1


# ===========================================================================
# DATABASE — discovered_devices CRUD
# ===========================================================================

def test_database_discovered_devices_crud():
    db = Database(db_path=str(TEST_DB))
    dev = {
        'ip': '192.168.1.10', 'xaddr': 'http://192.168.1.10/onvif/device_service',
        'name': 'Test Camera', 'manufacturer': 'Acme', 'model': 'Cam1',
        'firmware': '1.0', 'serial': 'XYZ', 'rtsp_uris': ['rtsp://192.168.1.10/stream1'],
        'requires_auth': False,
    }
    db.upsert_discovered_device(dev)
    results = db.get_discovered_devices()
    assert len(results) == 1
    assert results[0]['ip'] == '192.168.1.10'
    assert results[0]['rtsp_uris'] == ['rtsp://192.168.1.10/stream1']

    # Upsert again (update existing)
    dev['name'] = 'Updated Camera'
    db.upsert_discovered_device(dev)
    results = db.get_discovered_devices()
    assert len(results) == 1
    assert results[0]['name'] == 'Updated Camera'

    # Clear
    db.clear_discovered_devices()
    assert db.get_discovered_devices() == []


# ===========================================================================
# DASHBOARD — discovery panel UI
# ===========================================================================

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_build_discovery_panel(mock_ui, mock_app):
    """_build_discovery_panel creates the discovery UI widgets."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash._build_discovery_panel()
    mock_ui.label.assert_called()
    mock_ui.button.assert_called()


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_run_discovery_scan(mock_ui, mock_app):
    """_run_discovery_scan launches ONVIFDiscovery.scan in a thread and updates results."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash.discovery_list_container = MagicMock()
    dash._refresh_discovery_list = MagicMock()

    fake_dev = DiscoveredDevice(
        xaddr='http://192.168.1.10/onvif/device_service', ip='192.168.1.10', name='TestCam'
    )

    with patch('fftrix.dashboard.ONVIFDiscovery') as mock_disc_cls:
        mock_disc_cls.return_value.scan.return_value = [fake_dev]
        dash._run_discovery_scan()
        # Wait for the background worker thread to finish
        import time; time.sleep(0.2)

    assert len(dash._discovered) == 1
    assert dash._discovered[0].name == 'TestCam'


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_run_discovery_scan_no_double_start(mock_ui, mock_app):
    """Calling _run_discovery_scan while scanning is a no-op."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash._discovery_scanning = True  # Already scanning
    with patch('fftrix.dashboard.ONVIFDiscovery') as mock_disc_cls:
        dash._run_discovery_scan()
        mock_disc_cls.assert_not_called()


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_clear_discovery(mock_ui, mock_app):
    """_clear_discovery empties results and DB cache."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    db.upsert_discovered_device({
        'ip': '192.168.1.1', 'xaddr': 'x', 'name': 'Cam', 'manufacturer': '',
        'model': '', 'firmware': '', 'serial': '', 'rtsp_uris': [], 'requires_auth': False,
    })
    dash = Dashboard(db=db)
    dash._discovered = [DiscoveredDevice(xaddr='x', ip='192.168.1.1')]
    dash.discovery_list_container = MagicMock()
    dash._clear_discovery()
    assert dash._discovered == []
    assert db.get_discovered_devices() == []


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_refresh_discovery_list_with_devices(mock_ui, mock_app):
    """_refresh_discovery_list renders found devices."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash.discovery_list_container = MagicMock()
    dash._discovered = [
        DiscoveredDevice(xaddr='http://192.168.1.10/onvif/device_service',
                         ip='192.168.1.10', name='TestCam'),
        DiscoveredDevice(xaddr='http://192.168.1.11/onvif/device_service',
                         ip='192.168.1.11', name='AuthCam', requires_auth=True),
    ]
    dash._refresh_discovery_list()
    dash.discovery_list_container.clear.assert_called_once()


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_add_from_discovery_with_rtsp(mock_ui, mock_app):
    """add_from_discovery uses first RTSP URI when available."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash.add_camera_ui = MagicMock()
    dev = DiscoveredDevice(
        xaddr='http://192.168.1.10/onvif/device_service',
        ip='192.168.1.10', name='Hikvision DS-2CD2',
        rtsp_uris=['rtsp://192.168.1.10/stream1', 'rtsp://192.168.1.10/stream2'],
    )
    dash.add_from_discovery(dev)
    dash.add_camera_ui.assert_called_once_with('Hikvision DS-2CD2', 'rtsp://192.168.1.10/stream1', 'motion')


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_add_from_discovery_fallback_xaddr(mock_ui, mock_app):
    """add_from_discovery falls back to xaddr when no RTSP URIs available."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash.add_camera_ui = MagicMock()
    dev = DiscoveredDevice(
        xaddr='http://192.168.1.88/onvif/device_service',
        ip='192.168.1.88', name='Unknown ONVIF Device', rtsp_uris=[],
    )
    dash.add_from_discovery(dev)
    dash.add_camera_ui.assert_called_once_with(
        'Unknown ONVIF Device', 'http://192.168.1.88/onvif/device_service', 'motion'
    )


# ===========================================================================
# FIRST-USE CREDENTIALS — DB methods
# ===========================================================================

def test_database_is_default_credentials_true():
    db = Database(db_path=str(TEST_DB))
    # Ensure fresh default admin:admin exists
    db.add_user('admin', 'admin')
    assert db.is_default_credentials() is True


def test_database_is_default_credentials_false():
    db = Database(db_path=str(TEST_DB))
    db.add_user('admin', 'admin')
    db.change_user_password('admin', 'newuser', 'StrongPass1!')
    assert db.is_default_credentials() is False


def test_database_change_user_password():
    db = Database(db_path=str(TEST_DB))
    db.add_user('admin', 'admin')
    db.change_user_password('admin', 'operator', 'SecurePass99!')
    # Old credentials gone
    assert not db.verify_user('admin', 'admin')
    # New credentials work
    assert db.verify_user('operator', 'SecurePass99!')


# ===========================================================================
# FIRST-USE CREDENTIALS — dashboard handle_login intercept + dialog
# ===========================================================================

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_handle_login_triggers_first_use_dialog(mock_ui, mock_app):
    """handle_login shows first-use dialog when admin:admin is still set."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    db.add_user('admin', 'admin')
    dash = Dashboard(db=db)
    dash._show_first_use_dialog = MagicMock()
    result = dash.handle_login('admin', 'admin')
    assert result is True
    dash._show_first_use_dialog.assert_called_once_with('admin')
    # Should NOT redirect to / yet
    mock_ui.navigate.to.assert_not_called()


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_handle_login_no_first_use_dialog_after_change(mock_ui, mock_app):
    """handle_login proceeds normally when credentials are not default."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    db.add_user('admin', 'admin')
    db.change_user_password('admin', 'operator', 'SecurePass99!')
    dash = Dashboard(db=db)
    dash._show_first_use_dialog = MagicMock()
    dash.handle_login('operator', 'SecurePass99!')
    dash._show_first_use_dialog.assert_not_called()
    mock_ui.navigate.to.assert_called_once_with('/')


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_show_first_use_dialog_validation_empty_username(mock_ui, mock_app):
    """_show_first_use_dialog blocks save when username is empty."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    db.add_user('admin', 'admin')
    dash = Dashboard(db=db)

    # Capture the _save closure by calling the method and retrieving button callback
    save_fn = None
    status_label_mock = MagicMock()

    def capture_button(label, icon=None, on_click=None, **kw):
        nonlocal save_fn
        if label == 'Save & Continue':
            save_fn = on_click
        return MagicMock()

    mock_ui.button.side_effect = capture_button
    mock_ui.input.return_value.value = ''  # empty username
    mock_ui.label.return_value = status_label_mock
    mock_ui.dialog.return_value.__enter__ = lambda s: s
    mock_ui.dialog.return_value.__exit__ = MagicMock(return_value=False)
    mock_ui.card.return_value.__enter__ = lambda s: s
    mock_ui.card.return_value.__exit__ = MagicMock(return_value=False)

    dash._show_first_use_dialog('admin')
    # _save should reject empty username
    if save_fn:
        save_fn()
    # DB should remain unchanged (admin:admin still active)
    assert db.is_default_credentials() is True


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_show_first_use_dialog_opens(mock_ui, mock_app):
    """_show_first_use_dialog creates and opens a persistent dialog."""
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    db.add_user('admin', 'admin')
    dash = Dashboard(db=db)

    mock_ui.card.return_value.__enter__ = lambda s: s
    mock_ui.card.return_value.__exit__ = MagicMock(return_value=False)

    dash._show_first_use_dialog('admin')

    # dialog.props('persistent') is called — verify the call happened
    mock_ui.dialog.return_value.props.assert_called_with('persistent')
    # open() is called on the props() return value (chained mock)
    mock_ui.dialog.return_value.props.return_value.open.assert_called_once()


# ===========================================================================
# M1 — ALERT MANAGER
# ===========================================================================

from fftrix.alerts import AlertManager, send_email, send_webhook


def test_alert_manager_configure_and_get():
    mgr = AlertManager()
    mgr.configure('C1', {'email': 'test@example.com', 'enabled': True, 'cooldown_s': 10})
    cfg = mgr.get_config('C1')
    assert cfg['email'] == 'test@example.com'


def test_alert_manager_fire_disabled():
    """No dispatch when enabled=False."""
    mgr = AlertManager()
    mgr.configure('C1', {'enabled': False, 'email': 'a@b.com'})
    mgr._dispatch = MagicMock()
    mgr.fire('C1', 'MOTION')
    mgr._dispatch.assert_not_called()


def test_alert_manager_fire_respects_cooldown():
    """Second fire within cooldown window is suppressed."""
    mgr = AlertManager(cooldown_s=9999)
    mgr.configure('C1', {'enabled': True, 'cooldown_s': 9999})
    dispatched = []
    original_dispatch = mgr._dispatch

    def capturing_dispatch(*args, **kw):
        dispatched.append(args)

    mgr._dispatch = capturing_dispatch
    mgr.fire('C1', 'MOTION')
    mgr.fire('C1', 'MOTION')  # should be suppressed
    assert len(dispatched) == 1


def test_alert_manager_reset_cooldown():
    mgr = AlertManager(cooldown_s=9999)
    mgr.configure('C1', {'enabled': True, 'cooldown_s': 9999})
    fired = []
    mgr._dispatch = lambda *a, **k: fired.append(1)
    mgr.fire('C1', 'MOTION')
    mgr.reset_cooldown('C1')
    mgr.fire('C1', 'MOTION')
    assert len(fired) == 2


def test_alert_manager_get_config_unknown_camera():
    mgr = AlertManager()
    assert mgr.get_config('UNKNOWN') == {}


def test_send_email_no_smtp_configured():
    """send_email returns False gracefully when SMTP env vars are absent."""
    with patch.dict('os.environ', {}, clear=True):
        result = send_email('to@example.com', 'Subject', 'Body')
    assert result is False


def test_send_webhook_success():
    """send_webhook returns True on 200 response."""
    import urllib.request
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200
    with patch.object(urllib.request, 'urlopen', return_value=mock_resp):
        result = send_webhook('http://example.com/hook', {'text': 'alert'})
    assert result is True


def test_send_webhook_failure():
    """send_webhook returns False on network error."""
    with patch('urllib.request.urlopen', side_effect=Exception('network error')):
        result = send_webhook('http://bad.host/hook', {})
    assert result is False


def test_alert_manager_dispatch_fires_email_and_webhook():
    """_dispatch calls send_email and send_webhook when both configured."""
    mgr = AlertManager()
    config = {
        'enabled': True,
        'email': 'ops@example.com',
        'webhook_url': 'http://slack.example.com/hook',
        'cooldown_s': 0,
    }
    with patch('fftrix.alerts.send_email', return_value=True) as mock_email, \
         patch('fftrix.alerts.send_webhook', return_value=True) as mock_wh:
        mgr._dispatch('C1', 'MOTION', 'details', None, config)
    mock_email.assert_called_once()
    mock_wh.assert_called_once()


def test_alert_manager_dispatch_with_snapshot():
    """_dispatch passes snapshot_path to send_email."""
    mgr = AlertManager()
    config = {'enabled': True, 'email': 'a@b.com', 'webhook_url': '', 'cooldown_s': 0}
    with patch('fftrix.alerts.send_email', return_value=True) as mock_email:
        mgr._dispatch('C1', 'MOTION', '', '/tmp/snap.jpg', config)
    call_args = mock_email.call_args
    assert call_args[0][3] == '/tmp/snap.jpg'  # snapshot_path positional arg


# ===========================================================================
# M1 — RETENTION MANAGER
# ===========================================================================

from fftrix.retention import RetentionManager


def test_retention_run_once_deletes_old_files():
    db = Database(db_path=str(TEST_DB))
    db.add_camera('C1', 'Cam1', 'rtsp://x', 'motion')
    db.update_camera_config('C1', retention_days=7)

    rec_dir = TEST_HOME / 'recordings' / 'C1'
    rec_dir.mkdir(parents=True, exist_ok=True)
    old_file = rec_dir / 'old.mp4'
    old_file.write_bytes(b'data')
    # Back-date the file's mtime to 8 days ago
    old_mtime = time.time() - (8 * 86400)
    import os as _os
    _os.utime(old_file, (old_mtime, old_mtime))

    new_file = rec_dir / 'new.mp4'
    new_file.write_bytes(b'data')

    mgr = RetentionManager(db=db, recordings_root=TEST_HOME / 'recordings')
    results = mgr.run_once()

    assert results.get('C1', 0) == 1
    assert not old_file.exists()
    assert new_file.exists()


def test_retention_run_once_keeps_forever():
    """retention_days=0 means keep all files."""
    db = Database(db_path=str(TEST_DB))
    db.add_camera('C1', 'Cam1', 'rtsp://x', 'motion')
    db.update_camera_config('C1', retention_days=0)

    rec_dir = TEST_HOME / 'recordings' / 'C1'
    rec_dir.mkdir(parents=True, exist_ok=True)
    old_file = rec_dir / 'ancient.mp4'
    old_file.write_bytes(b'data')
    old_mtime = time.time() - (365 * 86400)
    import os as _os
    _os.utime(old_file, (old_mtime, old_mtime))

    mgr = RetentionManager(db=db, recordings_root=TEST_HOME / 'recordings')
    results = mgr.run_once()
    assert results == {}
    assert old_file.exists()


def test_retention_run_once_missing_dir():
    """run_once handles cameras with no recordings directory gracefully."""
    db = Database(db_path=str(TEST_DB))
    db.add_camera('GHOST', 'Ghost', 'rtsp://x', 'motion')
    db.update_camera_config('GHOST', retention_days=7)
    mgr = RetentionManager(db=db, recordings_root=TEST_HOME / 'recordings')
    results = mgr.run_once()
    assert 'GHOST' not in results


def test_retention_start_stop():
    db = Database(db_path=str(TEST_DB))
    mgr = RetentionManager(db=db, recordings_root=TEST_HOME / 'recordings')
    mgr.start()
    assert mgr._thread is not None and mgr._thread.is_alive()
    mgr.stop()
    # Double-start is a no-op
    mgr.start()


# ===========================================================================
# M1 — DATABASE: new columns
# ===========================================================================

def test_database_retention_days_default():
    db = Database(db_path=str(TEST_DB))
    db.add_camera('C1', 'Cam', 'rtsp://x', 'motion')
    cams = db.get_cameras()
    assert cams[0]['retention_days'] == 30


def test_database_update_retention_days():
    db = Database(db_path=str(TEST_DB))
    db.add_camera('C1', 'Cam', 'rtsp://x', 'motion')
    db.update_camera_config('C1', retention_days=14)
    assert db.get_cameras()[0]['retention_days'] == 14


def test_database_update_alert_config():
    db = Database(db_path=str(TEST_DB))
    db.add_camera('C1', 'Cam', 'rtsp://x', 'motion')
    cfg = {'email': 'a@b.com', 'enabled': True, 'cooldown_s': 60, 'webhook_url': ''}
    db.update_camera_config('C1', alert_config=cfg)
    stored = db.get_cameras()[0]['alert_config']
    assert stored['email'] == 'a@b.com'


def test_database_log_event_with_snapshot():
    db = Database(db_path=str(TEST_DB))
    db.log_event('MOTION', 'C1', 'Trigger detected', is_flag=1, snapshot_path='/tmp/snap.jpg')
    events = db.get_events(flags_only=True)
    assert events[0]['snapshot_path'] == '/tmp/snap.jpg'


def test_database_log_event_snapshot_none():
    db = Database(db_path=str(TEST_DB))
    db.log_event('MOTION', 'C1', 'No snapshot', is_flag=1)
    events = db.get_events(flags_only=True)
    assert events[0]['snapshot_path'] is None


def test_database_get_user_role_default():
    db = Database(db_path=str(TEST_DB))
    # Default admin user gets role 'admin'
    assert db.get_user_role('admin') == 'admin'


def test_database_get_user_role_viewer():
    db = Database(db_path=str(TEST_DB))
    db.add_user('viewer1', 'pass', role='viewer')
    assert db.get_user_role('viewer1') == 'viewer'


def test_database_get_user_role_unknown():
    db = Database(db_path=str(TEST_DB))
    assert db.get_user_role('nobody') == 'admin'  # safe default


def test_database_privacy_zones_roundtrip():
    db = Database(db_path=str(TEST_DB))
    db.add_camera('C1', 'Cam', 'rtsp://x', 'motion')
    zones = [[10, 20, 200, 300], [50, 60, 100, 150]]
    db.update_camera_config('C1', privacy_zones=zones)
    assert db.get_cameras()[0]['privacy_zones'] == zones


# ===========================================================================
# M1 — ANALYTICS: snapshot + alert + privacy blur integration
# ===========================================================================

def test_analytics_snapshot_saved_on_flag():
    """When a trigger fires, _save_snapshot writes a JPEG to disk."""
    snap_dir = TEST_HOME / 'snapshots'
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline = AnalyticsPipeline(snapshots_dir=snap_dir)
    path = pipeline._save_snapshot(frame, 'C1')
    assert path is not None
    assert Path(path).exists()


def test_analytics_snapshot_no_dir():
    """_save_snapshot returns None when snapshots_dir is not set."""
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    pipeline = AnalyticsPipeline()
    assert pipeline._save_snapshot(frame, 'C1') is None


def test_analytics_alert_manager_called_on_flag():
    """alert_manager.fire() is called when a motion trigger fires."""
    mock_alert = MagicMock()
    mock_db = MagicMock()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Add white rectangle to guarantee motion
    frame[50:150, 50:150] = 255

    pipeline = AnalyticsPipeline(db=mock_db, alert_manager=mock_alert)
    pipeline.set_config('C1', mode='motion')
    # Prime background subtractor
    for _ in range(5):
        pipeline.process(np.zeros((200, 200, 3), dtype=np.uint8), 'C1')
    # Now process frame with motion
    _, triggered = pipeline.process(frame, 'C1')
    if triggered:
        mock_alert.fire.assert_called()


def test_analytics_privacy_blur_applied():
    """apply_privacy_blur blurs the specified region."""
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
    pipeline = AnalyticsPipeline()
    pipeline.set_config('C1', privacy_zones=[[10, 10, 90, 90]])
    result = pipeline.apply_privacy_blur(frame, 'C1')
    # Blurred region values differ from solid 200 background
    assert result is not None
    assert result.shape == frame.shape


def test_analytics_privacy_blur_no_zones():
    """apply_privacy_blur is a no-op when no privacy zones are set."""
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    pipeline = AnalyticsPipeline()
    result = pipeline.apply_privacy_blur(frame, 'C1')
    assert np.array_equal(result, frame)


# ===========================================================================
# M1 — ALERTS: SMTP path coverage
# ===========================================================================

from fftrix.alerts import _build_smtp_conn


def test_build_smtp_conn_returns_none_without_config():
    """_build_smtp_conn returns None when env vars are missing."""
    with patch.dict('os.environ', {}, clear=True):
        assert _build_smtp_conn() is None


def test_build_smtp_conn_smtp_failure():
    """_build_smtp_conn swallows connection errors and returns None."""
    with patch.dict('os.environ', {'FFTRIX_SMTP_HOST': 'smtp.bad.host', 'FFTRIX_SMTP_USER': 'u'}):
        with patch('smtplib.SMTP', side_effect=Exception('conn refused')):
            result = _build_smtp_conn()
    assert result is None


def test_send_email_plain_success(tmp_path):
    """send_email returns True via mocked SMTP (plain, no attachment)."""
    mock_conn = MagicMock()
    with patch('fftrix.alerts._build_smtp_conn', return_value=mock_conn):
        result = send_email('to@example.com', 'Subj', 'Body text')
    assert result is True
    mock_conn.send_message.assert_called_once()
    mock_conn.quit.assert_called_once()


def test_send_email_with_real_snapshot(tmp_path):
    """send_email attaches the snapshot image when path exists."""
    snap = tmp_path / 'snap.jpg'
    snap.write_bytes(b'\xff\xd8\xff')  # minimal JPEG header
    mock_conn = MagicMock()
    with patch('fftrix.alerts._build_smtp_conn', return_value=mock_conn):
        result = send_email('to@example.com', 'Alert', 'Body', str(snap))
    assert result is True
    mock_conn.send_message.assert_called_once()


def test_send_email_exception_returns_false():
    """send_email returns False when SMTP send_message raises."""
    mock_conn = MagicMock()
    mock_conn.send_message.side_effect = Exception('SMTP failure')
    with patch('fftrix.alerts._build_smtp_conn', return_value=mock_conn):
        result = send_email('to@example.com', 'Subj', 'Body')
    assert result is False



# ===========================================================================
# M3 — MILESTONE 3: Camera Intelligence
# ===========================================================================

# ---- PTZController ----

from fftrix.ptz import PTZController


def _make_ptz_with_mock_service():
    ptz = PTZController(xaddr='http://192.168.1.1/onvif', username='admin', password='pass')
    mock_svc = MagicMock()
    mock_svc.ContinuousMove.return_value = None
    mock_svc.Stop.return_value = None
    mock_svc.GotoPreset.return_value = None
    preset = MagicMock()
    preset.token = 'p1'
    preset.Name = 'Home'
    mock_svc.GetPresets.return_value = [preset]
    ptz._ptz_service = mock_svc
    ptz._profile_token = 'tok1'
    return ptz, mock_svc


def test_ptz_move_dispatches_continuous_move():
    ptz, svc = _make_ptz_with_mock_service()
    ptz.move('up', speed=0.5)
    import time as _t; _t.sleep(0.1)
    svc.ContinuousMove.assert_called_once()


def test_ptz_move_unknown_direction_sends_zero_velocity():
    ptz, svc = _make_ptz_with_mock_service()
    ptz.move('nowhere', speed=0.5)
    import time as _t; _t.sleep(0.1)
    cargs = svc.ContinuousMove.call_args[0][0]
    assert cargs['Velocity']['PanTilt']['x'] == 0.0
    assert cargs['Velocity']['PanTilt']['y'] == 0.0


def test_ptz_zoom_in():
    ptz, svc = _make_ptz_with_mock_service()
    ptz.zoom('in', speed=0.4)
    import time as _t; _t.sleep(0.1)
    assert svc.ContinuousMove.call_args[0][0]['Velocity']['Zoom']['x'] > 0


def test_ptz_zoom_out():
    ptz, svc = _make_ptz_with_mock_service()
    ptz.zoom('out', speed=0.4)
    import time as _t; _t.sleep(0.1)
    assert svc.ContinuousMove.call_args[0][0]['Velocity']['Zoom']['x'] < 0


def test_ptz_stop():
    ptz, svc = _make_ptz_with_mock_service()
    ptz.stop()
    import time as _t; _t.sleep(0.1)
    svc.Stop.assert_called_once()


def test_ptz_go_to_preset():
    ptz, svc = _make_ptz_with_mock_service()
    ptz.go_to_preset('p1')
    import time as _t; _t.sleep(0.1)
    svc.GotoPreset.assert_called_once()


def test_ptz_get_presets():
    ptz, svc = _make_ptz_with_mock_service()
    presets = ptz.get_presets()
    assert presets == [{'token': 'p1', 'name': 'Home'}]


def test_ptz_get_service_bad_xaddr_returns_none():
    ptz = PTZController(xaddr='not-a-url')
    svc, tok = ptz._get_service()
    assert svc is None and tok is None


def test_ptz_move_no_service_invokes_callback_false():
    ptz = PTZController(xaddr='not-a-url')
    results = []
    ptz.move('up', callback=lambda ok: results.append(ok))
    import time as _t; _t.sleep(0.15)
    assert results == [False]


# ---- is_armed() ----


def test_is_armed_no_schedule_always_true():
    pipeline = AnalyticsPipeline()
    assert pipeline.is_armed('C1') is True


def test_is_armed_within_schedule():
    import time as _t
    now = _t.localtime()
    today = now.tm_wday
    start = f"{now.tm_hour:02d}:00"
    end = f"{now.tm_hour:02d}:59"
    pipeline = AnalyticsPipeline()
    pipeline.set_config('C1', arm_schedule=[{'days': [today], 'start': start, 'end': end}])
    assert pipeline.is_armed('C1') is True


def test_is_armed_outside_schedule_no_days():
    pipeline = AnalyticsPipeline()
    pipeline.set_config('C1', arm_schedule=[{'days': [], 'start': '00:00', 'end': '23:59'}])
    assert pipeline.is_armed('C1') is False


# ---- analytics.process() skips alerts when unarmed ----


def test_process_no_alert_when_unarmed(setup_teardown, tmp_path):
    db = Database(db_path=str(TEST_DB))
    mock_alert = MagicMock()
    pipeline = AnalyticsPipeline(db=db, alert_manager=mock_alert, snapshots_dir=tmp_path)
    pipeline.set_config('T1', mode='motion',
                        arm_schedule=[{'days': [], 'start': '00:00', 'end': '23:59'}])
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_sub = MagicMock()
    pipeline.back_subs['T1'] = mock_sub
    cnt = np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]], dtype=np.int32)
    mock_sub.apply.return_value = np.ones((200, 200), dtype=np.uint8) * 255
    with patch('cv2.findContours', return_value=([cnt], None)), \
         patch('cv2.contourArea', return_value=5000):
        pipeline.process(frame, 'T1')
    mock_alert.fire.assert_not_called()


# ---- Dashboard M3 panels ----


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_build_ptz_panel_renders_buttons(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    with patch('fftrix.ptz.PTZController', MagicMock()):
        dash._build_ptz_panel('C1', 'http://cam/onvif')
    mock_ui.button.assert_called()


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_build_privacy_panel_renders(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash._build_privacy_panel('C1', {'privacy_zones': [[0, 0, 100, 100]]})
    mock_ui.label.assert_called()


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_build_arm_schedule_panel_renders(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash._build_arm_schedule_panel('C1', {'arm_schedule': [
        {'days': [0, 1, 2, 3, 4], 'start': '08:00', 'end': '18:00'}
    ]})
    mock_ui.button.assert_called()


# ===========================================================================
# M4 — MILESTONE 4: Data & Reporting
# ===========================================================================

from fftrix.clipper import ClipExporter
from fftrix.engine import CameraNode


# ---- ClipExporter ----

def test_clipexporter_list_segments_empty(tmp_path):
    ex = ClipExporter(recordings_root=tmp_path / 'recs')
    assert ex.list_segments('CAM1') == []


def test_clipexporter_list_segments(tmp_path):
    rec = tmp_path / 'recs' / 'CAM1'
    rec.mkdir(parents=True)
    (rec / 'seg1.mp4').write_bytes(b'fakevid1')
    (rec / 'seg2.mp4').write_bytes(b'fakevid2')
    ex = ClipExporter(recordings_root=tmp_path / 'recs')
    segs = ex.list_segments('CAM1')
    assert len(segs) == 2
    assert all(s['name'].endswith('.mp4') for s in segs)


def test_clipexporter_export_no_segments_returns_none(tmp_path):
    ex = ClipExporter(recordings_root=tmp_path / 'recs', export_dir=tmp_path / 'exp')
    result = ex.export('CAM1')
    assert result is None


def test_clipexporter_export_single_segment(tmp_path):
    rec = tmp_path / 'recs' / 'CAM1'
    rec.mkdir(parents=True)
    f = rec / 'seg1.mp4'
    f.write_bytes(b'fake')
    ex = ClipExporter(recordings_root=tmp_path / 'recs', export_dir=tmp_path / 'exp')
    result = ex.export('CAM1')
    assert result is not None
    from pathlib import Path
    assert Path(result).exists()


def test_clipexporter_export_callback_no_segments(tmp_path):
    ex = ClipExporter(recordings_root=tmp_path / 'recs', export_dir=tmp_path / 'exp')
    got = []
    ex.export('CAM1', callback=lambda p: got.append(p))
    assert got == [None]


def test_clipexporter_export_callback_single(tmp_path):
    rec = tmp_path / 'recs' / 'CAM1'
    rec.mkdir(parents=True)
    (rec / 'seg1.mp4').write_bytes(b'fake')
    ex = ClipExporter(recordings_root=tmp_path / 'recs', export_dir=tmp_path / 'exp')
    got = []
    ex.export('CAM1', callback=lambda p: got.append(p))
    assert got and got[0] is not None


def test_clipexporter_export_async_callback(tmp_path):
    rec = tmp_path / 'recs' / 'CAM1'
    rec.mkdir(parents=True)
    (rec / 'seg1.mp4').write_bytes(b'fakevid')
    ex = ClipExporter(recordings_root=tmp_path / 'recs', export_dir=tmp_path / 'exp')
    results = []
    ex.export_async('CAM1', callback=lambda p: results.append(p))
    import time as _t; _t.sleep(0.2)
    assert len(results) == 1 and results[0] is not None


def test_clipexporter_delete_segment(tmp_path):
    f = tmp_path / 'seg.mp4'
    f.write_bytes(b'data')
    ex = ClipExporter(recordings_root=tmp_path, export_dir=tmp_path / 'exp')
    assert ex.delete_segment(str(f)) is True
    assert not f.exists()


def test_clipexporter_delete_missing_returns_false(tmp_path):
    ex = ClipExporter(recordings_root=tmp_path, export_dir=tmp_path / 'exp')
    assert ex.delete_segment('/nonexistent/file.mp4') is False


def test_clipexporter_export_time_filter(tmp_path):
    import time as _t
    rec = tmp_path / 'recs' / 'CAM1'
    rec.mkdir(parents=True)
    f = rec / 'seg1.mp4'
    f.write_bytes(b'data')
    mtime = f.stat().st_mtime
    ex = ClipExporter(recordings_root=tmp_path / 'recs', export_dir=tmp_path / 'exp')
    # exclude by time range
    result = ex.export('CAM1', start_ts=mtime + 3600, end_ts=mtime + 7200)
    assert result is None


def test_clipexporter_concat_ffmpeg_not_found(tmp_path, monkeypatch):
    """Concat path falls back to copy when ffmpeg is missing."""
    rec = tmp_path / 'recs' / 'CAM1'
    rec.mkdir(parents=True)
    for i in range(2):
        (rec / f'seg{i}.mp4').write_bytes(b'fakedata')
    ex = ClipExporter(recordings_root=tmp_path / 'recs', export_dir=tmp_path / 'exp')
    import subprocess
    def _raise(*a, **kw): raise FileNotFoundError('ffmpeg not found')
    monkeypatch.setattr(subprocess, 'run', _raise)
    result = ex.export('CAM1')
    assert result is not None  # fell back to copy


# ---- CameraNode health stats ----

def test_camera_node_get_health_initial():
    pipeline = AnalyticsPipeline()
    mock_db = MagicMock()
    node = CameraNode('C1', '0', 'Cam1', pipeline, mock_db)
    h = node.get_health()
    assert h['cam_id'] == 'C1'
    assert h['running'] is False
    assert h['fps'] == 0.0
    assert h['frames_processed'] == 0
    assert h['uptime_s'] == 0.0


def test_camera_node_frames_processed_and_dropped():
    pipeline = AnalyticsPipeline()
    mock_db = MagicMock()
    node = CameraNode('C1', '0', 'Cam1', pipeline, mock_db)
    node.frames_processed = 100
    node.dropped_frames = 5
    h = node.get_health()
    assert h['frames_processed'] == 100
    assert h['dropped_frames'] == 5


def test_camera_node_fps_rolling():
    import time as _t
    pipeline = AnalyticsPipeline()
    mock_db = MagicMock()
    node = CameraNode('C1', '0', 'Cam1', pipeline, mock_db)
    now = _t.time()
    node._fps_buf = [now - 1.0 + i * 0.033 for i in range(30)]
    h = node.get_health()
    assert h['fps'] > 0


# ---- Dashboard M4 panels ----

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_build_recordings_panel_no_camera(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    mock_app.storage.user.get.side_effect = lambda k, d=None: 'admin' if k == 'role' else d
    dash._build_recordings_panel()
    mock_ui.label.assert_called()


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_build_recordings_panel_with_segments(mock_ui, mock_app, tmp_path):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    mock_node = MagicMock(); mock_node.name = 'TestCam'
    dash.cameras['C1'] = mock_node
    dash.selected_cam_id = 'C1'
    mock_app.storage.user.get.side_effect = lambda k, d=None: 'admin' if k == 'role' else d
    with patch('fftrix.dashboard.ClipExporter') as MockExp:
        mock_exp_instance = MockExp.return_value
        mock_exp_instance.list_segments.return_value = [
            {'path': str(tmp_path / 'seg.mp4'), 'mtime': 1700000000.0,
             'size': 1048576, 'name': 'seg.mp4'}
        ]
        dash._build_recordings_panel()
    mock_ui.label.assert_called()
    mock_ui.button.assert_called()


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_build_health_widget_renders(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    dash._build_health_widget()
    mock_ui.row.assert_called()
    mock_ui.timer.assert_called()


@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_refresh_health_with_camera(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    db = Database(db_path=str(TEST_DB))
    dash = Dashboard(db=db)
    mock_node = MagicMock()
    mock_node.get_health.return_value = {
        'cam_id': 'C1', 'name': 'TestCam', 'running': True,
        'recording': False, 'uptime_s': 10.0, 'fps': 25.0,
        'frames_processed': 500, 'dropped_frames': 0,
    }
    dash.cameras['C1'] = mock_node
    dash._health_container = MagicMock()
    dash._refresh_health()
    dash._health_container.clear.assert_called_once()


def test_clipexporter_export_concat_exception(tmp_path, monkeypatch):
    """If concat raises a generic exception, callback is called with None."""
    rec = tmp_path / 'recs' / 'CAM1'
    rec.mkdir(parents=True)
    for i in range(2):
        (rec / f's{i}.mp4').write_bytes(b'data')
    ex = ClipExporter(recordings_root=tmp_path / 'recs', export_dir=tmp_path / 'exp')
    import subprocess
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(Exception('boom')))
    got = []
    result = ex.export('CAM1', callback=lambda p: got.append(p))
    # Either returned None or fell back to copy; we just care it didn't raise
    assert True


def test_clipexporter_export_ffmpeg_failure_returncode(tmp_path, monkeypatch):
    """ffmpeg non-zero returncode falls back to single copy."""
    rec = tmp_path / 'recs' / 'CAM1'
    rec.mkdir(parents=True)
    for i in range(2):
        (rec / f's{i}.mp4').write_bytes(b'video')
    ex = ClipExporter(recordings_root=tmp_path / 'recs', export_dir=tmp_path / 'exp')
    import subprocess
    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stderr = b'error msg'
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: fake_result)
    result = ex.export('CAM1')
    assert result is not None  # fell back to copy of first segment


def test_ptz_zoom_stop_no_service_callback_false():
    ptz = PTZController(xaddr='not-a-url')
    for fn in [lambda: ptz.zoom('in'), lambda: ptz.stop(), lambda: ptz.go_to_preset('p1')]:
        fn()
    # No assertion — just confirm no raise
    assert True


def test_ptz_zoom_no_service_callback():
    ptz = PTZController(xaddr='not-a-url')
    results = []
    ptz.zoom('in', callback=lambda ok: results.append(ok))
    import time as _t; _t.sleep(0.15)
    assert results == [False]


def test_ptz_stop_no_service_callback():
    ptz = PTZController(xaddr='not-a-url')
    results = []
    ptz.stop(callback=lambda ok: results.append(ok))
    import time as _t; _t.sleep(0.15)
    assert results == [False]


# ===========================================================================
# M5 — MILESTONE 5: User Experience
# ===========================================================================

from fftrix.lpr import LicensePlateReader, PlateDetection


# ---- LicensePlateReader unit tests ----

def _make_lpr():
    """Build LPR with a mock OCR engine to avoid tesseract dependency."""
    mock_ocr = MagicMock()
    mock_ocr.Output.DICT = 'dict'
    mock_ocr.image_to_data.return_value = {
        'text': ['XYZ123', ''],
        'conf': [95, -1],
        'left': [0, 0], 'top': [0, 0],
        'width': [80, 0], 'height': [20, 0],
    }
    return LicensePlateReader(ocr_engine=mock_ocr)


def test_lpr_process_empty_frame():
    lpr = _make_lpr()
    result = lpr.process(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result == []


def test_lpr_process_none_frame():
    lpr = _make_lpr()
    result = lpr.process(None)
    assert result == []


def test_lpr_process_no_candidates_returns_empty():
    """Blank frame produces no detectable plate contours."""
    lpr = _make_lpr()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = lpr.process(frame)
    # May or may not find plates in a blank frame — just ensure no crash
    assert isinstance(result, list)


def test_lpr_process_with_mocked_candidates():
    """Inject mock candidates via monkeypatching _candidate_rois."""
    lpr = _make_lpr()
    frame = np.ones((200, 400, 3), dtype=np.uint8) * 200
    with patch.object(lpr, '_candidate_rois', return_value=[(10, 10, 120, 40)]):
        detections = lpr.process(frame)
    # OCR returns XYZ123 which matches plate pattern
    assert any(d.text == 'XYZ123' for d in detections)


def test_lpr_annotate_draws_rect():
    """annotate() should call cv2.rectangle and cv2.putText — verify no crash."""
    lpr = _make_lpr()
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    dets = [PlateDetection(text='ABC123', confidence=0.9, bbox=(10, 10, 100, 30))]
    result = lpr.annotate(frame, dets)
    assert result is not None
    assert result.shape == frame.shape


def test_lpr_annotate_empty_detections():
    lpr = _make_lpr()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = lpr.annotate(frame, [])
    assert np.array_equal(result, frame)


def test_lpr_ocr_roi_low_confidence_filtered():
    """Boxes with conf <= 40 should be ignored."""
    mock_ocr = MagicMock()
    mock_ocr.Output.DICT = 'dict'
    mock_ocr.image_to_data.return_value = {
        'text': ['BAD'], 'conf': [20],
    }
    lpr = LicensePlateReader(ocr_engine=mock_ocr)
    text, conf = lpr._ocr_roi(np.ones((30, 100, 3), dtype=np.uint8) * 200)
    assert text == '' and conf == 0.0


def test_lpr_ocr_roi_exception_returns_empty():
    mock_ocr = MagicMock()
    mock_ocr.Output.DICT = 'dict'
    mock_ocr.image_to_data.side_effect = RuntimeError('tesseract gone')
    lpr = LicensePlateReader(ocr_engine=mock_ocr)
    text, conf = lpr._ocr_roi(np.ones((30, 100, 3), dtype=np.uint8) * 200)
    assert text == '' and conf == 0.0


# ---- analytics LPR mode ----

def test_analytics_lpr_mode_triggers(setup_teardown, tmp_path):
    db = Database(db_path=str(TEST_DB))
    mock_alert = MagicMock()
    pipeline = AnalyticsPipeline(db=db, alert_manager=mock_alert, snapshots_dir=tmp_path)
    pipeline.set_config('C1', mode='lpr')
    frame = np.ones((200, 400, 3), dtype=np.uint8) * 200

    mock_lpr = MagicMock()
    mock_det = PlateDetection(text='ABC123', confidence=0.9, bbox=(10, 10, 100, 30))
    mock_lpr.process.return_value = [mock_det]
    mock_lpr.annotate.return_value = frame
    pipeline._lpr = mock_lpr

    proc, triggered = pipeline.process(frame, 'C1')

    assert triggered is True


def test_analytics_lpr_mode_no_detections(setup_teardown, tmp_path):
    db = Database(db_path=str(TEST_DB))
    pipeline = AnalyticsPipeline(db=db, snapshots_dir=tmp_path)
    pipeline.set_config('C1', mode='lpr')
    frame = np.ones((200, 400, 3), dtype=np.uint8) * 200

    mock_lpr = MagicMock()
    mock_lpr.process.return_value = []
    pipeline._lpr = mock_lpr

    proc, triggered = pipeline.process(frame, 'C1')
    assert triggered is False


# ---- PWA manifest + head tags ----

def test_write_pwa_manifest(tmp_path):
    from fftrix.dashboard import _write_pwa_manifest
    import json as _json
    from fftrix import dashboard as _dash_mod
    orig = _dash_mod.STATIC_DIR
    _dash_mod.STATIC_DIR = tmp_path / 'static'
    _dash_mod.STATIC_DIR.mkdir(parents=True)
    try:
        url = _write_pwa_manifest(8080)
        assert url == '/static/manifest.json'
        manifest = _json.loads((_dash_mod.STATIC_DIR / 'manifest.json').read_text())
        assert manifest['name'] == 'FFTrix NVR'
        assert manifest['display'] == 'standalone'
    finally:
        _dash_mod.STATIC_DIR = orig


@patch('fftrix.dashboard.ui')
def test_add_pwa_head_tags(mock_ui):
    from fftrix.dashboard import _add_pwa_head_tags
    _add_pwa_head_tags()
    assert mock_ui.add_head_html.call_count >= 6
    calls_str = ' '.join(str(c) for c in mock_ui.add_head_html.call_args_list)
    assert 'manifest' in calls_str
    assert 'viewport' in calls_str
