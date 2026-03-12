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
from fftrix.database import Database, DB_PATH, FFTRIX_HOME, RECORDINGS_DIR
from fftrix.analytics import AnalyticsPipeline, WatermarkEngine
from fftrix.engine import CameraNode
from fftrix.__main__ import cli, handle_serve, handle_user_add
from fftrix import main as init_main
from click.testing import CliRunner

TEST_HOME = Path("./test_fftrix_home")
TEST_DB = TEST_HOME / "test_system.db"

@pytest.fixture(autouse=True)
def setup_teardown():
    if TEST_HOME.exists(): shutil.rmtree(TEST_HOME)
    TEST_HOME.mkdir(parents=True, exist_ok=True)
    (TEST_HOME / "recordings").mkdir()
    yield
    if TEST_HOME.exists(): shutil.rmtree(TEST_HOME)


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
    dash._build_sidebar()
    mock_ui.label.assert_called()
    mock_ui.input.assert_called()
    mock_ui.button.assert_called()

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
    mock_ui.timer.assert_called_once()
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
