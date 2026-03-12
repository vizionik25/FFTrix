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
    assert fftrix.__version__ == "1.0.0"
    assert hasattr(fftrix, 'Database')
    assert hasattr(fftrix, 'AnalyticsPipeline')
    assert hasattr(fftrix, 'CameraNode')
    assert hasattr(fftrix, 'Dashboard')


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
