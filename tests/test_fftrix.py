import pytest
import os
import shutil
import numpy as np
import cv2
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

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

# --- ANALYTICS FAILURE PATHS ---

def test_pipeline_exception_handling():
    p = AnalyticsPipeline()
    # Pass malformed data to trigger Exception block
    res, triggered = p.process(None, "C1")
    assert res is None
    assert triggered is False

def test_watermark_edge_cases():
    engine = WatermarkEngine()
    # Path exists but is invalid image
    bad_file = TEST_HOME / "bad.txt"
    bad_file.write_text("not an image")
    engine.set_config("C1", {'image_path': str(bad_file)})
    frame = np.zeros((10,10,3), dtype=np.uint8)
    res = engine.apply(frame, "C1")
    assert np.array_equal(res, frame)

# --- ENGINE FAILURE PATHS ---

@patch('fftrix.engine.VideoGear')
def test_engine_start_failure(mock_vidgear):
    mock_vidgear.side_effect = Exception("HW Error")
    db = Database(db_path=str(TEST_DB))
    n = CameraNode("C1", "0", "Cam", AnalyticsPipeline(), db)
    n.start() # Should hit the log_event ERROR path
    assert len(db.get_events()) > 0

# --- DASHBOARD LOGIC BRANCHES ---

@patch('fftrix.dashboard.app')
@patch('fftrix.dashboard.ui')
def test_dashboard_method_coverage(mock_ui, mock_app):
    from fftrix.dashboard import Dashboard
    with patch('fftrix.database.DB_PATH', str(TEST_DB)):
        with patch('fftrix.database.RECORDINGS_DIR', TEST_HOME):
            dash = Dashboard()
            dash.add_camera_ui("C1", "0", "none")
            cam_id = "C01"
            
            # 1. Refresh logic when container is missing
            dash.grid_container = None
            dash._refresh_grid() # Hits 'if not'
            
            # 2. Deletion when active
            dash.cameras[cam_id].is_running = True
            dash.delete_camera(cam_id)
            assert cam_id not in dash.cameras

# --- RE-RUN CORE INTEGRITY ---

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
        with patch('fftrix.__main__.run_dashboard') as mock_run:
            runner.invoke(cli, ['serve', '--no-ui'])
            assert mock_run.called
