"""FFTrix — Enterprise NVR, AI Surveillance and Streaming Server."""

__version__ = "1.0.0"
__all__ = ["main", "Database", "AnalyticsPipeline", "CameraNode", "Dashboard"]

from .database import Database
from .analytics import AnalyticsPipeline
from .engine import CameraNode
from .dashboard import Dashboard


def main() -> None:
    from .__main__ import cli
    cli()
