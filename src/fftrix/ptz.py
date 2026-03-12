"""FFTrix PTZ Controller — Pan/Tilt/Zoom control via ONVIF."""

from __future__ import annotations

import logging
import threading
from typing import Callable

log = logging.getLogger(__name__)


class PTZController:
    """Wraps ONVIF PTZ service calls for a single camera.

    Usage::

        ptz = PTZController(xaddr='http://192.168.1.10/onvif/device_service',
                            username='admin', password='12345')
        ptz.move('up', speed=0.5)
        ptz.zoom('in', speed=0.3)
        ptz.stop()
    """

    # Direction -> (pan_x, tilt_y) velocity components
    _PAN_TILT = {
        'up':         (0.0,  1.0),
        'down':       (0.0, -1.0),
        'left':       (-1.0, 0.0),
        'right':      ( 1.0, 0.0),
        'up-left':    (-0.7, 0.7),
        'up-right':   ( 0.7, 0.7),
        'down-left':  (-0.7, -0.7),
        'down-right': ( 0.7, -0.7),
    }

    def __init__(self, xaddr: str, username: str = '', password: str = ''):
        self.xaddr = xaddr
        self.username = username
        self.password = password
        self._lock = threading.Lock()
        self._ptz_service = None
        self._profile_token = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _get_service(self):
        """Lazily connect to the ONVIF device and return the PTZ service."""
        if self._ptz_service is not None:
            return self._ptz_service, self._profile_token

        try:
            from onvif import ONVIFCamera
            import re
            m = re.match(r'https?://([^/:]+)(?::(\d+))?', self.xaddr)
            if not m:
                raise ValueError(f"Cannot parse xaddr: {self.xaddr}")
            host = m.group(1)
            port = int(m.group(2) or 80)
            cam = ONVIFCamera(host, port, self.username, self.password)
            svc = cam.create_ptz_service()
            media = cam.create_media_service()
            profiles = media.GetProfiles()
            token = profiles[0].token if profiles else None
            self._ptz_service = svc
            self._profile_token = token
            return svc, token
        except Exception as exc:
            log.error("PTZ connection failed: %s", exc)
            return None, None

    # ------------------------------------------------------------------
    # Move / Zoom / Stop
    # ------------------------------------------------------------------

    def move(self, direction: str, speed: float = 0.5, callback: Callable | None = None) -> bool:
        """Start continuous pan/tilt movement.  Call stop() to halt.

        Args:
            direction: one of up/down/left/right/up-left/up-right/down-left/down-right
            speed: 0.0-1.0
            callback: optional callable(success: bool) invoked after the command

        Returns:
            True if the command was sent successfully.
        """
        def _run():
            svc, token = self._get_service()
            if svc is None or token is None:
                if callback: callback(False)
                return
            pan_x, tilt_y = self._PAN_TILT.get(direction, (0.0, 0.0))
            speed_clamped = max(0.0, min(1.0, speed))
            with self._lock:
                try:
                    svc.ContinuousMove({
                        'ProfileToken': token,
                        'Velocity': {
                            'PanTilt': {'x': pan_x * speed_clamped, 'y': tilt_y * speed_clamped},
                            'Zoom': {'x': 0.0},
                        },
                    })
                    if callback: callback(True)
                except Exception as exc:
                    log.error("PTZ move failed: %s", exc)
                    if callback: callback(False)

        threading.Thread(target=_run, daemon=True).start()
        return True

    def zoom(self, direction: str, speed: float = 0.3, callback: Callable | None = None) -> bool:
        """Start continuous zoom in or out.

        Args:
            direction: 'in' or 'out'
            speed: 0.0-1.0
        """
        z = speed if direction == 'in' else -speed

        def _run():
            svc, token = self._get_service()
            if svc is None or token is None:
                if callback: callback(False)
                return
            with self._lock:
                try:
                    svc.ContinuousMove({
                        'ProfileToken': token,
                        'Velocity': {
                            'PanTilt': {'x': 0.0, 'y': 0.0},
                            'Zoom': {'x': z},
                        },
                    })
                    if callback: callback(True)
                except Exception as exc:
                    log.error("PTZ zoom failed: %s", exc)
                    if callback: callback(False)

        threading.Thread(target=_run, daemon=True).start()
        return True

    def stop(self, callback: Callable | None = None) -> bool:
        """Stop all PTZ movement."""
        def _run():
            svc, token = self._get_service()
            if svc is None or token is None:
                if callback: callback(False)
                return
            with self._lock:
                try:
                    svc.Stop({
                        'ProfileToken': token,
                        'PanTilt': True,
                        'Zoom': True,
                    })
                    if callback: callback(True)
                except Exception as exc:
                    log.error("PTZ stop failed: %s", exc)
                    if callback: callback(False)

        threading.Thread(target=_run, daemon=True).start()
        return True

    def go_to_preset(self, preset_token: str) -> bool:
        """Move to a named preset position."""
        def _run():
            svc, token = self._get_service()
            if svc is None or token is None:
                return
            with self._lock:
                try:
                    svc.GotoPreset({
                        'ProfileToken': token,
                        'PresetToken': preset_token,
                        'Speed': {},
                    })
                except Exception as exc:
                    log.error("PTZ goto-preset failed: %s", exc)

        threading.Thread(target=_run, daemon=True).start()
        return True

    def get_presets(self) -> list[dict]:
        """Return list of {token, name} dicts for available presets."""
        svc, token = self._get_service()
        if svc is None or token is None:
            return []
        try:
            presets = svc.GetPresets({'ProfileToken': token})
            return [{'token': p.token, 'name': getattr(p, 'Name', p.token)} for p in presets]
        except Exception as exc:
            log.error("PTZ get-presets failed: %s", exc)
            return []
