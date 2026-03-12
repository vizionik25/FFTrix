"""FFTrix Alert Notifications — email (SMTP) and webhook dispatchers."""

from __future__ import annotations

import os
import time
import threading
import smtplib
import logging
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SMTP helpers
# ---------------------------------------------------------------------------

def _build_smtp_conn() -> smtplib.SMTP | None:
    """Return an authenticated SMTP connection from env vars, or None if not configured."""
    host = os.environ.get("FFTRIX_SMTP_HOST", "")
    port = int(os.environ.get("FFTRIX_SMTP_PORT", 587))
    user = os.environ.get("FFTRIX_SMTP_USER", "")
    password = os.environ.get("FFTRIX_SMTP_PASSWORD", "")
    if not host or not user:
        return None
    try:
        conn = smtplib.SMTP(host, port, timeout=10)
        conn.ehlo()
        conn.starttls()
        conn.login(user, password)
        return conn
    except Exception as exc:
        log.warning("SMTP connection failed: %s", exc)
        return None


def send_email(to: str, subject: str, body: str, snapshot_path: str | None = None) -> bool:
    """Send an alert email, optionally attaching a snapshot image.

    Returns True on success, False on failure.
    """
    from_addr = os.environ.get("FFTRIX_ALERT_FROM", os.environ.get("FFTRIX_SMTP_USER", "fftrix@localhost"))
    try:
        conn = _build_smtp_conn()
        if conn is None:
            log.warning("Email alert skipped — SMTP not configured.")
            return False

        if snapshot_path and Path(snapshot_path).exists():
            msg = MIMEMultipart()
            msg["Subject"] = subject
            msg["From"] = from_addr
            msg["To"] = to
            msg.attach(MIMEText(body, "plain"))
            with open(snapshot_path, "rb") as f:
                ext = Path(snapshot_path).suffix.lstrip(".").lower() or "jpeg"
                img = MIMEImage(f.read(), _subtype=ext)
                img.add_header("Content-Disposition", "attachment", filename=Path(snapshot_path).name)
                msg.attach(img)
            conn.send_message(msg)
        else:
            simple = EmailMessage()
            simple["Subject"] = subject
            simple["From"] = from_addr
            simple["To"] = to
            simple.set_content(body)
            conn.send_message(simple)

        conn.quit()
        return True
    except Exception as exc:
        log.error("Failed to send email alert: %s", exc)
        return False


def send_webhook(url: str, payload: dict) -> bool:
    """POST a JSON payload to a webhook URL (Slack, Discord, ntfy.sh, etc.).

    Returns True on success, False on failure.
    """
    try:
        import urllib.request, json as _json
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status < 400
    except Exception as exc:
        log.error("Webhook alert failed (%s): %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """Dispatches email and/or webhook alerts on flagged camera events.

    Attributes:
        cooldown_s: Default cooldown in seconds between alerts for the same camera.
    """

    def __init__(self, cooldown_s: int = 60):
        self.cooldown_s = cooldown_s
        self._last_fired: dict[str, float] = {}  # cam_id -> last alert timestamp
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(self, cam_id: str, config: dict) -> None:
        """Attach an alert config dict to a camera.

        Config keys:
            email (str): recipient address, or empty to disable
            webhook_url (str): HTTP URL to POST to, or empty to disable
            cooldown_s (int): per-camera cooldown override
            enabled (bool): master switch
        """
        # Store on the instance keyed by cam_id; config is read at fire() time
        # so callers can update it any time by calling configure() again.
        if not hasattr(self, "_configs"):
            self._configs: dict[str, dict] = {}
        self._configs[cam_id] = config

    def get_config(self, cam_id: str) -> dict:
        configs = getattr(self, "_configs", {})
        return configs.get(cam_id, {})

    def fire(
        self,
        cam_id: str,
        event_type: str,
        details: str = "",
        snapshot_path: str | None = None,
    ) -> None:
        """Fire alerts for a flagged event, respecting cooldown.

        Runs dispatch in a background thread so it never blocks the video pipeline.
        """
        config = self.get_config(cam_id)
        if not config.get("enabled", False):
            return

        cooldown = config.get("cooldown_s", self.cooldown_s)
        with self._lock:
            last = self._last_fired.get(cam_id, 0)
            if time.time() - last < cooldown:
                return  # still in cooldown
            self._last_fired[cam_id] = time.time()

        threading.Thread(
            target=self._dispatch,
            args=(cam_id, event_type, details, snapshot_path, config),
            daemon=True,
        ).start()

    def reset_cooldown(self, cam_id: str) -> None:
        """Force-reset the cooldown for a camera (useful in tests)."""
        with self._lock:
            self._last_fired.pop(cam_id, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        cam_id: str,
        event_type: str,
        details: str,
        snapshot_path: str | None,
        config: dict,
    ) -> None:
        subject = f"[FFTrix] {event_type} detected on {cam_id}"
        body = (
            f"Camera: {cam_id}\n"
            f"Event: {event_type}\n"
            f"Details: {details}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        payload = {
            "text": subject,
            "camera": cam_id,
            "event": event_type,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        email = config.get("email", "").strip()
        if email:
            send_email(email, subject, body, snapshot_path)

        webhook = config.get("webhook_url", "").strip()
        if webhook:
            send_webhook(webhook, payload)
