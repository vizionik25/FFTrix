"""ONVIF Device Auto-Discovery via WS-Discovery + ONVIF SOAP."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

from wsdiscovery import WSDiscovery
from wsdiscovery.service import Service


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredDevice:
    xaddr: str
    ip: str = ""
    port: int = 80
    name: str = "Unknown ONVIF Device"
    manufacturer: str = ""
    model: str = ""
    firmware: str = ""
    serial: str = ""
    rtsp_uris: list[str] = field(default_factory=list)
    requires_auth: bool = False

    def to_dict(self) -> dict:
        return {
            "xaddr": self.xaddr,
            "ip": self.ip,
            "port": self.port,
            "name": self.name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "firmware": self.firmware,
            "serial": self.serial,
            "rtsp_uris": self.rtsp_uris,
            "requires_auth": self.requires_auth,
        }

    @staticmethod
    def from_dict(d: dict) -> "DiscoveredDevice":
        return DiscoveredDevice(
            xaddr=d.get("xaddr", ""),
            ip=d.get("ip", ""),
            port=d.get("port", 80),
            name=d.get("name", "Unknown ONVIF Device"),
            manufacturer=d.get("manufacturer", ""),
            model=d.get("model", ""),
            firmware=d.get("firmware", ""),
            serial=d.get("serial", ""),
            rtsp_uris=d.get("rtsp_uris", []),
            requires_auth=d.get("requires_auth", False),
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_ip_port(xaddr: str) -> tuple[str, int]:
    """Extract (host, port) from an ONVIF XAddr URL."""
    try:
        parsed = urlparse(xaddr)
        host = parsed.hostname or ""
        port = parsed.port or 80
        return host, port
    except Exception:
        return "", 80


def _probe_device(xaddr: str, username: str, password: str) -> DiscoveredDevice | None:
    """Connect to a single ONVIF device and pull its metadata and stream URIs."""
    ip, port = _parse_ip_port(xaddr)
    if not ip:
        return None

    device = DiscoveredDevice(xaddr=xaddr, ip=ip, port=port)

    try:
        from onvif import ONVIFCamera  # imported here to make mocking easier

        cam = ONVIFCamera(ip, port, username, password)
        device_svc = cam.create_devicemgmt_service()

        # Pull device info
        info = device_svc.GetDeviceInformation()
        device.manufacturer = getattr(info, "Manufacturer", "") or ""
        device.model = getattr(info, "Model", "") or ""
        device.firmware = getattr(info, "FirmwareVersion", "") or ""
        device.serial = getattr(info, "SerialNumber", "") or ""
        device.name = f"{device.manufacturer} {device.model}".strip() or "Unknown ONVIF Device"

        # Pull RTSP stream URIs from all media profiles
        try:
            media_svc = cam.create_media_service()
            profiles = media_svc.GetProfiles()
            for profile in profiles:
                try:
                    stream_setup = media_svc.create_type("GetStreamUri")
                    stream_setup.ProfileToken = profile.token
                    stream_setup.StreamSetup = {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}}
                    uri_resp = media_svc.GetStreamUri(stream_setup)
                    uri = getattr(uri_resp, "Uri", "") or getattr(uri_resp, "uri", "")
                    if uri:
                        device.rtsp_uris.append(uri)
                except Exception:
                    pass
        except Exception:
            pass

    except Exception as exc:
        exc_str = str(exc).lower()
        if "401" in exc_str or "unauthorized" in exc_str or "auth" in exc_str:
            device.requires_auth = True
            # Keep partial device — we know IP and port at minimum
            return device
        # Any other error — device is unreachable or non-ONVIF
        return None

    return device


# ---------------------------------------------------------------------------
# Main discovery class
# ---------------------------------------------------------------------------

class ONVIFDiscovery:
    """Discover ONVIF cameras on the local network using WS-Discovery."""

    def scan(
        self,
        timeout: int = 5,
        username: str = "",
        password: str = "",
        progress_callback: Callable[[DiscoveredDevice], None] | None = None,
    ) -> list[DiscoveredDevice]:
        """
        Broadcast a WS-Discovery probe and interrogate each responding device.

        Args:
            timeout: Seconds to wait for WS-Discovery responses.
            username: ONVIF credential (common default: 'admin').
            password: ONVIF credential.
            progress_callback: Called with each DiscoveredDevice as it is found.

        Returns:
            List of DiscoveredDevice, deduplicated by IP.
        """
        # 1. Broadcast WS-Discovery probe
        wsd = WSDiscovery()
        wsd.start()
        try:
            services: list[Service] = wsd.searchServices(timeout=timeout)
        finally:
            wsd.stop()

        if not services:
            return []

        # 2. Collect unique XAddrs
        xaddrs: list[str] = []
        seen_ips: set[str] = set()
        for svc in services:
            for xaddr in (svc.getXAddrs() or []):
                ip, _ = _parse_ip_port(xaddr)
                if ip and ip not in seen_ips:
                    seen_ips.add(ip)
                    xaddrs.append(xaddr)

        if not xaddrs:
            return []

        # 3. Probe each device in parallel threads
        results: list[DiscoveredDevice] = []
        lock = threading.Lock()

        def _worker(xaddr: str) -> None:
            dev = _probe_device(xaddr, username, password)
            if dev is not None:
                with lock:
                    results.append(dev)
                if progress_callback:
                    progress_callback(dev)

        threads = [threading.Thread(target=_worker, args=(x,), daemon=True) for x in xaddrs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout + 2)

        return results
