import sqlite3
import time
import os
import json
import hashlib
from pathlib import Path

# Centralized Home Path Configuration
FFTRIX_HOME = Path.home() / ".fftrix"
DB_PATH = FFTRIX_HOME / "system.db"
RECORDINGS_DIR = FFTRIX_HOME / "recordings"
SNAPSHOTS_DIR = FFTRIX_HOME / "snapshots"
STATIC_DIR = FFTRIX_HOME / "static"

class Database:
    def __init__(self, db_path=None):
        # Ensure FFTRIX_HOME exists on first run
        FFTRIX_HOME.mkdir(parents=True, exist_ok=True)
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        STATIC_DIR.mkdir(parents=True, exist_ok=True)

        target_db = db_path if db_path else str(DB_PATH)
        self.conn = sqlite3.connect(target_db, check_same_thread=False)
        self.create_tables()
        self._init_default_user()
        
    def create_tables(self):
        self.conn.execute('''CREATE TABLE IF NOT EXISTS events
                             (id INTEGER PRIMARY KEY, timestamp REAL, type TEXT,
                              source TEXT, details TEXT, is_flag INTEGER DEFAULT 0,
                              snapshot_path TEXT)''')

        self.conn.execute('''CREATE TABLE IF NOT EXISTS cameras
                             (id TEXT PRIMARY KEY, name TEXT, url TEXT, mode TEXT,
                              zones TEXT, record_247 INTEGER DEFAULT 0,
                              watermark_config TEXT,
                              retention_days INTEGER DEFAULT 30,
                              alert_config TEXT,
                              privacy_zones TEXT,
                              arm_schedule TEXT,
                              xaddr TEXT,
                              lpr_watchlist TEXT,
                              role TEXT DEFAULT \'admin\')''')

        self.conn.execute('''CREATE TABLE IF NOT EXISTS users
                             (username TEXT PRIMARY KEY, password_hash TEXT,
                              role TEXT DEFAULT \'admin\')''')

        self.conn.execute('''CREATE TABLE IF NOT EXISTS discovered_devices
                             (ip TEXT PRIMARY KEY, xaddr TEXT, name TEXT,
                              manufacturer TEXT, model TEXT, firmware TEXT,
                              serial TEXT, rtsp_uris TEXT,
                              requires_auth INTEGER DEFAULT 0, last_seen REAL)''')

        # Migrate existing tables — add columns if they don't already exist
        self._migrate_tables()
        self.conn.commit()

    def _migrate_tables(self):
        """Add new columns to existing tables without destroying data."""
        migrations = [
            ("events",   "ALTER TABLE events ADD COLUMN snapshot_path TEXT"),
            ("cameras",  "ALTER TABLE cameras ADD COLUMN retention_days INTEGER DEFAULT 30"),
            ("cameras",  "ALTER TABLE cameras ADD COLUMN alert_config TEXT"),
            ("cameras",  "ALTER TABLE cameras ADD COLUMN privacy_zones TEXT"),
            ("cameras",  "ALTER TABLE cameras ADD COLUMN arm_schedule TEXT"),
            ("cameras",  "ALTER TABLE cameras ADD COLUMN xaddr TEXT"),
            ("cameras",  "ALTER TABLE cameras ADD COLUMN lpr_watchlist TEXT"),
            ("users",    "ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'admin'"),
        ]
        for _table, sql in migrations:
            try:
                self.conn.execute(sql)
            except Exception:
                pass  # column already exists

    def _init_default_user(self):
        """Create a default admin:admin account if no users exist."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        if cur.fetchone()[0] == 0:
            self.add_user("admin", "admin")

    def add_user(self, username, password, role: str = "admin"):
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        self.conn.execute(
            "INSERT OR REPLACE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, pw_hash, role),
        )
        self.conn.commit()

    def get_user_role(self, username: str) -> str:
        """Return the role ('admin' or 'viewer') for a user, defaulting to 'admin'."""
        cur = self.conn.cursor()
        cur.execute("SELECT role FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        return row[0] if row and row[0] else "admin"

    def verify_user(self, username, password):
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        cur = self.conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        return row and row[0] == pw_hash

    def is_default_credentials(self) -> bool:
        """Return True if the default admin:admin credentials are still active."""
        default_hash = hashlib.sha256(b"admin").hexdigest()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT 1 FROM users WHERE username=? AND password_hash=?",
            ("admin", default_hash),
        )
        return cur.fetchone() is not None

    def change_user_password(self, old_username: str, new_username: str, new_password: str) -> None:
        """Rename a user and set a new password atomically."""
        pw_hash = hashlib.sha256(new_password.encode()).hexdigest()
        self.conn.execute("DELETE FROM users WHERE username=?", (old_username,))
        self.conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (new_username, pw_hash),
        )
        self.conn.commit()


    def log_event(self, event_type, source, details="", is_flag=0, snapshot_path=None):
        self.conn.execute(
            "INSERT INTO events (timestamp, type, source, details, is_flag, snapshot_path) VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), event_type, source, details, is_flag, snapshot_path),
        )
        self.conn.commit()

    def get_events(self, limit=50, flags_only=False):
        cur = self.conn.cursor()
        query = "SELECT timestamp, type, source, details, is_flag, snapshot_path FROM events"
        if flags_only: query += " WHERE is_flag = 1"
        query += " ORDER BY timestamp DESC LIMIT ?"
        cur.execute(query, (limit,))
        return [
            {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[0])),
                'type': r[1], 'source': r[2], 'details': r[3],
                'is_flag': r[4], 'snapshot_path': r[5],
            }
            for r in cur.fetchall()
        ]

    def update_camera_config(self, cam_id, zones=None, record_247=None, watermark=None,
                             retention_days=None, alert_config=None, privacy_zones=None,
                             arm_schedule=None, xaddr=None, lpr_watchlist=None):
        updates = []
        params = []
        if zones is not None:         updates.append("zones=?");            params.append(json.dumps(zones))
        if record_247 is not None:    updates.append("record_247=?");       params.append(int(record_247))
        if watermark is not None:     updates.append("watermark_config=?"); params.append(json.dumps(watermark))
        if retention_days is not None:updates.append("retention_days=?");   params.append(int(retention_days))
        if alert_config is not None:  updates.append("alert_config=?");     params.append(json.dumps(alert_config))
        if privacy_zones is not None: updates.append("privacy_zones=?");    params.append(json.dumps(privacy_zones))
        if arm_schedule is not None:  updates.append("arm_schedule=?");     params.append(json.dumps(arm_schedule))
        if xaddr is not None:         updates.append("xaddr=?");            params.append(xaddr)
        if lpr_watchlist is not None: updates.append("lpr_watchlist=?");    params.append(json.dumps(lpr_watchlist))
        if updates:
            params.append(cam_id)
            self.conn.execute(f"UPDATE cameras SET {', '.join(updates)} WHERE id=?", params)
            self.conn.commit()

    def add_camera(self, cam_id, name, url, mode):
        self.conn.execute("REPLACE INTO cameras (id, name, url, mode) VALUES (?, ?, ?, ?)", (cam_id, name, url, mode))
        self.conn.commit()

    def get_cameras(self):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, name, url, mode, zones, record_247, watermark_config, "
            "retention_days, alert_config, privacy_zones, arm_schedule, xaddr, lpr_watchlist "
            "FROM cameras"
        )
        rows = cur.fetchall()
        return [
            {
                'id': r[0], 'name': r[1], 'url': r[2], 'mode': r[3],
                'zones': json.loads(r[4]) if r[4] else [],
                'record_247': bool(r[5]),
                'watermark': json.loads(r[6]) if r[6] else {},
                'retention_days': r[7] if r[7] is not None else 30,
                'alert_config': json.loads(r[8]) if r[8] else {},
                'privacy_zones': json.loads(r[9]) if r[9] else [],
                'arm_schedule': json.loads(r[10]) if r[10] else [],
                'xaddr': r[11] or '',
                'lpr_watchlist': json.loads(r[12]) if r[12] else [],
            }
            for r in rows
        ]

    def remove_camera(self, cam_id):
        self.conn.execute("DELETE FROM cameras WHERE id=?", (cam_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Discovered Devices (ONVIF scan cache)
    # ------------------------------------------------------------------

    def upsert_discovered_device(self, device: dict) -> None:
        """Insert or update a discovered ONVIF device record."""
        self.conn.execute(
            """INSERT OR REPLACE INTO discovered_devices
               (ip, xaddr, name, manufacturer, model, firmware, serial,
                rtsp_uris, requires_auth, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                device.get("ip", ""),
                device.get("xaddr", ""),
                device.get("name", ""),
                device.get("manufacturer", ""),
                device.get("model", ""),
                device.get("firmware", ""),
                device.get("serial", ""),
                json.dumps(device.get("rtsp_uris", [])),
                int(device.get("requires_auth", False)),
                time.time(),
            ),
        )
        self.conn.commit()

    def get_discovered_devices(self) -> list[dict]:
        """Return all cached discovered devices, most-recently-seen first."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT ip, xaddr, name, manufacturer, model, firmware, serial, "
            "rtsp_uris, requires_auth, last_seen FROM discovered_devices ORDER BY last_seen DESC"
        )
        return [
            {
                "ip": r[0], "xaddr": r[1], "name": r[2],
                "manufacturer": r[3], "model": r[4], "firmware": r[5],
                "serial": r[6], "rtsp_uris": json.loads(r[7]) if r[7] else [],
                "requires_auth": bool(r[8]),
                "last_seen": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[9])) if r[9] else "",
            }
            for r in cur.fetchall()
        ]

    def clear_discovered_devices(self) -> None:
        """Remove all cached discovered devices."""
        self.conn.execute("DELETE FROM discovered_devices")
        self.conn.commit()
