import sqlite3
import time
import os
import json
import hashlib

class Database:
    def __init__(self, db_path="fftrix_system.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        self._init_default_user()
        
    def create_tables(self):
        self.conn.execute('''CREATE TABLE IF NOT EXISTS events
                             (id INTEGER PRIMARY KEY, timestamp REAL, type TEXT, source TEXT, details TEXT, is_flag INTEGER DEFAULT 0)''')
        
        self.conn.execute('''CREATE TABLE IF NOT EXISTS cameras
                             (id TEXT PRIMARY KEY, name TEXT, url TEXT, mode TEXT, 
                              zones TEXT, record_247 INTEGER DEFAULT 0,
                              watermark_config TEXT)''')
        
        self.conn.execute('''CREATE TABLE IF NOT EXISTS users
                             (username TEXT PRIMARY KEY, password_hash TEXT)''')
        self.conn.commit()

    def _init_default_user(self):
        """Create a default admin:admin account if no users exist."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        if cur.fetchone()[0] == 0:
            self.add_user("admin", "admin")

    def add_user(self, username, password):
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        self.conn.execute("INSERT OR REPLACE INTO users (username, password_hash) VALUES (?, ?)", (username, pw_hash))
        self.conn.commit()

    def verify_user(self, username, password):
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        cur = self.conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        return row and row[0] == pw_hash

    def log_event(self, event_type, source, details="", is_flag=0):
        self.conn.execute("INSERT INTO events (timestamp, type, source, details, is_flag) VALUES (?, ?, ?, ?, ?)",
                          (time.time(), event_type, source, details, is_flag))
        self.conn.commit()

    def get_events(self, limit=50, flags_only=False):
        cur = self.conn.cursor()
        query = "SELECT timestamp, type, source, details, is_flag FROM events"
        if flags_only: query += " WHERE is_flag = 1"
        query += " ORDER BY timestamp DESC LIMIT ?"
        cur.execute(query, (limit,))
        return [{'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[0])), 
                 'type': r[1], 'source': r[2], 'details': r[3], 'is_flag': r[4]} for r in cur.fetchall()]

    def update_camera_config(self, cam_id, zones=None, record_247=None, watermark=None):
        if zones is not None:
            self.conn.execute("UPDATE cameras SET zones=? WHERE id=?", (json.dumps(zones), cam_id))
        if record_247 is not None:
            self.conn.execute("UPDATE cameras SET record_247=? WHERE id=?", (int(record_247), cam_id))
        if watermark is not None:
            self.conn.execute("UPDATE cameras SET watermark_config=? WHERE id=?", (json.dumps(watermark), cam_id))
        self.conn.commit()

    def add_camera(self, cam_id, name, url, mode):
        self.conn.execute("REPLACE INTO cameras (id, name, url, mode) VALUES (?, ?, ?, ?)", (cam_id, name, url, mode))
        self.conn.commit()

    def get_cameras(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, url, mode, zones, record_247, watermark_config FROM cameras")
        return [{'id': r[0], 'name': r[1], 'url': r[2], 'mode': r[3], 
                 'zones': json.loads(r[4]) if r[4] else [], 
                 'record_247': bool(r[5]),
                 'watermark': json.loads(r[6]) if r[6] else {}} for r in cur.fetchall()]
    
    def remove_camera(self, cam_id):
        self.conn.execute("DELETE FROM cameras WHERE id=?", (cam_id,))
        self.conn.commit()
