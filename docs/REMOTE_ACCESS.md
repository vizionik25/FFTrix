# Security & Remote Access

FFTrix provides enterprise-grade security for worldwide surveillance monitoring.

## 🔐 Operator Authentication

Access to the Command Center is restricted via a secure gateway.

### 1. Account Management (CLI)
FFTrix includes a built-in user management system. Use the following commands to manage access (assuming standard installation):

**Add or Update an Operator:**
```bash
fftrix user add [username]
```
*This will securely prompt you for a password and store the SHA-256 hash in the database.*

**List All Operators:**
```bash
fftrix user list
```

**Revoke Access:**
```bash
fftrix user delete [username]
```

---

### 2. Technical Security Details
- **Password Hashing:** FFTrix uses **SHA-256** with unique salts (inherent to the database schema logic) to ensure passwords are never stored in plain text.
- **Session Management:** Encrypted browser cookies manage operator sessions. Sessions are automatically invalidated when the server is restarted or the operator logs out.
- **Default Credentials:** The system initializes with `admin` / `admin`. **You must change this immediately** by running the `user add admin` command.

---

## 🌍 Zero-Trust Remote Access

FFTrix can be exposed to the public internet securely without opening ports on your router.

### Deploying the Secure Tunnel:
```bash
fftrix serve --remote
```

### How the Tunnel Works:
1. **Encrypted Reverse Proxy:** FFTrix uses `pyngrok` to create a secure tunnel between your local machine and the public internet.
2. **Dynamic URL:** A unique HTTPS URL is generated upon launch. This URL is the only entry point to your server.
3. **NAT Bypass:** This method works even behind strict firewalls and CGNAT, as the connection is established from the *inside* out.

---

## 🛡 Network Hardening

- **Host Binding:** By default, the server binds to `0.0.0.0`, making it accessible to any device on your local network via your machine's IP (e.g., `http://192.168.1.15:8080`).
- **Emergency Lockdown:** The `HALT ALL` button in the UI header instantly kills all background processing threads and halts streaming across all nodes.
- **Database Encryption:** The `fftrix_system.db` file should be treated as a sensitive asset. Ensure the file system permissions restrict access to the application's service account.
