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

FFTrix can be exposed to the public internet securely without opening ports on your router, utilizing a combination of **Tailscale Funnel** for secure egress/tunnelling and **Angie (NGINX)** for secured ingress.

### Deploying the Secure Tunnel:
```bash
fftrix serve --remote
```

### How the Tunnel Works:
1. **Secure Egress (Tailscale):** FFTrix uses Tailscale Funnel to create an encrypted tunnel from your node to the Tailscale edge. This bypasses NAT and CGNAT without requiring port forwarding.
2. **Secured Ingress (Angie):** The system generates an `angie.conf` configuration in your `~/.fftrix/` directory. Angie acts as a high-performance reverse proxy, providing an additional layer of security, WebSocket optimization, and header hardening.
3. **Global DNS:** Your server becomes accessible via your unique Tailnet DNS name (e.g., `https://your-node.tailnet-name.ts.net`).

### Optional: Automatic Authentication
If you are deploying in a headless environment or container, you can provide a `TS_AUTHKEY` in your `.env` file. FFTrix will automatically attempt to authenticate and bring the Tailscale node online before starting the Funnel.

```bash
# .env
TS_AUTHKEY=tskey-auth-xxxxxx
```

### Manual Ingress Setup (Optional):
If you have **Angie** installed on your system, you can manually launch the ingress proxy with the generated configuration found in your home directory:
```bash
sudo angie -c ~/.fftrix/angie.conf
```

---

## 🛡 Network Hardening

- **Host Binding:** By default, the server binds to `0.0.0.0`, making it accessible to any device on your local network via your machine's IP (e.g., `http://192.168.1.15:8080`).
- **Emergency Lockdown:** The `HALT ALL` button in the UI header instantly kills all background processing threads and halts streaming across all nodes.
- **Database Encryption:** The `fftrix_system.db` file should be treated as a sensitive asset. Ensure the file system permissions restrict access to the application's service account.
