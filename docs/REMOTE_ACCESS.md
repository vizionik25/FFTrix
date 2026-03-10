# Security & Remote Access

FFTrix is designed to be accessible from anywhere in the world while maintaining strict enterprise security standards.

## 🔐 Authentication

All remote interfaces are protected by a secure gateway.

### 1. Session Management
- **Persistence:** Uses encrypted browser cookies via NiceGUI's `app.storage.user`.
- **Redirects:** Any attempt to access monitoring routes without a valid session will result in a forced redirect to the `/login` portal.

### 2. User Security
- **Hashed Passwords:** Credentials are stored in the database using **SHA-256 hashing**. FFTrix never stores plain-text passwords.
- **Default Account:**
  - **Username:** `admin`
  - **Password:** `admin`
  - *Recommendation:* Add a new user or update the password in `database.py` for production deployments.

---

## 🌍 Worldwide Access (Zero-Trust Tunnel)

FFTrix includes an integrated reverse proxy feature that allows you to bypass firewalls and NAT without configuring your router.

### How to Deploy Remotely:
Run the server with the `--remote` flag:
```bash
uv run fftrix --remote
```

### What happens behind the scenes:
1. **Tunneling:** The system uses `pyngrok` to create a secure, encrypted tunnel from your local port `8080` to the `ngrok` cloud.
2. **Public URL:** A unique, random HTTPS URL is generated (e.g., `https://a1b2c3d4.ngrok-free.app`).
3. **Encrypted Path:** External traffic travels through the ngrok tunnel, hitting your server without ever exposing your home IP or requiring open ports.

---

## 🛡 Network Hardening

### 1. Host Binding
The server binds to `0.0.0.0` by default. This allows devices on your local Wi-Fi to connect via the server's local IP address (e.g., `http://192.168.1.50:8080`).

### 2. Disabling API Docs
For security, standard FastAPI documentation routes (`/docs` and `/redoc`) are disabled in the production UI launch to prevent information disclosure.

### 3. Lockdown Mode
You can instantly terminate all remote access and background streams by clicking the **HALT ALL** button in the Command Center header.
