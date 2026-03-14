# Secure Deployment

FFTrix is designed for high-security deployments using a tiered networking architecture. It leverages **Docker**, **Angie** (Nginx fork), and **Tailscale Funnel** to provide a fully encrypted, authenticated endpoint with zero open firewall ports.

## 🏗️ Networking Flow

The deployment uses a shared network stack via Docker's `network_mode: "service:tailscale"`.

1.  **FFTrix (App Layer)**: Listens on port `8080`.
2.  **Angie (Internal Proxy)**: Listens on port `80`. It intercepts internal traffic and relays it to the application at `http://localhost:8080`, applying security headers and WebSocket optimizations.
3.  **Tailscale (Secure Egress)**: 
    - **Internal**: Listens on port `80` within the shared container network.
    - **External**: Exposes the service to the internet on port `443` via **Tailscale Funnel**.
    - **Encryption**: Automatically manages SSL/TLS certificates via **Let's Encrypt** and provides a secure **MagicDNS** URL.

---

## 🔒 Security Features

### 1. Zero-Trust Ingress
Because FFTrix uses Tailscale Funnel, you do not need to open any ports on your router or firewall. All external traffic is routed through Tailscale’s secure edge nodes.

### 2. End-to-End Encryption
Tailscale handles the TLS termination on port 443. It automatically provisions and renews SSL certificates, ensuring that your data is encrypted from the user's browser to the Tailscale node.

### 3. Angie Reverse Proxy
Angie provides an additional layer of security by filtering requests and adding critical headers:
- `X-Frame-Options: "SAMEORIGIN"` (Prevents clickjacking)
- `X-XSS-Protection: "1; mode=block"` (Mitigates XSS attacks)
- `X-Content-Type-Options: "nosniff"` (Prevents MIME-sniffing)

---

## 🚀 Deployment Steps

### 1. Preparation
Ensure you have a Tailscale account and have enabled **MagicDNS** and **HTTPS Certificates** in your Tailscale admin console.

### 2. Environment Setup
Create a `.env` file in the project root:
```env
TS_AUTHKEY=tskey-auth-XXXXXXXXXXXX
FFTRIX_SECRET_KEY=your-random-secret
```

### 3. Configuration
The `funnel.json` file dictates how Tailscale exposes the internal port 80 to the public port 443:
```json
{
  "Web": {
    "80": {
      "Handlers": ["proxy"],
      "Funnel": true
    }
  }
}
```

### 4. Launch
```bash
docker compose up -d
```

---

## 💾 Data Persistence

FFTrix uses persistent Docker volumes to ensure data survives container restarts:
- `fftrix-data`: Maps to `/home/fftrix/.fftrix`, storing the `system.db`, recordings, and snapshots.
- `tailscale-data`: Stores the Tailscale node identity, ensuring your MagicDNS name remains consistent.
