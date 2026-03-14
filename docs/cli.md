# CLI Reference

The `fftrix` CLI is the primary interface for managing the surveillance server and authorized operators.

## 🚀 Server Management

### `fftrix serve`
Launches the FFTrix server and the NiceGUI-based dashboard.

**Options:**
- `--ui / --no-ui`: (Default: `--ui`) Automatically launch a browser to the dashboard. Use `--no-ui` for headless deployments.
- `--remote`: (Default: `False`) Binds the server to all network interfaces (`0.0.0.0`) instead of localhost.

---

## 👤 User Provisioning

All user-related tasks are grouped under the `fftrix user` command.

### `fftrix user add <username>`
Adds a new authorized operator. Prompts for a password.

**Options:**
- `--role [admin|viewer]`: (Default: `admin`)
  - `admin`: Full access to settings, cameras, and user management.
  - `viewer`: Access to live view and events only.

### `fftrix user list`
Displays all provisioned operators and their roles.

### `fftrix user reset-password <username>`
Changes the password for an existing user. Use this if an operator is locked out.

### `fftrix user delete <username>`
Immediately revokes access for an operator.

---

## 🆘 Maintenance & Emergency

### `fftrix user reset-admin`
**DANGER:** This command wipes the entire `users` table and restores the default `admin:admin` credentials. Use this as a factory reset if you lose access to all admin accounts. 

*On next login, the first-use dialog will force you to change these credentials.*
