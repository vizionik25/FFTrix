import click
import sys
import os
from .dashboard import run_dashboard
from .database import Database, DB_PATH

def handle_serve(ui=True, remote=False):
    """Business logic for serving dashboard."""
    if ui:
        run_dashboard(remote=remote)
    return True

def handle_user_add(username, password):
    db = Database(db_path=str(DB_PATH))
    db.add_user(username, password)
    return True

@click.group()
def cli():
    """FFTrix Enterprise Command Line Interface"""
    pass

@cli.command()
@click.option('--ui/--no-ui', is_flag=True, default=True)
@click.option('--remote', is_flag=True, default=False)
def serve(ui, remote):
    """Start the FFTrix Surveillance Server."""
    click.echo("🚀 Launching FFTrix...")
    handle_serve(ui=ui, remote=remote)

@cli.group()
def user():
    """Manage authorized operators."""
    pass

@user.command(name='add')
@click.argument('username')
@click.password_option()
def add_user(username, password):
    """Provision a new operator."""
    handle_user_add(username, password)
    click.echo(f"✅ Operator '{username}' provisioned.")

@user.command(name='list')
def list_users():
    """List operators."""
    db = Database(db_path=str(DB_PATH))
    cur = db.conn.cursor()
    cur.execute("SELECT username FROM users")
    users = cur.fetchall()
    for u in users: click.echo(f"- {u[0]}")

@user.command(name='delete')
@click.argument('username')
def delete_user(username):
    """Revoke access."""
    db = Database(db_path=str(DB_PATH))
    db.conn.execute("DELETE FROM users WHERE username=?", (username,))
    db.conn.commit()
    click.echo(f"❌ Revoked '{username}'.")

@user.command(name='reset-password')
@click.argument('username')
@click.password_option(prompt='New password', confirmation_prompt='Confirm new password')
def reset_password(username, password):
    """Reset an operator's password (use when locked out)."""
    db = Database(db_path=str(DB_PATH))
    cur = db.conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
    if not cur.fetchone():
        click.echo(f"❌ User '{username}' not found. Use 'fftrix user list' to see all users.", err=True)
        sys.exit(1)
    db.change_user_password(username, username, password)
    click.echo(f"✅ Password updated for '{username}'. You can now log in with the new password.")

@user.command(name='reset-admin')
@click.confirmation_option(prompt='⚠️  This will delete ALL users and reset admin:admin. Continue?')
def reset_admin():
    """Emergency factory reset: wipes all users, recreates admin:admin.\n\nOn next login the first-use dialog forces you to set a new password."""
    db = Database(db_path=str(DB_PATH))
    db.conn.execute("DELETE FROM users")
    db.conn.commit()
    db._init_default_user()  # recreates admin:admin
    click.echo("✅ All users cleared. Login with admin:admin — you'll be forced to change it immediately.")

if __name__ == "__main__":
    cli()
