import click
import sys
import os
from .vision import run_vision_loop
from .dashboard import run_dashboard
from .database import Database

@click.group()
def cli():
    """FFTrix Enterprise Command Line Interface"""
    pass

@cli.command()
@click.option('--ui/--no-ui', is_flag=True, default=True, help='Launch the streaming dashboard (default: True).')
@click.option('--cli-mode', is_flag=True, default=False, help='Force command-line vision mode.')
@click.option('--remote', is_flag=True, default=False, help='Expose to public internet via secure tunnel.')
@click.option('--mode', default='none', type=click.Choice(['none', 'edge', 'motion', 'face', 'object'], case_sensitive=False))
@click.option('--source', default='0', help='Video source.')
def serve(ui, cli_mode, remote, mode, source):
    """Start the FFTrix Server or CLI vision engine."""
    if cli_mode or not ui:
        processed_source = int(source) if source.isdigit() else source
        try:
            run_vision_loop(source=processed_source, mode=mode.lower())
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        click.echo("🚀 Launching FFTrix Enterprise Core...")
        run_dashboard(remote=remote)

@cli.group()
def user():
    """Manage authorized operators and passcodes."""
    pass

@user.command(name='add')
@click.argument('username')
@click.password_option()
def add_user(username, password):
    """Provision a new operator node."""
    db = Database()
    db.add_user(username, password)
    click.echo(f"✅ Operator '{username}' provisioned successfully.")

@user.command(name='list')
def list_users():
    """List all authorized operator identifiers."""
    db = Database()
    cur = db.conn.cursor()
    cur.execute("SELECT username FROM users")
    users = cur.fetchall()
    click.echo("\nAUTHORIZED OPERATORS:")
    for u in users:
        click.echo(f"- {u[0]}")
    click.echo("")

@user.command(name='delete')
@click.argument('username')
def delete_user(username):
    """Revoke access for an operator."""
    db = Database()
    db.conn.execute("DELETE FROM users WHERE username=?", (username,))
    db.conn.commit()
    click.echo(f"❌ Access revoked for '{username}'.")

if __name__ == "__main__":
    cli()
