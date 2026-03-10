import click
import sys
from .vision import run_vision_loop
from .dashboard import run_dashboard

@click.command()
@click.option('--ui', is_flag=True, default=True,
              help='Launch the streaming dashboard (default).')
@click.option('--cli', is_flag=True, default=False,
              help='Force command-line mode.')
@click.option('--mode', default='edge', 
              type=click.Choice(['edge', 'motion', 'face', 'recognize', 'ocr', 'rembg', 'chroma'], case_sensitive=False),
              help='Processing mode (CLI only).')
@click.option('--source', default='0',
              help='Video source (CLI only).')
@click.option('--record', type=click.Path(),
              help='Path to save output (CLI only).')
@click.option('--stream-to', type=str,
              help='Broadcasting address (CLI only).')
@click.option('--stabilize', is_flag=True, default=False,
              help='Enable stabilization (CLI only).')
def main(ui, cli, mode, source, record, stream_to, stabilize):
    """FFTrix Video Processing Suite (VidGear & NiceGUI Powered)"""
    
    # If --cli is set, or --ui is explicitly false, run CLI mode
    if cli:
        processed_source = int(source) if source.isdigit() else source
        try:
            run_vision_loop(
                source=processed_source, 
                mode=mode.lower(),
                record_path=record,
                stream_address=stream_to,
                stabilize=stabilize
            )
        except KeyboardInterrupt:
            click.echo("\nExiting...")
            sys.exit(0)
    else:
        # Default: Launch the Dashboard
        click.echo("Launching FFTrix Dashboard on http://localhost:8080")
        run_dashboard()

if __name__ == "__main__":
    main()
