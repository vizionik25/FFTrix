import click
import sys
from .vision import run_vision_loop

@click.command()
@click.option('--mode', default='edge', 
              type=click.Choice(['edge', 'motion', 'face', 'recognize', 'ocr', 'rembg', 'chroma'], case_sensitive=False),
              help='Processing mode.')
@click.option('--source', default='0',
              help='Video source (0 for webcam, path to file, or YouTube URL).')
@click.option('--record', type=click.Path(),
              help='Path to save the output video (e.g., output.mp4).')
@click.option('--stream-to', type=str,
              help='Broadcasting address for NetGear (e.g., 127.0.0.1:5555).')
@click.option('--stabilize', is_flag=True, default=False,
              help='Enable real-time video stabilization.')
def main(mode, source, record, stream_to, stabilize):
    """FFTrix Video Processing Suite (VidGear Powered)"""
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

if __name__ == "__main__":
    main()
