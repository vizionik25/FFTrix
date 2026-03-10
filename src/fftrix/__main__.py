import click
import sys
from .vision import run_vision_loop

@click.command()
@click.option('--mode', default='edge', 
              type=click.Choice(['edge', 'motion', 'face', 'recognize', 'object', 'ocr', 'rembg', 'chroma'], case_sensitive=False),
              help='Processing mode: edge, motion, face, recognize, object, ocr, rembg, chroma')
@click.option('--source', default='0',
              help='Video source (0 for webcam or path to file)')
def main(mode, source):
    """FFTrix Video Processing Suite"""
    processed_source = int(source) if source.isdigit() else source
    
    try:
        run_vision_loop(source=processed_source, mode=mode.lower())
    except KeyboardInterrupt:
        click.echo("\nExiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
