#!/usr/bin/env python3
"""
Script to mirror (horizontally flip) a video file.
"""

import argparse
import os
import subprocess
from pathlib import Path


def mirror_video(input_path, output_path=None):
    """
    Mirror a video horizontally using ffmpeg.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output video file (optional, defaults to input_mirrored.ext)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_mirrored{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Mirroring video:")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    
    # Use ffmpeg to horizontally flip the video
    # hflip filter flips the video horizontally
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', 'hflip',
        '-c:a', 'copy',  # Copy audio without re-encoding
        '-y',  # Overwrite output file if it exists
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"\n✓ Successfully created mirrored video: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running ffmpeg:")
        print(e.stderr)
        raise
    except FileNotFoundError:
        print("\n✗ Error: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Mirror (horizontally flip) a video file using ffmpeg."
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output video file (default: <input>_mirrored.<ext>)'
    )
    
    args = parser.parse_args()
    
    mirror_video(args.input, args.output)


if __name__ == '__main__':
    main()

