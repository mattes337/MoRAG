#!/usr/bin/env python3
"""
Create Test Video for Debugging

This script creates a simple test video from an audio file for debugging purposes.
"""

import sys
import subprocess
from pathlib import Path
import time

def create_test_video(audio_path: Path, output_path: Path, duration: int = 60):
    """Create a simple test video from audio file."""
    try:
        print(f"Creating test video from {audio_path}")
        print(f"Output: {output_path}")
        print(f"Duration: {duration} seconds")
        
        # Create a simple video with a static image and the audio
        # Generate a simple colored background
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f", "lavfi",
            "-i", f"color=c=blue:size=640x480:duration={duration}",  # Blue background
            "-i", str(audio_path),  # Audio input
            "-c:v", "libx264",  # Video codec
            "-c:a", "aac",  # Audio codec
            "-shortest",  # Stop when shortest input ends
            "-pix_fmt", "yuv420p",  # Pixel format for compatibility
            str(output_path)
        ]
        
        print("Running ffmpeg command...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Test video created successfully: {output_path}")
            print(f"File size: {output_path.stat().st_size:,} bytes")
            return True
        else:
            print(f"❌ ffmpeg failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating test video: {e}")
        return False


def main():
    """Main function."""
    # Use the existing audio file
    audio_path = Path("uploads/Sprache.mp3")
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Create output path
    output_path = Path("uploads/test_video.mp4")
    
    # Create a short test video (30 seconds to keep it small)
    success = create_test_video(audio_path, output_path, duration=30)
    
    if success:
        print(f"\n🎥 Test video ready for debugging!")
        print(f"You can now run:")
        print(f"python tests/manual/debug_video_duplication.py \"{output_path}\"")
    else:
        print(f"\n❌ Failed to create test video")
        sys.exit(1)


if __name__ == "__main__":
    main()
