#!/usr/bin/env python3
"""Generate a simple test audio file for testing transcription fixes."""

import sys
import wave
from pathlib import Path

import numpy as np


def generate_test_audio(filename: str = "test_audio.wav", duration: int = 10):
    """Generate a simple test audio file with speech-like patterns.

    Args:
        filename: Output filename
        duration: Duration in seconds
    """
    try:
        # Audio parameters
        sample_rate = 16000  # 16kHz sample rate (good for speech)

        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # Create speech-like audio with multiple frequency components
        # Simulate different speakers with different fundamental frequencies
        audio = np.zeros_like(t)

        # Speaker 1 (first half) - lower pitch
        speaker1_end = len(t) // 2
        fundamental1 = 120  # Hz
        audio[:speaker1_end] += 0.3 * np.sin(
            2 * np.pi * fundamental1 * t[:speaker1_end]
        )
        audio[:speaker1_end] += 0.2 * np.sin(
            2 * np.pi * fundamental1 * 2 * t[:speaker1_end]
        )
        audio[:speaker1_end] += 0.1 * np.sin(
            2 * np.pi * fundamental1 * 3 * t[:speaker1_end]
        )

        # Add some modulation to make it more speech-like
        modulation1 = 1 + 0.3 * np.sin(
            2 * np.pi * 5 * t[:speaker1_end]
        )  # 5Hz modulation
        audio[:speaker1_end] *= modulation1

        # Speaker 2 (second half) - higher pitch
        fundamental2 = 180  # Hz
        audio[speaker1_end:] += 0.3 * np.sin(
            2 * np.pi * fundamental2 * t[speaker1_end:]
        )
        audio[speaker1_end:] += 0.2 * np.sin(
            2 * np.pi * fundamental2 * 2 * t[speaker1_end:]
        )
        audio[speaker1_end:] += 0.1 * np.sin(
            2 * np.pi * fundamental2 * 3 * t[speaker1_end:]
        )

        # Add modulation for speaker 2
        modulation2 = 1 + 0.4 * np.sin(
            2 * np.pi * 7 * t[speaker1_end:]
        )  # 7Hz modulation
        audio[speaker1_end:] *= modulation2

        # Add some pauses to simulate natural speech
        pause_points = [int(0.3 * len(t)), int(0.7 * len(t))]
        for pause_point in pause_points:
            pause_length = int(0.5 * sample_rate)  # 0.5 second pause
            start_idx = max(0, pause_point - pause_length // 2)
            end_idx = min(len(audio), pause_point + pause_length // 2)
            audio[start_idx:end_idx] *= 0.1  # Reduce volume for pause

        # Add some noise to make it more realistic
        noise = 0.05 * np.random.normal(0, 1, len(audio))
        audio += noise

        # Normalize audio
        audio = audio / np.max(np.abs(audio)) * 0.8

        # Convert to 16-bit integers
        audio_int = (audio * 32767).astype(np.int16)

        # Write WAV file
        with wave.open(filename, "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())

        print(f"✅ Generated test audio file: {filename}")
        print(f"   Duration: {duration} seconds")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Format: 16-bit mono WAV")
        print(f"   Features: 2 simulated speakers with pauses")

        return True

    except Exception as e:
        print(f"❌ Failed to generate test audio: {e}")
        return False


def main():
    """Main function."""
    print("🎵 Generating test audio file for transcription testing")

    # Check if numpy is available
    try:
        import numpy as np
    except ImportError:
        print("❌ NumPy is required to generate test audio")
        print("   Install with: pip install numpy")
        return False

    # Generate test audio
    output_file = "test_audio.wav"

    # Check if file already exists
    if Path(output_file).exists():
        response = input(f"⚠️  {output_file} already exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("❌ Cancelled")
            return False

    success = generate_test_audio(output_file, duration=15)

    if success:
        print(f"\n🎯 Test audio file ready: {output_file}")
        print("   You can now run the transcription tests with this file")
        print("   Example: python tests/manual/test_transcription_fixes.py")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
