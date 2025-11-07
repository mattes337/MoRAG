#!/usr/bin/env python3
"""Test script to verify .env file loading for remote converter."""

import os
import sys
from pathlib import Path


def load_env_file(env_file=".env.test"):
    """Load environment variables from a .env file."""
    env_path = Path(env_file)
    if not env_path.exists():
        print(f"❌ Environment file {env_file} not found")
        return False

    print(f"📁 Loading environment from {env_file}")

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value
                print(f"   {key}={value}")

    return True


def test_remote_converter_config():
    """Test remote converter configuration with environment variables."""
    print("\n🧪 Testing Remote Converter Configuration...")

    # Add the packages to the path
    sys.path.insert(
        0, str(Path(__file__).parent.parent.parent / "packages" / "morag-audio" / "src")
    )

    try:
        from config import RemoteConverterConfig

        # Create config - should pick up environment variables
        config_manager = RemoteConverterConfig()
        config = config_manager.get_config()

        print(f"✅ Worker ID: {config.get('worker_id')}")
        print(f"✅ API Base URL: {config.get('api_base_url')}")
        print(f"✅ Content Types: {config.get('content_types')}")
        print(f"✅ Audio Environment Variables: {config.get('audio_env_vars', {})}")

        # Verify audio environment variables are captured
        audio_env = config.get("audio_env_vars", {})
        expected_vars = [
            "WHISPER_MODEL_SIZE",
            "MORAG_AUDIO_LANGUAGE",
            "MORAG_SPACY_MODEL",
        ]

        for var in expected_vars:
            if var in audio_env:
                print(f"✅ {var}: {audio_env[var]}")
            else:
                print(f"⚠️  {var} not found in audio config")

        return True

    except Exception as e:
        print(f"❌ Failed to test remote converter config: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_audio_processor_with_env():
    """Test AudioProcessor with environment variables loaded."""
    print("\n🧪 Testing AudioProcessor with Environment Variables...")

    # Add the packages to the path
    sys.path.insert(
        0, str(Path(__file__).parent.parent.parent / "packages" / "morag-audio" / "src")
    )

    try:
        from morag_audio.processor import AudioConfig

        # Create config - should pick up environment variables
        config = AudioConfig()

        print(f"✅ Whisper Model Size: {config.model_size}")
        print(f"✅ Audio Language: {config.language}")
        print(f"✅ Audio Device: {config.device}")
        print(f"✅ Speaker Diarization: {config.enable_diarization}")
        print(f"✅ Topic Segmentation: {config.enable_topic_segmentation}")

        # Verify the environment variables were applied
        assert (
            config.model_size == "large-v3"
        ), f"Expected 'large-v3', got '{config.model_size}'"
        assert config.language == "de", f"Expected 'de', got '{config.language}'"
        assert config.device == "cpu", f"Expected 'cpu', got '{config.device}'"

        return True

    except Exception as e:
        print(f"❌ Failed to test AudioProcessor: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the test."""
    print("🚀 Testing MoRAG Remote Converter Environment Configuration\n")

    # Load environment file
    if not load_env_file():
        return 1

    # Test remote converter config
    success1 = test_remote_converter_config()

    # Test audio processor
    success2 = test_audio_processor_with_env()

    if success1 and success2:
        print(
            "\n🎉 All tests passed! Environment file configuration is working correctly."
        )
        print("\n📝 Usage Summary:")
        print("   1. Copy .env.example to .env")
        print("   2. Set WHISPER_MODEL_SIZE=large-v3 in .env")
        print("   3. Set other audio configuration variables as needed")
        print("   4. Start the remote converter - it will use your configured models")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
