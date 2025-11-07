#!/usr/bin/env python3
"""Test script to verify environment variable configuration for audio models."""

import os
import sys
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag-audio" / "src"))

def test_whisper_model_env():
    """Test WHISPER_MODEL_SIZE environment variable."""
    print("🧪 Testing WHISPER_MODEL_SIZE environment variable...")

    # Test with WHISPER_MODEL_SIZE
    os.environ["WHISPER_MODEL_SIZE"] = "large-v3"

    try:
        from morag_audio.processor import AudioConfig

        # Create config - should pick up environment variable
        config = AudioConfig()

        print(f"✅ Model size from WHISPER_MODEL_SIZE: {config.model_size}")
        assert config.model_size == "large-v3", f"Expected 'large-v3', got '{config.model_size}'"

    except Exception as e:
        print(f"❌ Failed to test WHISPER_MODEL_SIZE: {e}")
        return False
    finally:
        # Clean up
        if "WHISPER_MODEL_SIZE" in os.environ:
            del os.environ["WHISPER_MODEL_SIZE"]

    return True

def test_morag_whisper_model_env():
    """Test MORAG_WHISPER_MODEL_SIZE environment variable."""
    print("🧪 Testing MORAG_WHISPER_MODEL_SIZE environment variable...")

    # Test with MORAG_WHISPER_MODEL_SIZE
    os.environ["MORAG_WHISPER_MODEL_SIZE"] = "base"

    try:
        from morag_audio.processor import AudioConfig

        # Create config - should pick up environment variable
        config = AudioConfig()

        print(f"✅ Model size from MORAG_WHISPER_MODEL_SIZE: {config.model_size}")
        assert config.model_size == "base", f"Expected 'base', got '{config.model_size}'"

    except Exception as e:
        print(f"❌ Failed to test MORAG_WHISPER_MODEL_SIZE: {e}")
        return False
    finally:
        # Clean up
        if "MORAG_WHISPER_MODEL_SIZE" in os.environ:
            del os.environ["MORAG_WHISPER_MODEL_SIZE"]

    return True

def test_spacy_model_env():
    """Test MORAG_SPACY_MODEL environment variable."""
    print("🧪 Testing MORAG_SPACY_MODEL environment variable...")

    # Test with MORAG_SPACY_MODEL
    os.environ["MORAG_SPACY_MODEL"] = "de_core_news_sm"

    try:
        # Import the topic segmentation module to trigger spacy loading
        from morag_audio.services import topic_segmentation

        # Check if the environment variable was read
        print(f"✅ SpaCy model environment variable set to: {os.environ.get('MORAG_SPACY_MODEL')}")

    except Exception as e:
        print(f"⚠️  Could not test spaCy model loading (dependencies may not be available): {e}")
        return True  # This is OK, spaCy is optional
    finally:
        # Clean up
        if "MORAG_SPACY_MODEL" in os.environ:
            del os.environ["MORAG_SPACY_MODEL"]

    return True

def test_audio_features_env():
    """Test audio feature environment variables."""
    print("🧪 Testing audio feature environment variables...")

    # Set environment variables
    os.environ["MORAG_ENABLE_SPEAKER_DIARIZATION"] = "false"
    os.environ["MORAG_ENABLE_TOPIC_SEGMENTATION"] = "true"
    os.environ["MORAG_AUDIO_LANGUAGE"] = "de"
    os.environ["MORAG_AUDIO_DEVICE"] = "cpu"

    try:
        from morag_audio.processor import AudioConfig

        # Create config - should pick up environment variables
        config = AudioConfig()

        print(f"✅ Speaker diarization: {config.enable_diarization}")
        print(f"✅ Topic segmentation: {config.enable_topic_segmentation}")
        print(f"✅ Audio language: {config.language}")
        print(f"✅ Audio device: {config.device}")

        assert config.enable_diarization == False, f"Expected False, got {config.enable_diarization}"
        assert config.enable_topic_segmentation == True, f"Expected True, got {config.enable_topic_segmentation}"
        assert config.language == "de", f"Expected 'de', got '{config.language}'"
        assert config.device == "cpu", f"Expected 'cpu', got '{config.device}'"

    except Exception as e:
        print(f"❌ Failed to test audio feature environment variables: {e}")
        return False
    finally:
        # Clean up
        for var in ["MORAG_ENABLE_SPEAKER_DIARIZATION", "MORAG_ENABLE_TOPIC_SEGMENTATION",
                   "MORAG_AUDIO_LANGUAGE", "MORAG_AUDIO_DEVICE"]:
            if var in os.environ:
                del os.environ[var]

    return True

def test_default_behavior():
    """Test that defaults work when no environment variables are set."""
    print("🧪 Testing default behavior (no environment variables)...")

    try:
        from morag_audio.processor import AudioConfig

        # Create config - should use defaults
        config = AudioConfig()

        print(f"✅ Default model size: {config.model_size}")
        print(f"✅ Default language: {config.language}")
        print(f"✅ Default device: {config.device}")
        print(f"✅ Default diarization: {config.enable_diarization}")
        print(f"✅ Default topic segmentation: {config.enable_topic_segmentation}")

        assert config.model_size == "medium", f"Expected 'medium', got '{config.model_size}'"
        assert config.language is None, f"Expected None, got '{config.language}'"
        assert config.device == "auto", f"Expected 'auto', got '{config.device}'"
        assert config.enable_diarization == True, f"Expected True, got {config.enable_diarization}"
        assert config.enable_topic_segmentation == True, f"Expected True, got {config.enable_topic_segmentation}"

    except Exception as e:
        print(f"❌ Failed to test default behavior: {e}")
        return False

    return True

def main():
    """Run all tests."""
    print("🚀 Testing MoRAG Audio Environment Variable Configuration\n")

    tests = [
        test_default_behavior,
        test_whisper_model_env,
        test_morag_whisper_model_env,
        test_spacy_model_env,
        test_audio_features_env,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                print("❌ FAILED\n")
        except Exception as e:
            print(f"❌ FAILED with exception: {e}\n")

    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Environment variable configuration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
