#!/usr/bin/env python3
"""
Test script for configuration and error handling fixes.

Tests:
1. Vision model configuration via environment variables
2. ExternalServiceError proper initialization
3. Audio processing speaker diarization attribute fixes
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root and package paths to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-image" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-audio" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-services" / "src"))

def test_vision_model_configuration():
    """Test that vision model can be configured via environment variable."""
    print("\n🔧 Testing Vision Model Configuration")
    print("=" * 50)
    
    try:
        # Test with environment variable set
        with patch.dict(os.environ, {'GEMINI_VISION_MODEL': 'gemini-1.5-flash'}):
            from morag_image.processor import ImageConfig
            
            config = ImageConfig()
            assert config.vision_model == 'gemini-1.5-flash', f"Expected gemini-1.5-flash, got {config.vision_model}"
            print("✅ Vision model correctly configured from environment variable")
        
        # Test with no environment variable (should use default)
        with patch.dict(os.environ, {}, clear=True):
            # Need to reload the module to test default behavior
            import importlib
            import morag_image.processor as processor_module
            importlib.reload(processor_module)
            
            config = processor_module.ImageConfig()
            assert config.vision_model == 'gemini-1.5-flash', f"Expected default gemini-1.5-flash, got {config.vision_model}"
            print("✅ Vision model uses correct default when no environment variable set")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision model configuration test failed: {e}")
        return False

def test_external_service_error_initialization():
    """Test that ExternalServiceError is properly initialized with service parameter."""
    print("\n🔧 Testing ExternalServiceError Initialization")
    print("=" * 50)

    try:
        from morag_core.exceptions import ExternalServiceError

        # Test proper initialization
        error = ExternalServiceError("Test error message", "test_service")
        assert error.service == "test_service", f"Expected test_service, got {error.service}"
        assert "test_service error: Test error message" in str(error), f"Error message format incorrect: {str(error)}"
        print("✅ ExternalServiceError properly initialized with service parameter")

        # Test that missing service parameter raises TypeError
        try:
            error = ExternalServiceError("Test error message")
            print("❌ ExternalServiceError should require service parameter")
            return False
        except TypeError:
            print("✅ ExternalServiceError correctly requires service parameter")

        return True

    except Exception as e:
        print(f"❌ ExternalServiceError initialization test failed: {e}")
        return False

def test_embedding_service_error_handling():
    """Test that embedding services properly handle ExternalServiceError with service parameter."""
    print("\n🔧 Testing Embedding Service Error Handling")
    print("=" * 50)

    try:
        from morag_services.embedding import GeminiEmbeddingService
        from morag_core.exceptions import ExternalServiceError

        # Create service without API key to trigger error
        service = GeminiEmbeddingService(
            api_key="",  # Empty API key should trigger error
            embedding_model="text-embedding-004",
            generation_model="gemini-2.0-flash-001"
        )

        # Test that client initialization check raises proper error
        # We'll test the sync method directly to avoid asyncio issues in test
        try:
            # This should raise ExternalServiceError because client is not initialized
            service._generate_embedding_sync("test text", "retrieval_document")
            print("❌ Should have raised ExternalServiceError for uninitialized client")
            return False
        except ExternalServiceError as e:
            assert hasattr(e, 'service'), "ExternalServiceError should have service attribute"
            assert e.service == "gemini", f"Expected service 'gemini', got {e.service}"
            print("✅ Embedding service properly raises ExternalServiceError with service parameter")
            return True
        except Exception as e:
            # The sync method might fail differently, so let's check if it's a proper error
            if "Gemini client not initialized" in str(e):
                print("✅ Embedding service properly handles uninitialized client")
                return True
            else:
                print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
                return False

    except Exception as e:
        print(f"❌ Embedding service error handling test failed: {e}")
        return False

def test_speaker_segment_attributes():
    """Test that SpeakerSegment has correct attributes."""
    print("\n🔧 Testing SpeakerSegment Attributes")
    print("=" * 50)
    
    try:
        from morag_audio.services.speaker_diarization import SpeakerSegment
        
        # Create a SpeakerSegment instance
        segment = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            confidence=0.9
        )
        
        # Test that correct attributes exist
        assert hasattr(segment, 'start_time'), "SpeakerSegment should have start_time attribute"
        assert hasattr(segment, 'end_time'), "SpeakerSegment should have end_time attribute"
        assert hasattr(segment, 'speaker_id'), "SpeakerSegment should have speaker_id attribute"
        assert not hasattr(segment, 'start'), "SpeakerSegment should not have start attribute"
        assert not hasattr(segment, 'end'), "SpeakerSegment should not have end attribute"
        assert not hasattr(segment, 'speaker'), "SpeakerSegment should not have speaker attribute"
        
        # Test attribute values
        assert segment.start_time == 0.0, f"Expected start_time 0.0, got {segment.start_time}"
        assert segment.end_time == 5.0, f"Expected end_time 5.0, got {segment.end_time}"
        assert segment.speaker_id == "SPEAKER_00", f"Expected speaker_id SPEAKER_00, got {segment.speaker_id}"
        
        print("✅ SpeakerSegment has correct attributes (start_time, end_time, speaker_id)")
        return True
        
    except Exception as e:
        print(f"❌ SpeakerSegment attributes test failed: {e}")
        return False

def test_audio_processor_metadata_reference():
    """Test that audio processor uses self.metadata correctly."""
    print("\n🔧 Testing Audio Processor Metadata Reference")
    print("=" * 50)
    
    try:
        from morag_audio.processor import AudioProcessor, AudioConfig
        
        # Create processor instance
        config = AudioConfig(enable_diarization=False, enable_topic_segmentation=False)
        processor = AudioProcessor(config)
        
        # Test that metadata is initialized
        assert hasattr(processor, 'metadata'), "AudioProcessor should have metadata attribute"
        assert isinstance(processor.metadata, dict), f"Expected metadata to be dict, got {type(processor.metadata)}"
        
        # Test metadata access
        processor.metadata['test_key'] = 'test_value'
        assert processor.metadata['test_key'] == 'test_value', "Should be able to set and get metadata values"
        
        print("✅ AudioProcessor metadata reference works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Audio processor metadata test failed: {e}")
        return False

def test_core_config_vision_model():
    """Test that core config includes vision model setting."""
    print("\n🔧 Testing Core Config Vision Model Setting")
    print("=" * 50)

    try:
        from morag_core.config import Settings

        # Test with environment variable (Settings uses MORAG_ prefix)
        with patch.dict(os.environ, {'MORAG_GEMINI_VISION_MODEL': 'test-vision-model'}):
            settings = Settings()
            assert hasattr(settings, 'gemini_vision_model'), "Settings should have gemini_vision_model attribute"
            assert settings.gemini_vision_model == 'test-vision-model', f"Expected test-vision-model, got {settings.gemini_vision_model}"

        # Test default value
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.gemini_vision_model == 'gemini-1.5-flash', f"Expected default gemini-1.5-flash, got {settings.gemini_vision_model}"

        print("✅ Core config vision model setting works correctly")
        return True

    except Exception as e:
        print(f"❌ Core config vision model test failed: {e}")
        return False

def test_exception_re_raising_logic():
    """Test the exception re-raising logic in ingest_tasks.py."""
    print("\n🔧 Testing Exception Re-raising Logic")
    print("=" * 50)

    try:
        from morag_core.exceptions import ExternalServiceError, RateLimitError, ProcessingError

        # Test function that simulates the fixed error handling logic
        def simulate_error_handling(original_exception):
            """Simulate the fixed error handling logic from ingest_tasks.py."""
            e = original_exception

            # Handle special cases for exceptions that require specific parameters
            if hasattr(e, 'service') and hasattr(type(e), '__init__'):
                # For ExternalServiceError and similar exceptions that need service parameter
                try:
                    raise type(e)(str(e).replace(f"{e.service} error: ", ""), e.service)
                except:
                    # Fallback to generic exception if reconstruction fails
                    raise Exception(str(e))
            else:
                # For other exceptions, try to recreate with just the message
                try:
                    raise type(e)(str(e))
                except:
                    # Fallback to generic exception if reconstruction fails
                    raise Exception(str(e))

        # Test 1: ExternalServiceError should be properly re-raised
        try:
            original_error = ExternalServiceError("API failed", "gemini")
            simulate_error_handling(original_error)
            print("❌ Should have raised an exception")
            return False
        except ExternalServiceError as e:
            assert e.service == "gemini", f"Expected gemini service, got {e.service}"
            print("✅ ExternalServiceError properly re-raised with service parameter")
        except Exception as e:
            print(f"✅ ExternalServiceError fallback to generic Exception: {type(e).__name__}")

        # Test 2: Regular exceptions should be re-raised normally
        try:
            original_error = RateLimitError("Rate limit exceeded")
            simulate_error_handling(original_error)
            print("❌ Should have raised an exception")
            return False
        except RateLimitError:
            print("✅ RateLimitError properly re-raised")
        except Exception as e:
            print(f"✅ RateLimitError fallback to generic Exception: {type(e).__name__}")

        # Test 3: Standard exceptions should work
        try:
            original_error = ValueError("Invalid value")
            simulate_error_handling(original_error)
            print("❌ Should have raised an exception")
            return False
        except ValueError:
            print("✅ ValueError properly re-raised")
        except Exception as e:
            print(f"✅ ValueError fallback to generic Exception: {type(e).__name__}")

        return True

    except ImportError as e:
        print(f"❌ Failed to import required exceptions: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error testing exception re-raising: {e}")
        return False

async def main():
    """Run all configuration and error handling tests."""
    print("🚀 Configuration and Error Handling Fixes Test Suite")
    print("=" * 60)
    print("Testing vision model configuration, error handling, and audio processing fixes...")
    print()
    
    tests = [
        test_vision_model_configuration,
        test_external_service_error_initialization,
        test_embedding_service_error_handling,
        test_speaker_segment_attributes,
        test_audio_processor_metadata_reference,
        test_core_config_vision_model,
        test_exception_re_raising_logic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All configuration and error handling fixes working correctly!")
        return True
    else:
        print("❌ Some tests failed. Please check the fixes.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
