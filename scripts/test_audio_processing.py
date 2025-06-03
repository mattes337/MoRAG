#!/usr/bin/env python3
"""Test script for audio processing functionality."""

import asyncio
import sys
import tempfile
from pathlib import Path
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.processors.audio import AudioProcessor, AudioConfig
from morag.services.whisper_service import WhisperService

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def create_test_wav_file() -> Path:
    """Create a minimal test WAV file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Write a minimal but valid WAV header
        f.write(b'RIFF')
        f.write((44).to_bytes(4, 'little'))  # File size
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write((16).to_bytes(4, 'little'))  # Format chunk size
        f.write((1).to_bytes(2, 'little'))   # Audio format (PCM)
        f.write((1).to_bytes(2, 'little'))   # Number of channels
        f.write((44100).to_bytes(4, 'little'))  # Sample rate
        f.write((88200).to_bytes(4, 'little'))  # Byte rate
        f.write((2).to_bytes(2, 'little'))   # Block align
        f.write((16).to_bytes(2, 'little'))  # Bits per sample
        f.write(b'data')
        f.write((8).to_bytes(4, 'little'))   # Data chunk size
        f.write(b'\x00' * 8)  # Audio data (silence)
        
        return Path(f.name)

async def test_audio_processor():
    """Test audio processor functionality."""
    logger.info("Testing AudioProcessor...")
    
    try:
        # Create test audio file
        test_file = create_test_wav_file()
        logger.info("Created test WAV file", file_path=str(test_file))
        
        # Create audio config (use tiny model for faster testing)
        config = AudioConfig(
            model_size="tiny",
            device="cpu",
            compute_type="int8"
        )
        logger.info("Created audio config", config=config.__dict__)
        
        # Create processor
        processor = AudioProcessor(config)
        logger.info("Created AudioProcessor")
        
        # Test file validation
        try:
            processor._validate_audio_file(test_file, config)
            logger.info("✓ File validation passed")
        except Exception as e:
            logger.error("✗ File validation failed", error=str(e))
            return False
        
        # Test metadata extraction
        try:
            metadata = await processor._extract_metadata(test_file)
            logger.info("✓ Metadata extraction passed", metadata=metadata)
        except Exception as e:
            logger.error("✗ Metadata extraction failed", error=str(e))
            return False
        
        # Test WAV conversion (should be no-op for WAV files)
        try:
            wav_path = await processor._convert_to_wav(test_file)
            logger.info("✓ WAV conversion passed", wav_path=str(wav_path))
            assert wav_path == test_file, "WAV file should not be converted"
        except Exception as e:
            logger.error("✗ WAV conversion failed", error=str(e))
            return False
        
        logger.info("✓ AudioProcessor basic functionality tests passed")
        return True
        
    except Exception as e:
        logger.error("✗ AudioProcessor test failed", error=str(e))
        return False
    finally:
        # Clean up
        if 'test_file' in locals() and test_file.exists():
            test_file.unlink()
            logger.info("Cleaned up test file")

async def test_whisper_service():
    """Test Whisper service functionality."""
    logger.info("Testing WhisperService...")
    
    try:
        # Create service
        service = WhisperService()
        logger.info("Created WhisperService")
        
        # Test available models
        models = service.get_available_models()
        logger.info("✓ Available models", models=models)
        assert "tiny" in models, "Tiny model should be available"
        assert "base" in models, "Base model should be available"
        
        # Test supported languages
        languages = service.get_supported_languages()
        logger.info("✓ Supported languages", language_count=len(languages))
        assert "en" in languages, "English should be supported"
        assert "es" in languages, "Spanish should be supported"
        assert len(languages) > 50, "Should support many languages"
        
        # Test model cleanup
        service.cleanup_models()
        logger.info("✓ Model cleanup completed")
        
        logger.info("✓ WhisperService basic functionality tests passed")
        return True
        
    except Exception as e:
        logger.error("✗ WhisperService test failed", error=str(e))
        return False

async def test_audio_config():
    """Test audio configuration."""
    logger.info("Testing AudioConfig...")
    
    try:
        # Test default config
        default_config = AudioConfig()
        logger.info("✓ Default config created", config=default_config.__dict__)
        
        # Test custom config
        custom_config = AudioConfig(
            model_size="small",
            language="es",
            device="cpu",
            compute_type="int8",
            chunk_duration=600,
            overlap_duration=60,
            quality_threshold=0.8,
            max_file_size=100 * 1024 * 1024  # 100MB for testing
        )
        logger.info("✓ Custom config created", config=custom_config.__dict__)
        
        # Verify supported formats
        assert "mp3" in default_config.supported_formats
        assert "wav" in default_config.supported_formats
        assert "m4a" in default_config.supported_formats
        logger.info("✓ Supported formats verified", formats=default_config.supported_formats)
        
        logger.info("✓ AudioConfig tests passed")
        return True
        
    except Exception as e:
        logger.error("✗ AudioConfig test failed", error=str(e))
        return False

async def main():
    """Run all audio processing tests."""
    logger.info("Starting audio processing tests...")
    
    tests = [
        ("AudioConfig", test_audio_config),
        ("AudioProcessor", test_audio_processor),
        ("WhisperService", test_whisper_service),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name} test PASSED")
            else:
                logger.error(f"✗ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} test FAILED with exception", error=str(e))
            results[test_name] = False
    
    # Summary
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    logger.info("Test Summary", 
               passed=passed, 
               total=total, 
               success_rate=f"{passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All audio processing tests PASSED!")
        return 0
    else:
        logger.error(f"❌ {total - passed} out of {total} tests FAILED")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
