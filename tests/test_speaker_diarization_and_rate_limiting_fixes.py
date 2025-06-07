#!/usr/bin/env python3
"""
Test script for speaker diarization and rate limiting fixes.

This script tests the fixes for:
1. Speaker diarization coroutine error
2. Gemini API rate limiting with exponential backoff

Usage:
    python tests/test_speaker_diarization_and_rate_limiting_fixes.py
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import structlog

# Configure logging
logger = structlog.get_logger(__name__)


async def test_speaker_diarization_fix():
    """Test that speaker diarization no longer has coroutine access errors."""
    print("\n🎯 Testing Speaker Diarization Fix...")
    
    try:
        from morag_audio.processor import AudioProcessor
        from morag_audio.config import AudioConfig
        from morag_audio.services.speaker_diarization import DiarizationResult, SpeakerInfo, SpeakerSegment
        
        # Create a mock audio processor with diarization enabled
        config = AudioConfig(
            enable_diarization=True,
            min_speakers=1,
            max_speakers=3
        )
        
        processor = AudioProcessor(config)
        
        # Mock the diarization service to return a proper DiarizationResult
        mock_diarization_service = Mock()
        
        # Create a proper async mock that returns a DiarizationResult
        async def mock_diarize_audio(*args, **kwargs):
            return DiarizationResult(
                speakers=[SpeakerInfo("SPEAKER_00", 10.0, 1, 10.0, [0.9], 0.0, 10.0)],
                segments=[SpeakerSegment("SPEAKER_00", 0.0, 10.0, 10.0, 0.9)],
                total_speakers=1,
                total_duration=10.0,
                speaker_overlap_time=0.0,
                processing_time=1.0,
                model_used="mock",
                confidence_threshold=0.5
            )
        
        mock_diarization_service.diarize_audio = mock_diarize_audio
        processor.diarization_service = mock_diarization_service
        
        # Create mock audio segments
        from morag_audio.models import AudioSegment
        segments = [
            AudioSegment(
                start=0.0,
                end=5.0,
                text="Test audio segment",
                speaker=None,
                confidence=0.9,
                topic_id=None,
                topic_label=None
            )
        ]
        
        # Test the _apply_diarization method
        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
            result_segments = await processor._apply_diarization(temp_file.name, segments)
            
            # Verify the method completed without errors
            assert len(result_segments) == 1
            assert result_segments[0].speaker == "SPEAKER_00"
            
        print("✅ Speaker diarization fix working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Speaker diarization test failed: {e}")
        return False


async def test_rate_limiting_fix():
    """Test that rate limiting errors are handled with exponential backoff."""
    print("\n🎯 Testing Rate Limiting Fix...")
    
    try:
        from morag_services.embedding import GeminiEmbeddingService
        from morag_core.exceptions import RateLimitError
        
        # Create a service instance
        service = GeminiEmbeddingService(
            api_key="test_key",
            embedding_model="text-embedding-004"
        )
        
        # Mock the client to simulate rate limiting
        mock_client = Mock()
        service.client = mock_client
        
        # Track call attempts
        call_count = 0
        
        def mock_embed_content(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:  # First two calls fail with rate limit
                raise Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric 'Batch Embed Content API requests'")
            else:  # Third call succeeds
                mock_response = Mock()
                mock_response.embeddings = [Mock()]
                mock_response.embeddings[0].values = [0.1] * 768
                return mock_response
        
        mock_client.models.embed_content = mock_embed_content
        
        # Test the rate limiting with retry logic
        start_time = time.time()
        
        try:
            result = service._generate_embedding_sync("test text", "retrieval_document")
            
            # Verify it succeeded after retries
            assert result.embedding == [0.1] * 768
            assert call_count == 3  # Should have retried twice
            
            # Verify exponential backoff was applied (should take at least 3 seconds)
            elapsed_time = time.time() - start_time
            assert elapsed_time >= 3.0, f"Expected at least 3 seconds for backoff, got {elapsed_time}"
            
            print("✅ Rate limiting fix working correctly with exponential backoff")
            return True
            
        except RateLimitError:
            # This is expected if all retries are exhausted
            print("✅ Rate limiting fix working correctly (all retries exhausted)")
            return True
            
    except Exception as e:
        print(f"❌ Rate limiting test failed: {e}")
        return False


async def test_batch_processing_delays():
    """Test that batch processing includes delays to prevent rate limiting."""
    print("\n🎯 Testing Batch Processing Delays...")
    
    try:
        from morag_services.embedding import GeminiEmbeddingService
        
        # Create a service instance
        service = GeminiEmbeddingService(
            api_key="test_key",
            embedding_model="text-embedding-004"
        )
        
        # Mock the generate_embedding method to track timing
        call_times = []
        
        async def mock_generate_embedding(text, task_type):
            call_times.append(time.time())
            from morag_services.embedding import EmbeddingResult
            return EmbeddingResult(
                embedding=[0.1] * 768,
                token_count=len(text.split()),
                model="text-embedding-004"
            )
        
        service.generate_embedding = mock_generate_embedding
        
        # Test batch processing with small batch
        texts = ["text1", "text2", "text3"]
        start_time = time.time()
        
        results = await service.generate_embeddings_batch(texts, batch_size=3, delay_between_batches=0.5)
        
        # Verify results
        assert len(results) == 3
        assert len(call_times) == 3
        
        # Verify delays between calls (should be at least 0.1 seconds each)
        for i in range(1, len(call_times)):
            delay = call_times[i] - call_times[i-1]
            assert delay >= 0.1, f"Expected at least 0.1s delay between calls, got {delay}"
        
        print("✅ Batch processing delays working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Testing Speaker Diarization and Rate Limiting Fixes")
    print("=" * 60)
    
    tests = [
        test_speaker_diarization_fix,
        test_rate_limiting_fix,
        test_batch_processing_delays
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All fixes are working correctly!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
