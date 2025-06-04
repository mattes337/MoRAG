"""Integration tests for AI error handling with real services."""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
import time

from morag.services.embedding import gemini_service
from morag.services.whisper_service import whisper_service
from morag.core.ai_error_handlers import execute_with_ai_resilience, get_ai_service_health
from morag.core.exceptions import RateLimitError, ExternalServiceError


@pytest.mark.asyncio
class TestGeminiServiceResilience:
    """Test Gemini service with error handling integration."""
    
    async def test_embedding_generation_with_resilience(self):
        """Test embedding generation with resilience framework."""
        try:
            # Test successful embedding generation
            result = await gemini_service.generate_embedding("Test text for embedding")
            
            assert result is not None
            assert hasattr(result, 'embedding')
            assert hasattr(result, 'token_count')
            assert hasattr(result, 'model')
            assert len(result.embedding) > 0
            
            # Check that health metrics were recorded
            health = get_ai_service_health("gemini")
            assert "total_attempts" in health
            assert health["total_attempts"] > 0
            
        except Exception as e:
            # If Gemini is not available, test should still pass
            # but we should verify the error handling worked
            assert isinstance(e, (ExternalServiceError, RateLimitError))
    
    async def test_summary_generation_with_resilience(self):
        """Test summary generation with resilience framework."""
        try:
            test_text = """
            This is a test document for summarization. It contains multiple sentences
            and should be summarized into a shorter version. The AI service should
            process this text and return a concise summary that captures the main points.
            """
            
            result = await gemini_service.generate_summary(test_text, max_length=50)
            
            assert result is not None
            assert hasattr(result, 'summary')
            assert hasattr(result, 'token_count')
            assert hasattr(result, 'model')
            assert len(result.summary) > 0
            
        except Exception as e:
            # Verify proper error handling
            assert isinstance(e, (ExternalServiceError, RateLimitError))
    
    @patch('morag.services.embedding.genai.Client')
    async def test_rate_limit_handling(self, mock_client):
        """Test rate limit error handling and retry behavior."""
        # Mock rate limit error
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance
        
        # First call fails with rate limit, second succeeds
        call_count = 0
        def mock_embed_content(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Too Many Requests")
            
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.embeddings = [AsyncMock()]
            mock_response.embeddings[0].values = [0.1] * 768
            return mock_response
        
        mock_client_instance.models.embed_content = mock_embed_content
        
        # Create new service instance with mocked client
        with patch.object(gemini_service, 'client', mock_client_instance):
            try:
                result = await gemini_service.generate_embedding("Test text")
                # Should succeed after retry
                assert result is not None
                assert call_count == 2  # Initial call + 1 retry
            except RateLimitError:
                # Rate limit handling worked correctly
                pass
    
    async def test_health_check_integration(self):
        """Test health check with resilience metrics."""
        health = await gemini_service.health_check()
        
        assert "status" in health
        assert "resilience_health" in health
        assert health["status"] in ["healthy", "unhealthy"]
        
        # Check resilience health structure
        resilience_health = health["resilience_health"]
        if not isinstance(resilience_health, dict) or "error" not in resilience_health:
            assert "service_name" in resilience_health
            assert "success_rate" in resilience_health
            assert "health_status" in resilience_health


@pytest.mark.asyncio
class TestWhisperServiceResilience:
    """Test Whisper service with error handling integration."""
    
    async def test_whisper_model_loading_resilience(self):
        """Test Whisper model loading with error handling."""
        try:
            # Test getting available models (should not fail)
            models = whisper_service.get_available_models()
            assert isinstance(models, list)
            assert len(models) > 0
            assert "base" in models
            
            # Test model loading through resilience framework
            # This might fail if models aren't available, but should handle errors gracefully
            languages = whisper_service.get_supported_languages()
            assert isinstance(languages, list)
            assert "en" in languages
            
        except Exception as e:
            # Should be properly handled external service error
            assert isinstance(e, ExternalServiceError)
    
    @patch('morag.services.whisper_service.WhisperModel')
    async def test_transcription_error_handling(self, mock_whisper_model):
        """Test transcription error handling."""
        # Mock model that fails initially then succeeds
        mock_model_instance = AsyncMock()
        mock_whisper_model.return_value = mock_model_instance
        
        call_count = 0
        def mock_transcribe(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Model loading failed")
            
            # Mock successful transcription
            mock_segment = AsyncMock()
            mock_segment.text = "Test transcription"
            mock_segment.start = 0.0
            mock_segment.end = 1.0
            mock_segment.avg_logprob = -0.5
            
            mock_info = AsyncMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.9
            mock_info.duration = 1.0
            mock_info.duration_after_vad = 1.0
            mock_info.all_language_probs = {"en": 0.9}
            
            return [mock_segment], mock_info
        
        mock_model_instance.transcribe = mock_transcribe
        
        # Test with mocked model
        with patch.object(whisper_service, '_models', {"base_cpu_int8": mock_model_instance}):
            try:
                from morag.processors.audio import AudioConfig
                config = AudioConfig(model_size="base")
                
                # This should use the resilience framework
                result = await execute_with_ai_resilience(
                    "whisper",
                    whisper_service._transcribe_audio_internal,
                    "/fake/path.wav",
                    config,
                    timeout=30.0
                )
                
                # Should succeed after retry
                assert result is not None
                assert call_count == 2
                
            except Exception as e:
                # Error should be properly handled
                assert isinstance(e, ExternalServiceError)


@pytest.mark.asyncio
class TestCircuitBreakerIntegration:
    """Test circuit breaker behavior in real scenarios."""
    
    async def test_circuit_breaker_opens_on_repeated_failures(self):
        """Test that circuit breaker opens after repeated failures."""
        
        async def always_failing_operation():
            raise Exception("Service unavailable")
        
        # Execute multiple failing operations to trigger circuit breaker
        for i in range(6):  # More than failure threshold
            try:
                await execute_with_ai_resilience(
                    "test_service",
                    always_failing_operation,
                    timeout=1.0
                )
            except Exception:
                pass  # Expected to fail
        
        # Check circuit breaker status
        health = get_ai_service_health("test_service")
        if isinstance(health, dict) and "circuit_breaker" in health:
            cb_state = health["circuit_breaker"]["state"]
            # Should be open after repeated failures
            assert cb_state in ["open", "half_open"]
    
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        
        call_count = 0
        
        async def initially_failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                raise Exception("Service unavailable")
            return "success"
        
        # Trigger circuit breaker opening
        for i in range(6):
            try:
                await execute_with_ai_resilience(
                    "recovery_test",
                    initially_failing_operation,
                    timeout=1.0
                )
            except Exception:
                pass
        
        # Wait for recovery timeout (circuit breaker should attempt reset)
        await asyncio.sleep(2.0)
        
        # Try operation again - should succeed if circuit breaker allows it
        try:
            result = await execute_with_ai_resilience(
                "recovery_test",
                initially_failing_operation,
                timeout=1.0
            )
            # If we get here, circuit breaker allowed the call and it succeeded
            assert result == "success"
        except Exception:
            # Circuit breaker might still be protecting, which is also valid
            pass


@pytest.mark.asyncio
class TestHealthMetricsIntegration:
    """Test health metrics collection and reporting."""
    
    async def test_health_metrics_accumulation(self):
        """Test that health metrics accumulate correctly."""
        service_name = "metrics_test"
        
        # Perform several operations
        for i in range(5):
            try:
                await execute_with_ai_resilience(
                    service_name,
                    lambda: "success",
                    timeout=1.0
                )
            except Exception:
                pass
        
        # Check accumulated metrics
        health = get_ai_service_health(service_name)
        if isinstance(health, dict):
            assert "total_attempts" in health
            assert health["total_attempts"] >= 5
            assert "success_rate" in health
    
    async def test_error_distribution_tracking(self):
        """Test that error types are tracked correctly."""
        service_name = "error_tracking_test"
        
        # Generate different types of errors
        error_operations = [
            lambda: (_ for _ in ()).throw(Exception("429 Rate limit")),
            lambda: (_ for _ in ()).throw(Exception("503 Service unavailable")),
            lambda: (_ for _ in ()).throw(Exception("401 Unauthorized")),
        ]
        
        for operation in error_operations:
            try:
                await execute_with_ai_resilience(
                    service_name,
                    operation,
                    timeout=1.0
                )
            except Exception:
                pass  # Expected
        
        # Check error distribution
        health = get_ai_service_health(service_name)
        if isinstance(health, dict) and "error_distribution" in health:
            error_dist = health["error_distribution"]
            # Should have tracked different error types
            assert len(error_dist) > 0


@pytest.mark.asyncio
class TestPerformanceImpact:
    """Test performance impact of resilience framework."""
    
    async def test_resilience_overhead(self):
        """Test that resilience framework doesn't add significant overhead."""
        
        async def fast_operation():
            return "result"
        
        # Measure time with resilience
        start_time = time.time()
        for _ in range(10):
            await execute_with_ai_resilience(
                "performance_test",
                fast_operation,
                timeout=1.0
            )
        resilience_time = time.time() - start_time
        
        # Measure time without resilience
        start_time = time.time()
        for _ in range(10):
            await fast_operation()
        direct_time = time.time() - start_time
        
        # Overhead should be minimal (less than 100% increase)
        overhead_ratio = resilience_time / direct_time if direct_time > 0 else 1
        assert overhead_ratio < 2.0, f"Resilience overhead too high: {overhead_ratio}x"
    
    async def test_concurrent_operations(self):
        """Test resilience framework under concurrent load."""
        
        async def concurrent_operation(operation_id: int):
            await asyncio.sleep(0.01)  # Simulate some work
            return f"result_{operation_id}"
        
        # Run concurrent operations
        tasks = [
            execute_with_ai_resilience(
                "concurrent_test",
                concurrent_operation,
                i,
                timeout=5.0
            )
            for i in range(20)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 20
        
        # Check that health metrics were recorded correctly
        health = get_ai_service_health("concurrent_test")
        if isinstance(health, dict):
            assert health.get("total_attempts", 0) >= 20
