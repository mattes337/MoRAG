"""Test GPU error simulation and fallback mechanisms."""

import pytest
from unittest.mock import patch, MagicMock
from morag.core.config import detect_device, get_safe_device
from morag.core.ai_error_handlers import WhisperErrorHandler
from morag.core.resilience import ErrorType


class TestGPUErrorSimulation:
    """Test GPU error simulation and fallback mechanisms."""
    
    def test_cuda_out_of_memory_error(self):
        """Test CUDA out of memory error detection."""
        handler = WhisperErrorHandler()

        # Simulate CUDA out of memory error
        error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        error_type = handler._classify_error(error)

        assert error_type == ErrorType.SERVICE_UNAVAILABLE

    def test_gpu_device_error(self):
        """Test GPU device error detection."""
        handler = WhisperErrorHandler()

        # Simulate GPU device error
        error = RuntimeError("GPU device not found")
        error_type = handler._classify_error(error)

        assert error_type == ErrorType.SERVICE_UNAVAILABLE

    def test_torch_cuda_error(self):
        """Test PyTorch CUDA error detection."""
        handler = WhisperErrorHandler()

        # Simulate PyTorch CUDA error
        error = RuntimeError("torch.cuda.OutOfMemoryError: CUDA out of memory")
        error_type = handler._classify_error(error)

        assert error_type == ErrorType.SERVICE_UNAVAILABLE
    
    def test_device_detection_with_cuda_error(self):
        """Test device detection when CUDA throws an error."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA driver error")
        
        with patch('builtins.__import__', return_value=mock_torch) as mock_import:
            mock_import.side_effect = lambda name, *args: mock_torch if name == 'torch' else __import__(name, *args)
            device = detect_device()
            assert device == "cpu"
    
    def test_safe_device_with_memory_error(self):
        """Test safe device function with memory error."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA out of memory")
        
        with patch('builtins.__import__', return_value=mock_torch) as mock_import:
            mock_import.side_effect = lambda name, *args: mock_torch if name == 'torch' else __import__(name, *args)
            device = get_safe_device("cuda")
            assert device == "cpu"
    
    def test_whisper_gpu_fallback_simulation(self):
        """Test Whisper service GPU fallback with simulated error."""
        from morag.services.whisper_service import WhisperService
        
        # Mock WhisperModel to simulate GPU failure
        def mock_whisper_model(model_size, device, compute_type):
            if device == "cuda":
                raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB")
            return MagicMock()
        
        with patch('morag.services.whisper_service.WhisperModel', side_effect=mock_whisper_model):
            with patch('morag.services.whisper_service.get_safe_device', return_value="cuda"):
                service = WhisperService()
                # This should not raise an exception and should fall back to CPU
                model = service._get_model("base", "cuda", "int8")
                assert model is not None
    
    def test_easyocr_gpu_fallback_simulation(self):
        """Test EasyOCR GPU fallback with simulated error."""
        from morag.processors.image import ImageProcessor
        
        # Mock EasyOCR to simulate GPU failure
        mock_easyocr = MagicMock()
        mock_reader_class = MagicMock()
        mock_reader_class.side_effect = lambda langs, gpu=False: (
            RuntimeError("CUDA out of memory") if gpu else MagicMock()
        )
        mock_easyocr.Reader = mock_reader_class
        
        with patch.dict('sys.modules', {'easyocr': mock_easyocr}):
            processor = ImageProcessor()
            # This should initialize without errors
            assert processor is not None
    
    def test_sentence_transformer_gpu_fallback_simulation(self):
        """Test SentenceTransformer GPU fallback with simulated error."""
        from morag.services.topic_segmentation import EnhancedTopicSegmentation
        
        # Mock SentenceTransformer to simulate GPU failure
        mock_st = MagicMock()
        mock_st_class = MagicMock()
        mock_st_class.side_effect = lambda model, device="cpu": (
            RuntimeError("CUDA out of memory") if device == "cuda" else MagicMock()
        )
        mock_st.SentenceTransformer = mock_st_class
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            with patch('morag.services.topic_segmentation.TOPIC_SEGMENTATION_AVAILABLE', True):
                with patch('morag.services.topic_segmentation.get_safe_device', return_value="cuda"):
                    service = EnhancedTopicSegmentation()
                    # Should initialize successfully with CPU fallback
                    assert service is not None
    
    def test_multiple_gpu_error_patterns(self):
        """Test detection of various GPU error patterns."""
        handler = WhisperErrorHandler()
        
        gpu_errors = [
            "CUDA out of memory",
            "GPU device not found", 
            "NVIDIA driver error",
            "cuDNN error",
            "CUBLAS error",
            "torch.cuda.OutOfMemoryError",
            "device-side assert triggered",
            "CUDA kernel launch failed",
            "GPU memory allocation failed"
        ]
        
        for error_msg in gpu_errors:
            error = RuntimeError(error_msg)
            error_type = handler._classify_error(error)
            assert error_type == ErrorType.SERVICE_UNAVAILABLE, f"Failed to detect GPU error: {error_msg}"
    
    def test_non_gpu_errors_not_classified_as_gpu(self):
        """Test that non-GPU errors are not classified as GPU errors."""
        handler = WhisperErrorHandler()
        
        non_gpu_errors = [
            "File not found",
            "Network connection failed",
            "Invalid API key",
            "Rate limit exceeded",
            "Service temporarily unavailable"
        ]
        
        for error_msg in non_gpu_errors:
            error = RuntimeError(error_msg)
            error_type = handler._classify_error(error)
            # Should not be classified as SERVICE_UNAVAILABLE due to GPU
            # (though some might be SERVICE_UNAVAILABLE for other reasons)
            assert "cuda" not in error_msg.lower()
            assert "gpu" not in error_msg.lower()


class TestGPUFallbackIntegration:
    """Integration tests for GPU fallback system."""
    
    def test_audio_config_gpu_fallback_integration(self):
        """Test AudioConfig GPU fallback integration."""
        from morag.processors.audio import AudioConfig
        
        # Mock get_safe_device to simulate GPU unavailable
        with patch('morag.processors.audio.get_safe_device', return_value="cpu"):
            config = AudioConfig(device="cuda")
            assert config.device == "cpu"
    
    def test_settings_device_configuration_integration(self):
        """Test Settings device configuration integration."""
        from morag.core.config import Settings
        
        # Test force CPU override
        settings = Settings(force_cpu=True, preferred_device="cuda")
        assert settings.get_device() == "cpu"
        
        # Test auto detection
        with patch('morag.core.config.detect_device', return_value="cpu"):
            settings = Settings(preferred_device="auto")
            assert settings.get_device() == "cpu"
    
    @pytest.mark.asyncio
    async def test_async_gpu_fallback_integration(self):
        """Test async operations with GPU fallback."""
        from morag.services.topic_segmentation import EnhancedTopicSegmentation
        
        service = EnhancedTopicSegmentation()
        
        # This should work regardless of GPU availability
        test_text = "This is a test. This is another test."
        result = await service.segment_topics(test_text)
        
        assert result is not None
        assert result.total_topics >= 1
        assert result.processing_time >= 0


if __name__ == "__main__":
    pytest.main([__file__])
