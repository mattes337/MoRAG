"""Test device detection and fallback mechanisms."""

from unittest.mock import MagicMock, patch

import pytest
from morag_core.config import Settings, detect_device, get_safe_device


class TestDeviceDetection:
    """Test device detection and fallback functionality."""

    def test_detect_device_no_torch(self):
        """Test device detection when PyTorch is not available."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: ImportError()
            if name == "torch"
            else __import__(name, *args),
        ):
            device = detect_device()
            assert device == "cpu"

    def test_detect_device_no_cuda(self):
        """Test device detection when CUDA is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("builtins.__import__", return_value=mock_torch) as mock_import:
            mock_import.side_effect = (
                lambda name, *args: mock_torch
                if name == "torch"
                else __import__(name, *args)
            )
            device = detect_device()
            assert device == "cpu"

    def test_detect_device_cuda_available(self):
        """Test device detection when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("builtins.__import__", return_value=mock_torch) as mock_import:
            mock_import.side_effect = (
                lambda name, *args: mock_torch
                if name == "torch"
                else __import__(name, *args)
            )
            device = detect_device()
            assert device == "cuda"

    def test_detect_device_exception(self):
        """Test device detection when an exception occurs."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("GPU error")

        with patch("builtins.__import__", return_value=mock_torch) as mock_import:
            mock_import.side_effect = (
                lambda name, *args: mock_torch
                if name == "torch"
                else __import__(name, *args)
            )
            device = detect_device()
            assert device == "cpu"

    def test_get_safe_device_cpu_preference(self):
        """Test safe device with CPU preference."""
        device = get_safe_device("cpu")
        assert device == "cpu"

    def test_get_safe_device_cuda_no_torch(self):
        """Test safe device with CUDA preference but no PyTorch."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: ImportError()
            if name == "torch"
            else __import__(name, *args),
        ):
            device = get_safe_device("cuda")
            assert device == "cpu"

    def test_get_safe_device_cuda_no_gpu(self):
        """Test safe device with CUDA preference but no GPU."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("builtins.__import__", return_value=mock_torch) as mock_import:
            mock_import.side_effect = (
                lambda name, *args: mock_torch
                if name == "torch"
                else __import__(name, *args)
            )
            device = get_safe_device("cuda")
            assert device == "cpu"

    def test_get_safe_device_cuda_available(self):
        """Test safe device with CUDA preference and GPU available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("builtins.__import__", return_value=mock_torch) as mock_import:
            mock_import.side_effect = (
                lambda name, *args: mock_torch
                if name == "torch"
                else __import__(name, *args)
            )
            device = get_safe_device("cuda")
            assert device == "cuda"

    def test_get_safe_device_auto_detection(self):
        """Test safe device with auto detection."""
        with patch("morag.core.config.detect_device", return_value="cpu"):
            device = get_safe_device(None)
            assert device == "cpu"

    def test_settings_get_device_force_cpu(self):
        """Test settings device configuration with force CPU."""
        settings = Settings(force_cpu=True, preferred_device="cuda")
        device = settings.get_device()
        assert device == "cpu"

    def test_settings_get_device_auto(self):
        """Test settings device configuration with auto detection."""
        settings = Settings(preferred_device="auto")
        with patch("morag.core.config.detect_device", return_value="cpu"):
            device = settings.get_device()
            assert device == "cpu"

    def test_settings_get_device_preferred(self):
        """Test settings device configuration with preferred device."""
        settings = Settings(preferred_device="cpu")
        with patch(
            "morag.core.config.get_safe_device", return_value="cpu"
        ) as mock_get_safe:
            device = settings.get_device()
            mock_get_safe.assert_called_once_with("cpu")
            assert device == "cpu"


class TestAudioConfigDeviceFallback:
    """Test audio configuration device fallback."""

    def test_audio_config_device_fallback(self):
        """Test that AudioConfig uses safe device fallback."""
        from morag_audio import AudioConfig

        with patch(
            "morag.processors.audio.get_safe_device", return_value="cpu"
        ) as mock_get_safe:
            config = AudioConfig(device="cuda")
            mock_get_safe.assert_called_once_with("cuda")
            assert config.device == "cpu"


class TestWhisperServiceDeviceFallback:
    """Test Whisper service device fallback."""

    @pytest.mark.asyncio
    async def test_whisper_model_gpu_fallback(self):
        """Test that Whisper model falls back to CPU when GPU fails."""
        from morag_audio.services import WhisperService

        # Mock WhisperModel to fail on GPU but succeed on CPU
        def mock_whisper_model(model_size, device, compute_type):
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return MagicMock()

        with patch(
            "morag.services.whisper_service.WhisperModel",
            side_effect=mock_whisper_model,
        ):
            with patch(
                "morag.services.whisper_service.get_safe_device", return_value="cuda"
            ):
                service = WhisperService()
                # This should not raise an exception and should fall back to CPU
                model = service._get_model("base", "cuda", "int8")
                assert model is not None


if __name__ == "__main__":
    pytest.main([__file__])
