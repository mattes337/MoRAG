"""Data models for audio processing."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os


@dataclass
class AudioSegment:
    """Represents a segment of audio with speaker information."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: float = 1.0
    topic_id: Optional[int] = None
    topic_label: Optional[str] = None


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    model_size: str = "medium"  # tiny, base, small, medium, large-v2, large-v3
    language: Optional[str] = None  # Auto-detect if None
    enable_diarization: bool = True  # Enable by default
    enable_topic_segmentation: bool = True  # Enable by default
    min_speakers: int = 1
    max_speakers: int = 5
    device: str = "auto"  # auto, cpu, cuda
    compute_type: str = "default"  # default, int8, float16, float32
    beam_size: int = 5
    vad_filter: bool = True
    vad_parameters: Dict[str, Any] = field(default_factory=lambda: {
        'threshold': 0.5,
        'min_speech_duration_ms': 250,
        'min_silence_duration_ms': 500
    })
    word_timestamps: bool = True
    include_metadata: bool = True
    # REST API configuration
    use_rest_api: bool = False  # Use REST API instead of local whisper
    openai_api_key: Optional[str] = None  # OpenAI API key for REST calls
    api_base_url: str = "https://api.openai.com/v1"  # OpenAI API base URL
    timeout: int = 3600  # API request timeout in seconds (60 minutes for long transcriptions)

    def __post_init__(self):
        """Load configuration from environment variables if not explicitly set."""
        import os

        # Override with environment variables if they exist
        # Support both WHISPER_MODEL_SIZE and MORAG_WHISPER_MODEL_SIZE
        env_model_size = (
            os.environ.get("WHISPER_MODEL_SIZE") or
            os.environ.get("MORAG_WHISPER_MODEL_SIZE")
        )
        if env_model_size and self.model_size == "medium":  # Only override if using default
            self.model_size = env_model_size

        # Override language if set in environment
        env_language = os.environ.get("MORAG_AUDIO_LANGUAGE")
        if env_language and self.language is None:
            self.language = env_language

        # Override device if set in environment
        env_device = os.environ.get("MORAG_AUDIO_DEVICE")
        if env_device and self.device == "auto":  # Only override if using default
            self.device = env_device

        # Override diarization setting
        env_diarization = os.environ.get("MORAG_ENABLE_SPEAKER_DIARIZATION")
        if env_diarization is not None:
            self.enable_diarization = env_diarization.lower() in ("true", "1", "yes", "on")

        # Override topic segmentation setting
        env_topic_seg = os.environ.get("MORAG_ENABLE_TOPIC_SEGMENTATION")
        if env_topic_seg is not None:
            self.enable_topic_segmentation = env_topic_seg.lower() in ("true", "1", "yes", "on")
        
        # Override REST API settings
        env_use_rest = os.environ.get("MORAG_USE_REST_TRANSCRIPTION")
        if env_use_rest is not None:
            self.use_rest_api = env_use_rest.lower() in ("true", "1", "yes", "on")
        
        # Override OpenAI API key
        env_openai_key = os.environ.get("OPENAI_API_KEY")
        if env_openai_key and self.openai_api_key is None:
            self.openai_api_key = env_openai_key
        
        # Override API base URL
        env_api_base = os.environ.get("OPENAI_API_BASE")
        if env_api_base and self.api_base_url == "https://api.openai.com/v1":
            self.api_base_url = env_api_base
        
        # Override timeout setting
        env_timeout = os.environ.get("MORAG_REST_TIMEOUT")
        if env_timeout and self.timeout == 3600:  # Only override if using default
            try:
                self.timeout = int(env_timeout)
            except ValueError:
                pass  # Keep default if invalid value