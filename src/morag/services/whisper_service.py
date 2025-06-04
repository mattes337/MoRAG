"""Whisper service for speech-to-text processing."""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import asyncio
import structlog
import time

from faster_whisper import WhisperModel
from morag.core.config import settings
from morag.core.exceptions import ExternalServiceError, ProcessingError
from morag.processors.audio import AudioConfig, AudioProcessingResult, AudioTranscriptSegment

logger = structlog.get_logger()

class WhisperService:
    """Service for managing Whisper models and transcription."""
    
    def __init__(self):
        """Initialize Whisper service."""
        self._models: Dict[str, WhisperModel] = {}
        self._default_config = AudioConfig()
        
        logger.info("Initialized WhisperService")
    
    def _get_model(self, model_size: str, device: str = "cpu", compute_type: str = "int8") -> WhisperModel:
        """Get or create Whisper model."""
        model_key = f"{model_size}_{device}_{compute_type}"
        
        if model_key not in self._models:
            try:
                logger.info("Loading Whisper model", 
                           model_size=model_size,
                           device=device,
                           compute_type=compute_type)
                
                model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type
                )
                
                self._models[model_key] = model
                logger.info("Whisper model loaded successfully", model_key=model_key)
                
            except Exception as e:
                logger.error("Failed to load Whisper model", 
                           model_size=model_size,
                           error=str(e))
                raise ExternalServiceError(f"Failed to load Whisper model: {str(e)}", "whisper")
        
        return self._models[model_key]
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        config: Optional[AudioConfig] = None
    ) -> AudioProcessingResult:
        """Transcribe audio file to text."""
        start_time = time.time()
        config = config or self._default_config
        audio_path = Path(audio_path)
        
        logger.info("Starting audio transcription", 
                   audio_path=str(audio_path),
                   model_size=config.model_size)
        
        try:
            # Get model
            model = self._get_model(
                config.model_size,
                config.device,
                config.compute_type
            )
            
            # Perform transcription
            segments, info = await asyncio.to_thread(
                self._transcribe_sync,
                model,
                str(audio_path),
                config
            )
            
            # Process results
            audio_segments = []
            full_text_parts = []
            
            for segment in segments:
                # Convert log probability to confidence (0-1 scale)
                confidence = max(0.0, min(1.0, (segment.avg_logprob + 1.0) / 2.0))

                audio_segment = AudioTranscriptSegment(
                    text=segment.text.strip(),
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=confidence,
                    language=info.language
                )
                audio_segments.append(audio_segment)
                full_text_parts.append(segment.text.strip())
            
            full_text = " ".join(full_text_parts)
            
            # Calculate overall confidence
            overall_confidence = (
                sum(seg.confidence for seg in audio_segments) / len(audio_segments)
                if audio_segments else 0.0
            )
            
            processing_time = time.time() - start_time
            
            result = AudioProcessingResult(
                text=full_text,
                language=info.language,
                confidence=overall_confidence,
                duration=info.duration,
                segments=audio_segments,
                metadata={
                    "language_probability": info.language_probability,
                    "duration_after_vad": info.duration_after_vad,
                    "all_language_probs": info.all_language_probs
                },
                processing_time=processing_time,
                model_used=config.model_size
            )
            
            logger.info("Audio transcription completed", 
                       audio_path=str(audio_path),
                       duration=info.duration,
                       language=info.language,
                       confidence=overall_confidence,
                       processing_time=processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Audio transcription failed", 
                        audio_path=str(audio_path),
                        error=str(e))
            raise ExternalServiceError(f"Audio transcription failed: {str(e)}", "whisper")
    
    def _transcribe_sync(
        self,
        model: WhisperModel,
        audio_path: str,
        config: AudioConfig
    ):
        """Synchronous transcription method with enhanced settings for better quality."""
        return model.transcribe(
            audio_path,
            language=config.language,
            beam_size=settings.whisper_beam_size,
            best_of=settings.whisper_best_of,
            temperature=settings.whisper_temperature,
            condition_on_previous_text=True,
            compression_ratio_threshold=settings.whisper_compression_ratio_threshold,
            log_prob_threshold=settings.whisper_log_prob_threshold,
            no_speech_threshold=settings.whisper_no_speech_threshold,
            initial_prompt=None,
            word_timestamps=True  # Enable word-level timestamps for better speaker alignment
        )
    
    async def transcribe_with_chunking(
        self,
        audio_path: Union[str, Path],
        config: Optional[AudioConfig] = None
    ) -> AudioProcessingResult:
        """Transcribe long audio file with chunking."""
        config = config or self._default_config
        audio_path = Path(audio_path)
        
        logger.info("Starting chunked audio transcription", 
                   audio_path=str(audio_path),
                   chunk_duration=config.chunk_duration)
        
        try:
            # For now, use regular transcription
            # TODO: Implement actual chunking for very long files
            return await self.transcribe_audio(audio_path, config)
            
        except Exception as e:
            logger.error("Chunked audio transcription failed", 
                        audio_path=str(audio_path),
                        error=str(e))
            raise ExternalServiceError(f"Chunked audio transcription failed: {str(e)}", "whisper")
    
    async def detect_language(
        self,
        audio_path: Union[str, Path],
        model_size: str = "base"
    ) -> Dict[str, Any]:
        """Detect language of audio file."""
        audio_path = Path(audio_path)
        
        logger.info("Detecting audio language", 
                   audio_path=str(audio_path),
                   model_size=model_size)
        
        try:
            model = self._get_model(model_size)
            
            # Detect language
            segments, info = await asyncio.to_thread(
                model.transcribe,
                str(audio_path),
                language=None,  # Auto-detect
                beam_size=1,
                best_of=1,
                temperature=0.0
            )
            
            result = {
                "language": info.language,
                "language_probability": info.language_probability,
                "all_language_probs": info.all_language_probs
            }
            
            logger.info("Language detection completed", 
                       audio_path=str(audio_path),
                       detected_language=info.language,
                       probability=info.language_probability)
            
            return result
            
        except Exception as e:
            logger.error("Language detection failed", 
                        audio_path=str(audio_path),
                        error=str(e))
            raise ExternalServiceError(f"Language detection failed: {str(e)}", "whisper")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Whisper model sizes."""
        return ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # This is a subset of languages supported by Whisper
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
            "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
            "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
            "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
            "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
            "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
            "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
            "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
    
    def cleanup_models(self) -> None:
        """Clean up loaded models to free memory."""
        logger.info("Cleaning up Whisper models", model_count=len(self._models))
        self._models.clear()

# Global instance
whisper_service = WhisperService()
