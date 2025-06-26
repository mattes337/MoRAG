"""Language detection and SpaCy model management service."""

import os
import structlog
from typing import Optional, Dict, Any, List
from functools import lru_cache

logger = structlog.get_logger(__name__)

# Language to SpaCy model mapping
SPACY_MODEL_MAPPING = {
    # English models
    "en": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
    "english": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
    
    # German models
    "de": ["de_core_news_sm", "de_core_news_md", "de_core_news_lg"],
    "german": ["de_core_news_sm", "de_core_news_md", "de_core_news_lg"],
    
    # French models
    "fr": ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"],
    "french": ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"],
    
    # Spanish models
    "es": ["es_core_news_sm", "es_core_news_md", "es_core_news_lg"],
    "spanish": ["es_core_news_sm", "es_core_news_md", "es_core_news_lg"],
    
    # Italian models
    "it": ["it_core_news_sm", "it_core_news_md", "it_core_news_lg"],
    "italian": ["it_core_news_sm", "it_core_news_md", "it_core_news_lg"],
    
    # Portuguese models
    "pt": ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"],
    "portuguese": ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"],
    
    # Dutch models
    "nl": ["nl_core_news_sm", "nl_core_news_md", "nl_core_news_lg"],
    "dutch": ["nl_core_news_sm", "nl_core_news_md", "nl_core_news_lg"],
    
    # Chinese models
    "zh": ["zh_core_web_sm", "zh_core_web_md", "zh_core_web_lg"],
    "chinese": ["zh_core_web_sm", "zh_core_web_md", "zh_core_web_lg"],
    
    # Japanese models
    "ja": ["ja_core_news_sm", "ja_core_news_md", "ja_core_news_lg"],
    "japanese": ["ja_core_news_sm", "ja_core_news_md", "ja_core_news_lg"],
    
    # Russian models
    "ru": ["ru_core_news_sm", "ru_core_news_md", "ru_core_news_lg"],
    "russian": ["ru_core_news_sm", "ru_core_news_md", "ru_core_news_lg"],
}

# Default fallback models (most commonly available)
DEFAULT_FALLBACK_MODELS = [
    "en_core_web_sm", 
    "en_core_web_md", 
    "de_core_news_sm", 
    "de_core_news_md",
    "fr_core_news_sm",
    "es_core_news_sm"
]


class LanguageDetectionService:
    """Service for detecting language and managing SpaCy models."""
    
    def __init__(self):
        self._spacy_available = False
        self._langdetect_available = False
        self._loaded_models: Dict[str, Any] = {}
        
        # Check for spaCy availability
        try:
            import spacy
            self._spacy_available = True
            logger.info("SpaCy is available for NLP processing")
        except ImportError:
            logger.warning("SpaCy not available, language-specific NLP features disabled")
        
        # Check for langdetect availability
        try:
            import langdetect
            self._langdetect_available = True
            logger.info("langdetect is available for language detection")
        except ImportError:
            logger.warning("langdetect not available, using basic language detection")
    
    def detect_language(self, text: str, fallback: str = "en") -> str:
        """Detect the language of the given text.
        
        Args:
            text: Text to analyze
            fallback: Fallback language if detection fails
            
        Returns:
            Detected language code (e.g., 'en', 'de', 'fr')
        """
        if not text or len(text.strip()) < 10:
            logger.debug("Text too short for reliable language detection, using fallback")
            return fallback
        
        # Try langdetect first (more accurate)
        if self._langdetect_available:
            try:
                from langdetect import detect
                detected = detect(text[:1000])  # Use first 1000 chars for detection
                logger.debug("Language detected using langdetect", language=detected)
                return detected
            except Exception as e:
                logger.warning("langdetect failed", error=str(e))
        
        # Try spaCy language detection if available
        if self._spacy_available:
            try:
                import spacy
                from spacy.lang.en import English
                
                # Use a simple English model for basic language detection
                nlp = English()
                doc = nlp(text[:500])  # Use first 500 chars
                
                # Basic heuristic: check for common language patterns
                # This is a simplified approach - in practice, you'd use a proper language detector
                if any(word in text.lower() for word in ['the', 'and', 'is', 'are', 'was', 'were']):
                    return "en"
                elif any(word in text.lower() for word in ['der', 'die', 'das', 'und', 'ist', 'sind']):
                    return "de"
                elif any(word in text.lower() for word in ['le', 'la', 'les', 'et', 'est', 'sont']):
                    return "fr"
                elif any(word in text.lower() for word in ['el', 'la', 'los', 'las', 'y', 'es', 'son']):
                    return "es"
                
            except Exception as e:
                logger.warning("SpaCy language detection failed", error=str(e))
        
        logger.debug("Language detection failed, using fallback", fallback=fallback)
        return fallback
    
    def get_spacy_model_for_language(self, language: str) -> Optional[Any]:
        """Get the best available SpaCy model for the given language.
        
        Args:
            language: Language code (e.g., 'en', 'de', 'fr')
            
        Returns:
            Loaded SpaCy model or None if not available
        """
        if not self._spacy_available:
            return None
        
        # Check if we already have a model loaded for this language
        if language in self._loaded_models:
            return self._loaded_models[language]
        
        # Check environment variable override first
        env_model = os.environ.get("MORAG_SPACY_MODEL")
        if env_model:
            model = self._try_load_model(env_model)
            if model:
                self._loaded_models[language] = model
                return model
        
        # Try language-specific models
        language_lower = language.lower()
        if language_lower in SPACY_MODEL_MAPPING:
            for model_name in SPACY_MODEL_MAPPING[language_lower]:
                model = self._try_load_model(model_name)
                if model:
                    self._loaded_models[language] = model
                    logger.info("Loaded language-specific SpaCy model", 
                               language=language, model=model_name)
                    return model
        
        # Try fallback models
        for model_name in DEFAULT_FALLBACK_MODELS:
            model = self._try_load_model(model_name)
            if model:
                self._loaded_models[language] = model
                logger.info("Loaded fallback SpaCy model", 
                           language=language, model=model_name)
                return model
        
        logger.warning("No SpaCy model available for language", language=language)
        return None
    
    def _try_load_model(self, model_name: str) -> Optional[Any]:
        """Try to load a specific SpaCy model.
        
        Args:
            model_name: Name of the SpaCy model to load
            
        Returns:
            Loaded model or None if loading failed
        """
        try:
            import spacy
            model = spacy.load(model_name)
            logger.debug("Successfully loaded SpaCy model", model=model_name)
            return model
        except OSError:
            logger.debug("SpaCy model not available", model=model_name)
            return None
        except Exception as e:
            logger.warning("Failed to load SpaCy model", model=model_name, error=str(e))
            return None
    
    def get_available_models(self) -> List[str]:
        """Get list of available SpaCy models on the system.
        
        Returns:
            List of available model names
        """
        if not self._spacy_available:
            return []
        
        available = []
        all_models = set()
        
        # Add all models from mapping
        for models in SPACY_MODEL_MAPPING.values():
            all_models.update(models)
        
        # Add fallback models
        all_models.update(DEFAULT_FALLBACK_MODELS)
        
        # Test which ones are actually available
        for model_name in all_models:
            if self._try_load_model(model_name):
                available.append(model_name)
        
        return sorted(available)
    
    @lru_cache(maxsize=32)
    def detect_and_get_model(self, text: str, fallback_language: str = "en") -> tuple[str, Optional[Any]]:
        """Detect language and get appropriate SpaCy model in one call.
        
        Args:
            text: Text to analyze
            fallback_language: Fallback language if detection fails
            
        Returns:
            Tuple of (detected_language, spacy_model)
        """
        detected_language = self.detect_language(text, fallback_language)
        spacy_model = self.get_spacy_model_for_language(detected_language)
        
        logger.debug("Language detection and model selection completed",
                    detected_language=detected_language,
                    model_available=spacy_model is not None)
        
        return detected_language, spacy_model


# Global instance for easy access
_language_service = None

def get_language_service() -> LanguageDetectionService:
    """Get the global language detection service instance."""
    global _language_service
    if _language_service is None:
        _language_service = LanguageDetectionService()
    return _language_service
