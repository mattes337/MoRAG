"""SpaCy-based entity extractor with multi-language support."""

import asyncio
import os
from typing import List, Optional, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor
import structlog

from ..models import Entity
from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)

# Try to import spaCy and language detection
try:
    import spacy
    from spacy.lang.en import English
    from spacy.lang.de import German
    from spacy.lang.es import Spanish
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    detect = None


class SpacyEntityExtractor:
    """SpaCy-based entity extractor with multi-language support and confidence scoring."""
    
    # Mapping of language codes to SpaCy model names
    LANGUAGE_MODELS = {
        'en': 'en_core_web_lg',
        'de': 'de_core_news_lg', 
        'es': 'es_core_news_lg'
    }
    
    # Entity type mapping from SpaCy to our schema
    ENTITY_TYPE_MAPPING = {
        'PERSON': 'PERSON',
        'PER': 'PERSON',
        'ORG': 'ORGANIZATION',
        'GPE': 'LOCATION',  # Geopolitical entity
        'LOC': 'LOCATION',
        'MISC': 'MISCELLANEOUS',
        'MONEY': 'MONETARY',
        'DATE': 'TEMPORAL',
        'TIME': 'TEMPORAL',
        'PERCENT': 'QUANTITY',
        'QUANTITY': 'QUANTITY',
        'ORDINAL': 'QUANTITY',
        'CARDINAL': 'QUANTITY',
        'EVENT': 'EVENT',
        'FAC': 'FACILITY',
        'LANGUAGE': 'LANGUAGE',
        'LAW': 'LAW',
        'NORP': 'GROUP',  # Nationalities, religious groups
        'PRODUCT': 'PRODUCT',
        'WORK_OF_ART': 'CREATIVE_WORK'
    }
    
    def __init__(self, 
                 min_confidence: float = 0.6,
                 supported_languages: Optional[List[str]] = None,
                 fallback_language: str = 'en',
                 enable_language_detection: bool = True,
                 max_workers: int = 2):
        """Initialize SpaCy entity extractor.
        
        Args:
            min_confidence: Minimum confidence threshold for entities
            supported_languages: List of supported language codes
            fallback_language: Default language when detection fails
            enable_language_detection: Whether to auto-detect language
            max_workers: Number of worker threads for processing
        """
        if not SPACY_AVAILABLE:
            raise ProcessingError("SpaCy is not available. Please install spacy>=3.7.0")
        
        self.settings = get_settings()
        self.min_confidence = min_confidence
        self.supported_languages = supported_languages or ['en', 'de', 'es']
        self.fallback_language = fallback_language
        self.enable_language_detection = enable_language_detection and LANGDETECT_AVAILABLE
        
        # Thread pool for CPU-intensive SpaCy operations
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="spacy_extract")
        
        # Cache for loaded models
        self._loaded_models: Dict[str, Any] = {}
        
        # Initialize models
        self._initialize_models()
        
        logger.info(
            "SpaCy entity extractor initialized",
            min_confidence=self.min_confidence,
            supported_languages=self.supported_languages,
            enable_language_detection=self.enable_language_detection,
            loaded_models=list(self._loaded_models.keys())
        )
    
    def _initialize_models(self) -> None:
        """Initialize SpaCy models for supported languages."""
        for lang_code in self.supported_languages:
            model_name = self.LANGUAGE_MODELS.get(lang_code)
            if model_name:
                try:
                    # Check if model is available
                    if spacy.util.is_package(model_name):
                        # Load model lazily when first needed
                        logger.info(f"SpaCy model {model_name} is available for {lang_code}")
                    else:
                        logger.warning(
                            f"SpaCy model {model_name} not found for {lang_code}. "
                            f"Install with: python -m spacy download {model_name}"
                        )
                except Exception as e:
                    logger.warning(f"Error checking SpaCy model {model_name}: {e}")
    
    def _get_model(self, language: str) -> Optional[Any]:
        """Get or load SpaCy model for the given language.
        
        Args:
            language: Language code (e.g., 'en', 'de', 'es')
            
        Returns:
            Loaded SpaCy model or None if not available
        """
        if language not in self.supported_languages:
            language = self.fallback_language
        
        # Return cached model if available
        if language in self._loaded_models:
            return self._loaded_models[language]
        
        model_name = self.LANGUAGE_MODELS.get(language)
        if not model_name:
            logger.warning(f"No model mapping for language: {language}")
            return None
        
        try:
            # Load model
            model = spacy.load(model_name)
            self._loaded_models[language] = model
            logger.info(f"Loaded SpaCy model {model_name} for {language}")
            return model
        except OSError as e:
            logger.warning(f"Failed to load SpaCy model {model_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading SpaCy model {model_name}: {e}")
            return None
    
    def _detect_language(self, text: str) -> str:
        """Detect language of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code or fallback language
        """
        if not self.enable_language_detection or not text.strip():
            return self.fallback_language
        
        try:
            # Use langdetect for language detection
            detected_lang = detect(text)
            
            # Map to our supported languages
            if detected_lang in self.supported_languages:
                return detected_lang
            
            # Handle common language variations
            lang_mapping = {
                'ca': 'es',  # Catalan -> Spanish
                'pt': 'es',  # Portuguese -> Spanish
                'fr': 'en',  # French -> English
                'it': 'en',  # Italian -> English
                'nl': 'de',  # Dutch -> German
            }
            
            mapped_lang = lang_mapping.get(detected_lang, self.fallback_language)
            if mapped_lang in self.supported_languages:
                return mapped_lang
                
        except (LangDetectError, Exception) as e:
            logger.debug(f"Language detection failed: {e}")
        
        return self.fallback_language
    
    async def extract(self, 
                     text: str, 
                     language: Optional[str] = None,
                     source_doc_id: Optional[str] = None,
                     **kwargs) -> List[Entity]:
        """Extract entities from text using SpaCy NER.
        
        Args:
            text: Text to extract entities from
            language: Optional language override
            source_doc_id: Optional source document ID
            **kwargs: Additional arguments
            
        Returns:
            List of Entity objects
        """
        if not text or not text.strip():
            return []
        
        try:
            # Detect or use provided language
            detected_language = language or self._detect_language(text)
            
            logger.debug(
                "Starting SpaCy entity extraction",
                text_length=len(text),
                language=detected_language,
                source_doc_id=source_doc_id
            )
            
            # Run extraction in thread pool
            entities = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._extract_entities_sync,
                text,
                detected_language,
                source_doc_id
            )
            
            logger.info(
                "SpaCy entity extraction completed",
                entities_found=len(entities),
                language=detected_language,
                source_doc_id=source_doc_id
            )
            
            return entities
            
        except Exception as e:
            logger.error(
                "SpaCy entity extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
                language=language
            )
            raise ProcessingError(f"SpaCy entity extraction failed: {e}")
    
    def _extract_entities_sync(self, 
                              text: str, 
                              language: str,
                              source_doc_id: Optional[str] = None) -> List[Entity]:
        """Synchronous entity extraction using SpaCy.
        
        Args:
            text: Text to process
            language: Language code
            source_doc_id: Optional source document ID
            
        Returns:
            List of Entity objects
        """
        # Get SpaCy model
        nlp = self._get_model(language)
        if not nlp:
            logger.warning(f"No SpaCy model available for {language}, skipping extraction")
            return []
        
        try:
            # Process text with SpaCy
            doc = nlp(text)
            
            entities = []
            seen_entities: Set[str] = set()
            
            for ent in doc.ents:
                # Skip empty or very short entities
                if not ent.text.strip() or len(ent.text.strip()) < 2:
                    continue
                
                # Normalize entity text
                entity_text = ent.text.strip()
                entity_key = f"{entity_text.lower()}_{ent.label_}"
                
                # Skip duplicates
                if entity_key in seen_entities:
                    continue
                seen_entities.add(entity_key)
                
                # Map entity type
                entity_type = self.ENTITY_TYPE_MAPPING.get(ent.label_, ent.label_)
                
                # Calculate confidence (SpaCy doesn't provide confidence, so we estimate)
                confidence = self._calculate_confidence(ent, doc)
                
                # Skip low confidence entities
                if confidence < self.min_confidence:
                    continue
                
                # Create entity
                entity = Entity(
                    name=entity_text,
                    type=entity_type,
                    source_doc_id=source_doc_id,
                    confidence=confidence,
                    attributes={
                        'extraction_method': 'spacy',
                        'spacy_label': ent.label_,
                        'language': language,
                        'start_pos': ent.start_char,
                        'end_pos': ent.end_char,
                        'start_token': ent.start,
                        'end_token': ent.end
                    }
                )
                
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in SpaCy processing: {e}")
            return []

    def _calculate_confidence(self, ent, doc) -> float:
        """Calculate confidence score for a SpaCy entity.

        Since SpaCy doesn't provide confidence scores directly, we estimate based on:
        - Entity length and complexity
        - Position in sentence
        - Entity type reliability
        - Context quality

        Args:
            ent: SpaCy entity
            doc: SpaCy document

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.7  # Base confidence for SpaCy NER

        # Adjust based on entity type reliability
        type_confidence = {
            'PERSON': 0.9,
            'ORG': 0.85,
            'GPE': 0.8,
            'LOC': 0.8,
            'MONEY': 0.95,
            'DATE': 0.9,
            'TIME': 0.9,
            'PERCENT': 0.95,
            'QUANTITY': 0.85,
            'ORDINAL': 0.85,
            'CARDINAL': 0.8
        }.get(ent.label_, 0.7)

        # Adjust based on entity length (longer entities often more reliable)
        length_factor = min(1.0, len(ent.text) / 10.0)
        length_bonus = length_factor * 0.1

        # Adjust based on capitalization (proper nouns more reliable)
        capitalization_bonus = 0.05 if ent.text[0].isupper() else 0.0

        # Adjust based on alphanumeric content
        alpha_ratio = sum(c.isalpha() for c in ent.text) / len(ent.text)
        alpha_bonus = 0.05 if alpha_ratio > 0.7 else 0.0

        # Calculate final confidence
        confidence = (base_confidence + type_confidence) / 2.0
        confidence += length_bonus + capitalization_bonus + alpha_bonus

        # Ensure confidence is within bounds
        return max(0.0, min(1.0, confidence))

    async def extract_with_context(self,
                                  text: str,
                                  context: Optional[str] = None,
                                  language: Optional[str] = None,
                                  source_doc_id: Optional[str] = None,
                                  **kwargs) -> List[Entity]:
        """Extract entities with additional context information.

        Args:
            text: Primary text to extract entities from
            context: Additional context text
            language: Optional language override
            source_doc_id: Optional source document ID
            **kwargs: Additional arguments

        Returns:
            List of Entity objects
        """
        # For SpaCy, we can combine text and context for better extraction
        combined_text = text
        if context:
            combined_text = f"{context}\n\n{text}"

        entities = await self.extract(
            combined_text,
            language=language,
            source_doc_id=source_doc_id,
            **kwargs
        )

        # Filter entities that appear in the main text (not just context)
        if context:
            context_length = len(context) + 2  # +2 for newlines
            filtered_entities = []

            for entity in entities:
                start_pos = entity.attributes.get('start_pos', 0)
                if start_pos >= context_length:
                    # Adjust position to be relative to main text
                    entity.attributes['start_pos'] = start_pos - context_length
                    entity.attributes['end_pos'] = entity.attributes.get('end_pos', 0) - context_length
                    filtered_entities.append(entity)

            return filtered_entities

        return entities

    def get_available_languages(self) -> List[str]:
        """Get list of available languages with loaded models.

        Returns:
            List of language codes
        """
        available = []
        for lang_code in self.supported_languages:
            model_name = self.LANGUAGE_MODELS.get(lang_code)
            if model_name and spacy.util.is_package(model_name):
                available.append(lang_code)
        return available

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models.

        Returns:
            Dictionary with model information
        """
        info = {
            'available_languages': self.get_available_languages(),
            'loaded_models': list(self._loaded_models.keys()),
            'supported_languages': self.supported_languages,
            'fallback_language': self.fallback_language,
            'language_detection_enabled': self.enable_language_detection,
            'min_confidence': self.min_confidence
        }

        # Add model details for loaded models
        for lang, model in self._loaded_models.items():
            if hasattr(model, 'meta'):
                info[f'{lang}_model_info'] = {
                    'name': model.meta.get('name', 'unknown'),
                    'version': model.meta.get('version', 'unknown'),
                    'lang': model.meta.get('lang', 'unknown')
                }

        return info

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)

            # Clear model cache
            self._loaded_models.clear()

            logger.info("SpaCy entity extractor closed")
        except Exception as e:
            logger.warning("Error during SpaCy extractor cleanup", error=str(e))

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
