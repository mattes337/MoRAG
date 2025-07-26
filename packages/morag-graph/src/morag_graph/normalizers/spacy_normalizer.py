"""SpaCy-specific entity normalizer with advanced linguistic processing."""

import asyncio
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from .entity_normalizer import NormalizedEntity

logger = structlog.get_logger(__name__)

# Try to import spaCy
try:
    import spacy
    from spacy.tokens import Token, Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


class SpacyNormalizer:
    """SpaCy-specific entity normalizer with advanced linguistic processing.
    
    This normalizer uses SpaCy's linguistic features for more accurate normalization:
    - Lemmatization for base forms
    - POS tagging for proper handling of different word types
    - Dependency parsing for context-aware normalization
    - Named entity recognition for proper noun preservation
    """
    
    def __init__(self, 
                 supported_languages: Optional[List[str]] = None,
                 fallback_language: str = 'en',
                 preserve_proper_nouns: bool = True,
                 max_workers: int = 2):
        """Initialize SpaCy normalizer.
        
        Args:
            supported_languages: List of supported language codes
            fallback_language: Default language when detection fails
            preserve_proper_nouns: Whether to preserve proper nouns as-is
            max_workers: Number of worker threads for processing
        """
        if not SPACY_AVAILABLE:
            raise ProcessingError("SpaCy is not available. Please install spacy>=3.7.0")
        
        self.settings = get_settings()
        self.supported_languages = supported_languages or ['en', 'de', 'es']
        self.fallback_language = fallback_language
        self.preserve_proper_nouns = preserve_proper_nouns
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="spacy_norm")
        
        # Cache for loaded models
        self._loaded_models: Dict[str, Any] = {}
        
        # Language-specific model mappings
        self.LANGUAGE_MODELS = {
            'en': 'en_core_web_lg',
            'de': 'de_core_news_lg',
            'es': 'es_core_news_lg'
        }
        
        # Proper noun POS tags by language
        self.PROPER_NOUN_TAGS = {
            'en': {'PROPN', 'NNP', 'NNPS'},
            'de': {'PROPN', 'NE'},
            'es': {'PROPN', 'NP00000'}
        }
        
        # Entity types that should preserve original form
        self.PRESERVE_ENTITY_TYPES = {
            'PERSON', 'ORGANIZATION', 'LOCATION', 'GPE', 'ORG', 'LOC', 'PER'
        }
        
        logger.info(
            "SpaCy normalizer initialized",
            supported_languages=self.supported_languages,
            preserve_proper_nouns=self.preserve_proper_nouns
        )
    
    def _get_model(self, language: str) -> Optional[Any]:
        """Get or load SpaCy model for the given language."""
        if language not in self.supported_languages:
            language = self.fallback_language
        
        # Return cached model if available
        if language in self._loaded_models:
            return self._loaded_models[language]
        
        model_name = self.LANGUAGE_MODELS.get(language)
        if not model_name:
            return None
        
        try:
            model = spacy.load(model_name)
            self._loaded_models[language] = model
            logger.debug(f"Loaded SpaCy model {model_name} for normalization")
            return model
        except OSError:
            logger.warning(f"SpaCy model {model_name} not available for normalization")
            return None
    
    async def normalize_entities(self,
                                entities: List[str],
                                entity_types: Optional[List[str]] = None,
                                language: Optional[str] = None,
                                source_doc_id: Optional[str] = None) -> List[NormalizedEntity]:
        """Normalize entities using SpaCy linguistic processing.
        
        Args:
            entities: List of entity texts to normalize
            entity_types: Optional list of entity types
            language: Language code for processing
            source_doc_id: Optional source document ID
            
        Returns:
            List of normalized entities
        """
        if not entities:
            return []
        
        try:
            # Use provided language or fallback
            processing_language = language or self.fallback_language
            
            logger.debug(
                "Starting SpaCy entity normalization",
                entity_count=len(entities),
                language=processing_language,
                source_doc_id=source_doc_id
            )
            
            # Run normalization in thread pool
            normalized_entities = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._normalize_entities_sync,
                entities,
                entity_types,
                processing_language
            )
            
            logger.info(
                "SpaCy entity normalization completed",
                input_entities=len(entities),
                normalized_entities=len(normalized_entities),
                language=processing_language
            )
            
            return normalized_entities
            
        except Exception as e:
            logger.error(
                "SpaCy entity normalization failed",
                error=str(e),
                error_type=type(e).__name__,
                entity_count=len(entities)
            )
            raise ProcessingError(f"SpaCy entity normalization failed: {e}")
    
    def _normalize_entities_sync(self,
                                entities: List[str],
                                entity_types: Optional[List[str]] = None,
                                language: str = 'en') -> List[NormalizedEntity]:
        """Synchronous entity normalization using SpaCy."""
        # Get SpaCy model
        nlp = self._get_model(language)
        if not nlp:
            logger.warning(f"No SpaCy model available for {language}, using rule-based fallback")
            return self._normalize_with_rules(entities, entity_types, language)
        
        normalized_entities = []
        
        for i, entity_text in enumerate(entities):
            entity_type = entity_types[i] if entity_types and i < len(entity_types) else None
            
            try:
                normalized = self._normalize_single_entity(entity_text, entity_type, nlp, language)
                normalized_entities.append(normalized)
            except Exception as e:
                logger.warning(f"Error normalizing entity '{entity_text}': {e}")
                # Fallback to original text
                normalized_entities.append(NormalizedEntity(
                    original_text=entity_text,
                    normalized_text=entity_text,
                    canonical_form=entity_text,
                    language=language,
                    entity_type=entity_type,
                    confidence=0.5,
                    normalization_method="error_fallback"
                ))
        
        return normalized_entities
    
    def _normalize_single_entity(self,
                                entity_text: str,
                                entity_type: Optional[str],
                                nlp: Any,
                                language: str) -> NormalizedEntity:
        """Normalize a single entity using SpaCy."""
        # Process with SpaCy
        doc = nlp(entity_text.strip())
        
        # Check if this should be preserved as proper noun
        should_preserve = self._should_preserve_entity(doc, entity_type, language)
        
        if should_preserve:
            # Preserve proper nouns but clean up formatting
            normalized_text = self._clean_proper_noun(entity_text)
            canonical_form = normalized_text
            confidence = 0.95
            method = "spacy_preserved"
        else:
            # Apply linguistic normalization
            normalized_text = self._apply_linguistic_normalization(doc, language)
            canonical_form = normalized_text.lower()
            confidence = 0.9
            method = "spacy_linguistic"
        
        # Generate variations
        variations = self._generate_spacy_variations(normalized_text, doc, language)
        
        return NormalizedEntity(
            original_text=entity_text,
            normalized_text=normalized_text,
            canonical_form=canonical_form,
            language=language,
            entity_type=entity_type,
            confidence=confidence,
            variations=variations,
            normalization_method=method,
            metadata={
                'spacy_processed': True,
                'pos_tags': [token.pos_ for token in doc],
                'lemmas': [token.lemma_ for token in doc],
                'is_proper_noun': should_preserve
            }
        )
    
    def _should_preserve_entity(self, doc: Doc, entity_type: Optional[str], language: str) -> bool:
        """Determine if entity should be preserved as proper noun."""
        if not self.preserve_proper_nouns:
            return False
        
        # Preserve based on entity type
        if entity_type and entity_type in self.PRESERVE_ENTITY_TYPES:
            return True
        
        # Preserve based on POS tags
        proper_noun_tags = self.PROPER_NOUN_TAGS.get(language, {'PROPN'})
        has_proper_noun = any(token.pos_ in proper_noun_tags for token in doc)
        
        # Preserve if mostly capitalized
        capitalized_ratio = sum(1 for token in doc if token.text[0].isupper()) / len(doc)
        
        return has_proper_noun or capitalized_ratio > 0.5
    
    def _clean_proper_noun(self, text: str) -> str:
        """Clean proper noun while preserving capitalization."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove leading/trailing punctuation but preserve internal punctuation
        cleaned = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', cleaned)
        
        return cleaned
    
    def _apply_linguistic_normalization(self, doc: Doc, language: str) -> str:
        """Apply linguistic normalization using SpaCy features."""
        normalized_tokens = []
        
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            
            # Use lemma for base form
            lemma = token.lemma_.lower()
            
            # Handle special cases
            if lemma == '-PRON-':  # SpaCy placeholder for pronouns
                lemma = token.text.lower()
            
            # Remove common prefixes/suffixes for better normalization
            lemma = self._clean_lemma(lemma, language)
            
            if lemma and lemma.isalpha():
                normalized_tokens.append(lemma)
        
        return ' '.join(normalized_tokens) if normalized_tokens else doc.text.lower()
    
    def _clean_lemma(self, lemma: str, language: str) -> str:
        """Clean lemma based on language-specific rules."""
        if not lemma:
            return lemma
        
        # Language-specific cleaning
        if language == 'de':
            # Remove German articles that might be attached
            lemma = re.sub(r'^(der|die|das|ein|eine)\s*', '', lemma)
        elif language == 'es':
            # Remove Spanish articles
            lemma = re.sub(r'^(el|la|los|las|un|una)\s*', '', lemma)
        elif language == 'en':
            # Remove English articles
            lemma = re.sub(r'^(the|a|an)\s*', '', lemma)
        
        return lemma.strip()
    
    def _generate_spacy_variations(self, normalized_text: str, doc: Doc, language: str) -> List[str]:
        """Generate variations using SpaCy linguistic features."""
        variations = [normalized_text]
        
        # Add case variations
        variations.extend([
            normalized_text.lower(),
            normalized_text.upper(),
            normalized_text.title()
        ])
        
        # Add lemma-based variations
        lemmas = [token.lemma_ for token in doc if not token.is_space and not token.is_punct]
        if lemmas:
            lemma_text = ' '.join(lemmas).lower()
            variations.append(lemma_text)
        
        # Add original tokens
        original_tokens = [token.text for token in doc if not token.is_space and not token.is_punct]
        if original_tokens:
            variations.append(' '.join(original_tokens))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var not in seen and var.strip():
                seen.add(var)
                unique_variations.append(var)
        
        return unique_variations
    
    def _normalize_with_rules(self,
                             entities: List[str],
                             entity_types: Optional[List[str]],
                             language: str) -> List[NormalizedEntity]:
        """Fallback rule-based normalization when SpaCy is not available."""
        normalized_entities = []
        
        for i, entity_text in enumerate(entities):
            entity_type = entity_types[i] if entity_types and i < len(entity_types) else None
            
            # Simple rule-based normalization
            normalized_text = entity_text.strip().lower()
            
            # Remove common articles and prepositions
            if language == 'en':
                normalized_text = re.sub(r'\b(the|a|an)\s+', '', normalized_text)
            elif language == 'es':
                normalized_text = re.sub(r'\b(el|la|los|las|un|una)\s+', '', normalized_text)
            elif language == 'de':
                normalized_text = re.sub(r'\b(der|die|das|ein|eine)\s+', '', normalized_text)
            
            # Normalize whitespace
            normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
            
            normalized_entities.append(NormalizedEntity(
                original_text=entity_text,
                normalized_text=normalized_text,
                canonical_form=normalized_text,
                language=language,
                entity_type=entity_type,
                confidence=0.7,
                variations=[normalized_text, entity_text.lower()],
                normalization_method="rule_based_fallback"
            ))
        
        return normalized_entities
    
    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            
            # Clear model cache
            self._loaded_models.clear()
            
            logger.info("SpaCy normalizer closed")
        except Exception as e:
            logger.warning("Error during SpaCy normalizer cleanup", error=str(e))
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
