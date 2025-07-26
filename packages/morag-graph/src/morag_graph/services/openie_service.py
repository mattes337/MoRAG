"""OpenIE wrapper service for MoRAG relation extraction."""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, NamedTuple, Union
from enum import Enum
import structlog
import nltk
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError, ValidationError

logger = structlog.get_logger(__name__)


class OpenIEBackend(Enum):
    """Supported OpenIE backends."""
    STANFORD = "stanford"
    ALLENNLP = "allennlp"
    SPACY = "spacy"


class ExtractionMetrics(NamedTuple):
    """Metrics for OpenIE extraction performance."""
    total_sentences: int
    processed_sentences: int
    total_triplets: int
    filtered_triplets: int
    processing_time: float
    average_confidence: float
    error_count: int


class OpenIETriplet(NamedTuple):
    """Represents an OpenIE extracted triplet."""
    subject: str
    predicate: str
    object: str
    confidence: float
    sentence: str
    start_pos: int = 0
    end_pos: int = 0
    backend: str = "unknown"
    extraction_time: float = 0.0
    metadata: Dict[str, Any] = {}


class OpenIEService:
    """Enhanced OpenIE service with multiple backend support and improved error handling."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenIE service with configuration.

        Args:
            config: Optional configuration dictionary. If None, uses global settings.
        """
        self.settings = get_settings()
        self.config = config or {}

        # Configuration from settings
        self.enabled = self.config.get('enabled', self.settings.openie_enabled)
        self.backend = OpenIEBackend(self.config.get('backend', self.settings.openie_implementation))
        self.confidence_threshold = self.config.get('confidence_threshold', self.settings.openie_confidence_threshold)
        self.max_triplets_per_sentence = self.config.get('max_triplets_per_sentence', self.settings.openie_max_triplets_per_sentence)
        self.batch_size = self.config.get('batch_size', self.settings.openie_batch_size)
        self.timeout_seconds = self.config.get('timeout_seconds', self.settings.openie_timeout_seconds)
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)

        # Performance settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_size = self.config.get('cache_size', 1000)
        self.enable_metrics = self.config.get('enable_metrics', True)

        # Initialize components
        self._openie_client = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="openie")
        self._initialized = False
        self._cache = {} if self.enable_caching else None
        self._metrics = []
        self._fallback_backends = self._get_fallback_backends()

        logger.info(
            "OpenIE service initialized",
            enabled=self.enabled,
            backend=self.backend.value,
            confidence_threshold=self.confidence_threshold,
            max_triplets_per_sentence=self.max_triplets_per_sentence,
            retry_attempts=self.retry_attempts,
            enable_caching=self.enable_caching
        )
    
    def _get_fallback_backends(self) -> List[OpenIEBackend]:
        """Get list of fallback backends in order of preference."""
        fallbacks = []
        for backend in OpenIEBackend:
            if backend != self.backend:
                fallbacks.append(backend)
        return fallbacks

    async def initialize(self) -> None:
        """Initialize OpenIE client and download required resources."""
        if self._initialized:
            return

        if not self.enabled:
            logger.info("OpenIE service disabled, skipping initialization")
            return

        # Try primary backend first, then fallbacks
        backends_to_try = [self.backend] + self._fallback_backends

        for backend in backends_to_try:
            try:
                logger.info("Initializing OpenIE service", backend=backend.value)

                # Download NLTK data if needed
                await self._ensure_nltk_data()

                # Initialize OpenIE client in thread pool
                await asyncio.get_event_loop().run_in_executor(
                    self._executor, self._init_openie_client, backend
                )

                self.backend = backend  # Update to working backend
                self._initialized = True
                logger.info("OpenIE service initialization completed", backend=backend.value)
                return

            except Exception as e:
                logger.warning(
                    "Failed to initialize backend, trying next",
                    backend=backend.value,
                    error=str(e),
                    error_type=type(e).__name__
                )
                continue

        # If all backends failed
        logger.error("All OpenIE backends failed to initialize")
        raise ValidationError("No OpenIE backend could be initialized")
    
    def _init_openie_client(self, backend: OpenIEBackend) -> None:
        """Initialize OpenIE client for specific backend (runs in thread pool)."""
        try:
            if backend == OpenIEBackend.STANFORD:
                self._init_stanford_client()
            elif backend == OpenIEBackend.ALLENNLP:
                self._init_allennlp_client()
            elif backend == OpenIEBackend.SPACY:
                self._init_spacy_client()
            else:
                raise ValidationError(f"Unsupported OpenIE backend: {backend.value}")

        except ImportError as e:
            raise ValidationError(f"OpenIE dependencies not available for {backend.value}: {e}")
        except Exception as e:
            raise ValidationError(f"Failed to initialize {backend.value} client: {e}")

    def _init_stanford_client(self) -> None:
        """Initialize Stanford OpenIE client."""
        from openie import StanfordOpenIE

        # Create temporary directory for Stanford OpenIE
        temp_dir = tempfile.mkdtemp(prefix="openie_stanford_")

        # Initialize with custom configuration
        self._openie_client = StanfordOpenIE(
            corenlp_home=temp_dir,
            memory="2g",  # Allocate 2GB memory
            timeout=self.timeout_seconds * 1000,  # Convert to milliseconds
            quiet=True
        )

        logger.info("Stanford OpenIE client initialized", temp_dir=temp_dir)

    def _init_allennlp_client(self) -> None:
        """Initialize AllenNLP OpenIE client."""
        try:
            from allennlp.predictors.predictor import Predictor

            # Load pre-trained OpenIE model
            self._openie_client = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
            )

            logger.info("AllenNLP OpenIE client initialized")

        except Exception as e:
            raise ValidationError(f"Failed to initialize AllenNLP client: {e}")

    def _init_spacy_client(self) -> None:
        """Initialize spaCy-based OpenIE client (simple rule-based)."""
        try:
            import spacy

            # Try to load English model
            try:
                self._openie_client = spacy.load("en_core_web_lg")
            except OSError:
                try:
                    self._openie_client = spacy.load("en_core_web_sm")
                except OSError:
                    raise ValidationError("No spaCy English model found. Install with: python -m spacy download en_core_web_sm")

            logger.info("spaCy OpenIE client initialized", model=self._openie_client.meta['name'])

        except ImportError:
            raise ValidationError("spaCy not available. Install with: pip install spacy")
    
    async def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is downloaded."""
        try:
            def download_nltk_data():
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    logger.info("Downloading NLTK punkt tokenizer")
                    nltk.download('punkt', quiet=True)
                
                try:
                    nltk.data.find('tokenizers/punkt_tab')
                except LookupError:
                    logger.info("Downloading NLTK punkt_tab tokenizer")
                    nltk.download('punkt_tab', quiet=True)
            
            await asyncio.get_event_loop().run_in_executor(
                self._executor, download_nltk_data
            )
            
        except Exception as e:
            logger.warning("Failed to download NLTK data", error=str(e))
            # Continue without NLTK data - we'll handle sentence splitting differently
    
    async def extract_triplets(self, text: str, source_doc_id: Optional[str] = None, chunk_info: Optional[Dict] = None) -> List[OpenIETriplet]:
        """Extract OpenIE triplets from text with enhanced error handling and caching.

        Args:
            text: Input text to process
            source_doc_id: Optional source document ID for tracking
            chunk_info: Optional chunk information for context

        Returns:
            List of extracted OpenIE triplets

        Raises:
            ProcessingError: If extraction fails
        """
        if not self.enabled:
            logger.debug("OpenIE disabled, returning empty triplets")
            return []

        if not text or not text.strip():
            return []

        # Check cache first
        cache_key = self._get_cache_key(text)
        if self._cache and cache_key in self._cache:
            logger.debug("Returning cached triplets", cache_key=cache_key[:16])
            return self._cache[cache_key]

        await self.initialize()

        start_time = time.time()
        error_count = 0

        for attempt in range(self.retry_attempts):
            try:
                logger.debug(
                    "Starting OpenIE extraction",
                    text_length=len(text),
                    source_doc_id=source_doc_id,
                    attempt=attempt + 1,
                    backend=self.backend.value
                )

                # Split text into sentences with improved processing
                sentences = await self._split_sentences_enhanced(text)

                # Process sentences in batches with retry logic
                all_triplets = []
                for i in range(0, len(sentences), self.batch_size):
                    batch = sentences[i:i + self.batch_size]
                    batch_triplets = await self._process_sentence_batch_enhanced(batch, source_doc_id, chunk_info)
                    all_triplets.extend(batch_triplets)

                # Filter by confidence and quality
                filtered_triplets = self._filter_and_score_triplets(all_triplets)

                # Cache results if enabled
                if self._cache and len(filtered_triplets) > 0:
                    self._update_cache(cache_key, filtered_triplets)

                # Record metrics
                processing_time = time.time() - start_time
                if self.enable_metrics:
                    self._record_metrics(len(sentences), len(all_triplets), len(filtered_triplets), processing_time, error_count)

                logger.info(
                    "OpenIE extraction completed",
                    total_sentences=len(sentences),
                    total_triplets=len(all_triplets),
                    filtered_triplets=len(filtered_triplets),
                    confidence_threshold=self.confidence_threshold,
                    processing_time=processing_time,
                    backend=self.backend.value,
                    source_doc_id=source_doc_id
                )

                return filtered_triplets

            except Exception as e:
                error_count += 1
                logger.warning(
                    "OpenIE extraction attempt failed",
                    attempt=attempt + 1,
                    error=str(e),
                    error_type=type(e).__name__,
                    text_length=len(text),
                    source_doc_id=source_doc_id
                )

                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    # Final attempt failed
                    logger.error(
                        "OpenIE extraction failed after all retries",
                        error=str(e),
                        error_type=type(e).__name__,
                        text_length=len(text),
                        source_doc_id=source_doc_id,
                        total_attempts=self.retry_attempts
                    )
                    raise ProcessingError(f"OpenIE extraction failed after {self.retry_attempts} attempts: {e}")

        # Should never reach here
        raise ProcessingError("OpenIE extraction failed unexpectedly")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _update_cache(self, key: str, triplets: List[OpenIETriplet]) -> None:
        """Update cache with new triplets."""
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = triplets

    def _record_metrics(self, total_sentences: int, total_triplets: int, filtered_triplets: int, processing_time: float, error_count: int) -> None:
        """Record extraction metrics."""
        avg_confidence = 0.0
        if filtered_triplets > 0:
            # This would need to be calculated from actual triplets
            avg_confidence = 0.8  # Placeholder

        metrics = ExtractionMetrics(
            total_sentences=total_sentences,
            processed_sentences=total_sentences - error_count,
            total_triplets=total_triplets,
            filtered_triplets=filtered_triplets,
            processing_time=processing_time,
            average_confidence=avg_confidence,
            error_count=error_count
        )

        self._metrics.append(metrics)

        # Keep only last 100 metrics
        if len(self._metrics) > 100:
            self._metrics = self._metrics[-100:]

    async def _split_sentences_enhanced(self, text: str) -> List[str]:
        """Enhanced sentence splitting with better handling of edge cases."""
        try:
            def split_with_enhanced_logic():
                try:
                    from nltk.tokenize import sent_tokenize
                    sentences = sent_tokenize(text)
                except Exception:
                    # Enhanced fallback with better regex
                    import re
                    # Split on sentence endings but preserve abbreviations
                    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+', text)

                # Clean and filter sentences
                cleaned_sentences = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    # Filter out very short sentences, but keep meaningful ones
                    if len(sentence) > 5 and len(sentence.split()) > 2:
                        cleaned_sentences.append(sentence)

                return cleaned_sentences

            sentences = await asyncio.get_event_loop().run_in_executor(
                self._executor, split_with_enhanced_logic
            )

            logger.debug(f"Split text into {len(sentences)} sentences")
            return sentences

        except Exception as e:
            logger.warning("Enhanced sentence splitting failed, using simple fallback", error=str(e))
            # Simple fallback
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    async def _process_sentence_batch_enhanced(self, sentences: List[str], source_doc_id: Optional[str] = None, chunk_info: Optional[Dict] = None) -> List[OpenIETriplet]:
        """Enhanced batch processing with better error handling and parallel processing."""
        try:
            def extract_from_sentences():
                triplets = []
                for i, sentence in enumerate(sentences):
                    try:
                        start_time = time.time()
                        sentence_triplets = self._extract_from_sentence_enhanced(sentence, source_doc_id, chunk_info)
                        extraction_time = time.time() - start_time

                        # Add timing information to triplets
                        enhanced_triplets = []
                        for triplet in sentence_triplets:
                            enhanced_triplet = triplet._replace(
                                backend=self.backend.value,
                                extraction_time=extraction_time,
                                metadata={
                                    'sentence_index': i,
                                    'batch_size': len(sentences),
                                    'source_doc_id': source_doc_id,
                                    'chunk_info': chunk_info
                                }
                            )
                            enhanced_triplets.append(enhanced_triplet)

                        triplets.extend(enhanced_triplets)

                    except Exception as e:
                        logger.warning(
                            "Failed to extract triplets from sentence",
                            sentence=sentence[:100],
                            sentence_index=i,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                return triplets

            # Run extraction in thread pool with timeout
            try:
                triplets = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor, extract_from_sentences
                    ),
                    timeout=self.timeout_seconds * 2  # Increased timeout for enhanced processing
                )

                logger.debug(f"Processed batch of {len(sentences)} sentences, extracted {len(triplets)} triplets")
                return triplets

            except asyncio.TimeoutError:
                logger.warning(
                    "OpenIE extraction timed out",
                    batch_size=len(sentences),
                    timeout_seconds=self.timeout_seconds * 2
                )
                return []

        except Exception as e:
            logger.error("Enhanced batch processing failed", error=str(e), batch_size=len(sentences))
            return []

    def _filter_and_score_triplets(self, triplets: List[OpenIETriplet]) -> List[OpenIETriplet]:
        """Filter and score triplets based on quality metrics."""
        if not triplets:
            return []

        filtered_triplets = []

        for triplet in triplets:
            # Basic confidence filtering
            if triplet.confidence < self.confidence_threshold:
                continue

            # Quality checks
            if not self._is_valid_triplet(triplet):
                continue

            # Additional scoring could be added here
            filtered_triplets.append(triplet)

        # Sort by confidence (highest first)
        filtered_triplets.sort(key=lambda t: t.confidence, reverse=True)

        return filtered_triplets

    def _is_valid_triplet(self, triplet: OpenIETriplet) -> bool:
        """Check if triplet meets quality criteria."""
        # Check for empty components
        if not triplet.subject.strip() or not triplet.predicate.strip() or not triplet.object.strip():
            return False

        # Check for minimum length
        if len(triplet.subject) < 2 or len(triplet.predicate) < 2 or len(triplet.object) < 2:
            return False

        # Check for common noise patterns
        noise_patterns = ['the', 'a', 'an', 'this', 'that', 'these', 'those']
        if triplet.subject.lower() in noise_patterns or triplet.object.lower() in noise_patterns:
            return False

        # Check for reasonable confidence
        if triplet.confidence <= 0.0 or triplet.confidence > 1.0:
            return False

        return True
    
    def _extract_from_sentence_enhanced(self, sentence: str, source_doc_id: Optional[str] = None, chunk_info: Optional[Dict] = None) -> List[OpenIETriplet]:
        """Enhanced extraction from a single sentence supporting multiple backends."""
        if not self._openie_client:
            return []

        try:
            if self.backend == OpenIEBackend.STANFORD:
                return self._extract_stanford(sentence, source_doc_id, chunk_info)
            elif self.backend == OpenIEBackend.ALLENNLP:
                return self._extract_allennlp(sentence, source_doc_id, chunk_info)
            elif self.backend == OpenIEBackend.SPACY:
                return self._extract_spacy(sentence, source_doc_id, chunk_info)
            else:
                logger.warning(f"Unknown backend: {self.backend}")
                return []

        except Exception as e:
            logger.warning(
                "Failed to extract from sentence",
                sentence=sentence[:100],
                backend=self.backend.value,
                error=str(e)
            )
            return []

    def _extract_stanford(self, sentence: str, source_doc_id: Optional[str] = None, chunk_info: Optional[Dict] = None) -> List[OpenIETriplet]:
        """Extract using Stanford OpenIE."""
        extractions = self._openie_client.annotate(sentence)

        triplets = []
        for extraction in extractions[:self.max_triplets_per_sentence]:
            # Parse extraction result
            subject = extraction.get('subject', '').strip()
            predicate = extraction.get('relation', '').strip()
            obj = extraction.get('object', '').strip()
            confidence = float(extraction.get('confidence', 0.0))

            if subject and predicate and obj:
                triplet = OpenIETriplet(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=confidence,
                    sentence=sentence,
                    backend=self.backend.value
                )
                triplets.append(triplet)

        return triplets

    def _extract_allennlp(self, sentence: str, source_doc_id: Optional[str] = None, chunk_info: Optional[Dict] = None) -> List[OpenIETriplet]:
        """Extract using AllenNLP OpenIE."""
        result = self._openie_client.predict(sentence=sentence)

        triplets = []
        for extraction in result.get('verbs', [])[:self.max_triplets_per_sentence]:
            # Parse AllenNLP format
            description = extraction.get('description', '')
            confidence = float(extraction.get('confidence', 0.8))  # AllenNLP doesn't always provide confidence

            # Simple parsing of description like "[ARG0: subject] [V: predicate] [ARG1: object]"
            import re
            arg0_match = re.search(r'\[ARG0: ([^\]]+)\]', description)
            v_match = re.search(r'\[V: ([^\]]+)\]', description)
            arg1_match = re.search(r'\[ARG1: ([^\]]+)\]', description)

            if arg0_match and v_match and arg1_match:
                subject = arg0_match.group(1).strip()
                predicate = v_match.group(1).strip()
                obj = arg1_match.group(1).strip()

                triplet = OpenIETriplet(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=confidence,
                    sentence=sentence,
                    backend=self.backend.value
                )
                triplets.append(triplet)

        return triplets

    def _extract_spacy(self, sentence: str, source_doc_id: Optional[str] = None, chunk_info: Optional[Dict] = None) -> List[OpenIETriplet]:
        """Extract using spaCy rule-based approach."""
        doc = self._openie_client(sentence)

        triplets = []
        for sent in doc.sents:
            # Simple rule-based extraction
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    # Find subject and object
                    subject = None
                    obj = None

                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subject = child.text
                        elif child.dep_ in ["dobj", "pobj"]:
                            obj = child.text

                    if subject and obj:
                        triplet = OpenIETriplet(
                            subject=subject,
                            predicate=token.lemma_,
                            object=obj,
                            confidence=0.7,  # Fixed confidence for rule-based
                            sentence=sentence,
                            backend=self.backend.value
                        )
                        triplets.append(triplet)

                        if len(triplets) >= self.max_triplets_per_sentence:
                            break

        return triplets
    
    def get_metrics(self) -> List[ExtractionMetrics]:
        """Get extraction metrics."""
        return self._metrics.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self._metrics:
            return {}

        total_sentences = sum(m.total_sentences for m in self._metrics)
        total_triplets = sum(m.total_triplets for m in self._metrics)
        total_time = sum(m.processing_time for m in self._metrics)
        total_errors = sum(m.error_count for m in self._metrics)

        return {
            'total_extractions': len(self._metrics),
            'total_sentences': total_sentences,
            'total_triplets': total_triplets,
            'total_processing_time': total_time,
            'total_errors': total_errors,
            'average_sentences_per_extraction': total_sentences / len(self._metrics) if self._metrics else 0,
            'average_triplets_per_sentence': total_triplets / total_sentences if total_sentences > 0 else 0,
            'average_processing_time': total_time / len(self._metrics) if self._metrics else 0,
            'error_rate': total_errors / total_sentences if total_sentences > 0 else 0,
            'backend': self.backend.value,
            'cache_size': len(self._cache) if self._cache else 0
        }

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._openie_client and hasattr(self._openie_client, 'close'):
                await asyncio.get_event_loop().run_in_executor(
                    self._executor, self._openie_client.close
                )

            if self._executor:
                self._executor.shutdown(wait=True)

            # Clear cache
            if self._cache:
                self._cache.clear()

            logger.info("OpenIE service closed", backend=self.backend.value)

        except Exception as e:
            logger.warning("Error during OpenIE service cleanup", error=str(e))

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
