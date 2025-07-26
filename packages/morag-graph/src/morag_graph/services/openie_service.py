"""OpenIE wrapper service for MoRAG relation extraction."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import structlog
import nltk
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError, ValidationError

logger = structlog.get_logger(__name__)


class OpenIETriplet(NamedTuple):
    """Represents an OpenIE extracted triplet."""
    subject: str
    predicate: str
    object: str
    confidence: float
    sentence: str
    start_pos: int = 0
    end_pos: int = 0


class OpenIEService:
    """Service wrapper for Stanford OpenIE with async support and error handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenIE service with configuration.
        
        Args:
            config: Optional configuration dictionary. If None, uses global settings.
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Configuration from settings
        self.enabled = self.config.get('enabled', self.settings.openie_enabled)
        self.implementation = self.config.get('implementation', self.settings.openie_implementation)
        self.confidence_threshold = self.config.get('confidence_threshold', self.settings.openie_confidence_threshold)
        self.max_triplets_per_sentence = self.config.get('max_triplets_per_sentence', self.settings.openie_max_triplets_per_sentence)
        self.batch_size = self.config.get('batch_size', self.settings.openie_batch_size)
        self.timeout_seconds = self.config.get('timeout_seconds', self.settings.openie_timeout_seconds)
        
        # Initialize components
        self._openie_client = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="openie")
        self._initialized = False
        
        logger.info(
            "OpenIE service initialized",
            enabled=self.enabled,
            implementation=self.implementation,
            confidence_threshold=self.confidence_threshold,
            max_triplets_per_sentence=self.max_triplets_per_sentence
        )
    
    async def initialize(self) -> None:
        """Initialize OpenIE client and download required resources."""
        if self._initialized:
            return
            
        if not self.enabled:
            logger.info("OpenIE service disabled, skipping initialization")
            return
            
        try:
            logger.info("Initializing OpenIE service", implementation=self.implementation)
            
            # Download NLTK data if needed
            await self._ensure_nltk_data()
            
            # Initialize OpenIE client in thread pool
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._init_openie_client
            )
            
            self._initialized = True
            logger.info("OpenIE service initialization completed")
            
        except Exception as e:
            logger.error("Failed to initialize OpenIE service", error=str(e), error_type=type(e).__name__)
            raise ValidationError(f"OpenIE initialization failed: {e}")
    
    def _init_openie_client(self) -> None:
        """Initialize OpenIE client (runs in thread pool)."""
        try:
            if self.implementation == "stanford":
                from openie import StanfordOpenIE
                
                # Create temporary directory for Stanford OpenIE
                temp_dir = tempfile.mkdtemp(prefix="openie_")
                
                # Initialize with custom configuration
                self._openie_client = StanfordOpenIE(
                    corenlp_home=temp_dir,
                    memory="2g",  # Allocate 2GB memory
                    timeout=self.timeout_seconds * 1000,  # Convert to milliseconds
                    quiet=True
                )
                
                logger.info("Stanford OpenIE client initialized", temp_dir=temp_dir)
            else:
                raise ValidationError(f"Unsupported OpenIE implementation: {self.implementation}")
                
        except ImportError as e:
            raise ConfigurationError(f"OpenIE dependencies not available: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenIE client: {e}")
    
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
    
    async def extract_triplets(self, text: str, source_doc_id: Optional[str] = None) -> List[OpenIETriplet]:
        """Extract OpenIE triplets from text.
        
        Args:
            text: Input text to process
            source_doc_id: Optional source document ID for tracking
            
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
            
        await self.initialize()
        
        try:
            logger.debug(
                "Starting OpenIE extraction",
                text_length=len(text),
                source_doc_id=source_doc_id
            )
            
            # Split text into sentences
            sentences = await self._split_sentences(text)
            
            # Process sentences in batches
            all_triplets = []
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i:i + self.batch_size]
                batch_triplets = await self._process_sentence_batch(batch, source_doc_id)
                all_triplets.extend(batch_triplets)
            
            # Filter by confidence
            filtered_triplets = [
                triplet for triplet in all_triplets
                if triplet.confidence >= self.confidence_threshold
            ]
            
            logger.info(
                "OpenIE extraction completed",
                total_sentences=len(sentences),
                total_triplets=len(all_triplets),
                filtered_triplets=len(filtered_triplets),
                confidence_threshold=self.confidence_threshold,
                source_doc_id=source_doc_id
            )
            
            return filtered_triplets
            
        except Exception as e:
            logger.error(
                "OpenIE extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
                source_doc_id=source_doc_id
            )
            raise ProcessingError(f"OpenIE extraction failed: {e}")
    
    async def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK or simple fallback."""
        try:
            def split_with_nltk():
                try:
                    from nltk.tokenize import sent_tokenize
                    return sent_tokenize(text)
                except Exception:
                    # Fallback to simple sentence splitting
                    import re
                    sentences = re.split(r'[.!?]+', text)
                    return [s.strip() for s in sentences if s.strip()]
            
            sentences = await asyncio.get_event_loop().run_in_executor(
                self._executor, split_with_nltk
            )
            
            return [s for s in sentences if len(s.strip()) > 10]  # Filter very short sentences
            
        except Exception as e:
            logger.warning("Sentence splitting failed, using fallback", error=str(e))
            # Simple fallback
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    async def _process_sentence_batch(self, sentences: List[str], source_doc_id: Optional[str] = None) -> List[OpenIETriplet]:
        """Process a batch of sentences for triplet extraction."""
        try:
            def extract_from_sentences():
                triplets = []
                for sentence in sentences:
                    try:
                        sentence_triplets = self._extract_from_sentence(sentence)
                        triplets.extend(sentence_triplets)
                    except Exception as e:
                        logger.warning(
                            "Failed to extract triplets from sentence",
                            sentence=sentence[:100],
                            error=str(e)
                        )
                return triplets
            
            # Run extraction in thread pool with timeout
            try:
                triplets = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor, extract_from_sentences
                    ),
                    timeout=self.timeout_seconds
                )
                return triplets
                
            except asyncio.TimeoutError:
                logger.warning(
                    "OpenIE extraction timed out",
                    batch_size=len(sentences),
                    timeout_seconds=self.timeout_seconds
                )
                return []
                
        except Exception as e:
            logger.error("Batch processing failed", error=str(e), batch_size=len(sentences))
            return []
    
    def _extract_from_sentence(self, sentence: str) -> List[OpenIETriplet]:
        """Extract triplets from a single sentence (runs in thread pool)."""
        if not self._openie_client:
            return []
            
        try:
            # Extract triplets using Stanford OpenIE
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
                        sentence=sentence
                    )
                    triplets.append(triplet)
            
            return triplets
            
        except Exception as e:
            logger.warning(
                "Failed to extract from sentence",
                sentence=sentence[:100],
                error=str(e)
            )
            return []
    
    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._openie_client and hasattr(self._openie_client, 'close'):
                await asyncio.get_event_loop().run_in_executor(
                    self._executor, self._openie_client.close
                )
            
            if self._executor:
                self._executor.shutdown(wait=True)
                
            logger.info("OpenIE service closed")
            
        except Exception as e:
            logger.warning("Error during OpenIE service cleanup", error=str(e))
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
