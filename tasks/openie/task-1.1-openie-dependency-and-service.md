# Task 1.1: OpenIE Dependency Integration and Service Wrapper

## Objective
Add OpenIE framework as a dependency and create a wrapper service that provides a clean interface for the MoRAG system to interact with OpenIE functionality for relation extraction.

## Scope
- Add OpenIE dependencies to the project
- Create a wrapper service in morag-graph package
- Implement basic configuration management
- Set up logging and error handling
- **MANDATORY**: Test thoroughly before proceeding to Task 1.2

## Implementation Details

### 1. Add Dependencies

**File**: `requirements.txt`
```txt
# Add OpenIE dependencies
openie>=1.0.0
stanford-openie>=1.3.0
nltk>=3.8
```

**File**: `packages/morag-graph/pyproject.toml`
```toml
[project]
dependencies = [
    # ... existing dependencies
    "openie>=1.0.0",
    "stanford-openie>=1.3.0", 
    "nltk>=3.8",
]
```

### 2. Create OpenIE Service

**File**: `packages/morag-graph/src/morag_graph/services/openie_service.py`

```python
"""OpenIE wrapper service for MoRAG relation extraction."""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import structlog
import nltk
from openie import StanfordOpenIE

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError, ConfigurationError

logger = structlog.get_logger(__name__)

class OpenIEService:
    """Service wrapper for Stanford OpenIE relation extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenIE service.
        
        Args:
            config: Optional configuration overrides
        """
        self.settings = get_settings()
        self.config = config or {}
        self._openie = None
        self._initialize_openie()
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt')
    
    def _initialize_openie(self) -> None:
        """Initialize OpenIE instance with configuration."""
        try:
            # Configuration parameters
            confidence_threshold = self.config.get('confidence_threshold', 0.7)
            max_triplets = self.config.get('max_triplets_per_sentence', 10)
            timeout = self.config.get('timeout_seconds', 30)
            
            # Initialize Stanford OpenIE
            self._openie = StanfordOpenIE(
                confidence_threshold=confidence_threshold,
                max_triplets=max_triplets,
                timeout=timeout
            )
            
            logger.info(
                "OpenIE service initialized", 
                confidence_threshold=confidence_threshold,
                max_triplets=max_triplets,
                timeout=timeout
            )
            
        except Exception as e:
            logger.error("Failed to initialize OpenIE service", error=str(e))
            raise ConfigurationError(f"OpenIE initialization failed: {str(e)}")
    
    async def extract_triplets(self, text: str) -> List[Dict[str, Any]]:
        """Extract relation triplets from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of triplet dictionaries with subject, predicate, object
            
        Raises:
            ProcessingError: If extraction fails
        """
        if not text or not text.strip():
            return []
        
        try:
            # Segment text into sentences
            sentences = await self._segment_sentences(text)
            
            # Extract triplets from each sentence
            all_triplets = []
            for sentence in sentences:
                sentence_triplets = await self._extract_sentence_triplets(sentence)
                all_triplets.extend(sentence_triplets)
            
            logger.info(
                "OpenIE extraction completed",
                sentences_processed=len(sentences),
                triplets_extracted=len(all_triplets)
            )
            
            return all_triplets
            
        except Exception as e:
            logger.error("OpenIE extraction failed", error=str(e), text_length=len(text))
            raise ProcessingError(f"OpenIE extraction failed: {str(e)}")
    
    async def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into individual sentences.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of sentence strings
        """
        try:
            # Run sentence segmentation in thread pool
            loop = asyncio.get_event_loop()
            sentences = await loop.run_in_executor(
                None,
                nltk.sent_tokenize,
                text
            )
            
            # Filter out empty or very short sentences
            filtered_sentences = [
                s.strip() for s in sentences 
                if s.strip() and len(s.strip()) > 10
            ]
            
            return filtered_sentences
            
        except Exception as e:
            logger.error("Sentence segmentation failed", error=str(e))
            raise ProcessingError(f"Sentence segmentation failed: {str(e)}")
    
    async def _extract_sentence_triplets(self, sentence: str) -> List[Dict[str, Any]]:
        """Extract triplets from a single sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of triplet dictionaries
        """
        try:
            # Run OpenIE extraction in thread pool
            loop = asyncio.get_event_loop()
            triplets = await loop.run_in_executor(
                None,
                self._openie.annotate,
                sentence
            )
            
            # Convert to standardized format
            formatted_triplets = []
            for triplet in triplets:
                formatted_triplet = {
                    'subject': triplet.get('subject', '').strip(),
                    'predicate': triplet.get('relation', '').strip(),
                    'object': triplet.get('object', '').strip(),
                    'confidence': triplet.get('confidence', 0.0),
                    'source_sentence': sentence,
                    'extraction_method': 'stanford_openie'
                }
                
                # Only include triplets with all components
                if all([
                    formatted_triplet['subject'],
                    formatted_triplet['predicate'], 
                    formatted_triplet['object']
                ]):
                    formatted_triplets.append(formatted_triplet)
            
            return formatted_triplets
            
        except Exception as e:
            logger.error(
                "Sentence triplet extraction failed", 
                error=str(e), 
                sentence=sentence[:100]
            )
            # Return empty list rather than failing entire batch
            return []
    
    async def extract_batch_triplets(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract triplets from multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of triplet lists, one per input text
        """
        try:
            batch_size = self.config.get('batch_size', 100)
            results = []
            
            # Process in batches to manage memory
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = await asyncio.gather(
                    *[self.extract_triplets(text) for text in batch],
                    return_exceptions=True
                )
                
                # Handle exceptions in batch results
                processed_results = []
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error("Batch extraction error", error=str(result))
                        processed_results.append([])
                    else:
                        processed_results.append(result)
                
                results.extend(processed_results)
            
            return results
            
        except Exception as e:
            logger.error("Batch triplet extraction failed", error=str(e))
            raise ProcessingError(f"Batch triplet extraction failed: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics and configuration.
        
        Returns:
            Dictionary with service statistics
        """
        return {
            'service_name': 'OpenIE',
            'implementation': 'stanford_openie',
            'confidence_threshold': self.config.get('confidence_threshold', 0.7),
            'max_triplets_per_sentence': self.config.get('max_triplets_per_sentence', 10),
            'timeout_seconds': self.config.get('timeout_seconds', 30),
            'batch_size': self.config.get('batch_size', 100),
            'status': 'initialized' if self._openie else 'not_initialized'
        }
    
    def __del__(self):
        """Cleanup OpenIE resources."""
        if self._openie:
            try:
                self._openie.close()
            except Exception:
                pass  # Ignore cleanup errors
```

### 3. Configuration Updates

**File**: `packages/morag-core/src/morag_core/config.py`

Add OpenIE-specific configuration:

```python
# OpenIE configuration
OPENIE_ENABLED: bool = Field(default=True, description="Enable OpenIE relation extraction")
OPENIE_IMPLEMENTATION: str = Field(default="stanford", description="OpenIE implementation to use")
OPENIE_CONFIDENCE_THRESHOLD: float = Field(default=0.7, description="Minimum confidence for triplet extraction")
OPENIE_MAX_TRIPLETS_PER_SENTENCE: int = Field(default=10, description="Maximum triplets per sentence")
OPENIE_TIMEOUT_SECONDS: int = Field(default=30, description="Timeout for OpenIE processing")
OPENIE_BATCH_SIZE: int = Field(default=100, description="Batch size for processing multiple texts")
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_openie_service.py`

```python
"""Tests for OpenIE service."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from morag_graph.services.openie_service import OpenIEService
from morag_core.exceptions import ProcessingError, ConfigurationError

class TestOpenIEService:
    
    def test_initialization(self):
        """Test service initialization."""
        service = OpenIEService()
        assert service._openie is not None
        assert service.config == {}
    
    def test_initialization_with_config(self):
        """Test service initialization with custom config."""
        config = {
            'confidence_threshold': 0.8,
            'max_triplets_per_sentence': 5,
            'timeout_seconds': 60
        }
        service = OpenIEService(config)
        assert service.config == config
    
    @pytest.mark.asyncio
    async def test_extract_triplets_success(self):
        """Test successful triplet extraction."""
        service = OpenIEService()
        
        with patch.object(service, '_segment_sentences') as mock_segment, \
             patch.object(service, '_extract_sentence_triplets') as mock_extract:
            
            mock_segment.return_value = ["John loves Mary."]
            mock_extract.return_value = [{
                'subject': 'John',
                'predicate': 'loves',
                'object': 'Mary',
                'confidence': 0.9,
                'source_sentence': 'John loves Mary.',
                'extraction_method': 'stanford_openie'
            }]
            
            result = await service.extract_triplets("John loves Mary.")
            
            assert len(result) == 1
            assert result[0]['subject'] == 'John'
            assert result[0]['predicate'] == 'loves'
            assert result[0]['object'] == 'Mary'
    
    @pytest.mark.asyncio
    async def test_extract_triplets_empty_text(self):
        """Test extraction with empty text."""
        service = OpenIEService()
        result = await service.extract_triplets("")
        assert result == []
    
    @pytest.mark.asyncio
    async def test_segment_sentences(self):
        """Test sentence segmentation."""
        service = OpenIEService()
        
        with patch('nltk.sent_tokenize') as mock_tokenize:
            mock_tokenize.return_value = [
                "This is sentence one.",
                "This is sentence two.",
                "Short."  # Should be filtered out
            ]
            
            result = await service._segment_sentences("Test text.")
            
            assert len(result) == 2
            assert "This is sentence one." in result
            assert "This is sentence two." in result
            assert "Short." not in result
    
    def test_get_statistics(self):
        """Test getting service statistics."""
        config = {'confidence_threshold': 0.8}
        service = OpenIEService(config)
        stats = service.get_statistics()
        
        assert stats['service_name'] == 'OpenIE'
        assert stats['implementation'] == 'stanford_openie'
        assert stats['confidence_threshold'] == 0.8
        assert stats['status'] == 'initialized'
```

## Acceptance Criteria

- [ ] OpenIE dependencies added to project requirements
- [ ] OpenIEService class implemented with proper error handling
- [ ] Configuration management for OpenIE options
- [ ] Async support for non-blocking operations
- [ ] Sentence segmentation functionality
- [ ] Triplet extraction with standardized format
- [ ] Batch processing capabilities
- [ ] Comprehensive unit tests with >90% coverage
- [ ] Proper logging and error handling
- [ ] Documentation for service usage

## Dependencies
- None (this is the foundation task)

## Estimated Effort
- **Development**: 6-8 hours
- **Testing**: 3-4 hours
- **Documentation**: 1-2 hours
- **Total**: 10-14 hours

## Notes
- This task establishes the foundation for all subsequent OpenIE integration
- Focus on clean interfaces and proper error handling
- Ensure async compatibility for integration with existing MoRAG services
- Consider future extensibility for additional OpenIE implementations
- Stanford OpenIE requires Java runtime environment
