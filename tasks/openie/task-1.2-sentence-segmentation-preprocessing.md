# Task 1.2: Sentence Segmentation and Preprocessing Pipeline

## Objective
Implement a robust sentence segmentation and preprocessing pipeline that prepares clean, well-structured text for OpenIE relation extraction, ensuring optimal triplet extraction quality.

## Scope
- Create sentence processor for text segmentation
- Implement text cleaning and normalization
- Add sentence quality assessment
- Integrate with existing text processing pipeline
- **MANDATORY**: Test thoroughly before proceeding to Task 1.3

## Implementation Details

### 1. Create Sentence Processor

**File**: `packages/morag-graph/src/morag_graph/processors/sentence_processor.py`

```python
"""Sentence processing and segmentation for OpenIE."""

import re
import asyncio
from typing import List, Dict, Any, Optional
import structlog
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)

class SentenceProcessor:
    """Processor for sentence segmentation and preprocessing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sentence processor.
        
        Args:
            config: Optional configuration overrides
        """
        self.settings = get_settings()
        self.config = config or {}
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is downloaded."""
        required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{data_name}')
                except LookupError:
                    try:
                        nltk.data.find(f'taggers/{data_name}')
                    except LookupError:
                        logger.info(f"Downloading NLTK {data_name}")
                        nltk.download(data_name)
    
    async def process_text(self, text: str) -> List[Dict[str, Any]]:
        """Process text into clean, segmented sentences.
        
        Args:
            text: Input text to process
            
        Returns:
            List of sentence dictionaries with metadata
            
        Raises:
            ProcessingError: If processing fails
        """
        if not text or not text.strip():
            return []
        
        try:
            # Clean and normalize text
            cleaned_text = await self._clean_text(text)
            
            # Segment into sentences
            raw_sentences = await self._segment_sentences(cleaned_text)
            
            # Process and validate sentences
            processed_sentences = []
            for i, sentence in enumerate(raw_sentences):
                sentence_data = await self._process_sentence(sentence, i)
                if sentence_data:
                    processed_sentences.append(sentence_data)
            
            logger.info(
                "Sentence processing completed",
                input_length=len(text),
                raw_sentences=len(raw_sentences),
                processed_sentences=len(processed_sentences)
            )
            
            return processed_sentences
            
        except Exception as e:
            logger.error("Sentence processing failed", error=str(e))
            raise ProcessingError(f"Sentence processing failed: {str(e)}")
    
    async def _clean_text(self, text: str) -> str:
        """Clean and normalize input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        try:
            # Run text cleaning in thread pool
            loop = asyncio.get_event_loop()
            cleaned = await loop.run_in_executor(None, self._clean_text_sync, text)
            return cleaned
            
        except Exception as e:
            logger.error("Text cleaning failed", error=str(e))
            return text  # Return original if cleaning fails
    
    def _clean_text_sync(self, text: str) -> str:
        """Synchronous text cleaning operations."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common punctuation issues
        text = re.sub(r'\.{2,}', '.', text)  # Multiple periods
        text = re.sub(r'\?{2,}', '?', text)  # Multiple question marks
        text = re.sub(r'!{2,}', '!', text)   # Multiple exclamation marks
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after punctuation
        
        # Remove markdown artifacts that might interfere
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Bold/italic
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Code spans
        text = re.sub(r'#{1,6}\s*', '', text)  # Headers
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    async def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using NLTK.
        
        Args:
            text: Cleaned input text
            
        Returns:
            List of sentence strings
        """
        try:
            # Run sentence tokenization in thread pool
            loop = asyncio.get_event_loop()
            sentences = await loop.run_in_executor(None, sent_tokenize, text)
            
            # Additional sentence boundary detection for edge cases
            refined_sentences = []
            for sentence in sentences:
                # Split on common abbreviations that NLTK might miss
                sub_sentences = self._split_complex_sentences(sentence)
                refined_sentences.extend(sub_sentences)
            
            return refined_sentences
            
        except Exception as e:
            logger.error("Sentence segmentation failed", error=str(e))
            raise ProcessingError(f"Sentence segmentation failed: {str(e)}")
    
    def _split_complex_sentences(self, sentence: str) -> List[str]:
        """Split complex sentences that NLTK might miss."""
        # Handle sentences with multiple clauses separated by semicolons
        if ';' in sentence and len(sentence) > 100:
            parts = sentence.split(';')
            return [part.strip() + '.' for part in parts if part.strip()]
        
        # Handle very long sentences (likely multiple sentences)
        if len(sentence) > 200:
            # Try to split on conjunctions in long sentences
            conjunctions = [' and ', ' but ', ' however ', ' therefore ', ' moreover ']
            for conj in conjunctions:
                if conj in sentence.lower():
                    parts = sentence.split(conj)
                    if len(parts) == 2 and all(len(p.strip()) > 20 for p in parts):
                        return [parts[0].strip() + '.', parts[1].strip().capitalize() + '.']
        
        return [sentence]
    
    async def _process_sentence(self, sentence: str, index: int) -> Optional[Dict[str, Any]]:
        """Process and validate a single sentence.
        
        Args:
            sentence: Raw sentence text
            index: Sentence index in document
            
        Returns:
            Sentence data dictionary or None if invalid
        """
        try:
            # Clean sentence
            cleaned = sentence.strip()
            if not cleaned:
                return None
            
            # Quality assessment
            quality_score = await self._assess_sentence_quality(cleaned)
            
            # Skip low-quality sentences
            min_quality = self.config.get('min_sentence_quality', 0.3)
            if quality_score < min_quality:
                logger.debug(
                    "Skipping low-quality sentence",
                    sentence=cleaned[:50],
                    quality_score=quality_score
                )
                return None
            
            # Tokenize for additional metadata
            loop = asyncio.get_event_loop()
            tokens = await loop.run_in_executor(None, word_tokenize, cleaned)
            
            return {
                'text': cleaned,
                'index': index,
                'length': len(cleaned),
                'word_count': len(tokens),
                'quality_score': quality_score,
                'tokens': tokens,
                'is_question': cleaned.endswith('?'),
                'is_exclamation': cleaned.endswith('!'),
                'has_entities': self._has_potential_entities(tokens)
            }
            
        except Exception as e:
            logger.error(
                "Sentence processing failed",
                error=str(e),
                sentence=sentence[:50]
            )
            return None
    
    async def _assess_sentence_quality(self, sentence: str) -> float:
        """Assess sentence quality for OpenIE processing.
        
        Args:
            sentence: Sentence to assess
            
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Length checks
        if len(sentence) < 10:
            score -= 0.5  # Too short
        elif len(sentence) > 300:
            score -= 0.3  # Very long
        
        # Word count checks
        words = sentence.split()
        if len(words) < 3:
            score -= 0.4  # Too few words
        elif len(words) > 50:
            score -= 0.2  # Too many words
        
        # Structural checks
        if not re.search(r'[.!?]$', sentence):
            score -= 0.2  # No proper ending
        
        if sentence.count('(') != sentence.count(')'):
            score -= 0.1  # Unbalanced parentheses
        
        # Content quality checks
        if re.search(r'^[A-Z][a-z]', sentence):
            score += 0.1  # Proper capitalization
        
        if re.search(r'\b(is|are|was|were|has|have|had|will|would|can|could|should|may|might)\b', sentence.lower()):
            score += 0.1  # Contains verbs likely to form relations
        
        # Penalize sentences with too many special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.!?]', sentence)) / len(sentence)
        if special_char_ratio > 0.2:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _has_potential_entities(self, tokens: List[str]) -> bool:
        """Check if sentence has potential named entities.
        
        Args:
            tokens: Tokenized sentence
            
        Returns:
            True if potential entities found
        """
        # Look for capitalized words (potential proper nouns)
        capitalized_words = [token for token in tokens if token[0].isupper() and len(token) > 1]
        
        # Look for common entity patterns
        entity_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Person names
            r'\b[A-Z][a-z]+ (Inc|Corp|Ltd|LLC)\b',  # Company names
            r'\b(Mr|Mrs|Dr|Prof)\. [A-Z][a-z]+\b',  # Titles with names
        ]
        
        sentence_text = ' '.join(tokens)
        for pattern in entity_patterns:
            if re.search(pattern, sentence_text):
                return True
        
        return len(capitalized_words) >= 2
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics and configuration.
        
        Returns:
            Dictionary with processor statistics
        """
        return {
            'processor_name': 'SentenceProcessor',
            'min_sentence_quality': self.config.get('min_sentence_quality', 0.3),
            'max_sentence_length': self.config.get('max_sentence_length', 300),
            'min_sentence_length': self.config.get('min_sentence_length', 10),
            'nltk_data_available': True
        }
```

### 2. Integration with OpenIE Service

**File**: Update `packages/morag-graph/src/morag_graph/services/openie_service.py`

Add sentence processor integration:

```python
# Add import
from morag_graph.processors.sentence_processor import SentenceProcessor

# Update OpenIEService.__init__
def __init__(self, config: Optional[Dict[str, Any]] = None):
    # ... existing code ...
    self.sentence_processor = SentenceProcessor(config)

# Update extract_triplets method
async def extract_triplets(self, text: str) -> List[Dict[str, Any]]:
    """Extract relation triplets from text."""
    if not text or not text.strip():
        return []
    
    try:
        # Process text into clean sentences
        processed_sentences = await self.sentence_processor.process_text(text)
        
        # Extract triplets from each processed sentence
        all_triplets = []
        for sentence_data in processed_sentences:
            sentence_triplets = await self._extract_sentence_triplets(
                sentence_data['text']
            )
            
            # Add sentence metadata to triplets
            for triplet in sentence_triplets:
                triplet.update({
                    'sentence_index': sentence_data['index'],
                    'sentence_quality': sentence_data['quality_score'],
                    'sentence_word_count': sentence_data['word_count']
                })
            
            all_triplets.extend(sentence_triplets)
        
        logger.info(
            "OpenIE extraction completed",
            sentences_processed=len(processed_sentences),
            triplets_extracted=len(all_triplets)
        )
        
        return all_triplets
        
    except Exception as e:
        logger.error("OpenIE extraction failed", error=str(e))
        raise ProcessingError(f"OpenIE extraction failed: {str(e)}")
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_sentence_processor.py`

```python
"""Tests for sentence processor."""

import pytest
from unittest.mock import patch, AsyncMock

from morag_graph.processors.sentence_processor import SentenceProcessor
from morag_core.exceptions import ProcessingError

class TestSentenceProcessor:
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = SentenceProcessor()
        assert processor.config == {}
    
    @pytest.mark.asyncio
    async def test_process_text_success(self):
        """Test successful text processing."""
        processor = SentenceProcessor()
        
        text = "John loves Mary. She is very kind."
        result = await processor.process_text(text)
        
        assert len(result) == 2
        assert result[0]['text'] == "John loves Mary."
        assert result[1]['text'] == "She is very kind."
        assert all('quality_score' in sentence for sentence in result)
    
    @pytest.mark.asyncio
    async def test_process_empty_text(self):
        """Test processing empty text."""
        processor = SentenceProcessor()
        result = await processor.process_text("")
        assert result == []
    
    def test_clean_text_sync(self):
        """Test text cleaning functionality."""
        processor = SentenceProcessor()
        
        dirty_text = "This  is   a    test.....Multiple   spaces!!!"
        cleaned = processor._clean_text_sync(dirty_text)
        
        assert "  " not in cleaned
        assert "....." not in cleaned
        assert "!!!" not in cleaned
    
    @pytest.mark.asyncio
    async def test_assess_sentence_quality(self):
        """Test sentence quality assessment."""
        processor = SentenceProcessor()
        
        # Good sentence
        good_score = await processor._assess_sentence_quality("John loves Mary very much.")
        assert good_score > 0.7
        
        # Poor sentence
        poor_score = await processor._assess_sentence_quality("a")
        assert poor_score < 0.5
    
    def test_has_potential_entities(self):
        """Test entity detection."""
        processor = SentenceProcessor()
        
        # Sentence with entities
        tokens_with_entities = ["John", "loves", "Mary", "Smith"]
        assert processor._has_potential_entities(tokens_with_entities)
        
        # Sentence without entities
        tokens_without_entities = ["the", "cat", "sat", "on", "mat"]
        assert not processor._has_potential_entities(tokens_without_entities)
```

## Acceptance Criteria

- [ ] SentenceProcessor class implemented with text cleaning
- [ ] Robust sentence segmentation using NLTK
- [ ] Quality assessment for sentence filtering
- [ ] Integration with OpenIE service
- [ ] Handling of edge cases (long sentences, special characters)
- [ ] Async support for non-blocking operations
- [ ] Comprehensive unit tests with >90% coverage
- [ ] Proper logging and error handling
- [ ] Performance optimization for large texts

## Dependencies
- Task 1.1: OpenIE Dependency Integration and Service Wrapper

## Estimated Effort
- **Development**: 5-7 hours
- **Testing**: 3-4 hours
- **Integration**: 2-3 hours
- **Total**: 10-14 hours

## Notes
- Focus on sentence quality to improve OpenIE extraction accuracy
- Handle multilingual text considerations (Spanish, German, English)
- Optimize for processing large documents efficiently
- Consider sentence context preservation for better relation extraction
