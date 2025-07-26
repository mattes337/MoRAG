"""Sentence segmentation and preprocessing for OpenIE pipeline."""

import asyncio
import re
from typing import List, Dict, Any, Optional, NamedTuple
from concurrent.futures import ThreadPoolExecutor
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


class ProcessedSentence(NamedTuple):
    """Represents a processed sentence with metadata."""
    text: str
    original_text: str
    start_pos: int
    end_pos: int
    sentence_id: str
    quality_score: float
    language: Optional[str] = None
    metadata: Dict[str, Any] = {}


class SentenceProcessor:
    """Advanced sentence segmentation and preprocessing for OpenIE."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sentence processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Configuration
        self.min_sentence_length = self.config.get('min_sentence_length', 10)
        self.max_sentence_length = self.config.get('max_sentence_length', 1000)
        self.enable_cleaning = self.config.get('enable_cleaning', True)
        self.enable_quality_scoring = self.config.get('enable_quality_scoring', True)
        self.batch_size = self.config.get('batch_size', 100)
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sentence_proc")
        
        # Compiled regex patterns for efficiency
        self._sentence_patterns = self._compile_sentence_patterns()
        self._cleaning_patterns = self._compile_cleaning_patterns()
        
        logger.info(
            "Sentence processor initialized",
            min_length=self.min_sentence_length,
            max_length=self.max_sentence_length,
            enable_cleaning=self.enable_cleaning,
            enable_quality_scoring=self.enable_quality_scoring
        )
    
    def _compile_sentence_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for sentence segmentation."""
        return {
            # Basic sentence endings
            'sentence_end': re.compile(r'[.!?]+\s+'),
            
            # Abbreviations that shouldn't end sentences
            'abbreviations': re.compile(r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|Inc|Ltd|Corp|Co|etc|vs|e\.g|i\.e|cf|al|et)\.\s+', re.IGNORECASE),
            
            # Numbers and decimals
            'numbers': re.compile(r'\d+\.\d+'),
            
            # URLs and emails (shouldn't be split)
            'urls': re.compile(r'https?://[^\s]+|www\.[^\s]+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # Quotations and parentheses
            'quoted_text': re.compile(r'"[^"]*"'),
            'parentheses': re.compile(r'\([^)]*\)'),
            
            # Multiple whitespace
            'whitespace': re.compile(r'\s+'),
            
            # Sentence quality indicators
            'has_verb': re.compile(r'\b(?:is|are|was|were|have|has|had|do|does|did|will|would|can|could|should|may|might|must)\b', re.IGNORECASE),
            'has_subject': re.compile(r'\b(?:I|you|he|she|it|we|they|this|that|these|those|[A-Z][a-z]+)\b'),
        }
    
    def _compile_cleaning_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for text cleaning."""
        return {
            # Remove extra whitespace
            'extra_whitespace': re.compile(r'\s+'),
            
            # Remove special characters that interfere with processing
            'control_chars': re.compile(r'[\x00-\x1f\x7f-\x9f]'),
            
            # Normalize punctuation
            'multiple_periods': re.compile(r'\.{2,}'),
            'multiple_exclamation': re.compile(r'!{2,}'),
            'multiple_question': re.compile(r'\?{2,}'),
            
            # Remove markdown-like formatting
            'markdown_bold': re.compile(r'\*\*(.*?)\*\*'),
            'markdown_italic': re.compile(r'\*(.*?)\*'),
            'markdown_code': re.compile(r'`(.*?)`'),
            
            # Remove HTML-like tags
            'html_tags': re.compile(r'<[^>]+>'),
        }
    
    async def process_text(self, text: str, source_doc_id: Optional[str] = None) -> List[ProcessedSentence]:
        """Process text into clean, segmented sentences.
        
        Args:
            text: Input text to process
            source_doc_id: Optional source document ID for tracking
            
        Returns:
            List of processed sentences with metadata
            
        Raises:
            ProcessingError: If processing fails
        """
        if not text or not text.strip():
            return []
        
        try:
            logger.debug(
                "Starting sentence processing",
                text_length=len(text),
                source_doc_id=source_doc_id
            )
            
            # Clean text if enabled
            if self.enable_cleaning:
                cleaned_text = await self._clean_text(text)
            else:
                cleaned_text = text
            
            # Segment into sentences
            raw_sentences = await self._segment_sentences(cleaned_text)
            
            # Process sentences in batches
            processed_sentences = []
            for i in range(0, len(raw_sentences), self.batch_size):
                batch = raw_sentences[i:i + self.batch_size]
                batch_processed = await self._process_sentence_batch(batch, source_doc_id)
                processed_sentences.extend(batch_processed)
            
            # Filter by length and quality
            filtered_sentences = self._filter_sentences(processed_sentences)
            
            logger.info(
                "Sentence processing completed",
                original_length=len(text),
                raw_sentences=len(raw_sentences),
                processed_sentences=len(processed_sentences),
                filtered_sentences=len(filtered_sentences),
                source_doc_id=source_doc_id
            )
            
            return filtered_sentences
            
        except Exception as e:
            logger.error(
                "Sentence processing failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
                source_doc_id=source_doc_id
            )
            raise ProcessingError(f"Sentence processing failed: {e}")
    
    async def _clean_text(self, text: str) -> str:
        """Clean text for better sentence segmentation."""
        def clean_text_sync():
            cleaned = text
            
            # Remove control characters
            cleaned = self._cleaning_patterns['control_chars'].sub('', cleaned)
            
            # Remove HTML tags
            cleaned = self._cleaning_patterns['html_tags'].sub('', cleaned)
            
            # Remove markdown formatting (keep content)
            cleaned = self._cleaning_patterns['markdown_bold'].sub(r'\1', cleaned)
            cleaned = self._cleaning_patterns['markdown_italic'].sub(r'\1', cleaned)
            cleaned = self._cleaning_patterns['markdown_code'].sub(r'\1', cleaned)
            
            # Normalize punctuation
            cleaned = self._cleaning_patterns['multiple_periods'].sub('...', cleaned)
            cleaned = self._cleaning_patterns['multiple_exclamation'].sub('!', cleaned)
            cleaned = self._cleaning_patterns['multiple_question'].sub('?', cleaned)
            
            # Normalize whitespace
            cleaned = self._cleaning_patterns['extra_whitespace'].sub(' ', cleaned)
            
            return cleaned.strip()
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, clean_text_sync
        )
    
    async def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using advanced rules."""
        def segment_sync():
            # Try NLTK first if available
            try:
                import nltk
                from nltk.tokenize import sent_tokenize
                return sent_tokenize(text)
            except (ImportError, LookupError):
                # Fallback to rule-based segmentation
                return self._rule_based_segmentation(text)
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, segment_sync
        )
    
    def _rule_based_segmentation(self, text: str) -> List[str]:
        """Rule-based sentence segmentation as fallback."""
        # Protect abbreviations, URLs, emails, etc.
        protected_spans = []
        
        # Find abbreviations
        for match in self._sentence_patterns['abbreviations'].finditer(text):
            protected_spans.append((match.start(), match.end()))
        
        # Find numbers
        for match in self._sentence_patterns['numbers'].finditer(text):
            protected_spans.append((match.start(), match.end()))
        
        # Find URLs and emails
        for match in self._sentence_patterns['urls'].finditer(text):
            protected_spans.append((match.start(), match.end()))
        for match in self._sentence_patterns['emails'].finditer(text):
            protected_spans.append((match.start(), match.end()))
        
        # Sort protected spans
        protected_spans.sort()
        
        # Split on sentence endings, but avoid protected spans
        sentences = []
        current_start = 0
        
        for match in self._sentence_patterns['sentence_end'].finditer(text):
            end_pos = match.start()
            
            # Check if this ending is in a protected span
            is_protected = any(start <= end_pos < end for start, end in protected_spans)
            
            if not is_protected:
                sentence = text[current_start:match.end()].strip()
                if sentence:
                    sentences.append(sentence)
                current_start = match.end()
        
        # Add remaining text as last sentence
        if current_start < len(text):
            remaining = text[current_start:].strip()
            if remaining:
                sentences.append(remaining)
        
        return sentences
    
    async def _process_sentence_batch(self, sentences: List[str], source_doc_id: Optional[str] = None) -> List[ProcessedSentence]:
        """Process a batch of sentences."""
        def process_batch_sync():
            processed = []
            for i, sentence in enumerate(sentences):
                try:
                    processed_sentence = self._process_single_sentence(sentence, i, source_doc_id)
                    if processed_sentence:
                        processed.append(processed_sentence)
                except Exception as e:
                    logger.warning(
                        "Failed to process sentence",
                        sentence=sentence[:100],
                        error=str(e)
                    )
            return processed
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, process_batch_sync
        )
    
    def _process_single_sentence(self, sentence: str, index: int, source_doc_id: Optional[str] = None) -> Optional[ProcessedSentence]:
        """Process a single sentence."""
        original_text = sentence
        cleaned_text = sentence.strip()
        
        if not cleaned_text:
            return None
        
        # Calculate quality score
        quality_score = 1.0
        if self.enable_quality_scoring:
            quality_score = self._calculate_quality_score(cleaned_text)
        
        # Generate sentence ID
        sentence_id = f"{source_doc_id or 'unknown'}_{index}"
        
        # Create metadata
        metadata = {
            'word_count': len(cleaned_text.split()),
            'char_count': len(cleaned_text),
            'has_punctuation': bool(re.search(r'[.!?]', cleaned_text)),
            'processing_index': index
        }
        
        return ProcessedSentence(
            text=cleaned_text,
            original_text=original_text,
            start_pos=0,  # Would need full text context to calculate
            end_pos=len(cleaned_text),
            sentence_id=sentence_id,
            quality_score=quality_score,
            metadata=metadata
        )
    
    def _calculate_quality_score(self, sentence: str) -> float:
        """Calculate quality score for a sentence."""
        score = 1.0
        
        # Length penalty for very short or very long sentences
        length = len(sentence)
        if length < 20:
            score *= 0.7
        elif length > 500:
            score *= 0.8
        
        # Bonus for having verbs and subjects
        if self._sentence_patterns['has_verb'].search(sentence):
            score *= 1.1
        else:
            score *= 0.8
        
        if self._sentence_patterns['has_subject'].search(sentence):
            score *= 1.1
        else:
            score *= 0.9
        
        # Penalty for too many special characters
        special_char_ratio = len(re.findall(r'[^\w\s]', sentence)) / len(sentence)
        if special_char_ratio > 0.3:
            score *= 0.7
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _filter_sentences(self, sentences: List[ProcessedSentence]) -> List[ProcessedSentence]:
        """Filter sentences by length and quality."""
        filtered = []
        
        for sentence in sentences:
            # Length filter
            if not (self.min_sentence_length <= len(sentence.text) <= self.max_sentence_length):
                continue
            
            # Quality filter (if enabled)
            if self.enable_quality_scoring and sentence.quality_score < 0.5:
                continue
            
            filtered.append(sentence)
        
        return filtered
    
    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("Sentence processor closed")
        except Exception as e:
            logger.warning("Error during sentence processor cleanup", error=str(e))
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
