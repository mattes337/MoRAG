"""Enhanced topic segmentation service with advanced features."""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
import re
import structlog
import numpy as np
from collections import Counter

from morag_core.exceptions import ProcessingError, ExternalServiceError
from morag_core.utils import get_safe_device

logger = structlog.get_logger(__name__)

# Try to import optional dependencies
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Fallback to small model if main model not available
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            logger.warning("spaCy models not available, using basic NLP processing")
            nlp = None
            SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    logger.warning("spaCy not available, using basic NLP processing")

try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        NLTK_AVAILABLE = True
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            NLTK_AVAILABLE = True
        except Exception:
            NLTK_AVAILABLE = False
            logger.warning("NLTK punkt tokenizer not available, using basic sentence splitting")
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using basic sentence splitting")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using basic topic segmentation")

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using basic topic segmentation")


@dataclass
class TopicSegment:
    """Represents a topic segment with timing and metadata."""
    topic_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    sentences: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    speaker_distribution: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    
    def __post_init__(self):
        # Calculate duration if not provided
        if self.duration == 0.0 and self.start_time is not None and self.end_time is not None:
            self.duration = self.end_time - self.start_time


@dataclass
class TopicSegmentationResult:
    """Result of topic segmentation process."""
    segments: List[TopicSegment]
    total_topics: int
    total_duration: float
    processing_time: float
    model_used: str
    similarity_threshold: float
    min_segment_length: int
    max_segments: int


class TopicSegmentationService:
    """Enhanced topic segmentation with advanced features."""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.5,
                 min_segment_length: int = 3,
                 max_segments: int = 10,
                 device: str = "auto",
                 summarization_service: Optional[Any] = None):
        """Initialize the topic segmentation service.
        
        Args:
            embedding_model: Name of the sentence-transformers model to use
            similarity_threshold: Threshold for topic boundary detection
            min_segment_length: Minimum number of sentences per segment
            max_segments: Maximum number of segments to create
            device: Device to use for inference (auto, cpu, cuda)
            summarization_service: Optional service for generating summaries
        """
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_segment_length = min_segment_length
        self.max_segments = max_segments
        self.preferred_device = device
        self.summarization_service = summarization_service
        
        self.embedding_model = None
        self.model_loaded = False
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model with safe device configuration."""
        try:
            safe_device = get_safe_device(self.preferred_device)
            logger.info("Initializing sentence embedding model",
                       model=self.embedding_model_name,
                       device=safe_device)

            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device=safe_device)
                self.model_loaded = True
                logger.info("Sentence embedding model initialized successfully", device=safe_device)
            except Exception as init_error:
                if safe_device != "cpu":
                    logger.warning("GPU model initialization failed, trying CPU", error=str(init_error))
                    # Force CPU initialization
                    self.embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")
                    self.model_loaded = True
                    logger.info("Sentence embedding model initialized on CPU fallback")
                else:
                    raise

        except Exception as e:
            logger.warning("Failed to initialize sentence embedding model",
                          error=str(e))
            self.embedding_model = None
            self.model_loaded = False
    
    async def segment_transcript(self,
                               transcript: str,
                               transcript_segments: Optional[List[Dict[str, Any]]] = None,
                               similarity_threshold: Optional[float] = None,
                               min_segment_length: Optional[int] = None,
                               max_segments: Optional[int] = None) -> TopicSegmentationResult:
        """Segment transcript into topics.
        
        Args:
            transcript: Full transcript text
            transcript_segments: Optional list of transcript segments with timing and speaker info
            similarity_threshold: Threshold for topic boundary detection
            min_segment_length: Minimum number of sentences per segment
            max_segments: Maximum number of segments to create
            
        Returns:
            TopicSegmentationResult with topic segments
        """
        start_time = time.time()
        
        # Use instance defaults if not provided
        similarity_threshold = similarity_threshold or self.similarity_threshold
        min_segment_length = min_segment_length or self.min_segment_length
        max_segments = max_segments or self.max_segments
        
        try:
            logger.info("Starting topic segmentation",
                       transcript_length=len(transcript),
                       segments_count=len(transcript_segments) if transcript_segments else 0)
            
            # Extract sentences from transcript
            sentences = await self._extract_sentences(transcript)
            
            # If transcript is too short, return single topic
            if len(sentences) < min_segment_length * 2:
                logger.info("Transcript too short for segmentation, creating single topic")
                return await self._create_single_topic_result(
                    transcript, sentences, transcript_segments, time.time() - start_time
                )
            
            # Generate embeddings for sentences
            if self.model_loaded and self.embedding_model:
                embeddings = await self._generate_embeddings(sentences)
                
                # Detect topic boundaries
                boundaries = await self._detect_topic_boundaries(
                    embeddings, 
                    similarity_threshold,
                    min_segment_length,
                    max_segments
                )
                
                # Create topic segments
                segments = await self._create_topic_segments(
                    sentences, 
                    boundaries, 
                    transcript_segments
                )
                
                # Generate summaries if summarization service is available
                if self.summarization_service and len(segments) > 0:
                    segments = await self._generate_topic_summaries(segments)
            else:
                # Fallback to simple segmentation
                logger.info("Using fallback topic segmentation")
                segments = await self._fallback_segmentation(
                    sentences, 
                    transcript_segments,
                    max_segments
                )
            
            # Calculate total duration
            total_duration = 0.0
            if segments and segments[-1].end_time > 0:
                total_duration = segments[-1].end_time
            
            result = TopicSegmentationResult(
                segments=segments,
                total_topics=len(segments),
                total_duration=total_duration,
                processing_time=time.time() - start_time,
                model_used=self.embedding_model_name if self.model_loaded else "fallback",
                similarity_threshold=similarity_threshold,
                min_segment_length=min_segment_length,
                max_segments=max_segments
            )
            
            logger.info("Topic segmentation completed",
                       topics_detected=result.total_topics,
                       processing_time=result.processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Topic segmentation failed",
                        error=str(e))
            # Fallback to simple segmentation
            return await self._create_single_topic_result(
                transcript, 
                sentences if 'sentences' in locals() else [], 
                transcript_segments, 
                time.time() - start_time
            )
    
    async def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using available NLP libraries."""
        if not text or text.strip() == "":
            return []
        
        # Try spaCy first
        if SPACY_AVAILABLE and nlp:
            try:
                # Process in chunks to avoid memory issues with long texts
                max_length = 100000  # spaCy default
                if len(text) > max_length:
                    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                    all_sentences = []
                    for chunk in chunks:
                        doc = nlp(chunk)
                        all_sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])
                    return all_sentences
                else:
                    doc = nlp(text)
                    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception as e:
                logger.warning("spaCy sentence extraction failed", error=str(e))
        
        # Try NLTK if spaCy fails
        if NLTK_AVAILABLE:
            try:
                from nltk.tokenize import sent_tokenize
                return [sent.strip() for sent in sent_tokenize(text) if sent.strip()]
            except Exception as e:
                logger.warning("NLTK sentence extraction failed", error=str(e))
        
        # Fallback to simple splitting
        logger.info("Using simple sentence splitting")
        # Split on common sentence terminators followed by space and capital letter
        simple_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [sent.strip() for sent in simple_sentences if sent.strip()]
    
    async def _generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for sentences using sentence transformer."""
        if not sentences:
            return np.array([])
        
        try:
            # Run embedding generation in thread pool to avoid blocking
            embeddings = await asyncio.to_thread(
                self.embedding_model.encode,
                sentences,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e))
            # Return empty array on error
            return np.array([])
    
    async def _detect_topic_boundaries(self,
                                     embeddings: np.ndarray,
                                     similarity_threshold: float,
                                     min_segment_length: int,
                                     max_segments: int) -> List[int]:
        """Detect topic boundaries based on embedding similarity."""
        if len(embeddings) == 0:
            return []
        
        try:
            # Calculate cosine similarity between consecutive sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                similarities.append(sim)
            
            # Find potential boundaries where similarity drops below threshold
            potential_boundaries = []
            for i, sim in enumerate(similarities):
                if sim < similarity_threshold:
                    potential_boundaries.append(i + 1)  # +1 because boundary is after sentence i
            
            # Apply constraints (min segment length, max segments)
            boundaries = [0]  # Start with beginning of text
            last_boundary = 0
            
            for boundary in sorted(potential_boundaries):
                # Check if adding this boundary would create a segment that's too small
                if boundary - last_boundary < min_segment_length:
                    continue
                
                # Add boundary
                boundaries.append(boundary)
                last_boundary = boundary
                
                # Stop if we've reached max segments
                if len(boundaries) >= max_segments:
                    break
            
            # Add end boundary if not already included
            if boundaries[-1] != len(embeddings):
                boundaries.append(len(embeddings))
            
            return boundaries
            
        except Exception as e:
            logger.error("Failed to detect topic boundaries", error=str(e))
            # Fallback to simple equal division
            num_segments = min(max_segments, len(embeddings) // min_segment_length)
            if num_segments <= 1:
                return [0, len(embeddings)]
            
            segment_size = len(embeddings) // num_segments
            boundaries = [0]
            for i in range(1, num_segments):
                boundaries.append(i * segment_size)
            boundaries.append(len(embeddings))
            
            return boundaries
    
    async def _create_topic_segments(self,
                                   sentences: List[str],
                                   boundaries: List[int],
                                   transcript_segments: Optional[List[Dict[str, Any]]] = None) -> List[TopicSegment]:
        """Create topic segments from boundaries."""
        if not sentences or not boundaries or len(boundaries) < 2:
            return []
        
        topic_segments = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            
            # Get sentences for this segment
            segment_sentences = sentences[start_idx:end_idx]
            if not segment_sentences:
                continue
            
            # Create topic segment
            topic_segment = TopicSegment(
                topic_id=f"TOPIC_{i:02d}",
                sentences=segment_sentences,
                confidence=0.8  # Default confidence
            )
            
            # Extract keywords
            topic_segment.keywords = await self._extract_keywords(segment_sentences)
            
            # Calculate timing if transcript segments are available
            if transcript_segments:
                await self._calculate_topic_timing(topic_segment, transcript_segments)
                
                # Calculate speaker distribution if available
                if any('speaker' in segment for segment in transcript_segments):
                    await self._calculate_speaker_distribution(topic_segment, transcript_segments)
            
            topic_segments.append(topic_segment)
        
        return topic_segments
    
    async def _calculate_topic_timing(self,
                                    topic_segment: TopicSegment,
                                    transcript_segments: List[Dict[str, Any]]):
        """Calculate timing for topic segment based on transcript segments."""
        # Join all sentences in the topic
        topic_text = " ".join(topic_segment.sentences)
        
        # Try to find matching transcript segments
        matching_segments = []
        for segment in transcript_segments:
            if self._simple_text_match(segment.get('text', ''), topic_text):
                matching_segments.append(segment)
        
        if matching_segments:
            # Calculate timing based on matching segments
            topic_segment.start_time = min(seg.get('start', 0.0) for seg in matching_segments)
            topic_segment.end_time = max(seg.get('end', 0.0) for seg in matching_segments)
            topic_segment.duration = topic_segment.end_time - topic_segment.start_time
        else:
            # Fallback: Map proportionally to transcript length
            total_transcript_length = sum(len(segment.get('text', '')) for segment in transcript_segments)
            topic_length = len(topic_text)
            
            if total_transcript_length > 0:
                topic_ratio = topic_length / total_transcript_length
                total_duration = transcript_segments[-1].get('end', 0.0) - transcript_segments[0].get('start', 0.0)
                
                # Estimate position in transcript
                topic_sentences_joined = " ".join(topic_segment.sentences)
                full_transcript = " ".join(segment.get('text', '') for segment in transcript_segments)
                
                try:
                    # Find approximate position
                    first_sentence = topic_segment.sentences[0]
                    position = full_transcript.find(first_sentence)
                    
                    if position >= 0:
                        position_ratio = position / len(full_transcript)
                        topic_segment.start_time = transcript_segments[0].get('start', 0.0) + (position_ratio * total_duration)
                        topic_segment.end_time = topic_segment.start_time + (topic_ratio * total_duration)
                    else:
                        # Couldn't find exact match, use proportional mapping
                        raise ValueError("Exact match not found")
                except Exception:
                    # Simple proportional mapping as ultimate fallback
                    segments_count = len(transcript_segments)
                    segments_per_topic = segments_count // (len(topic_segment.sentences) or 1)
                    
                    if segments_per_topic > 0:
                        start_segment_idx = topic_segment.topic_id.split('_')[1]
                        start_segment_idx = int(start_segment_idx) * segments_per_topic
                        end_segment_idx = min(start_segment_idx + segments_per_topic, segments_count - 1)
                        
                        topic_segment.start_time = transcript_segments[start_segment_idx].get('start', 0.0)
                        topic_segment.end_time = transcript_segments[end_segment_idx].get('end', 0.0)
                    else:
                        # Ultimate fallback
                        topic_segment.start_time = 0.0
                        topic_segment.end_time = total_duration
                
                topic_segment.duration = topic_segment.end_time - topic_segment.start_time
    
    def _simple_text_match(self, text1: str, text2: str) -> bool:
        """Simple text matching to find if text1 is part of text2 or vice versa."""
        text1 = text1.lower()
        text2 = text2.lower()
        
        return text1 in text2 or text2 in text1
    
    async def _calculate_speaker_distribution(self,
                                            topic_segment: TopicSegment,
                                            transcript_segments: List[Dict[str, Any]]):
        """Calculate speaker distribution within topic segment."""
        speaker_times = {}
        
        # Find segments that overlap with topic timing
        for segment in transcript_segments:
            if 'speaker' not in segment:
                continue
                
            segment_start = segment.get('start', 0.0)
            segment_end = segment.get('end', 0.0)
            
            # Check for overlap
            if segment_end < topic_segment.start_time or segment_start > topic_segment.end_time:
                continue
                
            # Calculate overlap duration
            overlap_start = max(segment_start, topic_segment.start_time)
            overlap_end = min(segment_end, topic_segment.end_time)
            overlap_duration = overlap_end - overlap_start
            
            if overlap_duration <= 0:
                continue
                
            speaker = segment.get('speaker', 'UNKNOWN')
            if speaker not in speaker_times:
                speaker_times[speaker] = 0.0
                
            speaker_times[speaker] += overlap_duration
        
        # Calculate percentages
        total_time = sum(speaker_times.values())
        if total_time > 0:
            topic_segment.speaker_distribution = {
                speaker: time / total_time for speaker, time in speaker_times.items()
            }
    
    async def _extract_keywords(self, sentences: List[str], max_keywords: int = 10) -> List[str]:
        """Extract keywords from sentences."""
        if not sentences:
            return []
        
        # Join sentences
        text = " ".join(sentences)
        
        # Try spaCy for keyword extraction
        if SPACY_AVAILABLE and nlp:
            try:
                doc = nlp(text)
                # Extract nouns and proper nouns as keywords
                keywords = []
                for token in doc:
                    if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
                        keywords.append(token.text.lower())
                
                # Count frequencies and get top keywords
                keyword_counts = Counter(keywords)
                return [kw for kw, _ in keyword_counts.most_common(max_keywords)]
            except Exception as e:
                logger.warning("spaCy keyword extraction failed", error=str(e))
        
        # Simple fallback
        words = text.lower().split()
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                     "be", "been", "being", "in", "on", "at", "to", "for", "with", 
                     "by", "about", "against", "between", "into", "through", "during", 
                     "before", "after", "above", "below", "from", "up", "down", "of", 
                     "off", "over", "under", "again", "further", "then", "once", "here", 
                     "there", "when", "where", "why", "how", "all", "any", "both", "each", 
                     "few", "more", "most", "other", "some", "such", "no", "nor", "not", 
                     "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
                     "will", "just", "don", "should", "now"}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        word_counts = Counter(filtered_words)
        
        return [word for word, _ in word_counts.most_common(max_keywords)]
    
    async def _generate_topic_summaries(self, segments: List[TopicSegment]) -> List[TopicSegment]:
        """Generate summaries for topic segments."""
        if not self.summarization_service or not segments:
            return segments
        
        try:
            for segment in segments:
                # Join sentences for this segment
                segment_text = " ".join(segment.sentences)
                
                # Generate summary and title
                try:
                    summary_result = await self.summarization_service.summarize(
                        segment_text,
                        max_length=100,
                        generate_title=True
                    )
                    
                    if summary_result:
                        segment.summary = summary_result.get('summary')
                        segment.title = summary_result.get('title')
                except Exception as e:
                    logger.warning("Failed to generate summary for topic",
                                 topic_id=segment.topic_id,
                                 error=str(e))
        except Exception as e:
            logger.error("Topic summary generation failed", error=str(e))
        
        return segments
    
    async def _create_single_topic_result(self,
                                        transcript: str,
                                        sentences: List[str],
                                        transcript_segments: Optional[List[Dict[str, Any]]],
                                        processing_time: float) -> TopicSegmentationResult:
        """Create a single topic result when segmentation is not meaningful."""
        # Extract sentences if not provided
        if not sentences:
            sentences = await self._extract_sentences(transcript)
        
        # Create single topic segment
        topic_segment = TopicSegment(
            topic_id="TOPIC_00",
            sentences=sentences,
            confidence=1.0
        )
        
        # Extract keywords
        topic_segment.keywords = await self._extract_keywords(sentences)
        
        # Calculate timing if transcript segments are available
        if transcript_segments:
            topic_segment.start_time = transcript_segments[0].get('start', 0.0)
            topic_segment.end_time = transcript_segments[-1].get('end', 0.0)
            topic_segment.duration = topic_segment.end_time - topic_segment.start_time
            
            # Calculate speaker distribution if available
            if any('speaker' in segment for segment in transcript_segments):
                await self._calculate_speaker_distribution(topic_segment, transcript_segments)
        
        # Generate summary if summarization service is available
        if self.summarization_service:
            segments_with_summary = await self._generate_topic_summaries([topic_segment])
            topic_segment = segments_with_summary[0]
        
        return TopicSegmentationResult(
            segments=[topic_segment],
            total_topics=1,
            total_duration=topic_segment.duration,
            processing_time=processing_time,
            model_used="single_topic",
            similarity_threshold=self.similarity_threshold,
            min_segment_length=self.min_segment_length,
            max_segments=1
        )
    
    async def _fallback_segmentation(self,
                                   sentences: List[str],
                                   transcript_segments: Optional[List[Dict[str, Any]]],
                                   max_segments: int) -> List[TopicSegment]:
        """Provide fallback segmentation when models are unavailable."""
        if not sentences:
            return []
        
        try:
            # Determine number of segments based on text length
            num_segments = min(max_segments, max(2, len(sentences) // 10))
            if num_segments > 3:
                num_segments = 3  # Limit to 2-3 segments for fallback
            
            segment_size = len(sentences) // num_segments
            topic_segments = []
            
            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(sentences)
                
                segment_sentences = sentences[start_idx:end_idx]
                if not segment_sentences:
                    continue
                
                topic_segment = TopicSegment(
                    topic_id=f"TOPIC_{i:02d}",
                    sentences=segment_sentences,
                    confidence=0.6  # Lower confidence for fallback
                )
                
                # Extract keywords
                topic_segment.keywords = await self._extract_keywords(segment_sentences)
                
                # Calculate timing if transcript segments are available
                if transcript_segments:
                    await self._calculate_topic_timing(topic_segment, transcript_segments)
                    
                    # Calculate speaker distribution if available
                    if any('speaker' in segment for segment in transcript_segments):
                        await self._calculate_speaker_distribution(topic_segment, transcript_segments)
                
                topic_segments.append(topic_segment)
            
            return topic_segments
            
        except Exception as e:
            logger.error("Fallback segmentation failed", error=str(e))
            # Create a single segment as ultimate fallback
            return [TopicSegment(
                topic_id="TOPIC_00",
                sentences=sentences,
                confidence=0.5
            )]