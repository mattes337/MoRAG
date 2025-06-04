"""Enhanced topic segmentation service with semantic understanding."""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import structlog
import numpy as np

from morag.core.config import settings, get_safe_device
from morag.core.exceptions import ProcessingError, ExternalServiceError

logger = structlog.get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    from nltk.tokenize import sent_tokenize
    TOPIC_SEGMENTATION_AVAILABLE = True
except ImportError:
    TOPIC_SEGMENTATION_AVAILABLE = False
    # Create dummy classes for when dependencies are not available
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, *args, **kwargs):
            return []
    logger.warning("Topic segmentation dependencies not available")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class TopicSegment:
    """Represents a topic segment with content and metadata."""
    topic_id: str
    title: str
    summary: str
    sentences: List[str]
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    confidence: float = 0.0
    keywords: List[str] = None
    speaker_distribution: Dict[str, float] = None


@dataclass
class TopicSegmentationResult:
    """Result of topic segmentation process."""
    topics: List[TopicSegment]
    total_topics: int
    processing_time: float
    model_used: str
    similarity_threshold: float
    segmentation_method: str


class EnhancedTopicSegmentation:
    """Enhanced topic segmentation with semantic understanding."""
    
    def __init__(self):
        """Initialize the topic segmentation service."""
        self.embedding_model = None
        self.nlp = None
        self.model_loaded = False
        
        if TOPIC_SEGMENTATION_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the embedding and NLP models."""
        try:
            logger.info("Initializing topic segmentation models",
                       embedding_model=settings.topic_embedding_model)
            
            # Initialize sentence transformer with safe device
            safe_device = get_safe_device(settings.preferred_device)
            logger.info("Initializing SentenceTransformer",
                       model=settings.topic_embedding_model,
                       device=safe_device)

            try:
                self.embedding_model = SentenceTransformer(settings.topic_embedding_model, device=safe_device)
                logger.info("SentenceTransformer initialized successfully", device=safe_device)
            except Exception as e:
                if safe_device != "cpu":
                    logger.warning("SentenceTransformer GPU initialization failed, trying CPU", error=str(e))
                    self.embedding_model = SentenceTransformer(settings.topic_embedding_model, device="cpu")
                    logger.info("SentenceTransformer initialized on CPU fallback")
                else:
                    raise
            
            # Initialize spaCy if available
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy English model not found, using basic tokenization")
                    self.nlp = None
            
            # Download NLTK data if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer")
                nltk.download('punkt', quiet=True)
            
            self.model_loaded = True
            logger.info("Topic segmentation models initialized successfully")
            
        except Exception as e:
            logger.warning("Failed to initialize topic segmentation models",
                          error=str(e))
            self.model_loaded = False
    
    async def segment_topics(
        self,
        text: str,
        speaker_segments: Optional[List] = None,
        transcript_segments: Optional[List] = None,
        similarity_threshold: Optional[float] = None,
        max_topics: Optional[int] = None
    ) -> TopicSegmentationResult:
        """Perform topic segmentation on text.
        
        Args:
            text: Input text to segment
            speaker_segments: Optional speaker diarization segments
            transcript_segments: Optional transcript segments with timing
            similarity_threshold: Similarity threshold for topic boundaries
            max_topics: Maximum number of topics to detect
            
        Returns:
            TopicSegmentationResult with topic information
        """
        if not self.model_loaded:
            return await self._fallback_segmentation(text)
        
        start_time = time.time()
        
        # Use settings defaults if not provided
        similarity_threshold = similarity_threshold or settings.topic_similarity_threshold
        max_topics = max_topics or settings.max_topics
        
        try:
            logger.info("Starting topic segmentation",
                       text_length=len(text),
                       similarity_threshold=similarity_threshold,
                       max_topics=max_topics)
            
            # Preprocess text and extract sentences
            sentences = await self._extract_sentences(text)
            
            if len(sentences) < settings.min_topic_sentences:
                # Too few sentences for meaningful segmentation
                return await self._create_single_topic_result(
                    sentences, text, time.time() - start_time
                )
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(sentences)
            
            # Perform topic segmentation
            topic_boundaries = await self._detect_topic_boundaries(
                embeddings, similarity_threshold
            )
            
            # Create topic segments
            topics = await self._create_topic_segments(
                sentences,
                topic_boundaries,
                speaker_segments,
                transcript_segments,
                max_topics
            )
            
            # Skip topic summarization - user doesn't want summaries
            # if settings.use_llm_topic_summarization:
            #     topics = await self._generate_topic_summaries(topics)
            
            processing_time = time.time() - start_time
            
            result = TopicSegmentationResult(
                topics=topics,
                total_topics=len(topics),
                processing_time=processing_time,
                model_used=settings.topic_embedding_model,
                similarity_threshold=similarity_threshold,
                segmentation_method="semantic_embedding"
            )
            
            logger.info("Topic segmentation completed",
                       topics_detected=len(topics),
                       processing_time=processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Topic segmentation failed", error=str(e))
            return await self._fallback_segmentation(text)
    
    async def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using best available method."""
        try:
            if self.nlp:
                # Use spaCy for better sentence segmentation
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                # Fallback to NLTK
                sentences = sent_tokenize(text)
                sentences = [s.strip() for s in sentences if s.strip()]
            
            logger.debug("Extracted sentences", count=len(sentences))
            return sentences
            
        except Exception as e:
            logger.warning("Sentence extraction failed, using simple split", error=str(e))
            # Ultimate fallback
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            return sentences
    
    async def _generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for sentences."""
        try:
            # Run embedding generation in thread pool
            embeddings = await asyncio.to_thread(
                self.embedding_model.encode,
                sentences,
                show_progress_bar=False
            )
            
            logger.debug("Generated embeddings", shape=embeddings.shape)
            return embeddings
            
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise ProcessingError(f"Failed to generate embeddings: {str(e)}")
    
    async def _detect_topic_boundaries(
        self,
        embeddings: np.ndarray,
        similarity_threshold: float
    ) -> List[int]:
        """Detect topic boundaries using similarity analysis."""
        try:
            boundaries = [0]  # Always start with first sentence
            
            # Calculate cosine similarities between consecutive sentences
            for i in range(1, len(embeddings)):
                similarity = cosine_similarity(
                    embeddings[i-1:i],
                    embeddings[i:i+1]
                )[0][0]
                
                # If similarity drops below threshold, it's a topic boundary
                if similarity < similarity_threshold:
                    boundaries.append(i)
            
            # Always end with last sentence
            if boundaries[-1] != len(embeddings):
                boundaries.append(len(embeddings))
            
            logger.debug("Detected topic boundaries", boundaries=boundaries)
            return boundaries
            
        except Exception as e:
            logger.error("Topic boundary detection failed", error=str(e))
            # Fallback: split into equal parts
            num_parts = min(3, len(embeddings) // settings.min_topic_sentences)
            part_size = len(embeddings) // num_parts
            return [i * part_size for i in range(num_parts + 1)]
    
    async def _create_topic_segments(
        self,
        sentences: List[str],
        boundaries: List[int],
        speaker_segments: Optional[List] = None,
        transcript_segments: Optional[List] = None,
        max_topics: int = 10
    ) -> List[TopicSegment]:
        """Create topic segments from boundaries."""
        topics = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            topic_sentences = sentences[start_idx:end_idx]
            
            # Skip if too few sentences
            if len(topic_sentences) < settings.min_topic_sentences:
                continue
            
            # Calculate timing if transcript segments available
            start_time, end_time, duration = self._calculate_topic_timing(
                start_idx, end_idx, transcript_segments, topic_sentences
            )

            # Calculate speaker distribution if available
            speaker_dist = self._calculate_speaker_distribution(
                start_time, end_time, speaker_segments
            )
            
            # Extract keywords
            keywords = await self._extract_keywords(topic_sentences)
            
            topic = TopicSegment(
                topic_id=f"topic_{i+1}",
                title=f"Topic {i+1}",
                summary="",  # Will be filled by LLM if enabled
                sentences=topic_sentences,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                confidence=0.8,  # Default confidence
                keywords=keywords,
                speaker_distribution=speaker_dist
            )
            
            topics.append(topic)
            
            # Limit number of topics
            if len(topics) >= max_topics:
                break
        
        return topics
    
    def _calculate_topic_timing(
        self,
        start_idx: int,
        end_idx: int,
        transcript_segments: Optional[List],
        topic_sentences: List[str]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate timing for topic segment."""
        if not transcript_segments or not topic_sentences:
            logger.debug("No transcript segments or topic sentences for timing calculation")
            return None, None, None

        try:
            logger.debug("Calculating topic timing",
                        start_idx=start_idx,
                        end_idx=end_idx,
                        topic_sentences_count=len(topic_sentences),
                        transcript_segments_count=len(transcript_segments))

            # Try to match topic sentences with transcript segments for better timing
            start_time = None
            end_time = None
            matches_found = 0

            # Find the earliest and latest matching transcript segments
            for i, sentence in enumerate(topic_sentences):
                sentence_clean = sentence.strip().lower()
                logger.debug(f"Looking for timing for sentence {i}", sentence=sentence_clean[:50])

                for j, segment in enumerate(transcript_segments):
                    if hasattr(segment, 'text') and hasattr(segment, 'start_time') and hasattr(segment, 'end_time'):
                        segment_text = segment.text.strip().lower()

                        # Check for text similarity with more lenient matching
                        if (sentence_clean in segment_text or
                            segment_text in sentence_clean or
                            self._simple_text_match(sentence_clean, segment_text)):

                            logger.debug(f"Found timing match for sentence {i}",
                                       segment_start=segment.start_time,
                                       segment_end=segment.end_time,
                                       segment_text=segment_text[:30])

                            if start_time is None or segment.start_time < start_time:
                                start_time = segment.start_time
                            if end_time is None or segment.end_time > end_time:
                                end_time = segment.end_time
                            matches_found += 1
                            break

            logger.debug("Timing matching results",
                        matches_found=matches_found,
                        start_time=start_time,
                        end_time=end_time)

            # If we found matches, return the timing
            if start_time is not None and end_time is not None:
                duration = end_time - start_time
                logger.debug("Using matched timing",
                           start_time=start_time,
                           end_time=end_time,
                           duration=duration)
                return start_time, end_time, duration

            # Fallback to proportional mapping based on segment indices
            logger.debug("No timing matches found, using proportional mapping")

            # Use the actual number of segments for proportional calculation
            total_segments = len(transcript_segments)
            if total_segments > 0:
                # Map topic sentence indices to transcript segment indices
                segment_start_idx = min(start_idx, total_segments - 1)
                segment_end_idx = min(end_idx, total_segments)

                # Get timing from the corresponding transcript segments
                if segment_start_idx < total_segments:
                    start_segment = transcript_segments[segment_start_idx]
                    if hasattr(start_segment, 'start_time'):
                        start_time = start_segment.start_time

                if segment_end_idx > 0 and segment_end_idx <= total_segments:
                    end_segment = transcript_segments[segment_end_idx - 1]
                    if hasattr(end_segment, 'end_time'):
                        end_time = end_segment.end_time

                # If we still don't have timing, use proportional calculation
                if start_time is None or end_time is None:
                    total_duration = max(seg.end_time for seg in transcript_segments if hasattr(seg, 'end_time'))
                    segment_ratio_start = segment_start_idx / total_segments
                    segment_ratio_end = segment_end_idx / total_segments

                    start_time = segment_ratio_start * total_duration
                    end_time = segment_ratio_end * total_duration

                if start_time is not None and end_time is not None:
                    duration = end_time - start_time
                    logger.debug("Using proportional timing",
                               start_time=start_time,
                               end_time=end_time,
                               duration=duration)
                    return start_time, end_time, duration

        except Exception as e:
            logger.warning("Failed to calculate topic timing", error=str(e), exc_info=True)

        logger.warning("Could not calculate topic timing, returning None")
        return None, None, None

    def _simple_text_match(self, text1: str, text2: str) -> bool:
        """Simple text matching for timing calculation."""
        try:
            # First check for direct substring matches (more lenient)
            if len(text1) > 10 and len(text2) > 10:
                # Check if either text contains a significant portion of the other
                if text1 in text2 or text2 in text1:
                    return True

                # Check for partial matches (at least 50% of shorter text)
                shorter_text = text1 if len(text1) < len(text2) else text2
                longer_text = text2 if len(text1) < len(text2) else text1

                # Split shorter text into words and check how many are in longer text
                shorter_words = shorter_text.split()
                if len(shorter_words) >= 3:
                    matches = sum(1 for word in shorter_words if len(word) > 3 and word in longer_text)
                    if matches / len(shorter_words) >= 0.5:
                        return True

            # Check if they share significant words
            words1 = set(text1.split())
            words2 = set(text2.split())

            if len(words1) < 2 or len(words2) < 2:
                return False

            # Remove very common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
            words1 = words1 - common_words
            words2 = words2 - common_words

            if not words1 or not words2:
                return False

            # Check if they share at least 25% of words (reduced threshold for better matching)
            intersection = words1.intersection(words2)
            match_ratio = len(intersection) / min(len(words1), len(words2))

            logger.debug("Text matching analysis",
                        text1_words=len(words1),
                        text2_words=len(words2),
                        intersection_words=len(intersection),
                        match_ratio=match_ratio)

            return match_ratio >= 0.25

        except Exception as e:
            logger.debug("Text matching failed", error=str(e))
            return False
    
    def _calculate_speaker_distribution(
        self,
        start_time: Optional[float],
        end_time: Optional[float],
        speaker_segments: Optional[List]
    ) -> Optional[Dict[str, float]]:
        """Calculate speaker distribution for topic segment."""
        if not start_time or not end_time or not speaker_segments:
            return None
        
        try:
            speaker_time = {}
            total_time = end_time - start_time
            
            for segment in speaker_segments:
                # Check overlap with topic segment
                overlap_start = max(start_time, segment.start_time)
                overlap_end = min(end_time, segment.end_time)
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    speaker_id = segment.speaker_id
                    
                    if speaker_id not in speaker_time:
                        speaker_time[speaker_id] = 0.0
                    
                    speaker_time[speaker_id] += overlap_duration
            
            # Convert to percentages
            if speaker_time and total_time > 0:
                return {
                    speaker: (time / total_time) * 100
                    for speaker, time in speaker_time.items()
                }
        
        except Exception as e:
            logger.warning("Failed to calculate speaker distribution", error=str(e))
        
        return None
    
    async def _extract_keywords(self, sentences: List[str]) -> List[str]:
        """Extract keywords from topic sentences."""
        try:
            if self.nlp:
                # Use spaCy for keyword extraction
                text = " ".join(sentences)
                doc = self.nlp(text)
                
                # Extract important tokens (nouns, proper nouns, adjectives)
                keywords = []
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and
                        not token.is_stop and
                        not token.is_punct and
                        len(token.text) > 2):
                        keywords.append(token.lemma_.lower())
                
                # Remove duplicates and limit
                keywords = list(dict.fromkeys(keywords))[:10]
                return keywords
            else:
                # Simple keyword extraction
                text = " ".join(sentences).lower()
                words = text.split()
                # Filter common words and short words
                keywords = [w for w in words if len(w) > 3 and w.isalpha()]
                return list(dict.fromkeys(keywords))[:10]
        
        except Exception as e:
            logger.warning("Keyword extraction failed", error=str(e))
            return []
    
    async def _generate_topic_summaries(self, topics: List[TopicSegment]) -> List[TopicSegment]:
        """Generate summaries for topics using LLM."""
        try:
            # Import here to avoid circular imports
            from morag.services.summarization import enhanced_summarization_service
            
            for topic in topics:
                if len(topic.sentences) > 0:
                    text = " ".join(topic.sentences)
                    
                    # Generate summary
                    summary_result = await enhanced_summarization_service.summarize_text(
                        text,
                        max_length=100,
                        summary_type="extractive"
                    )
                    
                    topic.summary = summary_result.summary
                    
                    # Generate title from summary
                    if topic.summary:
                        title_words = topic.summary.split()[:5]
                        topic.title = " ".join(title_words) + "..."
            
            return topics
            
        except Exception as e:
            logger.warning("Topic summary generation failed", error=str(e))
            return topics
    
    async def _create_single_topic_result(
        self,
        sentences: List[str],
        text: str,
        processing_time: float
    ) -> TopicSegmentationResult:
        """Create result with single topic when segmentation is not meaningful."""
        topic = TopicSegment(
            topic_id="topic_1",
            title="Main Content",
            summary=text[:100] + "..." if len(text) > 100 else text,
            sentences=sentences,
            confidence=1.0,
            keywords=[]
        )
        
        return TopicSegmentationResult(
            topics=[topic],
            total_topics=1,
            processing_time=processing_time,
            model_used="single_topic",
            similarity_threshold=0.0,
            segmentation_method="single_topic"
        )
    
    async def _fallback_segmentation(self, text: str) -> TopicSegmentationResult:
        """Fallback segmentation when models are not available."""
        logger.info("Using fallback topic segmentation")
        
        try:
            # Simple sentence-based segmentation
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            if len(sentences) <= settings.min_topic_sentences:
                return await self._create_single_topic_result(sentences, text, 0.1)
            
            # Split into 2-3 topics based on length
            num_topics = min(3, len(sentences) // settings.min_topic_sentences)
            sentences_per_topic = len(sentences) // num_topics
            
            topics = []
            for i in range(num_topics):
                start_idx = i * sentences_per_topic
                end_idx = start_idx + sentences_per_topic if i < num_topics - 1 else len(sentences)
                
                topic_sentences = sentences[start_idx:end_idx]
                
                topic = TopicSegment(
                    topic_id=f"topic_{i+1}",
                    title=f"Topic {i+1}",
                    summary=" ".join(topic_sentences[:2]) + "...",
                    sentences=topic_sentences,
                    confidence=0.6,
                    keywords=[]
                )
                topics.append(topic)
            
            return TopicSegmentationResult(
                topics=topics,
                total_topics=len(topics),
                processing_time=0.1,
                model_used="fallback",
                similarity_threshold=0.5,
                segmentation_method="simple_split"
            )
            
        except Exception as e:
            logger.error("Fallback segmentation failed", error=str(e))
            return await self._create_single_topic_result([text], text, 0.1)


# Global instance
topic_segmentation_service = EnhancedTopicSegmentation()
