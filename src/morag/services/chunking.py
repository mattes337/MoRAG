from typing import List, Dict, Any, Optional, Tuple
import structlog
from dataclasses import dataclass
import re
from collections import defaultdict

from morag.core.config import settings
from morag.core.exceptions import ProcessingError

logger = structlog.get_logger()

# Import spaCy only when available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available - using basic chunking only")

@dataclass
class ChunkInfo:
    """Information about a text chunk."""
    text: str
    start_char: int
    end_char: int
    sentence_count: int
    word_count: int
    entities: List[Dict[str, Any]]
    topics: List[str]
    chunk_type: str = "semantic"

class SemanticChunker:
    """Advanced semantic chunking using spaCy."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = None

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                logger.info("spaCy model loaded", model=model_name)
            except OSError:
                logger.warning(f"spaCy model {model_name} not found, falling back to basic chunking")
                self.nlp = None

        self.max_chunk_size = settings.max_chunk_size
        self.min_chunk_size = 100
        self.overlap_size = 50

    async def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        strategy: str = "semantic"
    ) -> List[ChunkInfo]:
        """Chunk text using specified strategy."""

        chunk_size = chunk_size or self.max_chunk_size

        if not text.strip():
            return []

        if self.nlp is None or strategy == "simple":
            return await self._simple_chunking(text, chunk_size)

        if strategy == "semantic":
            return await self._semantic_chunking(text, chunk_size)
        elif strategy == "sentence":
            return await self._sentence_chunking(text, chunk_size)
        elif strategy == "paragraph":
            return await self._paragraph_chunking(text, chunk_size)
        elif strategy == "page":
            return await self._page_chunking(text, chunk_size)
        else:
            logger.warning(f"Unknown chunking strategy: {strategy}, using semantic")
            return await self._semantic_chunking(text, chunk_size)

    async def _semantic_chunking(self, text: str, chunk_size: int) -> List[ChunkInfo]:
        """Advanced semantic chunking using NLP analysis."""

        try:
            # Process text with spaCy
            doc = self.nlp(text)

            # Group sentences by semantic similarity and topics
            sentence_groups = self._group_sentences_semantically(doc)

            # Create chunks from sentence groups
            chunks = []
            current_chunk_text = ""
            current_start = 0
            current_sentences = []

            for group in sentence_groups:
                group_text = " ".join([sent.text for sent in group])

                # Check if adding this group would exceed chunk size
                if (len(current_chunk_text) + len(group_text) > chunk_size and
                    len(current_chunk_text) > self.min_chunk_size):

                    # Finalize current chunk
                    if current_chunk_text:
                        chunk_info = self._create_chunk_info(
                            current_chunk_text,
                            current_start,
                            current_start + len(current_chunk_text),
                            current_sentences
                        )
                        chunks.append(chunk_info)

                    # Start new chunk
                    current_chunk_text = group_text
                    current_start = group[0].start_char
                    current_sentences = group
                else:
                    # Add to current chunk
                    if current_chunk_text:
                        current_chunk_text += " " + group_text
                    else:
                        current_chunk_text = group_text
                        current_start = group[0].start_char
                    current_sentences.extend(group)

            # Add final chunk
            if current_chunk_text:
                chunk_info = self._create_chunk_info(
                    current_chunk_text,
                    current_start,
                    current_start + len(current_chunk_text),
                    current_sentences
                )
                chunks.append(chunk_info)

            # Add overlap between chunks if needed
            chunks = self._add_overlap(chunks, text)

            logger.info(
                "Semantic chunking completed",
                original_length=len(text),
                chunks_created=len(chunks),
                avg_chunk_size=sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
            )

            return chunks

        except Exception as e:
            logger.error("Semantic chunking failed", error=str(e))
            # Fallback to simple chunking
            return await self._simple_chunking(text, chunk_size)

    def _group_sentences_semantically(self, doc) -> List[List]:
        """Group sentences by semantic similarity and topic coherence."""

        sentences = list(doc.sents)
        if not sentences:
            return []

        groups = []
        current_group = [sentences[0]]

        for i in range(1, len(sentences)):
            current_sent = sentences[i]
            prev_sent = sentences[i-1]

            # Calculate semantic similarity
            similarity = self._calculate_sentence_similarity(current_sent, prev_sent)

            # Check for topic continuity
            topic_continuity = self._check_topic_continuity(current_sent, prev_sent)

            # Decide whether to continue current group or start new one
            if similarity > 0.3 or topic_continuity:
                current_group.append(current_sent)
            else:
                groups.append(current_group)
                current_group = [current_sent]

        # Add final group
        if current_group:
            groups.append(current_group)

        return groups

    def _calculate_sentence_similarity(self, sent1, sent2) -> float:
        """Calculate semantic similarity between sentences."""

        try:
            # Use spaCy's built-in similarity if available
            if sent1.vector.any() and sent2.vector.any():
                return sent1.similarity(sent2)

            # Fallback: lexical overlap
            words1 = set(token.lemma_.lower() for token in sent1 if not token.is_stop and token.is_alpha)
            words2 = set(token.lemma_.lower() for token in sent2 if not token.is_stop and token.is_alpha)

            if not words1 or not words2:
                return 0.0

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            return len(intersection) / len(union) if union else 0.0

        except Exception:
            return 0.0

    def _check_topic_continuity(self, sent1, sent2) -> bool:
        """Check if sentences share similar topics or entities."""

        # Check for shared named entities
        entities1 = set(ent.text.lower() for ent in sent1.ents)
        entities2 = set(ent.text.lower() for ent in sent2.ents)

        if entities1 and entities2:
            shared_entities = entities1.intersection(entities2)
            if len(shared_entities) > 0:
                return True

        # Check for shared noun phrases
        noun_phrases1 = set(chunk.text.lower() for chunk in sent1.noun_chunks)
        noun_phrases2 = set(chunk.text.lower() for chunk in sent2.noun_chunks)

        if noun_phrases1 and noun_phrases2:
            shared_phrases = noun_phrases1.intersection(noun_phrases2)
            if len(shared_phrases) > 0:
                return True

        return False

    def _create_chunk_info(
        self,
        text: str,
        start_char: int,
        end_char: int,
        sentences: List
    ) -> ChunkInfo:
        """Create ChunkInfo object with extracted metadata."""

        # Extract entities
        entities = []
        topics = []

        if self.nlp:
            doc = self.nlp(text)

            # Extract named entities
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

            # Extract key topics (noun phrases)
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) > 1:  # Multi-word phrases
                    topics.append(chunk.text.lower())

        return ChunkInfo(
            text=text.strip(),
            start_char=start_char,
            end_char=end_char,
            sentence_count=len(sentences),
            word_count=len(text.split()),
            entities=entities,
            topics=list(set(topics))[:5],  # Top 5 unique topics
            chunk_type="semantic"
        )

    def _add_overlap(self, chunks: List[ChunkInfo], original_text: str) -> List[ChunkInfo]:
        """Add overlap between consecutive chunks for better context."""

        if len(chunks) <= 1 or self.overlap_size <= 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no overlap needed
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i-1]

                # Get last N characters from previous chunk
                overlap_text = prev_chunk.text[-self.overlap_size:].strip()

                # Find a good break point (sentence boundary)
                sentences = overlap_text.split('. ')
                if len(sentences) > 1:
                    overlap_text = '. '.join(sentences[1:])

                # Create new chunk with overlap
                new_text = overlap_text + " " + chunk.text if overlap_text else chunk.text

                new_chunk = ChunkInfo(
                    text=new_text,
                    start_char=chunk.start_char - len(overlap_text),
                    end_char=chunk.end_char,
                    sentence_count=chunk.sentence_count,
                    word_count=len(new_text.split()),
                    entities=chunk.entities,
                    topics=chunk.topics,
                    chunk_type=chunk.chunk_type
                )

                overlapped_chunks.append(new_chunk)

        return overlapped_chunks

    async def _sentence_chunking(self, text: str, chunk_size: int) -> List[ChunkInfo]:
        """Chunk text by sentences."""

        if self.nlp:
            doc = self.nlp(text)
            sentences = list(doc.sents)
        else:
            # Simple sentence splitting
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        chunks = []
        current_chunk = ""
        current_start = 0
        current_sentences = []

        for i, sent in enumerate(sentences):
            sent_text = sent.text if hasattr(sent, 'text') else sent

            if len(current_chunk) + len(sent_text) > chunk_size and current_chunk:
                # Create chunk
                chunk_info = ChunkInfo(
                    text=current_chunk.strip(),
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    sentence_count=len(current_sentences),
                    word_count=len(current_chunk.split()),
                    entities=[],
                    topics=[],
                    chunk_type="sentence"
                )
                chunks.append(chunk_info)

                # Start new chunk
                current_chunk = sent_text
                current_start = sent.start_char if hasattr(sent, 'start_char') else 0
                current_sentences = [sent]
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sent_text
                else:
                    current_chunk = sent_text
                    current_start = sent.start_char if hasattr(sent, 'start_char') else 0
                current_sentences.append(sent)

        # Add final chunk
        if current_chunk:
            chunk_info = ChunkInfo(
                text=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                sentence_count=len(current_sentences),
                word_count=len(current_chunk.split()),
                entities=[],
                topics=[],
                chunk_type="sentence"
            )
            chunks.append(chunk_info)

        return chunks

    async def _paragraph_chunking(self, text: str, chunk_size: int) -> List[ChunkInfo]:
        """Chunk text by paragraphs."""

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""
        current_start = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                # Create chunk
                chunk_info = ChunkInfo(
                    text=current_chunk.strip(),
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    sentence_count=len(current_chunk.split('. ')),
                    word_count=len(current_chunk.split()),
                    entities=[],
                    topics=[],
                    chunk_type="paragraph"
                )
                chunks.append(chunk_info)

                # Start new chunk
                current_chunk = para
                current_start = text.find(para)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = text.find(para)

        # Add final chunk
        if current_chunk:
            chunk_info = ChunkInfo(
                text=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                sentence_count=len(current_chunk.split('. ')),
                word_count=len(current_chunk.split()),
                entities=[],
                topics=[],
                chunk_type="paragraph"
            )
            chunks.append(chunk_info)

        return chunks

    async def _simple_chunking(self, text: str, chunk_size: int) -> List[ChunkInfo]:
        """Simple character-based chunking with word boundaries."""

        words = text.split()
        chunks = []
        current_chunk = ""
        current_start = 0

        for word in words:
            if len(current_chunk) + len(word) + 1 > chunk_size and current_chunk:
                # Create chunk
                chunk_info = ChunkInfo(
                    text=current_chunk.strip(),
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    sentence_count=len(current_chunk.split('. ')),
                    word_count=len(current_chunk.split()),
                    entities=[],
                    topics=[],
                    chunk_type="simple"
                )
                chunks.append(chunk_info)

                # Start new chunk
                current_chunk = word
                current_start = text.find(word, current_start + len(current_chunk))
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
                    current_start = text.find(word)

        # Add final chunk
        if current_chunk:
            chunk_info = ChunkInfo(
                text=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                sentence_count=len(current_chunk.split('. ')),
                word_count=len(current_chunk.split()),
                entities=[],
                topics=[],
                chunk_type="simple"
            )
            chunks.append(chunk_info)

        return chunks

    async def _page_chunking(self, text: str, chunk_size: int) -> List[ChunkInfo]:
        """Chunk text treating the entire text as a single page-based chunk.

        This method is designed for use with pre-parsed document chunks that
        should be grouped by page rather than further subdivided.
        """

        # For page-based chunking, we treat the entire text as one chunk
        # unless it exceeds the maximum size
        if len(text) <= chunk_size:
            # Single chunk for the entire text
            chunk_info = ChunkInfo(
                text=text.strip(),
                start_char=0,
                end_char=len(text),
                sentence_count=len([s for s in re.split(r'[.!?]+', text) if s.strip()]),
                word_count=len(text.split()),
                entities=[],
                topics=[],
                chunk_type="page"
            )
            return [chunk_info]
        else:
            # If text is too large, fall back to paragraph-based chunking
            # but with larger chunks to maintain page-like grouping
            logger.info(f"Text too large for single page chunk ({len(text)} chars), using paragraph chunking")
            return await self._paragraph_chunking(text, chunk_size)


class ChunkingService:
    """Service wrapper for semantic chunking functionality."""

    def __init__(self):
        self.chunker = SemanticChunker()
        self.max_chunk_size = settings.max_chunk_size

    async def semantic_chunk(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        strategy: str = "semantic"
    ) -> List[str]:
        """Perform semantic chunking and return text chunks (backward compatibility)."""

        chunk_infos = await self.chunker.chunk_text(text, chunk_size, strategy)
        return [chunk.text for chunk in chunk_infos]

    async def chunk_with_metadata(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        strategy: str = "semantic"
    ) -> List[ChunkInfo]:
        """Perform chunking and return full metadata."""

        return await self.chunker.chunk_text(text, chunk_size, strategy)

    async def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure and provide chunking recommendations."""

        if not text.strip():
            return {
                "word_count": 0,
                "sentence_count": 0,
                "paragraph_count": 0,
                "avg_sentence_length": 0,
                "recommended_strategy": "simple",
                "estimated_chunks": 0,
                "text_complexity": "empty",
                "spacy_available": SPACY_AVAILABLE
            }

        # Basic analysis
        word_count = len(text.split())
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])

        # Determine complexity
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        if avg_sentence_length > 25:
            complexity = "high"
            recommended_strategy = "semantic"
        elif avg_sentence_length > 15:
            complexity = "medium"
            recommended_strategy = "sentence"
        else:
            complexity = "low"
            recommended_strategy = "paragraph"

        # Estimate chunks
        estimated_chunks = max(1, word_count // (self.max_chunk_size // 5))  # Rough estimate

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_sentence_length": avg_sentence_length,
            "text_complexity": complexity,
            "recommended_strategy": recommended_strategy,
            "estimated_chunks": estimated_chunks,
            "spacy_available": SPACY_AVAILABLE
        }


# Global instance
chunking_service = ChunkingService()
