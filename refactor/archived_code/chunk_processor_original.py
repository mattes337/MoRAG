"""Chunk processor for MoRAG ingestion system.

Handles content chunking, embedding generation, and result processing.
"""

import json
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import uuid

from morag_core.models import ProcessingResult
from morag_services.embedding import GeminiEmbeddingService
from morag_graph.services.fact_extraction_service import FactExtractionService
from morag_graph.services.enhanced_fact_processing_service import EnhancedFactProcessingService
from morag_graph.utils.id_generation import UnifiedIDGenerator
from morag_core.utils.logging import get_logger

logger = get_logger(__name__)


class ChunkProcessor:
    """Handles content chunking, processing, and result generation."""

    def __init__(self):
        """Initialize the chunk processor."""
        self.embedding_service = None
        self.fact_extractor = None

    async def initialize(self):
        """Initialize services."""
        try:
            # Initialize embedding service
            self.embedding_service = GeminiEmbeddingService()
            await self.embedding_service.initialize()

            # Initialize fact extraction service
            self.fact_extractor = FactExtractionService()

            logger.info("Chunk processor initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize chunk processor", error=str(e))
            raise

    def create_chunks(self, content: str, chunk_size: int, chunk_overlap: int, content_type: str = 'document', metadata: Dict[str, Any] = None) -> List[str]:
        """Create chunks from content based on content type and metadata."""
        if metadata is None:
            metadata = {}

        # Determine chunking strategy based on content type and metadata
        chunk_type = self._determine_chunk_type(content_type, metadata)

        if chunk_type == 'topic_based':
            return self._create_topic_based_chunks(content, chunk_size, chunk_overlap, metadata)
        elif chunk_type == 'timestamp':
            return self._create_timestamp_chunks(content, chunk_size, chunk_overlap)
        elif chunk_type == 'image_section':
            return self._create_image_section_chunks(content, chunk_size, chunk_overlap)
        elif chunk_type == 'web_article':
            return self._create_web_article_chunks(content, chunk_size, chunk_overlap)
        elif chunk_type == 'code_structural':
            return self._create_code_structural_chunks(content, chunk_size, chunk_overlap)
        elif chunk_type == 'archive_file':
            return self._create_archive_file_chunks(content, chunk_size, chunk_overlap)
        elif chunk_type == 'text_semantic':
            return self._create_text_semantic_chunks(content, chunk_size, chunk_overlap)
        elif chunk_type == 'document':
            return self._create_document_chunks(content, chunk_size, chunk_overlap)
        else:
            # Default to character-based chunking
            return self._create_character_chunks(content, chunk_size, chunk_overlap)

    def _create_topic_based_chunks(self, content: str, chunk_size: int, chunk_overlap: int, metadata: Dict[str, Any]) -> List[str]:
        """Create topic-based chunks for audio/video content with timestamps."""
        logger.info("Creating topic-based chunks", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunks = []

        # Look for topic markers like timestamps or section headers
        topic_pattern = r'\n(?=(?:\d{1,2}:\d{2}(?::\d{2})?|\[?\d+:\d+(?::\d+)?\]?|\*\*[^*]+\*\*|\#{1,3}\s+[^\n]+))'
        topics = re.split(topic_pattern, content)

        if len(topics) <= 1:
            # No clear topics found, fall back to timestamp-based chunking
            return self._create_timestamp_chunks(content, chunk_size, chunk_overlap)

        current_chunk = ""

        for i, topic in enumerate(topics):
            topic = topic.strip()
            if not topic:
                continue

            # Check if adding this topic would exceed chunk size
            if len(current_chunk) + len(topic) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + topic
                else:
                    current_chunk = topic
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If topic is too large, split it
                if len(topic) > chunk_size:
                    topic_chunks = self._split_topic_at_timestamps(topic, chunk_size, chunk_overlap)
                    chunks.extend(topic_chunks[:-1])  # Add all but the last
                    current_chunk = topic_chunks[-1] if topic_chunks else ""
                else:
                    current_chunk = topic

        # Add the final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info("Created topic-based chunks", total_chunks=len(chunks))
        return chunks

    def _create_timestamp_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks based on timestamps for audio/video content."""
        logger.info("Creating timestamp-based chunks", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Split by timestamp patterns
        timestamp_pattern = r'\n(?=\d{1,2}:\d{2}(?::\d{2})?|\[?\d+:\d+(?::\d+)?\]?)'
        segments = re.split(timestamp_pattern, content)

        chunks = []
        current_chunk = ""

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Check if adding this segment would exceed chunk size
            if len(current_chunk) + len(segment) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + segment
                else:
                    current_chunk = segment
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, chunk_overlap)
                if len(segment) > chunk_size:
                    # Split large segment
                    segment_chunks = self._create_character_chunks(segment, chunk_size - len(overlap_text), chunk_overlap)
                    current_chunk = overlap_text + "\n\n" + segment_chunks[0] if overlap_text else segment_chunks[0]
                    chunks.extend([current_chunk] + segment_chunks[1:])
                    current_chunk = ""
                else:
                    current_chunk = overlap_text + "\n\n" + segment if overlap_text else segment

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info("Created timestamp-based chunks", total_chunks=len(chunks))
        return chunks

    def _split_topic_at_timestamps(self, topic_content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split a large topic at timestamp boundaries."""
        # Look for timestamps within the topic
        timestamp_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?|\[?\d+:\d+(?::\d+)?\]?)'

        # Split while preserving timestamps
        parts = re.split(f'({timestamp_pattern})', topic_content)

        chunks = []
        current_chunk = ""

        i = 0
        while i < len(parts):
            part = parts[i].strip()

            # Skip empty parts
            if not part:
                i += 1
                continue

            # Check if this is a timestamp
            if re.match(timestamp_pattern, part):
                # Include timestamp with following text
                timestamp_text = part
                if i + 1 < len(parts):
                    timestamp_text += " " + parts[i + 1]
                    i += 1  # Skip the next part as we've included it

                if len(current_chunk) + len(timestamp_text) + 2 <= chunk_size:
                    if current_chunk:
                        current_chunk += "\n" + timestamp_text
                    else:
                        current_chunk = timestamp_text
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = timestamp_text
            else:
                # Regular text part
                if len(current_chunk) + len(part) + 2 <= chunk_size:
                    if current_chunk:
                        current_chunk += " " + part
                    else:
                        current_chunk = part
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part

            i += 1

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _create_image_section_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks optimized for image content with descriptions."""
        logger.info("Creating image section chunks", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Split by image markers and section headers
        section_pattern = r'\n(?=(?:Image \d+|Figure \d+|Screenshot \d+|\[Image.*?\]|\*\*[^*]+\*\*|##?\s+[^\n]+))'
        sections = re.split(section_pattern, content)

        chunks = []
        current_chunk = ""

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Preserve image descriptions and their context together
            if len(current_chunk) + len(section) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk
                if len(section) > chunk_size:
                    # Split large section but try to preserve image-text relationships
                    section_chunks = self._create_character_chunks(section, chunk_size, chunk_overlap)
                    chunks.extend(section_chunks[:-1])
                    current_chunk = section_chunks[-1] if section_chunks else ""
                else:
                    current_chunk = section

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info("Created image section chunks", total_chunks=len(chunks))
        return chunks

    def _create_web_article_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks optimized for web article content."""
        logger.info("Creating web article chunks", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Split by common web article structures
        section_patterns = [
            r'\n(?=#{1,6}\s+[^\n]+)',  # Markdown headers
            r'\n(?=\*\*[^*]+\*\*)',    # Bold titles
            r'\n(?=\d+\.\s+[^\n]+)',   # Numbered sections
            r'\n(?=•\s+[^\n]+)',       # Bullet points
            r'\n\n(?=\S)',             # Paragraph breaks
        ]

        # Try each pattern to find the best splitting approach
        best_chunks = []
        best_score = float('inf')

        for pattern in section_patterns:
            try:
                sections = re.split(pattern, content)
                chunks = self._process_sections_into_chunks(sections, chunk_size, chunk_overlap)

                # Score based on chunk size variance (prefer more uniform chunks)
                if chunks:
                    sizes = [len(chunk) for chunk in chunks]
                    avg_size = sum(sizes) / len(sizes)
                    variance = sum((size - avg_size) ** 2 for size in sizes) / len(sizes)

                    if variance < best_score:
                        best_score = variance
                        best_chunks = chunks
            except:
                continue

        # If no pattern worked well, fall back to semantic chunking
        if not best_chunks:
            best_chunks = self._create_text_semantic_chunks(content, chunk_size, chunk_overlap)

        logger.info("Created web article chunks", total_chunks=len(best_chunks))
        return best_chunks

    def _create_text_semantic_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create semantically coherent text chunks."""
        logger.info("Creating semantic text chunks", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Split by paragraphs and sentences for better semantic coherence
        paragraphs = content.split('\n\n')

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(paragraph) + 2 > chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Handle large paragraphs
                if len(paragraph) > chunk_size:
                    # Split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    sentence_chunk = ""

                    for sentence in sentences:
                        if len(sentence_chunk) + len(sentence) + 1 <= chunk_size:
                            sentence_chunk += " " + sentence if sentence_chunk else sentence
                        else:
                            if sentence_chunk:
                                chunks.append(sentence_chunk.strip())

                            # Handle very long sentences
                            if len(sentence) > chunk_size:
                                # Split by clauses or character limit
                                clause_chunks = self._create_character_chunks(sentence, chunk_size, chunk_overlap)
                                chunks.extend(clause_chunks[:-1])
                                sentence_chunk = clause_chunks[-1] if clause_chunks else ""
                            else:
                                sentence_chunk = sentence

                    if sentence_chunk:
                        current_chunk = sentence_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info("Created semantic text chunks", total_chunks=len(chunks))
        return chunks

    def _create_code_structural_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks optimized for code content."""
        logger.info("Creating code structural chunks", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Split by common code structures
        code_patterns = [
            r'\n(?=class\s+\w+)',      # Class definitions
            r'\n(?=def\s+\w+)',        # Function definitions
            r'\n(?=function\s+\w+)',   # JavaScript functions
            r'\n(?=export\s+)',        # ES6 exports
            r'\n(?=import\s+)',        # Import statements
            r'\n(?=\/\*\*)',           # Documentation blocks
            r'\n(?=\/\*)',             # Comment blocks
            r'\n(?=#[^\n]*)',          # Comments/preprocessor
            r'\n\n(?=\S)',             # Code blocks separated by blank lines
        ]

        best_chunks = []

        # Try structural splitting first
        for pattern in code_patterns[:5]:  # Use first 5 patterns for structure
            try:
                sections = re.split(pattern, content)
                if len(sections) > 1:
                    chunks = self._process_sections_into_chunks(sections, chunk_size, chunk_overlap)
                    if chunks:
                        best_chunks = chunks
                        break
            except:
                continue

        # If no structural pattern worked, fall back to line-based chunking
        if not best_chunks:
            lines = content.split('\n')
            chunks = []
            current_chunk = ""

            for line in lines:
                if len(current_chunk) + len(line) + 1 <= chunk_size:
                    current_chunk += "\n" + line if current_chunk else line
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    # Handle very long lines
                    if len(line) > chunk_size:
                        line_chunks = self._create_character_chunks(line, chunk_size, chunk_overlap)
                        chunks.extend(line_chunks[:-1])
                        current_chunk = line_chunks[-1] if line_chunks else ""
                    else:
                        current_chunk = line

            if current_chunk:
                chunks.append(current_chunk.strip())

            best_chunks = chunks

        logger.info("Created code structural chunks", total_chunks=len(best_chunks))
        return best_chunks

    def _create_archive_file_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks optimized for archive file listings."""
        logger.info("Creating archive file chunks", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Split by file entries or directory structures
        file_patterns = [
            r'\n(?=\S+/)',             # Directory paths
            r'\n(?=\d{4}-\d{2}-\d{2})', # Date stamps
            r'\n(?=\w+\.\w+)',         # File names with extensions
            r'\n(?=[-d][-rwx]{9})',    # Unix permissions
        ]

        chunks = []

        # Try each pattern
        for pattern in file_patterns:
            try:
                sections = re.split(pattern, content)
                if len(sections) > 1:
                    chunks = self._process_sections_into_chunks(sections, chunk_size, chunk_overlap)
                    if chunks:
                        break
            except:
                continue

        # Fallback to line-based chunking
        if not chunks:
            lines = content.split('\n')
            current_chunk = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if len(current_chunk) + len(line) + 1 <= chunk_size:
                    current_chunk += "\n" + line if current_chunk else line
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = line

            if current_chunk:
                chunks.append(current_chunk.strip())

        logger.info("Created archive file chunks", total_chunks=len(chunks))
        return chunks

    def _create_document_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks optimized for document content."""
        logger.info("Creating document chunks", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Split by document structures (headings, paragraphs, etc.)
        doc_patterns = [
            r'\n(?=#{1,6}\s+[^\n]+)',      # Markdown headers
            r'\n(?=Chapter\s+\d+)',        # Chapter headings
            r'\n(?=Section\s+\d+)',        # Section headings
            r'\n(?=\d+\.\s+[A-Z][^\n]+)',  # Numbered headings
            r'\n\n(?=\S)',                 # Paragraph breaks
        ]

        best_chunks = []

        # Try structural patterns
        for pattern in doc_patterns:
            try:
                sections = re.split(pattern, content)
                if len(sections) > 1:
                    chunks = self._process_sections_into_chunks(sections, chunk_size, chunk_overlap)
                    if chunks:
                        best_chunks = chunks
                        break
            except:
                continue

        # Fallback to semantic chunking
        if not best_chunks:
            best_chunks = self._create_text_semantic_chunks(content, chunk_size, chunk_overlap)

        logger.info("Created document chunks", total_chunks=len(best_chunks))
        return best_chunks

    def _create_character_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks based on character count with overlap."""
        logger.info("Creating character-based chunks",
                   content_length=len(content),
                   chunk_size=chunk_size,
                   chunk_overlap=chunk_overlap)

        if len(content) <= chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            if end >= len(content):
                # Last chunk
                chunks.append(content[start:].strip())
                break

            # Try to break at a natural boundary
            chunk_text = content[start:end]

            # Look for good break points near the end
            break_chars = ['\n\n', '\n', '. ', '! ', '? ', ', ', ' ']
            break_point = -1

            for break_char in break_chars:
                pos = chunk_text.rfind(break_char)
                if pos > chunk_size * 0.8:  # Only use breaks in the last 20% of chunk
                    break_point = start + pos + len(break_char)
                    break

            if break_point == -1:
                # No good break point found, use hard limit
                break_point = end

            chunk = content[start:break_point].strip()
            if chunk:
                chunks.append(chunk)

            # Calculate next start position with overlap
            start = max(break_point - chunk_overlap, start + 1)

        logger.info("Created character chunks", total_chunks=len(chunks))
        return chunks

    def _process_sections_into_chunks(self, sections: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """Process text sections into appropriately sized chunks."""
        chunks = []
        current_chunk = ""

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(current_chunk) + len(section) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if len(section) > chunk_size:
                    section_chunks = self._create_character_chunks(section, chunk_size, chunk_overlap)
                    chunks.extend(section_chunks[:-1])
                    current_chunk = section_chunks[-1] if section_chunks else ""
                else:
                    current_chunk = section

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= overlap_size:
            return text

        overlap_text = text[-overlap_size:]

        # Try to start overlap at a natural boundary
        break_chars = ['\n\n', '\n', '. ', '! ', '? ']
        for break_char in break_chars:
            pos = overlap_text.find(break_char)
            if pos != -1:
                return overlap_text[pos + len(break_char):].strip()

        return overlap_text.strip()

    def _determine_chunk_type(self, content_type: str, metadata: Dict[str, Any]) -> str:
        """Determine the best chunking strategy based on content type and metadata."""
        # Check metadata for specific indicators
        if metadata.get('has_timestamps') or metadata.get('duration'):
            return 'timestamp'

        if metadata.get('has_topics') or metadata.get('topic_segments'):
            return 'topic_based'

        if metadata.get('has_images') or content_type in ['image', 'screenshot']:
            return 'image_section'

        if metadata.get('source_type') == 'web' or content_type == 'web':
            return 'web_article'

        if content_type in ['code', 'python', 'javascript', 'java', 'cpp']:
            return 'code_structural'

        if content_type in ['archive', 'zip', 'tar'] or metadata.get('is_archive'):
            return 'archive_file'

        if content_type in ['document', 'pdf', 'docx', 'txt']:
            return 'document'

        # Default to semantic text chunking
        return 'text_semantic'

    async def generate_embeddings_and_metadata(
        self,
        content: str,
        chunks: List[str],
        source_path: str,
        content_type: str,
        metadata: Dict[str, Any],
        document_id: str
    ) -> Dict[str, Any]:
        """Generate embeddings and metadata for content and chunks."""
        logger.info("Generating embeddings and metadata",
                   chunk_count=len(chunks),
                   content_length=len(content))

        try:
            # Generate document summary
            summary = await self._generate_document_summary(content, content_type)

            # Generate chunk embeddings
            chunk_embeddings = []
            if self.embedding_service:
                chunk_texts = [chunk for chunk in chunks if chunk.strip()]
                if chunk_texts:
                    embeddings = await self.embedding_service.generate_embeddings_batch(chunk_texts)
                    chunk_embeddings = embeddings if embeddings else []

            # Prepare chunks data with metadata
            chunks_data = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                chunk_metadata = self._add_chunk_metadata(chunk, i, content_type, metadata)
                chunk_data = {
                    'text': chunk,
                    'chunk_index': i,
                    'embedding': chunk_embeddings[i] if i < len(chunk_embeddings) else None,
                    'metadata': chunk_metadata,
                    'document_id': document_id
                }
                chunks_data.append(chunk_data)

            # Generate content embedding
            content_embedding = None
            if self.embedding_service and content.strip():
                content_embeddings = await self.embedding_service.generate_embeddings_batch([content[:8000]])  # Limit for embedding
                content_embedding = content_embeddings[0] if content_embeddings else None

            embeddings_data = {
                'content_embedding': content_embedding,
                'chunk_embeddings': chunk_embeddings,
                'chunks_data': chunks_data,
                'summary': summary,
                'embedding_dimension': len(chunk_embeddings[0]) if chunk_embeddings else 768,
                'total_chunks': len(chunks_data)
            }

            logger.info("Generated embeddings and metadata",
                       content_embedding_generated=content_embedding is not None,
                       chunk_embeddings_generated=len(chunk_embeddings),
                       embedding_dimension=embeddings_data['embedding_dimension'])

            return embeddings_data

        except Exception as e:
            logger.error("Failed to generate embeddings and metadata", error=str(e))
            raise

    def _add_chunk_metadata(self, chunk_text: str, chunk_index: int, content_type: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add chunk-specific metadata."""
        chunk_metadata = base_metadata.copy()

        # Add basic chunk info
        chunk_metadata.update({
            'chunk_index': chunk_index,
            'chunk_length': len(chunk_text),
            'chunk_type': self._determine_chunk_type(content_type, base_metadata)
        })

        # Add content-specific metadata
        if content_type in ['audio', 'video']:
            chunk_metadata.update(self._add_audio_video_chunk_metadata(chunk_text, chunk_index, base_metadata))
        elif content_type in ['document', 'pdf', 'text']:
            chunk_metadata.update(self._add_document_chunk_metadata(chunk_text, chunk_index, base_metadata))

        return chunk_metadata

    def _add_audio_video_chunk_metadata(self, chunk_text: str, chunk_index: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata specific to audio/video chunks."""
        chunk_metadata = {}

        # Extract timestamp information
        timestamp_patterns = [
            r'(\d{1,2}:\d{2}(?::\d{2})?)',  # MM:SS or HH:MM:SS
            r'\[(\d+:\d+(?::\d+)?)\]',      # [MM:SS] or [HH:MM:SS]
        ]

        timestamps = []
        for pattern in timestamp_patterns:
            matches = re.findall(pattern, chunk_text)
            timestamps.extend(matches)

        if timestamps:
            # Convert first timestamp to seconds for ordering
            first_timestamp = timestamps[0]
            try:
                time_parts = first_timestamp.split(':')
                if len(time_parts) == 2:  # MM:SS
                    seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                elif len(time_parts) == 3:  # HH:MM:SS
                    seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                else:
                    seconds = 0

                chunk_metadata.update({
                    'start_time': first_timestamp,
                    'start_seconds': seconds,
                    'timestamps': timestamps,
                    'timestamp_formatted': self._seconds_to_timestamp(seconds)
                })
            except (ValueError, IndexError):
                pass

        # Add speaker/topic information if available
        speaker_pattern = r'(?:Speaker\s*\d+|[A-Z][a-z]+\s*:)'
        speakers = re.findall(speaker_pattern, chunk_text)
        if speakers:
            chunk_metadata['speakers'] = speakers

        # Add topic information
        if metadata.get('topics'):
            # Try to match chunk to topics based on content
            for topic in metadata['topics']:
                if any(keyword in chunk_text.lower() for keyword in topic.lower().split()[:3]):
                    chunk_metadata['related_topic'] = topic
                    break

        return chunk_metadata

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"

    def _add_document_chunk_metadata(self, chunk_text: str, chunk_index: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata specific to document chunks."""
        chunk_metadata = {}

        # Detect headings and structure
        heading_patterns = [
            (r'^#{1,6}\s+(.+)$', 'markdown_heading'),
            (r'^(.+)\n=+$', 'heading_1'),
            (r'^(.+)\n-+$', 'heading_2'),
            (r'^\d+\.\s+(.+)$', 'numbered_section'),
            (r'^[A-Z][A-Z\s]+$', 'caps_heading'),
        ]

        headings = []
        for pattern, heading_type in heading_patterns:
            matches = re.findall(pattern, chunk_text, re.MULTILINE)
            for match in matches:
                headings.append({
                    'text': match,
                    'type': heading_type
                })

        if headings:
            chunk_metadata['headings'] = headings
            chunk_metadata['primary_heading'] = headings[0]['text']

        # Count structural elements
        chunk_metadata.update({
            'paragraph_count': len(re.findall(r'\n\n+', chunk_text)) + 1,
            'sentence_count': len(re.findall(r'[.!?]+', chunk_text)),
            'word_count': len(chunk_text.split()),
            'has_lists': bool(re.search(r'^\s*[-*•]\s+', chunk_text, re.MULTILINE)),
            'has_code': bool(re.search(r'```|`[^`]+`', chunk_text)),
            'has_urls': bool(re.search(r'https?://', chunk_text)),
        })

        return chunk_metadata

    async def _generate_document_summary(self, content: str, content_type: str) -> str:
        """Generate a summary of the document content."""
        try:
            # Limit content for summary generation
            summary_content = content[:4000]  # Limit to first 4000 characters

            if len(summary_content) < 100:
                return "Brief content - no summary needed"

            # Extract key information for summary
            lines = summary_content.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]

            if len(non_empty_lines) < 5:
                return non_empty_lines[0] if non_empty_lines else "Content summary unavailable"

            # Create a simple extractive summary
            summary_parts = []

            # Add first line if it looks like a title
            if non_empty_lines and len(non_empty_lines[0]) < 100:
                summary_parts.append(non_empty_lines[0])

            # Add key sentences (those with important keywords)
            key_indicators = [
                'summary', 'conclusion', 'important', 'key', 'main', 'primary',
                'essential', 'critical', 'overview', 'introduction'
            ]

            key_sentences = []
            for line in non_empty_lines[1:]:
                if any(indicator in line.lower() for indicator in key_indicators):
                    key_sentences.append(line)
                    if len(key_sentences) >= 3:
                        break

            summary_parts.extend(key_sentences)

            # If we have very little, add more content
            if len(' '.join(summary_parts)) < 200:
                summary_parts.extend(non_empty_lines[:5])

            summary = ' '.join(summary_parts)

            # Limit summary length
            if len(summary) > 500:
                summary = summary[:500] + "..."

            return summary or "Content processed successfully"

        except Exception as e:
            logger.warning("Failed to generate document summary", error=str(e))
            return f"{content_type.title()} content processed"

    def create_ingest_result(
        self,
        source_path: str,
        document_id: str,
        content: str,
        content_type: str,
        metadata: Dict[str, Any],
        embeddings_data: Dict[str, Any],
        graph_data: Optional[Dict[str, Any]] = None,
        processing_result: Optional[ProcessingResult] = None
    ) -> Dict[str, Any]:
        """Create comprehensive ingest result structure."""
        timestamp = datetime.now(timezone.utc).isoformat()

        ingest_result = {
            'document_id': document_id,
            'source_path': source_path,
            'content_type': content_type,
            'timestamp': timestamp,
            'processing_info': {
                'success': True,
                'content_length': len(content),
                'chunk_count': embeddings_data.get('total_chunks', 0),
                'embedding_dimension': embeddings_data.get('embedding_dimension', 768),
                'processing_time': processing_result.processing_time if processing_result else 0.0
            },
            'content_metadata': metadata,
            'summary': embeddings_data.get('summary', ''),
            'chunks': embeddings_data.get('chunks_data', []),
            'embeddings': {
                'content_embedding': embeddings_data.get('content_embedding'),
                'chunk_embeddings': embeddings_data.get('chunk_embeddings', [])
            }
        }

        # Add graph data if available
        if graph_data:
            ingest_result['graph_data'] = {
                'entities': graph_data.get('entities', []),
                'relations': graph_data.get('relations', []),
                'facts': graph_data.get('facts', []),
                'enhanced_processing': self._serialize_enhanced_processing(graph_data.get('enhanced_processing', {})),
                'entity_embeddings': graph_data.get('entity_embeddings', {}),
                'fact_embeddings': graph_data.get('fact_embeddings', {})
            }

        return ingest_result

    def _serialize_enhanced_processing(self, enhanced_processing: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize enhanced processing data to be JSON-compatible."""
        if not enhanced_processing:
            return {}

        serialized = {}

        for key, value in enhanced_processing.items():
            if key == 'entities' and isinstance(value, list):
                # Serialize Entity objects
                serialized[key] = []
                for entity in value:
                    if hasattr(entity, 'to_dict'):
                        serialized[key].append(entity.to_dict())
                    elif isinstance(entity, dict):
                        serialized[key].append(entity)
                    else:
                        # Convert Entity object to dict manually
                        entity_dict = {
                            'id': getattr(entity, 'id', str(uuid.uuid4())),
                            'name': getattr(entity, 'name', ''),
                            'type': getattr(entity, 'type', 'Unknown'),
                            'properties': getattr(entity, 'properties', {}),
                            'confidence': getattr(entity, 'confidence', 0.0),
                            'embedding': getattr(entity, 'embedding', None)
                        }
                        serialized[key].append(entity_dict)

            elif isinstance(value, (list, dict, str, int, float, bool)):
                serialized[key] = value
            else:
                # Convert other objects to string representation
                serialized[key] = str(value)

        return serialized

    def write_ingest_result_file(self, source_path: str, ingest_result: Dict[str, Any]) -> str:
        """Write the ingest_result.json file."""
        source_path_obj = Path(source_path)
        result_file_path = source_path_obj.parent / f"{source_path_obj.stem}.ingest_result.json"

        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(ingest_result, f, indent=2, ensure_ascii=False)

        return str(result_file_path)

    def create_ingest_data(
        self,
        document_id: str,
        embeddings_data: Dict[str, Any],
        graph_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create ingest_data.json structure for database writing."""
        ingest_data = {
            'document_id': document_id,
            'chunks': embeddings_data.get('chunks_data', []),
            'content_embedding': embeddings_data.get('content_embedding'),
            'summary': embeddings_data.get('summary', '')
        }

        # Add graph data if available
        if graph_data:
            ingest_data['graph_data'] = {
                'entities': graph_data.get('entities', []),
                'relations': graph_data.get('relations', []),
                'facts': graph_data.get('facts', []),
                'enhanced_processing': self._serialize_enhanced_processing(graph_data.get('enhanced_processing', {})),
                'entity_embeddings': graph_data.get('entity_embeddings', {}),
                'fact_embeddings': graph_data.get('fact_embeddings', {})
            }

        return ingest_data

    def write_ingest_data_file(self, source_path: str, ingest_data: Dict[str, Any]) -> str:
        """Write the ingest_data.json file."""
        source_path_obj = Path(source_path)
        data_file_path = source_path_obj.parent / f"{source_path_obj.stem}.ingest_data.json"

        with open(data_file_path, 'w', encoding='utf-8') as f:
            json.dump(ingest_data, f, indent=2, ensure_ascii=False)

        return str(data_file_path)
