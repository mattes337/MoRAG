"""Embedding and metadata processing for chunks."""

import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone

from morag_services.embedding import GeminiEmbeddingService
from morag_graph.services.fact_extraction_service import FactExtractionService
from morag_core.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProcessor:
    """Handles chunk embedding generation and metadata processing."""

    def __init__(self):
        """Initialize the embedding processor."""
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

            logger.info("Embedding processor initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize embedding processor", error=str(e))
            raise

    async def generate_embeddings_and_metadata(
        self,
        chunks: List[str],
        content_type: str = 'document',
        base_metadata: Dict[str, Any] = None
    ) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        """Generate embeddings and metadata for chunks with optimized batching.

        Args:
            chunks: List of text chunks
            content_type: Type of content being processed
            base_metadata: Base metadata to include with each chunk

        Returns:
            Tuple of (chunks, embeddings, metadata_list)
        """
        if not chunks:
            return [], [], []

        if base_metadata is None:
            base_metadata = {}

        try:
            # Use Gemini's optimal batch size (documented as 100)
            OPTIMAL_BATCH_SIZE = 100

            # Process in parallel batches
            embedding_tasks = []
            for i in range(0, len(chunks), OPTIMAL_BATCH_SIZE):
                batch = chunks[i:i + OPTIMAL_BATCH_SIZE]
                embedding_tasks.append(self.embedding_service.generate_batch(batch))

            # Execute all batches concurrently
            batch_results = await asyncio.gather(*embedding_tasks)

            # Flatten results
            embeddings = [emb for batch in batch_results for emb in batch]

            # Generate metadata in parallel with embeddings
            metadata_list = await asyncio.gather(*[
                self._generate_chunk_metadata_async(chunk, i, content_type, base_metadata)
                for i, chunk in enumerate(chunks)
            ])

            logger.info("Generated embeddings and metadata with optimized batching",
                       chunk_count=len(chunks),
                       content_type=content_type,
                       batch_count=len(embedding_tasks))

            return chunks, embeddings, metadata_list

        except Exception as e:
            logger.error("Failed to generate embeddings and metadata",
                        error=str(e),
                        chunk_count=len(chunks))
            raise

    async def _generate_chunk_metadata_async(self, chunk_text: str, chunk_index: int, content_type: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chunk metadata asynchronously for parallel processing."""
        return self._add_chunk_metadata(chunk_text, chunk_index, content_type, base_metadata)

    def _add_chunk_metadata(self, chunk_text: str, chunk_index: int, content_type: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to a chunk based on content type."""
        metadata = base_metadata.copy()

        # Add common metadata
        metadata.update({
            'chunk_index': chunk_index,
            'chunk_length': len(chunk_text),
            'content_type': content_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })

        # Add content-type specific metadata
        if content_type in ['audio', 'video']:
            metadata.update(self._add_audio_video_chunk_metadata(chunk_text, chunk_index, base_metadata))
        elif content_type == 'document':
            metadata.update(self._add_document_chunk_metadata(chunk_text, chunk_index, base_metadata))

        return metadata

    def _add_audio_video_chunk_metadata(self, chunk_text: str, chunk_index: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add audio/video specific metadata to chunk."""
        chunk_metadata = {}

        # Extract timestamp information from chunk text
        timestamp_matches = re.findall(r'\[?(\d{1,2}):(\d{2}):(\d{2})\]?', chunk_text)
        if timestamp_matches:
            # Get first and last timestamps
            first_match = timestamp_matches[0]
            last_match = timestamp_matches[-1]

            start_seconds = int(first_match[0]) * 3600 + int(first_match[1]) * 60 + int(first_match[2])
            end_seconds = int(last_match[0]) * 3600 + int(last_match[1]) * 60 + int(last_match[2])

            chunk_metadata.update({
                'start_time': start_seconds,
                'end_time': end_seconds,
                'duration': end_seconds - start_seconds,
                'start_timestamp': self._seconds_to_timestamp(start_seconds),
                'end_timestamp': self._seconds_to_timestamp(end_seconds),
            })

        # Extract speaker information if available
        speaker_matches = re.findall(r'(Speaker \w+|[A-Z][a-z]+ [A-Z][a-z]+):', chunk_text)
        if speaker_matches:
            unique_speakers = list(set(speaker_matches))
            chunk_metadata['speakers'] = unique_speakers
            chunk_metadata['speaker_count'] = len(unique_speakers)

        # Calculate word count and speaking rate
        words = chunk_text.split()
        chunk_metadata['word_count'] = len(words)

        if 'duration' in chunk_metadata and chunk_metadata['duration'] > 0:
            speaking_rate = len(words) / (chunk_metadata['duration'] / 60)  # words per minute
            chunk_metadata['speaking_rate'] = round(speaking_rate, 1)

        # Detect content themes
        if any(keyword in chunk_text.lower() for keyword in ['question', 'answer', 'q:', 'a:']):
            chunk_metadata['contains_qa'] = True

        if any(keyword in chunk_text.lower() for keyword in ['music', 'song', 'melody']):
            chunk_metadata['contains_music'] = True

        return chunk_metadata

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _add_document_chunk_metadata(self, chunk_text: str, chunk_index: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add document specific metadata to chunk."""
        chunk_metadata = {}

        # Extract structural information
        headers = re.findall(r'^#+\s+(.+)$', chunk_text, re.MULTILINE)
        if headers:
            chunk_metadata['headers'] = headers
            chunk_metadata['header_count'] = len(headers)
            chunk_metadata['primary_header'] = headers[0]

        # Count different content types
        bullet_points = len(re.findall(r'^\s*[-*•]\s+', chunk_text, re.MULTILINE))
        numbered_items = len(re.findall(r'^\s*\d+\.\s+', chunk_text, re.MULTILINE))
        tables = len(re.findall(r'^\|.*\|$', chunk_text, re.MULTILINE))
        code_blocks = len(re.findall(r'```', chunk_text))

        chunk_metadata.update({
            'bullet_points': bullet_points,
            'numbered_items': numbered_items,
            'tables': tables,
            'code_blocks': code_blocks // 2,  # Divide by 2 since each block has opening and closing
            'word_count': len(chunk_text.split()),
            'character_count': len(chunk_text),
            'line_count': len(chunk_text.split('\n')),
        })

        # Detect content characteristics
        if bullet_points > 0 or numbered_items > 0:
            chunk_metadata['has_lists'] = True

        if tables > 0:
            chunk_metadata['has_tables'] = True

        if code_blocks > 0:
            chunk_metadata['has_code'] = True

        # Estimate reading time (average 200 words per minute)
        word_count = chunk_metadata['word_count']
        reading_time_minutes = word_count / 200
        chunk_metadata['estimated_reading_time'] = round(reading_time_minutes, 1)

        # Extract page numbers if present in metadata
        if 'page_number' in metadata:
            chunk_metadata['page_number'] = metadata['page_number']

        if 'section' in metadata:
            chunk_metadata['section'] = metadata['section']

        return chunk_metadata

    async def _generate_document_summary(self, content: str, content_type: str) -> str:
        """Generate a summary of the document content."""
        try:
            if not self.fact_extractor:
                return f"Summary of {content_type} content ({len(content.split())} words)"

            # Use fact extractor to generate summary
            summary_result = await self.fact_extractor.generate_summary(content, max_length=200)

            if summary_result and 'summary' in summary_result:
                return summary_result['summary']
            else:
                return f"Summary of {content_type} content ({len(content.split())} words)"

        except Exception as e:
            logger.warning("Failed to generate document summary", error=str(e))
            return f"Summary of {content_type} content ({len(content.split())} words)"


__all__ = ["EmbeddingProcessor"]
