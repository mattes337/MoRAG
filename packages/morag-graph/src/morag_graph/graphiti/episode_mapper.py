"""Document to Episode mapping service for Graphiti integration."""

import asyncio
from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime
import structlog
from enum import Enum

from morag_core.models import Document, DocumentChunk
from .config import GraphitiConfig
from .connection import GraphitiConnectionService

logger = structlog.get_logger(__name__)


class EpisodeStrategy(Enum):
    """Episode creation strategies."""
    DOCUMENT_ONLY = "document_only"  # One episode per document
    CHUNK_ONLY = "chunk_only"       # One episode per chunk
    HYBRID = "hybrid"                # Both document and chunk episodes with rich context
    CONTEXTUAL_CHUNKS = "contextual_chunks"  # Chunks with enhanced contextual summaries


class ContextLevel(Enum):
    """Context enrichment levels."""
    MINIMAL = "minimal"      # Basic metadata only
    STANDARD = "standard"    # Document summary + chunk relationships
    RICH = "rich"           # Full context with surrounding chunks and semantic analysis
    COMPREHENSIVE = "comprehensive"  # All available context including cross-document relationships


class DocumentEpisodeMapper:
    """Service for mapping MoRAG documents to Graphiti episodes with hybrid strategy support."""

    def __init__(
        self,
        config: Optional[GraphitiConfig] = None,
        strategy: EpisodeStrategy = EpisodeStrategy.HYBRID,
        context_level: ContextLevel = ContextLevel.RICH,
        enable_ai_summarization: bool = True
    ):
        """Initialize the document episode mapper.

        Args:
            config: Optional Graphiti configuration
            strategy: Episode creation strategy
            context_level: Level of context enrichment
            enable_ai_summarization: Whether to use AI for contextual summaries
        """
        self.config = config
        self.connection_service = GraphitiConnectionService(config)
        self.strategy = strategy
        self.context_level = context_level
        self.enable_ai_summarization = enable_ai_summarization

        # Initialize AI services if available
        self._embedding_service = None
        self._summarization_agent = None
        self._init_ai_services()

    def _init_ai_services(self):
        """Initialize AI services for contextual processing."""
        try:
            # Try to import and initialize embedding service
            if self.enable_ai_summarization:
                try:
                    from morag_services.embedding import EmbeddingService
                    from morag_core.ai.summarization_agent import SummarizationAgent

                    # Initialize services if available
                    self._embedding_service = EmbeddingService()
                    self._summarization_agent = SummarizationAgent()

                    logger.info("AI services initialized for contextual processing")
                except ImportError:
                    logger.warning("AI services not available - contextual summaries will be basic")
                except Exception as e:
                    logger.warning("Failed to initialize AI services", error=str(e))
        except Exception as e:
            logger.warning("AI service initialization failed", error=str(e))

    async def map_document_hybrid(
        self,
        document: Document,
        episode_name_prefix: Optional[str] = None,
        source_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Map document using hybrid strategy - both document and chunk episodes.

        Args:
            document: MoRAG document to map
            episode_name_prefix: Optional prefix for episode names
            source_description: Optional source description

        Returns:
            Dictionary with mapping results for both document and chunks
        """
        try:
            results = {
                "strategy": "hybrid",
                "document_episode": None,
                "chunk_episodes": [],
                "success": False,
                "total_episodes": 0
            }

            # Create document-level episode
            doc_result = await self.map_document_to_episode(
                document, episode_name_prefix, source_description
            )
            results["document_episode"] = doc_result

            # Create contextual chunk episodes
            chunk_results = await self.map_document_chunks_to_contextual_episodes(
                document, episode_name_prefix
            )
            results["chunk_episodes"] = chunk_results

            # Calculate success
            doc_success = doc_result.get("success", False)
            chunk_success = all(r.get("success", False) for r in chunk_results)
            results["success"] = doc_success and chunk_success
            results["total_episodes"] = (1 if doc_success else 0) + len([r for r in chunk_results if r.get("success")])

            logger.info(
                "Hybrid document mapping completed",
                document_id=document.id,
                total_episodes=results["total_episodes"],
                success=results["success"]
            )

            return results

        except Exception as e:
            logger.error("Hybrid document mapping failed", document_id=document.id, error=str(e))
            return {
                "strategy": "hybrid",
                "success": False,
                "error": str(e),
                "document_id": document.id
            }

    async def map_document_to_episode(
        self,
        document: Document,
        episode_name: Optional[str] = None,
        source_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Map a MoRAG document to a Graphiti episode.
        
        Args:
            document: MoRAG document to map
            episode_name: Optional custom episode name
            source_description: Optional source description
            
        Returns:
            Dictionary with episode creation result
        """
        try:
            # Generate episode name if not provided
            if not episode_name:
                episode_name = self._generate_episode_name(document)
            
            # Generate episode content from document
            episode_content = self._generate_episode_content(document)
            
            # Generate source description if not provided
            if not source_description:
                source_description = self._generate_source_description(document)
            
            # Create episode metadata
            episode_metadata = self._generate_episode_metadata(document)
            
            # Connect to Graphiti and create episode
            async with self.connection_service as conn:
                success = await conn.create_episode(
                    name=episode_name,
                    content=episode_content,
                    source_description=source_description,
                    metadata=episode_metadata
                )
                
                if success:
                    logger.info(
                        "Document mapped to episode successfully",
                        document_id=document.id,
                        episode_name=episode_name
                    )
                    return {
                        "success": True,
                        "episode_name": episode_name,
                        "document_id": document.id,
                        "content_length": len(episode_content),
                        "metadata": episode_metadata
                    }
                else:
                    logger.error(
                        "Failed to create episode",
                        document_id=document.id,
                        episode_name=episode_name
                    )
                    return {
                        "success": False,
                        "error": "Failed to create episode in Graphiti",
                        "document_id": document.id
                    }
                    
        except Exception as e:
            logger.error(
                "Document to episode mapping failed",
                document_id=document.id,
                error=str(e)
            )
            return {
                "success": False,
                "error": str(e),
                "document_id": document.id
            }

    async def map_document_chunks_to_contextual_episodes(
        self,
        document: Document,
        chunk_episode_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Map document chunks to contextual Graphiti episodes with rich metadata.

        Args:
            document: MoRAG document with chunks
            chunk_episode_prefix: Optional prefix for chunk episode names

        Returns:
            List of contextual episode creation results
        """
        results = []

        if not document.chunks:
            logger.warning("Document has no chunks to map", document_id=document.id)
            return results

        try:
            # Generate document summary for context
            document_summary = await self._generate_document_summary(document)

            async with self.connection_service as conn:
                for i, chunk in enumerate(document.chunks):
                    try:
                        # Generate contextual summary for this chunk
                        contextual_summary = await self._generate_chunk_contextual_summary(
                            chunk, document, document_summary, i
                        )

                        # Generate episode name for chunk
                        if chunk_episode_prefix:
                            episode_name = f"{chunk_episode_prefix}_chunk_{i+1}"
                        else:
                            episode_name = f"{document.metadata.title or 'document'}_{document.id}_chunk_{i+1}"

                        # Create enhanced episode content
                        enhanced_content = self._create_enhanced_chunk_content(
                            chunk, contextual_summary, document, i
                        )

                        # Create episode with rich metadata
                        success = await conn.create_episode(
                            name=episode_name,
                            content=enhanced_content,
                            source_description=f"Contextual chunk {i+1} from document: {document.metadata.title or document.id}",
                            metadata=self._generate_contextual_chunk_metadata(
                                document, chunk, i, contextual_summary, document_summary
                            )
                        )

                        result = {
                            "success": success,
                            "episode_name": episode_name,
                            "chunk_id": chunk.id,
                            "chunk_index": i,
                            "document_id": document.id,
                            "contextual_summary": contextual_summary,
                            "enhanced_content_length": len(enhanced_content),
                            "context_level": self.context_level.value
                        }

                        if success:
                            logger.debug(
                                "Contextual chunk episode created",
                                episode_name=episode_name,
                                chunk_index=i,
                                context_length=len(contextual_summary)
                            )
                        else:
                            logger.error(
                                "Failed to create contextual chunk episode",
                                episode_name=episode_name,
                                chunk_index=i
                            )

                        results.append(result)

                    except Exception as e:
                        logger.error(
                            "Contextual chunk episode creation failed",
                            chunk_index=i,
                            chunk_id=chunk.id,
                            error=str(e)
                        )
                        results.append({
                            "success": False,
                            "error": str(e),
                            "chunk_id": chunk.id,
                            "chunk_index": i,
                            "document_id": document.id
                        })

        except Exception as e:
            logger.error(
                "Contextual chunk mapping failed",
                document_id=document.id,
                error=str(e)
            )

        return results

    async def map_document_chunks_to_episodes(
        self,
        document: Document,
        chunk_episode_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Map document chunks to individual Graphiti episodes.
        
        Args:
            document: MoRAG document with chunks
            chunk_episode_prefix: Optional prefix for chunk episode names
            
        Returns:
            List of episode creation results
        """
        results = []
        
        if not document.chunks:
            logger.warning("Document has no chunks to map", document_id=document.id)
            return results
        
        try:
            async with self.connection_service as conn:
                for i, chunk in enumerate(document.chunks):
                    try:
                        # Generate episode name for chunk
                        if chunk_episode_prefix:
                            episode_name = f"{chunk_episode_prefix}_chunk_{i+1}"
                        else:
                            episode_name = f"{document.metadata.title or 'document'}_{document.id}_chunk_{i+1}"
                        
                        # Create episode for chunk
                        success = await conn.create_episode(
                            name=episode_name,
                            content=chunk.content,
                            source_description=f"Chunk {i+1} from document: {document.metadata.title or document.id}",
                            metadata=self._generate_chunk_episode_metadata(document, chunk, i)
                        )
                        
                        result = {
                            "success": success,
                            "episode_name": episode_name,
                            "chunk_id": chunk.id,
                            "chunk_index": i,
                            "document_id": document.id
                        }
                        
                        if not success:
                            result["error"] = "Failed to create episode in Graphiti"
                        
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(
                            "Failed to map chunk to episode",
                            document_id=document.id,
                            chunk_index=i,
                            error=str(e)
                        )
                        results.append({
                            "success": False,
                            "error": str(e),
                            "chunk_id": chunk.id,
                            "chunk_index": i,
                            "document_id": document.id
                        })
                        
        except Exception as e:
            logger.error(
                "Document chunks to episodes mapping failed",
                document_id=document.id,
                error=str(e)
            )
            
        return results

    def _generate_episode_name(self, document: Document) -> str:
        """Generate episode name from document.
        
        Args:
            document: MoRAG document
            
        Returns:
            Generated episode name
        """
        # Use document title if available, otherwise use source_name or ID
        if hasattr(document.metadata, 'title') and document.metadata.title:
            base_name = document.metadata.title
        elif hasattr(document.metadata, 'source_name') and document.metadata.source_name:
            base_name = document.metadata.source_name
        else:
            base_name = f"document_{document.id}"
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"

    def _generate_episode_content(self, document: Document) -> str:
        """Generate episode content from document.
        
        Args:
            document: MoRAG document
            
        Returns:
            Generated episode content
        """
        content_parts = []
        
        # Add document metadata as context
        if hasattr(document.metadata, 'title') and document.metadata.title:
            content_parts.append(f"Document Title: {document.metadata.title}")
        
        if hasattr(document.metadata, 'source_name') and document.metadata.source_name:
            content_parts.append(f"Source File: {document.metadata.source_name}")
        
        # Add document summary if available
        if hasattr(document, 'raw_text') and document.raw_text:
            # Use first 1000 characters of raw text as summary
            summary = document.raw_text[:1000]
            if len(document.raw_text) > 1000:
                summary += "..."
            content_parts.append(f"Content Summary: {summary}")
        
        # Add chunk contents
        if document.chunks:
            content_parts.append(f"Document contains {len(document.chunks)} chunks:")
            for i, chunk in enumerate(document.chunks[:5]):  # Limit to first 5 chunks
                chunk_preview = chunk.content[:200]
                if len(chunk.content) > 200:
                    chunk_preview += "..."
                content_parts.append(f"Chunk {i+1}: {chunk_preview}")
            
            if len(document.chunks) > 5:
                content_parts.append(f"... and {len(document.chunks) - 5} more chunks")
        
        return "\n\n".join(content_parts)

    def _generate_source_description(self, document: Document) -> str:
        """Generate source description from document.
        
        Args:
            document: MoRAG document
            
        Returns:
            Generated source description
        """
        parts = ["MoRAG document"]
        
        if hasattr(document.metadata, 'source_type'):
            parts.append(f"type: {document.metadata.source_type}")
        
        if hasattr(document.metadata, 'source_name') and document.metadata.source_name:
            parts.append(f"file: {document.metadata.source_name}")
        
        return " - ".join(parts)

    def _generate_episode_metadata(self, document: Document) -> Dict[str, Any]:
        """Generate episode metadata from document.
        
        Args:
            document: MoRAG document
            
        Returns:
            Generated episode metadata
        """
        metadata = {
            "morag_document_id": document.id,
            "morag_source": "document_episode_mapper",
            "created_at": datetime.now().isoformat(),
            "chunk_count": len(document.chunks) if document.chunks else 0
        }
        
        # Add document metadata
        if hasattr(document, 'metadata') and document.metadata:
            doc_meta = document.metadata
            if hasattr(doc_meta, 'title') and doc_meta.title:
                metadata["document_title"] = doc_meta.title
            if hasattr(doc_meta, 'source_name') and doc_meta.source_name:
                metadata["document_filename"] = doc_meta.source_name
            if hasattr(doc_meta, 'source_type'):
                metadata["document_source_type"] = doc_meta.source_type
            if hasattr(doc_meta, 'file_size') and doc_meta.file_size:
                metadata["document_file_size"] = doc_meta.file_size
        
        # Add processing metadata
        if hasattr(document, 'processed_at') and document.processed_at:
            metadata["document_processed_at"] = document.processed_at.isoformat()
        
        return metadata

    def _generate_chunk_episode_metadata(
        self,
        document: Document,
        chunk: DocumentChunk,
        chunk_index: int
    ) -> Dict[str, Any]:
        """Generate episode metadata for a document chunk.
        
        Args:
            document: Parent MoRAG document
            chunk: Document chunk
            chunk_index: Index of the chunk
            
        Returns:
            Generated episode metadata
        """
        metadata = {
            "morag_document_id": document.id,
            "morag_chunk_id": chunk.id,
            "morag_source": "chunk_episode_mapper",
            "created_at": datetime.now().isoformat(),
            "chunk_index": chunk_index,
            "chunk_length": len(chunk.content)
        }
        
        # Add chunk-specific metadata
        if hasattr(chunk, 'chunk_index'):
            metadata["original_chunk_index"] = chunk.chunk_index
        if hasattr(chunk, 'page_number') and chunk.page_number:
            metadata["page_number"] = chunk.page_number
        if hasattr(chunk, 'section') and chunk.section:
            metadata["section"] = chunk.section
        
        # Add document context
        if hasattr(document.metadata, 'title') and document.metadata.title:
            metadata["document_title"] = document.metadata.title
        if hasattr(document.metadata, 'source_name') and document.metadata.source_name:
            metadata["document_filename"] = document.metadata.source_name
        
        return metadata

    async def _generate_document_summary(self, document: Document) -> str:
        """Generate a comprehensive document summary for context.

        Args:
            document: Document to summarize

        Returns:
            Document summary text
        """
        try:
            # Use AI summarization if available
            if self._summarization_agent and self.enable_ai_summarization:
                # Get full document text
                full_text = document.raw_text
                if not full_text and document.chunks:
                    full_text = "\n\n".join([chunk.content for chunk in document.chunks])

                if full_text:
                    summary_result = await self._summarization_agent.summarize_document(
                        text=full_text,
                        title=getattr(document.metadata, 'title', None),
                        document_type=getattr(document.metadata, 'source_type', None),
                        max_length=500
                    )
                    return summary_result.summary

            # Fallback to basic summary
            return self._generate_basic_document_summary(document)

        except Exception as e:
            logger.warning("AI document summarization failed, using basic summary", error=str(e))
            return self._generate_basic_document_summary(document)

    def _generate_basic_document_summary(self, document: Document) -> str:
        """Generate a basic document summary without AI.

        Args:
            document: Document to summarize

        Returns:
            Basic document summary
        """
        summary_parts = []

        # Add document metadata
        if hasattr(document.metadata, 'title') and document.metadata.title:
            summary_parts.append(f"Document: {document.metadata.title}")

        if hasattr(document.metadata, 'source_type') and document.metadata.source_type:
            summary_parts.append(f"Type: {document.metadata.source_type}")

        # Add content overview
        if document.chunks:
            summary_parts.append(f"Contains {len(document.chunks)} sections")

            # Add preview of first chunk
            if document.chunks[0].content:
                preview = document.chunks[0].content[:200]
                if len(document.chunks[0].content) > 200:
                    preview += "..."
                summary_parts.append(f"Content preview: {preview}")

        return ". ".join(summary_parts) if summary_parts else "Document content available for analysis."

    async def _generate_chunk_contextual_summary(
        self,
        chunk: DocumentChunk,
        document: Document,
        document_summary: str,
        chunk_index: int
    ) -> str:
        """Generate contextual summary for a chunk.

        Args:
            chunk: Chunk to summarize
            document: Parent document
            document_summary: Overall document summary
            chunk_index: Index of the chunk

        Returns:
            Contextual summary for the chunk
        """
        try:
            # Use AI for contextual summary if available
            if self._summarization_agent and self.enable_ai_summarization:
                return await self._generate_ai_chunk_context(
                    chunk, document, document_summary, chunk_index
                )

            # Fallback to basic contextual summary
            return self._generate_basic_chunk_context(
                chunk, document, document_summary, chunk_index
            )

        except Exception as e:
            logger.warning("AI chunk contextualization failed, using basic context", error=str(e))
            return self._generate_basic_chunk_context(
                chunk, document, document_summary, chunk_index
            )

    async def _generate_ai_chunk_context(
        self,
        chunk: DocumentChunk,
        document: Document,
        document_summary: str,
        chunk_index: int
    ) -> str:
        """Generate AI-powered contextual summary for a chunk.

        Args:
            chunk: Chunk to contextualize
            document: Parent document
            document_summary: Overall document summary
            chunk_index: Index of the chunk

        Returns:
            AI-generated contextual summary
        """
        # Build context prompt
        context_parts = [
            f"Document Summary: {document_summary}",
            f"This is section {chunk_index + 1} of {len(document.chunks)} in the document."
        ]

        # Add surrounding chunk context based on context level
        if self.context_level in [ContextLevel.RICH, ContextLevel.COMPREHENSIVE]:
            surrounding_context = self._get_surrounding_chunks_context(
                document.chunks, chunk_index
            )
            if surrounding_context:
                context_parts.append(f"Surrounding context: {surrounding_context}")

        context_prompt = f"""
{' '.join(context_parts)}

Please provide a brief contextual summary (max 300 characters) for the following text chunk,
explaining how it relates to the overall document and what specific information it contains:

Chunk Content:
{chunk.content[:1000]}...
"""

        try:
            summary_result = await self._summarization_agent.summarize_text(
                text=context_prompt,
                max_length=300,
                style="concise",
                context="chunk_contextualization"
            )
            return summary_result.summary
        except Exception as e:
            logger.warning("AI chunk context generation failed", error=str(e))
            return self._generate_basic_chunk_context(chunk, document, document_summary, chunk_index)

    def _generate_basic_chunk_context(
        self,
        chunk: DocumentChunk,
        document: Document,
        document_summary: str,
        chunk_index: int
    ) -> str:
        """Generate basic contextual summary for a chunk.

        Args:
            chunk: Chunk to contextualize
            document: Parent document
            document_summary: Overall document summary
            chunk_index: Index of the chunk

        Returns:
            Basic contextual summary
        """
        context_parts = []

        # Add position context
        total_chunks = len(document.chunks)
        if total_chunks > 1:
            position = "beginning" if chunk_index == 0 else "end" if chunk_index == total_chunks - 1 else "middle"
            context_parts.append(f"Section {chunk_index + 1} of {total_chunks} ({position} of document)")

        # Add section information if available
        if hasattr(chunk, 'section') and chunk.section:
            context_parts.append(f"Section: {chunk.section}")

        # Add content preview
        content_preview = chunk.content[:150]
        if len(chunk.content) > 150:
            content_preview += "..."
        context_parts.append(f"Content: {content_preview}")

        return ". ".join(context_parts)

    def _get_surrounding_chunks_context(
        self,
        chunks: List[DocumentChunk],
        current_index: int,
        window_size: int = 1
    ) -> str:
        """Get context from surrounding chunks.

        Args:
            chunks: All document chunks
            current_index: Index of current chunk
            window_size: Number of chunks before/after to include

        Returns:
            Context from surrounding chunks
        """
        context_parts = []

        # Previous chunks
        for i in range(max(0, current_index - window_size), current_index):
            preview = chunks[i].content[:100]
            if len(chunks[i].content) > 100:
                preview += "..."
            context_parts.append(f"Previous section: {preview}")

        # Next chunks
        for i in range(current_index + 1, min(len(chunks), current_index + window_size + 1)):
            preview = chunks[i].content[:100]
            if len(chunks[i].content) > 100:
                preview += "..."
            context_parts.append(f"Next section: {preview}")

        return ". ".join(context_parts)

    def _create_enhanced_chunk_content(
        self,
        chunk: DocumentChunk,
        contextual_summary: str,
        document: Document,
        chunk_index: int
    ) -> str:
        """Create enhanced content for chunk episode.

        Args:
            chunk: Original chunk
            contextual_summary: Contextual summary
            document: Parent document
            chunk_index: Index of the chunk

        Returns:
            Enhanced content with context
        """
        content_parts = []

        # Add contextual header
        content_parts.append(f"=== CONTEXTUAL SUMMARY ===")
        content_parts.append(contextual_summary)
        content_parts.append("")

        # Add document context if comprehensive
        if self.context_level == ContextLevel.COMPREHENSIVE:
            content_parts.append(f"=== DOCUMENT CONTEXT ===")
            content_parts.append(f"Document: {getattr(document.metadata, 'title', 'Untitled')}")
            content_parts.append(f"Section {chunk_index + 1} of {len(document.chunks)}")
            if hasattr(chunk, 'section') and chunk.section:
                content_parts.append(f"Section Title: {chunk.section}")
            content_parts.append("")

        # Add original content
        content_parts.append(f"=== ORIGINAL CONTENT ===")
        content_parts.append(chunk.content)

        return "\n".join(content_parts)

    def _generate_contextual_chunk_metadata(
        self,
        document: Document,
        chunk: DocumentChunk,
        chunk_index: int,
        contextual_summary: str,
        document_summary: str
    ) -> Dict[str, Any]:
        """Generate rich metadata for contextual chunk episode.

        Args:
            document: Parent document
            chunk: Document chunk
            chunk_index: Index of the chunk
            contextual_summary: Generated contextual summary
            document_summary: Document summary

        Returns:
            Rich metadata dictionary
        """
        metadata = {
            # Core identifiers
            "morag_document_id": document.id,
            "morag_chunk_id": chunk.id,
            "morag_source": "contextual_chunk_episode_mapper",
            "created_at": datetime.now().isoformat(),

            # Chunk information
            "chunk_index": chunk_index,
            "chunk_length": len(chunk.content),
            "total_chunks": len(document.chunks),

            # Context information
            "contextual_summary": contextual_summary,
            "context_level": self.context_level.value,
            "strategy": self.strategy.value,
            "has_ai_context": self.enable_ai_summarization and self._summarization_agent is not None,

            # Document context
            "document_summary": document_summary,
            "document_title": getattr(document.metadata, 'title', None),
            "document_type": getattr(document.metadata, 'source_type', None),
        }

        # Add chunk-specific metadata
        if hasattr(chunk, 'section') and chunk.section:
            metadata["section"] = chunk.section

        if hasattr(chunk, 'page_number') and chunk.page_number:
            metadata["page_number"] = chunk.page_number

        # Add original chunk metadata
        if hasattr(chunk, 'metadata') and chunk.metadata:
            metadata["original_chunk_metadata"] = chunk.metadata

        # Add document metadata
        if hasattr(document.metadata, 'source_name') and document.metadata.source_name:
            metadata["document_filename"] = document.metadata.source_name

        if hasattr(document.metadata, 'mime_type') and document.metadata.mime_type:
            metadata["document_mime_type"] = document.metadata.mime_type

        return metadata


# Convenience functions for creating mapper
def create_episode_mapper(
    config: Optional[GraphitiConfig] = None,
    strategy: EpisodeStrategy = EpisodeStrategy.HYBRID,
    context_level: ContextLevel = ContextLevel.RICH,
    enable_ai_summarization: bool = True
) -> DocumentEpisodeMapper:
    """Create a document episode mapper with specified strategy.

    Args:
        config: Optional Graphiti configuration
        strategy: Episode creation strategy
        context_level: Level of context enrichment
        enable_ai_summarization: Whether to use AI for contextual summaries

    Returns:
        DocumentEpisodeMapper instance
    """
    return DocumentEpisodeMapper(
        config=config,
        strategy=strategy,
        context_level=context_level,
        enable_ai_summarization=enable_ai_summarization
    )


def create_hybrid_episode_mapper(
    config: Optional[GraphitiConfig] = None,
    context_level: ContextLevel = ContextLevel.RICH
) -> DocumentEpisodeMapper:
    """Create a hybrid episode mapper with rich contextual processing.

    Args:
        config: Optional Graphiti configuration
        context_level: Level of context enrichment

    Returns:
        DocumentEpisodeMapper configured for hybrid strategy
    """
    return DocumentEpisodeMapper(
        config=config,
        strategy=EpisodeStrategy.HYBRID,
        context_level=context_level,
        enable_ai_summarization=True
    )


def create_contextual_chunk_mapper(
    config: Optional[GraphitiConfig] = None,
    context_level: ContextLevel = ContextLevel.COMPREHENSIVE
) -> DocumentEpisodeMapper:
    """Create a mapper focused on contextual chunk episodes.

    Args:
        config: Optional Graphiti configuration
        context_level: Level of context enrichment

    Returns:
        DocumentEpisodeMapper configured for contextual chunks
    """
    return DocumentEpisodeMapper(
        config=config,
        strategy=EpisodeStrategy.CONTEXTUAL_CHUNKS,
        context_level=context_level,
        enable_ai_summarization=True
    )
