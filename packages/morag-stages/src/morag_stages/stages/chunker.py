"""Chunker stage implementation."""

import json
import re
from datetime import datetime
from typing import List, Dict, Any, TYPE_CHECKING
from pathlib import Path
import structlog

from ..models import Stage, StageType, StageStatus, StageResult, StageContext, StageMetadata
from ..exceptions import StageExecutionError, StageValidationError

# Import services with graceful fallback
if TYPE_CHECKING:
    from agents import get_agent, SummarizationAgent
    from morag_embedding import GeminiEmbeddingService
    from morag_core.config.unified import ChunkerConfig

try:
    from agents import get_agent as _get_agent, SummarizationAgent as _SummarizationAgent
    from morag_embedding import GeminiEmbeddingService as _GeminiEmbeddingService
    from morag_core.config.unified import ChunkerConfig as _ChunkerConfig
    get_agent = _get_agent
    SummarizationAgent = _SummarizationAgent
    GeminiEmbeddingService = _GeminiEmbeddingService
    ChunkerConfig = _ChunkerConfig
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    get_agent = None  # type: ignore
    SummarizationAgent = None  # type: ignore
    GeminiEmbeddingService = None  # type: ignore
    ChunkerConfig = None  # type: ignore

logger = structlog.get_logger(__name__)


class ChunkerStage(Stage):
    """Stage that creates chunks and embeddings from markdown content."""
    
    def __init__(self, stage_type: StageType = StageType.CHUNKER):
        """Initialize chunker stage."""
        super().__init__(stage_type)
        
        if not SERVICES_AVAILABLE:
            logger.warning("Services not available for chunking")
        
        # Initialize embedding service with API key from environment
        if SERVICES_AVAILABLE and GeminiEmbeddingService is not None:
            import os
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                self.embedding_service = GeminiEmbeddingService(api_key=api_key)
            else:
                logger.warning("GEMINI_API_KEY not found - embedding generation disabled")
                self.embedding_service = None
        else:
            self.embedding_service = None
        self.summarization_agent = None

    def _get_config_value(self, config, key: str, default=None):
        """Get configuration value from either dict or config object."""
        if hasattr(config, key):
            return getattr(config, key)
        elif hasattr(config, 'get'):
            return config.get(key, default)
        else:
            return default
    
    async def execute(self, 
                     input_files: List[Path], 
                     context: StageContext) -> StageResult:
        """Execute chunking on input markdown files.
        
        Args:
            input_files: List of input markdown files
            context: Stage execution context
            
        Returns:
            Stage execution result
        """
        if len(input_files) != 1:
            raise StageValidationError(
                "Chunker stage requires exactly one input file",
                stage_type=self.stage_type.value,
                invalid_files=[str(f) for f in input_files]
            )
        
        input_file = input_files[0]

        # Load configuration from environment variables with context overrides
        context_config = context.get_stage_config(self.stage_type)
        if SERVICES_AVAILABLE and ChunkerConfig:
            config = ChunkerConfig.from_env_and_overrides(context_config)
        else:
            # Fallback to dictionary config
            config = context_config
        
        logger.info("Starting chunking", 
                   input_file=str(input_file),
                   config=config)
        
        try:
            # Generate output filename
            output_file = context.output_dir / f"{input_file.stem}.chunks.json"
            context.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Read input markdown
            markdown_content = input_file.read_text(encoding='utf-8')
            
            # Extract metadata and content
            metadata, content = self._extract_metadata_and_content(markdown_content)
            
            # Generate document summary
            summary = ""
            if self._get_config_value(config, 'generate_summary', True):
                summary = await self._generate_summary(content, metadata, config)
            
            # Create chunks based on strategy
            chunks = await self._create_chunks(content, metadata, config)
            
            # Generate embeddings for chunks
            if self.embedding_service:
                chunks = await self._add_embeddings(chunks, config)
            
            # Add contextual information
            chunks = self._add_context_to_chunks(chunks, config)
            
            # Create output data
            output_data = {
                "summary": summary,
                "chunks": chunks,
                "metadata": {
                    "total_chunks": len(chunks),
                    "chunk_strategy": self._get_config_value(config, 'chunk_strategy', 'semantic'),
                    "chunk_size": self._get_config_value(config, 'chunk_size', 4000),
                    "overlap": self._get_config_value(config, 'overlap', 200),
                    "embedding_model": self._get_config_value(config, 'embedding_model', 'text-embedding-004'),
                    "source_file": str(input_file),
                    "source_metadata": metadata,
                    "created_at": datetime.now().isoformat()
                }
            }
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Create metadata
            # Convert config to dict if it's a config object
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config

            stage_metadata = StageMetadata(
                execution_time=0.0,  # Will be set by manager
                start_time=datetime.now(),
                input_files=[str(input_file)],
                output_files=[str(output_file)],
                config_used=config_dict,
                metrics={
                    "total_chunks": len(chunks),
                    "average_chunk_size": sum(len(chunk['content']) for chunk in chunks) / len(chunks) if chunks else 0,
                    "content_length": len(content),
                    "has_summary": bool(summary),
                    "has_embeddings": any('embedding' in chunk for chunk in chunks)
                }
            )
            
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file],
                metadata=stage_metadata,
                data={
                    "total_chunks": len(chunks),
                    "summary_length": len(summary),
                    "chunk_strategy": self._get_config_value(config, 'chunk_strategy', 'semantic')
                }
            )
            
        except Exception as e:
            logger.error("Chunking failed", 
                        input_file=str(input_file), 
                        error=str(e))
            raise StageExecutionError(
                f"Chunking failed: {e}",
                stage_type=self.stage_type.value,
                original_error=e
            )
    
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for chunking.
        
        Args:
            input_files: List of input file paths
            
        Returns:
            True if inputs are valid
        """
        if len(input_files) != 1:
            return False
        
        input_file = input_files[0]
        
        # Check if file exists and is markdown
        if not input_file.exists():
            return False
        
        if input_file.suffix.lower() not in ['.md', '.markdown']:
            return False
        
        return True
    
    def get_dependencies(self) -> List[StageType]:
        """Get stage dependencies.
        
        Returns:
            List containing markdown-conversion stage (markdown-optimizer is optional)
        """
        return [StageType.MARKDOWN_CONVERSION]
    
    def get_expected_outputs(self, input_files: List[Path], context: StageContext) -> List[Path]:
        """Get expected output file paths.

        Args:
            input_files: List of input file paths
            context: Stage execution context

        Returns:
            List of expected output file paths
        """
        if len(input_files) != 1:
            return []

        input_file = input_files[0]
        from ..file_manager import sanitize_filename
        sanitized_name = sanitize_filename(input_file.stem)
        output_file = context.output_dir / f"{sanitized_name}.chunks.json"
        return [output_file]
    
    def _extract_metadata_and_content(self, markdown: str) -> tuple[Dict[str, Any], str]:
        """Extract metadata and content from markdown (supports both YAML frontmatter and H1+H2 format).

        Args:
            markdown: Full markdown content

        Returns:
            Tuple of (metadata dict, content string)
        """
        # Check for YAML frontmatter (legacy format)
        if markdown.startswith('---\n'):
            parts = markdown.split('---\n', 2)
            if len(parts) >= 3:
                yaml_content = parts[1]
                content = parts[2]

                # Parse YAML metadata (simple parsing)
                metadata = {}
                for line in yaml_content.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        metadata[key] = value

                return metadata, content

        # Check for new H1+H2 format
        lines = markdown.split('\n')
        if lines and lines[0].startswith('# '):
            # Extract title from H1
            title_line = lines[0][2:].strip()
            metadata = {'title': title_line}

            # Look for information section
            info_section_start = -1
            content_section_start = -1

            for i, line in enumerate(lines):
                if line.startswith('## ') and 'Information' in line:
                    info_section_start = i
                elif line.startswith('## ') and ('Content' in line or 'Transcript' in line):
                    content_section_start = i
                    break

            if info_section_start > 0:
                # Extract metadata from information section
                for i in range(info_section_start + 1, len(lines)):
                    line = lines[i].strip()
                    if line.startswith('- **') and '**:' in line:
                        # Parse metadata line: - **Key**: Value
                        key_end = line.find('**:', 4)
                        if key_end > 4:
                            key = line[4:key_end].strip()
                            value = line[key_end + 3:].strip()
                            metadata[key.lower().replace(' ', '_')] = value
                    elif line.startswith('## '):
                        # End of information section
                        break

            # Extract content from content/transcript section
            if content_section_start > 0:
                content_lines = lines[content_section_start + 2:]  # Skip section header and empty line
                content = '\n'.join(content_lines)
            else:
                content = markdown  # Fallback to full content

            return metadata, content

        return {}, markdown
    
    async def _generate_summary(self, content: str, metadata: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate document summary using LLM.

        Args:
            content: Document content
            metadata: Document metadata
            config: Stage configuration

        Returns:
            Document summary
        """
        if not SERVICES_AVAILABLE or get_agent is None:
            return ""

        # Check if API key is available before attempting to create agent
        import os
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.info("No API key available for summary generation - skipping")
            return ""

        try:
            if not self.summarization_agent:
                # Use the agents framework to get a summarization agent
                self.summarization_agent = get_agent("summarization")
            
            # Create summarization using the agents framework
            content_type = metadata.get('type', 'text')

            response = await self.summarization_agent.summarize(
                text=content[:8000],  # Limit content length
                content_type=content_type
            )
            # The agents framework returns a SummarizationResult
            return response.summary if hasattr(response, 'summary') else str(response)
            
        except Exception as e:
            logger.warning("Summary generation failed", error=str(e))
            return ""
    
    async def _create_chunks(self, content: str, metadata: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from content based on strategy.
        
        Args:
            content: Document content
            metadata: Document metadata
            config: Stage configuration
            
        Returns:
            List of chunk dictionaries
        """
        strategy = self._get_config_value(config, 'chunk_strategy', 'semantic')
        chunk_size = self._get_config_value(config, 'chunk_size', 4000)
        overlap = self._get_config_value(config, 'overlap', 200)
        
        if strategy == 'semantic':
            return await self._semantic_chunking(content, metadata, chunk_size, overlap)
        elif strategy == 'page-level':
            return self._page_level_chunking(content, metadata, chunk_size)
        elif strategy == 'topic-based':
            return self._topic_based_chunking(content, metadata, chunk_size)
        else:  # fixed-size
            return self._fixed_size_chunking(content, chunk_size, overlap)
    
    def _fixed_size_chunking(self, content: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Create fixed-size chunks with overlap.
        
        Args:
            content: Content to chunk
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        chunk_id = 1
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # Try to break at word boundary
            if end < len(content):
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append({
                    "id": f"chunk_{chunk_id:03d}",
                    "content": chunk_content,
                    "metadata": {
                        "start_char": start,
                        "end_char": end,
                        "chunk_size": len(chunk_content)
                    }
                })
                chunk_id += 1
            
            start = max(start + 1, end - overlap)
        
        return chunks

    async def _semantic_chunking(self, content: str, metadata: Dict[str, Any], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Create semantic chunks based on content structure.

        Args:
            content: Content to chunk
            metadata: Document metadata
            chunk_size: Target chunk size
            overlap: Overlap between chunks

        Returns:
            List of chunk dictionaries
        """
        # For now, use paragraph-based chunking as a semantic approximation
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""
        chunk_id = 1

        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > chunk_size:
                chunks.append({
                    "id": f"chunk_{chunk_id:03d}",
                    "content": current_chunk.strip(),
                    "metadata": {
                        "chunk_type": "semantic",
                        "chunk_size": len(current_chunk.strip()),
                        "has_context": bool(overlap > 0)
                    }
                })
                chunk_id += 1

                # Start new chunk with overlap
                if overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-overlap // 10:]  # Approximate word overlap
                    current_chunk = " ".join(overlap_words) + " " + paragraph
                else:
                    current_chunk = paragraph

                # If the paragraph itself is too large, split it further
                if len(current_chunk) > chunk_size:
                    # Split large paragraph into smaller chunks
                    large_chunks = self._split_large_content(current_chunk, chunk_size, overlap)
                    for large_chunk in large_chunks:
                        chunks.append({
                            "id": f"chunk_{chunk_id:03d}",
                            "content": large_chunk.strip(),
                            "metadata": {
                                "chunk_type": "semantic",
                                "chunk_size": len(large_chunk.strip()),
                                "has_context": bool(overlap > 0)
                            }
                        })
                        chunk_id += 1
                    current_chunk = ""
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk.strip():
            # Check if final chunk is too large and split if necessary
            if len(current_chunk.strip()) > chunk_size:
                large_chunks = self._split_large_content(current_chunk.strip(), chunk_size, overlap)
                for large_chunk in large_chunks:
                    chunks.append({
                        "id": f"chunk_{chunk_id:03d}",
                        "content": large_chunk.strip(),
                        "metadata": {
                            "chunk_type": "semantic",
                            "chunk_size": len(large_chunk.strip()),
                            "has_context": bool(overlap > 0)
                        }
                    })
                    chunk_id += 1
            else:
                chunks.append({
                    "id": f"chunk_{chunk_id:03d}",
                    "content": current_chunk.strip(),
                    "metadata": {
                        "chunk_type": "semantic",
                        "chunk_size": len(current_chunk.strip()),
                        "has_context": bool(overlap > 0)
                    }
                })

        return chunks

    def _split_large_content(self, content: str, chunk_size: int, overlap: int) -> List[str]:
        """Split large content into smaller chunks respecting the size limit.

        Args:
            content: Content to split
            chunk_size: Maximum chunk size
            overlap: Overlap between chunks

        Returns:
            List of content chunks
        """
        if len(content) <= chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))

            # Try to break at sentence boundary first
            if end < len(content):
                # Look for sentence endings within the last 200 characters
                sentence_break = content.rfind('.', start, end)
                if sentence_break > start + chunk_size // 2:  # Only if it's not too early
                    end = sentence_break + 1
                else:
                    # Fall back to word boundary
                    word_break = content.rfind(' ', start, end)
                    if word_break > start:
                        end = word_break

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)

            # Move start position with overlap consideration
            start = max(start + 1, end - overlap)

        return chunks

    def _page_level_chunking(self, content: str, metadata: Dict[str, Any], chunk_size: int) -> List[Dict[str, Any]]:
        """Create page-level chunks for documents.

        Args:
            content: Content to chunk
            metadata: Document metadata
            chunk_size: Target chunk size

        Returns:
            List of chunk dictionaries
        """
        # Look for page markers or use fixed-size as fallback
        page_pattern = r'\[Page \d+\]|\f|<!-- page \d+ -->'
        page_breaks = list(re.finditer(page_pattern, content, re.IGNORECASE))

        if page_breaks:
            chunks = []
            chunk_id = 1
            start = 0

            for i, page_break in enumerate(page_breaks):
                page_content = content[start:page_break.start()].strip()
                if page_content:
                    chunks.append({
                        "id": f"chunk_{chunk_id:03d}",
                        "content": page_content,
                        "metadata": {
                            "chunk_type": "page",
                            "page": i + 1,
                            "chunk_size": len(page_content)
                        }
                    })
                    chunk_id += 1
                start = page_break.end()

            # Add final page
            final_content = content[start:].strip()
            if final_content:
                chunks.append({
                    "id": f"chunk_{chunk_id:03d}",
                    "content": final_content,
                    "metadata": {
                        "chunk_type": "page",
                        "page": len(page_breaks) + 1,
                        "chunk_size": len(final_content)
                    }
                })

            return chunks
        else:
            # Fallback to fixed-size chunking
            return self._fixed_size_chunking(content, chunk_size, 200)

    def _topic_based_chunking(self, content: str, metadata: Dict[str, Any], chunk_size: int) -> List[Dict[str, Any]]:
        """Create topic-based chunks for audio/video content.

        Args:
            content: Content to chunk
            metadata: Document metadata
            chunk_size: Target chunk size

        Returns:
            List of chunk dictionaries
        """
        # Look for timestamp patterns and topic headers
        timestamp_pattern = r'\[(\d{2}:\d{2}(?::\d{2})?)\s*-\s*(\d{2}:\d{2}(?::\d{2})?)\]'
        topic_pattern = r'^#+\s+(.+)$'

        lines = content.split('\n')
        chunks = []
        current_chunk = ""
        current_topic = None
        current_start_time = None
        current_end_time = None
        chunk_id = 1

        for line in lines:
            # Check for topic headers
            topic_match = re.match(topic_pattern, line.strip())
            if topic_match:
                # Finalize previous chunk
                if current_chunk.strip():
                    chunks.append({
                        "id": f"chunk_{chunk_id:03d}",
                        "content": current_chunk.strip(),
                        "metadata": {
                            "chunk_type": "topic",
                            "topic": current_topic,
                            "start_time": current_start_time,
                            "end_time": current_end_time,
                            "chunk_size": len(current_chunk.strip())
                        }
                    })
                    chunk_id += 1

                # Start new chunk
                current_topic = topic_match.group(1)
                current_chunk = line
                current_start_time = None
                current_end_time = None
                continue

            # Check for timestamps
            timestamp_match = re.search(timestamp_pattern, line)
            if timestamp_match:
                if current_start_time is None:
                    current_start_time = timestamp_match.group(1)
                current_end_time = timestamp_match.group(2)

            # Add line to current chunk
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line

            # Check if chunk is getting too large
            if len(current_chunk) > chunk_size:
                chunks.append({
                    "id": f"chunk_{chunk_id:03d}",
                    "content": current_chunk.strip(),
                    "metadata": {
                        "chunk_type": "topic",
                        "topic": current_topic,
                        "start_time": current_start_time,
                        "end_time": current_end_time,
                        "chunk_size": len(current_chunk.strip())
                    }
                })
                chunk_id += 1
                current_chunk = ""
                current_start_time = None
                current_end_time = None

        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "id": f"chunk_{chunk_id:03d}",
                "content": current_chunk.strip(),
                "metadata": {
                    "chunk_type": "topic",
                    "topic": current_topic,
                    "start_time": current_start_time,
                    "end_time": current_end_time,
                    "chunk_size": len(current_chunk.strip())
                }
            })

        return chunks

    async def _add_embeddings(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add embeddings to chunks.

        Args:
            chunks: List of chunk dictionaries
            config: Stage configuration

        Returns:
            Chunks with embeddings added
        """
        if not self.embedding_service:
            return chunks

        try:
            # Extract content for embedding
            texts = [chunk['content'] for chunk in chunks]

            # Generate embeddings in batches
            batch_size = self._get_config_value(config, 'batch_size', 50)
            model = self._get_config_value(config, 'embedding_model', 'text-embedding-004')

            # Process in batches for better performance
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.debug("Generating embeddings for batch",
                           batch_start=i, batch_size=len(batch_texts), model=model)

                # Use the correct method name for batch embeddings
                batch_result = await self.embedding_service.generate_batch_embeddings(batch_texts)
                all_embeddings.extend(batch_result.embeddings)

            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                if i < len(all_embeddings):
                    chunk['embedding'] = all_embeddings[i]

            logger.info("Generated embeddings for chunks",
                       total_chunks=len(chunks),
                       total_embeddings=len(all_embeddings),
                       batch_size=batch_size,
                       model=model)

            return chunks

        except Exception as e:
            logger.warning("Failed to generate embeddings", error=str(e))
            return chunks

    def _add_context_to_chunks(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add contextual information to chunks.

        Args:
            chunks: List of chunk dictionaries
            config: Stage configuration

        Returns:
            Chunks with context added
        """
        if not self._get_config_value(config, 'include_context', True):
            return chunks

        context_window = self._get_config_value(config, 'context_window', 2)

        for i, chunk in enumerate(chunks):
            # Add surrounding context
            context_before = []
            context_after = []

            # Get context before
            for j in range(max(0, i - context_window), i):
                context_before.append(chunks[j]['content'][:200])  # First 200 chars

            # Get context after
            for j in range(i + 1, min(len(chunks), i + context_window + 1)):
                context_after.append(chunks[j]['content'][:200])  # First 200 chars

            # Create context summary
            context_summary = ""
            if context_before:
                context_summary += "Previous context: " + " ... ".join(context_before) + " ... "
            if context_after:
                context_summary += "Following context: " + " ... ".join(context_after)

            chunk['context_summary'] = context_summary.strip()
            chunk['metadata']['has_context'] = bool(context_summary)

        return chunks
