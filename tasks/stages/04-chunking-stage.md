# Task 4: Implement chunker Stage

## Overview
Implement the chunker stage that creates summary, chunks, and contextual embeddings from markdown content. This completely replaces all existing chunking logic.

## Objectives
- Create document summary generation with contextual understanding
- Implement configurable chunking strategies (semantic, page-level, topic-based)
- **Generate contextual embeddings for each chunk with surrounding context**
- **Create contextual summaries that understand chunk relationships**
- Add chunk metadata with source references
- Exclude metadata from chunking process
- **REMOVE ALL LEGACY CHUNKING CODE**

## Deliverables

### 1. chunker Stage Implementation (Complete Replacement)
```python
from morag_stages.models import Stage, StageType, StageResult, StageContext, StageStatus
from morag_services import GeminiEmbeddingService, GeminiLLMService
from morag_chunking import ChunkingService, ChunkingStrategy
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import yaml
import json

class ChunkerStage(Stage):
    def __init__(self,
                 embedding_service: GeminiEmbeddingService,
                 llm_service: GeminiLLMService,
                 chunking_service: ChunkingService):
        super().__init__(StageType.CHUNKER)
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.chunking_service = chunking_service
    
    async def execute(self, 
                     input_files: List[Path], 
                     context: StageContext) -> StageResult:
        """Create chunks and embeddings from markdown content."""
        start_time = time.time()
        
        try:
            if len(input_files) != 1:
                raise ValueError("Stage 3 requires exactly one markdown input file")
            
            input_file = input_files[0]
            
            # Parse markdown and separate metadata from content
            content, metadata = self._parse_markdown_file(input_file)
            
            # Generate document summary
            config = context.config.get('stage3', {})
            summary = await self._generate_summary(content, metadata, config)
            
            # Create chunks (excluding metadata)
            chunks = await self._create_chunks(content, metadata, config)
            
            # Generate contextual embeddings for chunks
            chunk_embeddings = await self._generate_chunk_embeddings(chunks, config)

            # Generate contextual summaries for chunks
            contextual_summaries = await self._generate_contextual_summaries(chunks, config)

            # Create chunk data structure
            chunk_data = self._create_chunk_data(
                chunks, chunk_embeddings, contextual_summaries, summary, metadata, input_file
            )
            
            # Generate output file
            output_file = self._generate_chunks_output(
                chunk_data, input_file, context
            )
            
            # Create chunking report
            report_file = self._create_chunking_report(
                chunk_data, input_file, context
            )
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file, report_file],
                metadata={
                    "chunk_count": len(chunks),
                    "total_tokens": sum(chunk.get('token_count', 0) for chunk in chunks),
                    "chunking_strategy": config.get('chunk_strategy', 'semantic'),
                    "embedding_model": config.get('embedding_model', 'text-embedding-004'),
                    "summary_length": len(summary.split()) if summary else 0,
                    "processing_time": execution_time
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for Stage 3."""
        if len(input_files) != 1:
            return False
        
        input_file = input_files[0]
        
        # Check if file exists and is markdown
        return input_file.exists() and input_file.suffix in ['.md']
    
    def get_dependencies(self) -> List[StageType]:
        """chunker depends on markdown-conversion (can use optimized or original)."""
        return [StageType.MARKDOWN_CONVERSION]
    
    def _parse_markdown_file(self, input_file: Path) -> tuple[str, Dict[str, Any]]:
        """Parse markdown file and separate metadata from content."""
        content = input_file.read_text(encoding='utf-8')
        
        # Extract YAML frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata = yaml.safe_load(parts[1])
                content = parts[2].strip()  # Content without metadata
            else:
                metadata = {}
        else:
            metadata = {}
        
        return content, metadata
    
    async def _generate_summary(self, 
                               content: str, 
                               metadata: Dict[str, Any], 
                               config: Dict[str, Any]) -> Optional[str]:
        """Generate document summary using LLM."""
        if not config.get('generate_summary', True):
            return None
        
        # Prepare summary prompt
        prompt = f"""Please provide a comprehensive summary of the following content. 
        Focus on the main topics, key points, and important information.
        
        Content Type: {metadata.get('content_type', 'unknown')}
        Title: {metadata.get('title', 'Unknown')}
        
        Content:
        {content[:8000]}  # Limit content for summary
        
        Provide a summary that captures the essence and main points of this content."""
        
        try:
            response = await self.llm_service.generate_text(
                prompt=prompt,
                model=config.get('summary_model', 'gemini-pro'),
                temperature=0.3,
                max_tokens=1000
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return None
    
    async def _create_chunks(self, 
                            content: str, 
                            metadata: Dict[str, Any], 
                            config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from content using specified strategy."""
        
        strategy = config.get('chunk_strategy', 'semantic')
        chunk_size = config.get('chunk_size', 4000)
        chunk_overlap = config.get('chunk_overlap', 200)
        
        # Determine chunking strategy based on content type
        content_type = metadata.get('content_type', 'unknown')
        
        if content_type in ['video', 'audio'] and strategy == 'semantic':
            # Use topic-based chunking for audio/video with timestamps
            return await self._chunk_by_topics(content, metadata, config)
        elif content_type == 'document' and strategy == 'semantic':
            # Use page/chapter-based chunking for documents
            return await self._chunk_by_pages(content, metadata, config)
        else:
            # Use semantic chunking for all other cases
            return await self._chunk_semantically(content, metadata, config)
    
    async def _chunk_by_topics(self, 
                              content: str, 
                              metadata: Dict[str, Any], 
                              config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk audio/video content by topics with timestamps."""
        chunks = []
        
        # Split by timestamp patterns
        import re
        timestamp_pattern = r'\[(\d{2}:\d{2}(?::\d{2})?(?:\s*-\s*\d{2}:\d{2}(?::\d{2})?)?)\]'
        
        # Find all timestamp sections
        sections = re.split(timestamp_pattern, content)
        
        current_chunk = ""
        current_timestamp = None
        chunk_index = 0
        
        for i, section in enumerate(sections):
            if i % 2 == 1:  # This is a timestamp
                current_timestamp = section
            else:  # This is content
                if section.strip():
                    chunk_content = f"[{current_timestamp}] {section.strip()}" if current_timestamp else section.strip()
                    
                    if len(current_chunk) + len(chunk_content) > config.get('chunk_size', 4000):
                        # Save current chunk
                        if current_chunk:
                            chunks.append(self._create_chunk_object(
                                current_chunk, chunk_index, metadata, current_timestamp
                            ))
                            chunk_index += 1
                        current_chunk = chunk_content
                    else:
                        current_chunk += "\n\n" + chunk_content if current_chunk else chunk_content
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_object(
                current_chunk, chunk_index, metadata, current_timestamp
            ))
        
        return chunks
    
    async def _chunk_by_pages(self, 
                             content: str, 
                             metadata: Dict[str, Any], 
                             config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk document content by pages/chapters."""
        chunks = []
        
        # Split by page/chapter markers
        import re
        page_pattern = r'(?:^|\n)(?:Page \d+|Chapter \d+|# .+)(?:\n|$)'
        
        sections = re.split(page_pattern, content, flags=re.MULTILINE)
        
        chunk_index = 0
        for section in sections:
            if section.strip():
                # Further split large sections
                if len(section) > config.get('chunk_size', 4000):
                    sub_chunks = self._split_large_section(section, config)
                    for sub_chunk in sub_chunks:
                        chunks.append(self._create_chunk_object(
                            sub_chunk, chunk_index, metadata
                        ))
                        chunk_index += 1
                else:
                    chunks.append(self._create_chunk_object(
                        section.strip(), chunk_index, metadata
                    ))
                    chunk_index += 1
        
        return chunks
    
    async def _chunk_semantically(self, 
                                 content: str, 
                                 metadata: Dict[str, Any], 
                                 config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk content semantically using the chunking service."""
        
        # Use existing chunking service
        chunk_strategy = ChunkingStrategy(
            strategy=config.get('chunk_strategy', 'semantic'),
            chunk_size=config.get('chunk_size', 4000),
            chunk_overlap=config.get('chunk_overlap', 200)
        )
        
        raw_chunks = await self.chunking_service.chunk_text(content, chunk_strategy)
        
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append(self._create_chunk_object(
                chunk_text, i, metadata
            ))
        
        return chunks
    
    def _create_chunk_object(self, 
                            content: str, 
                            index: int, 
                            metadata: Dict[str, Any], 
                            timestamp: Optional[str] = None) -> Dict[str, Any]:
        """Create standardized chunk object."""
        return {
            'index': index,
            'content': content,
            'token_count': len(content.split()),  # Approximate token count
            'character_count': len(content),
            'timestamp': timestamp,
            'source_metadata': {
                'title': metadata.get('title'),
                'content_type': metadata.get('content_type'),
                'source_file': metadata.get('source_file'),
                'language': metadata.get('language')
            }
        }
    
    async def _generate_chunk_embeddings(self,
                                        chunks: List[Dict[str, Any]],
                                        config: Dict[str, Any]) -> List[List[float]]:
        """Generate contextual embeddings for all chunks with surrounding context."""
        embeddings = []

        # Batch process embeddings for efficiency
        batch_size = config.get('embedding_batch_size', 50)
        contextual_window = config.get('contextual_embedding_window', 2)

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]

            # Create contextual texts that include surrounding chunks
            batch_texts = []
            for j, chunk in enumerate(batch_chunks):
                actual_index = i + j
                contextual_text = self._create_contextual_text(chunks, actual_index, contextual_window)
                batch_texts.append(contextual_text)

            try:
                batch_embeddings = await self.embedding_service.generate_embeddings_batch(
                    batch_texts,
                    model=config.get('embedding_model', 'text-embedding-004')
                )
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i}: {e}")
                # Add empty embeddings as fallback
                embeddings.extend([[] for _ in batch_chunks])

        return embeddings

    def _create_contextual_text(self,
                               chunks: List[Dict[str, Any]],
                               chunk_index: int,
                               window_size: int) -> str:
        """Create contextual text including surrounding chunks for better embeddings."""

        # Get surrounding chunks
        start_idx = max(0, chunk_index - window_size)
        end_idx = min(len(chunks), chunk_index + window_size + 1)

        contextual_parts = []

        for i in range(start_idx, end_idx):
            chunk = chunks[i]
            if i == chunk_index:
                # Mark the main chunk
                contextual_parts.append(f"[MAIN CHUNK] {chunk['content']}")
            else:
                # Add context chunks
                contextual_parts.append(f"[CONTEXT] {chunk['content'][:200]}...")  # Truncate context

        return "\n\n".join(contextual_parts)

    async def _generate_contextual_summaries(self,
                                           chunks: List[Dict[str, Any]],
                                           config: Dict[str, Any]) -> List[str]:
        """Generate contextual summaries for each chunk understanding its relationships."""

        if not config.get('generate_contextual_summaries', True):
            return ["" for _ in chunks]

        summaries = []
        contextual_window = config.get('contextual_embedding_window', 2)

        for i, chunk in enumerate(chunks):
            try:
                # Create contextual prompt
                contextual_text = self._create_contextual_text(chunks, i, contextual_window)

                system_message = """You are an expert at creating contextual summaries. Create a brief summary of the main chunk that explains its role and relationship to the surrounding context."""

                user_message = f"""Please create a contextual summary for the main chunk, considering its relationship to the surrounding context:

{contextual_text}

Focus on:
1. What the main chunk is about
2. How it relates to the surrounding content
3. Its role in the overall narrative/document
4. Key concepts and their connections

Provide a concise 2-3 sentence summary."""

                summary = await self.llm_service.generate_text_with_system_message(
                    system_message=system_message,
                    user_message=user_message,
                    model=config.get('summary_model', 'gemini-pro'),
                    temperature=0.3,
                    max_tokens=200
                )

                summaries.append(summary.strip())

            except Exception as e:
                logger.warning(f"Failed to generate contextual summary for chunk {i}: {e}")
                summaries.append("")

        return summaries
    
    def _create_chunk_data(self,
                          chunks: List[Dict[str, Any]],
                          embeddings: List[List[float]],
                          contextual_summaries: List[str],
                          summary: Optional[str],
                          metadata: Dict[str, Any],
                          input_file: Path) -> Dict[str, Any]:
        """Create comprehensive chunk data structure with contextual information."""

        # Combine chunks with embeddings and contextual summaries
        for i, chunk in enumerate(chunks):
            if i < len(embeddings):
                chunk['embedding'] = embeddings[i]
            else:
                chunk['embedding'] = []

            if i < len(contextual_summaries):
                chunk['contextual_summary'] = contextual_summaries[i]
            else:
                chunk['contextual_summary'] = ""

        return {
            'document_metadata': metadata,
            'source_file': str(input_file),
            'summary': summary,
            'chunk_count': len(chunks),
            'total_tokens': sum(chunk.get('token_count', 0) for chunk in chunks),
            'chunks': chunks,
            'processing_info': {
                'stage': 'chunker',
                'stage_name': 'chunker',
                'processed_at': time.time(),
                'embedding_model': metadata.get('embedding_model', 'text-embedding-004'),
                'contextual_embeddings': True,
                'contextual_summaries': len([s for s in contextual_summaries if s])
            }
        }
```

## Implementation Steps

1. **Create chunker package structure**
2. **Implement ChunkerStage class with contextual capabilities**
3. **Add summary generation with LLM**
4. **Implement multiple chunking strategies**
5. **Add contextual embedding generation with surrounding context**
6. **Implement contextual summary generation for each chunk**
7. **Create chunk metadata and source tracking with contextual information**
8. **Add configuration support**
9. **REMOVE ALL LEGACY CHUNKING CODE**
10. **Implement validation, error handling, comprehensive testing, and performance monitoring**

## Testing Requirements

- Unit tests for chunking logic with contextual features
- Contextual embedding generation tests
- Contextual summary generation validation
- Different chunking strategy tests
- Performance tests for large documents with contextual processing
- Error handling and edge cases

## Files to Create

- `packages/morag-stages/src/morag_stages/chunker/__init__.py`
- `packages/morag-stages/src/morag_stages/chunker/implementation.py`
- `packages/morag-stages/src/morag_stages/chunker/chunking_strategies.py`
- `packages/morag-stages/src/morag_stages/chunker/contextual_processing.py`
- `packages/morag-stages/tests/test_chunker.py`

## Success Criteria

- Chunks are created with appropriate size and overlap
- Contextual embeddings include surrounding chunk information
- Contextual summaries understand chunk relationships
- Embeddings are generated efficiently in batches with context
- Summary captures key document information
- Metadata is preserved and excluded from chunking
- **ALL LEGACY CHUNKING CODE IS REMOVED**
- Performance is acceptable for large documents with contextual processing
- All tests pass with good coverage
