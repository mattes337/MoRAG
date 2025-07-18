# Step 2: Document to Episode Mapping

**Duration**: 3-4 days  
**Phase**: Proof of Concept  
**Prerequisites**: Step 1 completed, working Graphiti connection

## Objective

Create an adapter layer that converts MoRAG's Document and DocumentChunk models into Graphiti episodes, enabling seamless integration while preserving all document metadata and relationships.

## Deliverables

1. Document-to-Episode adapter classes
2. Chunk-to-Episode conversion logic
3. Metadata preservation and mapping
4. Entity and relation extraction integration
5. Comprehensive test suite with validation

## Implementation

### 1. Create Document Episode Adapter

**File**: `packages/morag-graph/src/morag_graph/graphiti/adapters.py`

```python
"""Adapters for converting MoRAG models to Graphiti episodes."""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from graphiti_core.nodes import EpisodeType
from morag_graph.models import Document, DocumentChunk, Entity, Relation

logger = logging.getLogger(__name__)


class DocumentEpisodeAdapter:
    """Converts MoRAG Documents to Graphiti episodes."""
    
    def __init__(self):
        self.episode_type_mapping = {
            'text/plain': EpisodeType.text,
            'text/markdown': EpisodeType.text,
            'application/pdf': EpisodeType.text,
            'text/html': EpisodeType.text,
            'application/json': EpisodeType.json,
            'default': EpisodeType.text
        }
    
    def document_to_episode(
        self, 
        document: Document, 
        chunks: List[DocumentChunk],
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None
    ) -> Dict[str, Any]:
        """Convert a Document with chunks to a Graphiti episode.
        
        Args:
            document: MoRAG Document instance
            chunks: List of DocumentChunk instances
            entities: Optional list of extracted entities
            relations: Optional list of extracted relations
            
        Returns:
            Dictionary with episode data for Graphiti
        """
        # Determine episode type based on MIME type
        episode_type = self.episode_type_mapping.get(
            document.mime_type, 
            self.episode_type_mapping['default']
        )
        
        # Combine all chunk text for episode body
        episode_body = self._combine_chunks_to_body(chunks)
        
        # Create comprehensive metadata
        metadata = self._create_episode_metadata(document, chunks, entities, relations)
        
        # Generate episode name
        episode_name = self._generate_episode_name(document)
        
        # Create source description
        source_description = self._create_source_description(document)
        
        return {
            'name': episode_name,
            'episode_body': episode_body,
            'source_description': source_description,
            'episode_type': episode_type,
            'metadata': metadata
        }
    
    def _combine_chunks_to_body(self, chunks: List[DocumentChunk]) -> str:
        """Combine document chunks into a single episode body.
        
        Args:
            chunks: List of DocumentChunk instances
            
        Returns:
            Combined text with chunk boundaries marked
        """
        if not chunks:
            return ""
        
        # Sort chunks by index to maintain order
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)
        
        body_parts = []
        for chunk in sorted_chunks:
            # Add chunk boundary marker for later reference
            chunk_marker = f"\n--- CHUNK {chunk.chunk_index} (ID: {chunk.id}) ---\n"
            body_parts.append(chunk_marker + chunk.text)
        
        return "\n\n".join(body_parts)
    
    def _create_episode_metadata(
        self, 
        document: Document, 
        chunks: List[DocumentChunk],
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for the episode.
        
        Args:
            document: Document instance
            chunks: List of chunks
            entities: Optional entities
            relations: Optional relations
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            # Document metadata
            'morag_document_id': document.id,
            'source_file': document.source_file,
            'file_name': document.file_name,
            'file_size': document.file_size,
            'checksum': document.checksum,
            'mime_type': document.mime_type,
            'ingestion_timestamp': document.ingestion_timestamp.isoformat() if document.ingestion_timestamp else None,
            'last_modified': document.last_modified.isoformat() if document.last_modified else None,
            'model': document.model,
            'summary': document.summary,
            'original_metadata': document.metadata,
            
            # Chunk information
            'chunk_count': len(chunks),
            'chunk_ids': [chunk.id for chunk in chunks],
            'chunk_mapping': {
                chunk.id: {
                    'index': chunk.chunk_index,
                    'text_length': len(chunk.text),
                    'metadata': chunk.metadata
                }
                for chunk in chunks
            },
            
            # Entity and relation counts
            'entity_count': len(entities) if entities else 0,
            'relation_count': len(relations) if relations else 0,
            
            # Processing metadata
            'adapter_version': '1.0',
            'conversion_timestamp': datetime.now().isoformat(),
            'morag_integration': True
        }
        
        # Add entity information if available
        if entities:
            metadata['entities'] = [
                {
                    'id': entity.id,
                    'name': entity.name,
                    'type': str(entity.type),
                    'confidence': entity.confidence
                }
                for entity in entities
            ]
        
        # Add relation information if available
        if relations:
            metadata['relations'] = [
                {
                    'id': relation.id,
                    'source_entity_id': relation.source_entity_id,
                    'target_entity_id': relation.target_entity_id,
                    'relation_type': str(relation.relation_type),
                    'confidence': relation.confidence
                }
                for relation in relations
            ]
        
        return metadata
    
    def _generate_episode_name(self, document: Document) -> str:
        """Generate a descriptive name for the episode.
        
        Args:
            document: Document instance
            
        Returns:
            Episode name string
        """
        base_name = document.name or document.file_name or "Unknown Document"
        
        # Add file type if available
        if document.mime_type and document.mime_type != 'unknown':
            file_type = document.mime_type.split('/')[-1].upper()
            return f"{base_name} ({file_type})"
        
        return base_name
    
    def _create_source_description(self, document: Document) -> str:
        """Create a source description for the episode.
        
        Args:
            document: Document instance
            
        Returns:
            Source description string
        """
        parts = ["MoRAG Document Ingestion"]
        
        if document.source_file:
            parts.append(f"Source: {document.source_file}")
        
        if document.file_size:
            size_mb = document.file_size / (1024 * 1024)
            parts.append(f"Size: {size_mb:.2f}MB")
        
        if document.model:
            parts.append(f"Model: {document.model}")
        
        return " | ".join(parts)


class ChunkEpisodeAdapter:
    """Converts individual DocumentChunks to Graphiti episodes."""
    
    def chunk_to_episode(
        self, 
        chunk: DocumentChunk, 
        document: Document,
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None
    ) -> Dict[str, Any]:
        """Convert a single DocumentChunk to a Graphiti episode.
        
        Args:
            chunk: DocumentChunk instance
            document: Parent Document instance
            entities: Optional entities found in this chunk
            relations: Optional relations found in this chunk
            
        Returns:
            Dictionary with episode data for Graphiti
        """
        # Create episode name
        episode_name = f"{document.file_name} - Chunk {chunk.chunk_index}"
        
        # Use chunk text as episode body
        episode_body = chunk.text
        
        # Create metadata
        metadata = {
            'morag_chunk_id': chunk.id,
            'morag_document_id': chunk.document_id,
            'chunk_index': chunk.chunk_index,
            'parent_document': {
                'id': document.id,
                'name': document.name,
                'source_file': document.source_file,
                'file_name': document.file_name
            },
            'original_metadata': chunk.metadata,
            'text_length': len(chunk.text),
            'entity_count': len(entities) if entities else 0,
            'relation_count': len(relations) if relations else 0,
            'adapter_version': '1.0',
            'conversion_timestamp': datetime.now().isoformat(),
            'morag_integration': True,
            'episode_type': 'chunk'
        }
        
        # Add entity and relation information
        if entities:
            metadata['entities'] = [
                {
                    'id': entity.id,
                    'name': entity.name,
                    'type': str(entity.type),
                    'confidence': entity.confidence
                }
                for entity in entities
            ]
        
        if relations:
            metadata['relations'] = [
                {
                    'id': relation.id,
                    'source_entity_id': relation.source_entity_id,
                    'target_entity_id': relation.target_entity_id,
                    'relation_type': str(relation.relation_type),
                    'confidence': relation.confidence
                }
                for relation in relations
            ]
        
        # Create source description
        source_description = f"MoRAG Chunk | Document: {document.file_name} | Chunk: {chunk.chunk_index}"
        
        return {
            'name': episode_name,
            'episode_body': episode_body,
            'source_description': source_description,
            'episode_type': EpisodeType.text,
            'metadata': metadata
        }


class EpisodeDocumentReverseAdapter:
    """Converts Graphiti episodes back to MoRAG models for compatibility."""
    
    def episode_to_document_info(self, episode_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract document information from episode metadata.
        
        Args:
            episode_metadata: Metadata from Graphiti episode
            
        Returns:
            Document information dictionary or None if not a MoRAG episode
        """
        if not episode_metadata.get('morag_integration'):
            return None
        
        return {
            'document_id': episode_metadata.get('morag_document_id'),
            'source_file': episode_metadata.get('source_file'),
            'file_name': episode_metadata.get('file_name'),
            'file_size': episode_metadata.get('file_size'),
            'checksum': episode_metadata.get('checksum'),
            'mime_type': episode_metadata.get('mime_type'),
            'summary': episode_metadata.get('summary'),
            'chunk_count': episode_metadata.get('chunk_count', 0),
            'entity_count': episode_metadata.get('entity_count', 0),
            'relation_count': episode_metadata.get('relation_count', 0)
        }
    
    def extract_chunk_mapping(self, episode_metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract chunk mapping from episode metadata.
        
        Args:
            episode_metadata: Metadata from Graphiti episode
            
        Returns:
            Chunk mapping dictionary
        """
        return episode_metadata.get('chunk_mapping', {})
```

### 2. Create Episode Ingestion Service

**File**: `packages/morag-graph/src/morag_graph/graphiti/ingestion_service.py`

```python
"""Graphiti-based ingestion service for MoRAG documents."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from graphiti_core import Graphiti

from .adapters import DocumentEpisodeAdapter, ChunkEpisodeAdapter
from .config import create_graphiti_instance, GraphitiConfig
from morag_graph.models import Document, DocumentChunk, Entity, Relation

logger = logging.getLogger(__name__)


class GraphitiIngestionService:
    """Service for ingesting MoRAG documents using Graphiti."""
    
    def __init__(self, config: Optional[GraphitiConfig] = None):
        self.config = config
        self.graphiti = create_graphiti_instance(config)
        self.document_adapter = DocumentEpisodeAdapter()
        self.chunk_adapter = ChunkEpisodeAdapter()
        
    async def ingest_document_as_episode(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None,
        strategy: str = "document"  # "document" or "chunks"
    ) -> Dict[str, Any]:
        """Ingest a document and its chunks as Graphiti episodes.
        
        Args:
            document: Document to ingest
            chunks: List of document chunks
            entities: Optional extracted entities
            relations: Optional extracted relations
            strategy: "document" for single episode, "chunks" for multiple episodes
            
        Returns:
            Ingestion results dictionary
        """
        results = {
            "strategy": strategy,
            "document_id": document.id,
            "episode_ids": [],
            "success": False,
            "error": None,
            "metadata": {}
        }
        
        try:
            if strategy == "document":
                # Ingest as single episode containing all chunks
                episode_id = await self._ingest_as_single_episode(
                    document, chunks, entities, relations
                )
                results["episode_ids"] = [episode_id]
                
            elif strategy == "chunks":
                # Ingest each chunk as separate episode
                episode_ids = await self._ingest_as_chunk_episodes(
                    document, chunks, entities, relations
                )
                results["episode_ids"] = episode_ids
                
            else:
                raise ValueError(f"Unknown ingestion strategy: {strategy}")
            
            results["success"] = True
            results["metadata"] = {
                "chunk_count": len(chunks),
                "entity_count": len(entities) if entities else 0,
                "relation_count": len(relations) if relations else 0,
                "episode_count": len(results["episode_ids"])
            }
            
            logger.info(
                f"Successfully ingested document {document.id} using {strategy} strategy. "
                f"Created {len(results['episode_ids'])} episodes."
            )
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Failed to ingest document {document.id}: {e}")
        
        return results
    
    async def _ingest_as_single_episode(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None
    ) -> str:
        """Ingest document as a single episode containing all chunks.
        
        Args:
            document: Document to ingest
            chunks: List of chunks
            entities: Optional entities
            relations: Optional relations
            
        Returns:
            Episode ID
        """
        episode_data = self.document_adapter.document_to_episode(
            document, chunks, entities, relations
        )
        
        episode_id = await self.graphiti.add_episode(**episode_data)
        
        logger.debug(f"Created single episode {episode_id} for document {document.id}")
        return episode_id
    
    async def _ingest_as_chunk_episodes(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None
    ) -> List[str]:
        """Ingest each chunk as a separate episode.
        
        Args:
            document: Parent document
            chunks: List of chunks to ingest
            entities: Optional entities (will be distributed to relevant chunks)
            relations: Optional relations (will be distributed to relevant chunks)
            
        Returns:
            List of episode IDs
        """
        episode_ids = []
        
        # Create mapping of entities/relations to chunks
        chunk_entities = self._distribute_entities_to_chunks(chunks, entities or [])
        chunk_relations = self._distribute_relations_to_chunks(chunks, relations or [])
        
        for chunk in chunks:
            chunk_entity_list = chunk_entities.get(chunk.id, [])
            chunk_relation_list = chunk_relations.get(chunk.id, [])
            
            episode_data = self.chunk_adapter.chunk_to_episode(
                chunk, document, chunk_entity_list, chunk_relation_list
            )
            
            episode_id = await self.graphiti.add_episode(**episode_data)
            episode_ids.append(episode_id)
            
            logger.debug(f"Created chunk episode {episode_id} for chunk {chunk.id}")
        
        return episode_ids
    
    def _distribute_entities_to_chunks(
        self, 
        chunks: List[DocumentChunk], 
        entities: List[Entity]
    ) -> Dict[str, List[Entity]]:
        """Distribute entities to their relevant chunks based on text content.
        
        Args:
            chunks: List of chunks
            entities: List of entities
            
        Returns:
            Dictionary mapping chunk IDs to entity lists
        """
        chunk_entities = {chunk.id: [] for chunk in chunks}
        
        for entity in entities:
            # Simple text-based matching (could be enhanced with more sophisticated methods)
            for chunk in chunks:
                if entity.name.lower() in chunk.text.lower():
                    chunk_entities[chunk.id].append(entity)
        
        return chunk_entities
    
    def _distribute_relations_to_chunks(
        self, 
        chunks: List[DocumentChunk], 
        relations: List[Relation]
    ) -> Dict[str, List[Relation]]:
        """Distribute relations to their relevant chunks.
        
        Args:
            chunks: List of chunks
            relations: List of relations
            
        Returns:
            Dictionary mapping chunk IDs to relation lists
        """
        chunk_relations = {chunk.id: [] for chunk in chunks}
        
        # For now, assign relations to all chunks (could be enhanced)
        for relation in relations:
            for chunk in chunks:
                chunk_relations[chunk.id].append(relation)
        
        return chunk_relations
    
    async def search_episodes(
        self, 
        query: str, 
        limit: int = 10,
        filter_morag_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Search episodes with optional filtering for MoRAG content.
        
        Args:
            query: Search query
            limit: Maximum results
            filter_morag_only: Only return MoRAG-ingested episodes
            
        Returns:
            List of search results
        """
        results = await self.graphiti.search(query=query, limit=limit)
        
        formatted_results = []
        for result in results:
            result_data = {
                "score": result.score,
                "content": result.content,
                "metadata": getattr(result, 'metadata', {})
            }
            
            # Filter for MoRAG episodes if requested
            if filter_morag_only:
                metadata = result_data.get("metadata", {})
                if not metadata.get("morag_integration"):
                    continue
            
            formatted_results.append(result_data)
        
        return formatted_results
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_graphiti_adapters.py`

```python
"""Unit tests for Graphiti adapters."""

import pytest
from datetime import datetime
from morag_graph.models import Document, DocumentChunk, Entity, Relation, EntityType, RelationType
from morag_graph.graphiti.adapters import DocumentEpisodeAdapter, ChunkEpisodeAdapter


class TestDocumentEpisodeAdapter:
    """Test document to episode conversion."""
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        return Document(
            id="doc_123",
            name="Test Document",
            source_file="/path/to/test.pdf",
            file_name="test.pdf",
            file_size=1024000,
            checksum="abc123",
            mime_type="application/pdf",
            summary="A test document for validation"
        )
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_123",
                chunk_index=0,
                text="This is the first chunk of text.",
                metadata={"page": 1}
            ),
            DocumentChunk(
                id="chunk_2",
                document_id="doc_123",
                chunk_index=1,
                text="This is the second chunk of text.",
                metadata={"page": 2}
            )
        ]
    
    def test_document_to_episode_basic(self, sample_document, sample_chunks):
        """Test basic document to episode conversion."""
        adapter = DocumentEpisodeAdapter()
        result = adapter.document_to_episode(sample_document, sample_chunks)
        
        assert result["name"] == "Test Document (PDF)"
        assert "CHUNK 0" in result["episode_body"]
        assert "CHUNK 1" in result["episode_body"]
        assert result["metadata"]["morag_document_id"] == "doc_123"
        assert result["metadata"]["chunk_count"] == 2
        assert result["metadata"]["morag_integration"] is True
    
    def test_episode_metadata_structure(self, sample_document, sample_chunks):
        """Test episode metadata structure and content."""
        adapter = DocumentEpisodeAdapter()
        result = adapter.document_to_episode(sample_document, sample_chunks)
        
        metadata = result["metadata"]
        
        # Check required fields
        assert "morag_document_id" in metadata
        assert "chunk_count" in metadata
        assert "chunk_ids" in metadata
        assert "chunk_mapping" in metadata
        assert "adapter_version" in metadata
        assert "conversion_timestamp" in metadata
        
        # Check chunk mapping
        assert len(metadata["chunk_mapping"]) == 2
        assert "chunk_1" in metadata["chunk_mapping"]
        assert "chunk_2" in metadata["chunk_mapping"]


class TestChunkEpisodeAdapter:
    """Test chunk to episode conversion."""
    
    @pytest.fixture
    def sample_chunk(self):
        """Create sample chunk for testing."""
        return DocumentChunk(
            id="chunk_1",
            document_id="doc_123",
            chunk_index=0,
            text="This is a test chunk.",
            metadata={"page": 1}
        )
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        return Document(
            id="doc_123",
            name="Test Document",
            file_name="test.pdf"
        )
    
    def test_chunk_to_episode_basic(self, sample_chunk, sample_document):
        """Test basic chunk to episode conversion."""
        adapter = ChunkEpisodeAdapter()
        result = adapter.chunk_to_episode(sample_chunk, sample_document)
        
        assert result["name"] == "test.pdf - Chunk 0"
        assert result["episode_body"] == "This is a test chunk."
        assert result["metadata"]["morag_chunk_id"] == "chunk_1"
        assert result["metadata"]["chunk_index"] == 0
        assert result["metadata"]["episode_type"] == "chunk"


@pytest.mark.integration
class TestGraphitiIngestionService:
    """Integration tests for Graphiti ingestion service."""
    
    @pytest.mark.asyncio
    async def test_document_ingestion_mock(self):
        """Test document ingestion with mocked Graphiti."""
        from unittest.mock import Mock, patch
        from morag_graph.graphiti.ingestion_service import GraphitiIngestionService
        
        # Mock Graphiti instance
        with patch('morag_graph.graphiti.ingestion_service.create_graphiti_instance') as mock_create:
            mock_graphiti = Mock()
            mock_graphiti.add_episode = Mock(return_value="episode_123")
            mock_create.return_value = mock_graphiti
            
            service = GraphitiIngestionService()
            
            # Create test data
            document = Document(id="doc_1", name="Test", file_name="test.txt")
            chunks = [DocumentChunk(id="chunk_1", document_id="doc_1", chunk_index=0, text="Test text")]
            
            # Test ingestion
            result = await service.ingest_document_as_episode(document, chunks, strategy="document")
            
            assert result["success"] is True
            assert result["episode_ids"] == ["episode_123"]
            mock_graphiti.add_episode.assert_called_once()
```

## Validation Checklist

- [ ] Document to episode conversion works correctly
- [ ] Chunk to episode conversion preserves all metadata
- [ ] Entity and relation information is properly embedded
- [ ] Both single-document and multi-chunk strategies work
- [ ] Search functionality returns MoRAG episodes
- [ ] Metadata structure is comprehensive and consistent
- [ ] Unit tests pass for all adapter classes
- [ ] Integration tests work with mocked Graphiti
- [ ] Error handling for malformed documents
- [ ] Performance acceptable for large documents

## Success Criteria

1. **Functional**: Documents and chunks convert to valid Graphiti episodes
2. **Preserves Data**: All metadata, entities, and relations are maintained
3. **Flexible**: Supports both document-level and chunk-level ingestion
4. **Searchable**: Episodes can be found through Graphiti search
5. **Testable**: Comprehensive test coverage with mocks and integration tests

## Next Steps

After completing this step:
1. Validate conversion accuracy with sample documents
2. Test performance with various document sizes
3. Proceed to [Step 3: Basic Search Implementation](./step-03-basic-search.md)

## Performance Considerations

- Large documents may need chunking strategy for memory efficiency
- Metadata size should be monitored to avoid storage bloat
- Search performance depends on episode content structure
- Consider batch ingestion for multiple documents
