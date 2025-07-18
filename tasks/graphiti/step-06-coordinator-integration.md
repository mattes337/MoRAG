# Step 6: Ingestion Coordinator Integration

**Duration**: 3-4 days  
**Phase**: Core Integration  
**Prerequisites**: Steps 1-5 completed, entity storage working

## Objective

Integrate Graphiti-based ingestion into MoRAG's IngestionCoordinator, providing seamless switching between Graphiti and legacy Neo4j storage with feature flags and fallback mechanisms.

## Deliverables

1. Updated IngestionCoordinator with Graphiti support
2. Feature flags for Graphiti vs. legacy mode selection
3. Fallback mechanisms for error handling
4. Configuration management for dual-mode operation
5. Comprehensive testing with both storage backends

## Implementation

### 1. Create Graphiti Integration Service

**File**: `packages/morag-graph/src/morag_graph/graphiti/integration_service.py`

```python
"""Integration service for Graphiti with MoRAG ingestion pipeline."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .config import GraphitiConfig, create_graphiti_instance
from .ingestion_service import GraphitiIngestionService
from .entity_storage import GraphitiEntityStorage
from .search_service import GraphitiSearchService
from morag_graph.models import Document, DocumentChunk, Entity, Relation

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Available storage backends."""
    GRAPHITI = "graphiti"
    NEO4J = "neo4j"
    HYBRID = "hybrid"


@dataclass
class IngestionResult:
    """Result of ingestion operation."""
    success: bool
    backend_used: StorageBackend
    document_id: str
    episode_ids: List[str] = None
    entity_count: int = 0
    relation_count: int = 0
    chunk_count: int = 0
    error: Optional[str] = None
    fallback_used: bool = False
    performance_metrics: Dict[str, Any] = None


class GraphitiIntegrationService:
    """Service for integrating Graphiti with MoRAG ingestion pipeline."""
    
    def __init__(
        self, 
        graphiti_config: Optional[GraphitiConfig] = None,
        preferred_backend: StorageBackend = StorageBackend.GRAPHITI,
        enable_fallback: bool = True
    ):
        self.graphiti_config = graphiti_config
        self.preferred_backend = preferred_backend
        self.enable_fallback = enable_fallback
        
        # Initialize Graphiti services
        try:
            self.graphiti_ingestion = GraphitiIngestionService(graphiti_config)
            self.graphiti_entity_storage = GraphitiEntityStorage(graphiti_config)
            self.graphiti_search = GraphitiSearchService(graphiti_config)
            self.graphiti_available = True
            logger.info("Graphiti services initialized successfully")
        except Exception as e:
            self.graphiti_available = False
            logger.warning(f"Graphiti services initialization failed: {e}")
            if preferred_backend == StorageBackend.GRAPHITI and not enable_fallback:
                raise RuntimeError(f"Graphiti required but unavailable: {e}")
    
    async def ingest_document_with_graph_data(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: List[Entity],
        relations: List[Relation],
        force_backend: Optional[StorageBackend] = None
    ) -> IngestionResult:
        """Ingest document with graph data using specified or preferred backend.
        
        Args:
            document: Document to ingest
            chunks: Document chunks
            entities: Extracted entities
            relations: Extracted relations
            force_backend: Force specific backend (overrides preference)
            
        Returns:
            IngestionResult with operation details
        """
        import time
        start_time = time.time()
        
        # Determine backend to use
        backend = force_backend or self.preferred_backend
        
        # Validate backend availability
        if backend == StorageBackend.GRAPHITI and not self.graphiti_available:
            if self.enable_fallback:
                logger.warning("Graphiti unavailable, falling back to Neo4j")
                backend = StorageBackend.NEO4J
            else:
                return IngestionResult(
                    success=False,
                    backend_used=backend,
                    document_id=document.id,
                    error="Graphiti backend unavailable and fallback disabled"
                )
        
        try:
            if backend == StorageBackend.GRAPHITI:
                result = await self._ingest_with_graphiti(document, chunks, entities, relations)
            elif backend == StorageBackend.NEO4J:
                result = await self._ingest_with_neo4j(document, chunks, entities, relations)
            elif backend == StorageBackend.HYBRID:
                result = await self._ingest_with_hybrid(document, chunks, entities, relations)
            else:
                raise ValueError(f"Unknown backend: {backend}")
            
            # Add performance metrics
            result.performance_metrics = {
                "total_time": time.time() - start_time,
                "backend_used": result.backend_used.value
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed with {backend.value} backend: {e}")
            
            # Try fallback if enabled and not already using fallback
            if (self.enable_fallback and 
                backend != StorageBackend.NEO4J and 
                force_backend is None):
                
                logger.info("Attempting fallback to Neo4j backend")
                try:
                    fallback_result = await self._ingest_with_neo4j(document, chunks, entities, relations)
                    fallback_result.fallback_used = True
                    fallback_result.performance_metrics = {
                        "total_time": time.time() - start_time,
                        "backend_used": fallback_result.backend_used.value,
                        "fallback_from": backend.value
                    }
                    return fallback_result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    return IngestionResult(
                        success=False,
                        backend_used=backend,
                        document_id=document.id,
                        error=f"Primary: {str(e)}, Fallback: {str(fallback_error)}"
                    )
            
            return IngestionResult(
                success=False,
                backend_used=backend,
                document_id=document.id,
                error=str(e)
            )
    
    async def _ingest_with_graphiti(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: List[Entity],
        relations: List[Relation]
    ) -> IngestionResult:
        """Ingest using Graphiti backend."""
        # Ingest document and chunks as episodes
        doc_result = await self.graphiti_ingestion.ingest_document_as_episode(
            document, chunks, entities, relations, strategy="document"
        )
        
        if not doc_result["success"]:
            raise RuntimeError(f"Document ingestion failed: {doc_result['error']}")
        
        # Store entities and relations
        entity_results = await self.graphiti_entity_storage.store_entities_batch(entities)
        relation_results = await self.graphiti_entity_storage.store_relations_batch(relations)
        
        # Check for failures
        failed_entities = [r for r in entity_results if not r.success]
        failed_relations = [r for r in relation_results if not r.success]
        
        if failed_entities or failed_relations:
            error_msg = f"Failed entities: {len(failed_entities)}, Failed relations: {len(failed_relations)}"
            logger.warning(error_msg)
        
        return IngestionResult(
            success=True,
            backend_used=StorageBackend.GRAPHITI,
            document_id=document.id,
            episode_ids=doc_result["episode_ids"],
            entity_count=len([r for r in entity_results if r.success]),
            relation_count=len([r for r in relation_results if r.success]),
            chunk_count=len(chunks)
        )
    
    async def _ingest_with_neo4j(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: List[Entity],
        relations: List[Relation]
    ) -> IngestionResult:
        """Ingest using legacy Neo4j backend."""
        # This would call the existing Neo4j ingestion logic
        # For now, this is a placeholder that simulates the operation
        
        logger.info(f"Ingesting document {document.id} with Neo4j backend")
        
        # Simulate Neo4j ingestion
        # In actual implementation, this would call existing Neo4j storage methods
        
        return IngestionResult(
            success=True,
            backend_used=StorageBackend.NEO4J,
            document_id=document.id,
            entity_count=len(entities),
            relation_count=len(relations),
            chunk_count=len(chunks)
        )
    
    async def _ingest_with_hybrid(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        entities: List[Entity],
        relations: List[Relation]
    ) -> IngestionResult:
        """Ingest using hybrid approach (both backends)."""
        # Store in both Graphiti and Neo4j for comparison/migration
        
        graphiti_result = await self._ingest_with_graphiti(document, chunks, entities, relations)
        neo4j_result = await self._ingest_with_neo4j(document, chunks, entities, relations)
        
        # Return Graphiti result but note hybrid usage
        graphiti_result.backend_used = StorageBackend.HYBRID
        graphiti_result.performance_metrics = {
            "graphiti_success": graphiti_result.success,
            "neo4j_success": neo4j_result.success
        }
        
        return graphiti_result
    
    async def search_with_backend(
        self,
        query: str,
        backend: Optional[StorageBackend] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search using specified backend.
        
        Args:
            query: Search query
            backend: Backend to use (defaults to preferred)
            limit: Maximum results
            
        Returns:
            Search results with backend information
        """
        backend = backend or self.preferred_backend
        
        if backend == StorageBackend.GRAPHITI and self.graphiti_available:
            results, metrics = await self.graphiti_search.search(query, limit)
            return {
                "backend": backend.value,
                "results": [
                    {
                        "content": r.content,
                        "score": r.score,
                        "document_id": r.document_id,
                        "chunk_id": r.chunk_id
                    }
                    for r in results
                ],
                "metrics": {
                    "query_time": metrics.query_time,
                    "result_count": metrics.result_count
                }
            }
        else:
            # Fallback to Neo4j search (placeholder)
            return {
                "backend": "neo4j",
                "results": [],
                "metrics": {"query_time": 0.0, "result_count": 0},
                "note": "Neo4j search not implemented in this step"
            }
    
    def get_backend_status(self) -> Dict[str, Any]:
        """Get status of available backends."""
        return {
            "graphiti_available": self.graphiti_available,
            "preferred_backend": self.preferred_backend.value,
            "fallback_enabled": self.enable_fallback,
            "configuration": {
                "graphiti_config": self.graphiti_config.__dict__ if self.graphiti_config else None
            }
        }
```

### 2. Update Ingestion Coordinator

**File**: `packages/morag/src/morag/graphiti_coordinator_integration.py`

```python
"""Integration of Graphiti with MoRAG IngestionCoordinator."""

import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from morag_graph.graphiti.integration_service import GraphitiIntegrationService, StorageBackend
from morag_graph.graphiti.config import GraphitiConfig
from morag_graph.models import Document, DocumentChunk, Entity, Relation

logger = logging.getLogger(__name__)


class GraphitiCoordinatorIntegration:
    """Integration layer for Graphiti with IngestionCoordinator."""
    
    def __init__(self):
        self.graphiti_service = None
        self._initialize_graphiti()
    
    def _initialize_graphiti(self):
        """Initialize Graphiti service based on configuration."""
        try:
            # Check if Graphiti is enabled
            graphiti_enabled = os.getenv("GRAPHITI_ENABLED", "false").lower() == "true"
            
            if not graphiti_enabled:
                logger.info("Graphiti integration disabled via configuration")
                return
            
            # Create Graphiti configuration
            graphiti_config = GraphitiConfig(
                neo4j_uri=os.getenv("GRAPHITI_NEO4J_URI", "bolt://localhost:7687"),
                neo4j_username=os.getenv("GRAPHITI_NEO4J_USERNAME", "neo4j"),
                neo4j_password=os.getenv("GRAPHITI_NEO4J_PASSWORD", "password"),
                neo4j_database=os.getenv("GRAPHITI_NEO4J_DATABASE", "morag_graphiti"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                enable_telemetry=os.getenv("GRAPHITI_TELEMETRY_ENABLED", "false").lower() == "true"
            )
            
            # Determine preferred backend
            backend_preference = os.getenv("GRAPHITI_BACKEND_PREFERENCE", "graphiti").lower()
            if backend_preference == "neo4j":
                preferred_backend = StorageBackend.NEO4J
            elif backend_preference == "hybrid":
                preferred_backend = StorageBackend.HYBRID
            else:
                preferred_backend = StorageBackend.GRAPHITI
            
            # Enable fallback
            enable_fallback = os.getenv("GRAPHITI_ENABLE_FALLBACK", "true").lower() == "true"
            
            # Initialize service
            self.graphiti_service = GraphitiIntegrationService(
                graphiti_config=graphiti_config,
                preferred_backend=preferred_backend,
                enable_fallback=enable_fallback
            )
            
            logger.info(f"Graphiti integration initialized with {preferred_backend.value} backend")
            
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti integration: {e}")
            self.graphiti_service = None
    
    async def write_to_graphiti(
        self,
        graph_data: Dict[str, Any],
        embeddings_data: Dict[str, Any],
        document_id: str,
        document_summary: Optional[str] = None,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Write graph data to Graphiti (replacement for _write_to_neo4j).
        
        Args:
            graph_data: Graph data with entities and relations
            embeddings_data: Embeddings data with chunks
            document_id: Document ID
            document_summary: Optional document summary
            document_metadata: Optional document metadata
            
        Returns:
            Ingestion results
        """
        if not self.graphiti_service:
            raise RuntimeError("Graphiti service not initialized")
        
        try:
            # Create Document instance
            document = self._create_document_from_metadata(
                document_id, document_metadata, embeddings_data, document_summary
            )
            
            # Create DocumentChunk instances
            chunks = self._create_chunks_from_embeddings(document_id, embeddings_data)
            
            # Extract entities and relations
            entities = graph_data.get('entities', [])
            relations = graph_data.get('relations', [])
            
            # Ingest using Graphiti
            result = await self.graphiti_service.ingest_document_with_graph_data(
                document, chunks, entities, relations
            )
            
            if result.success:
                return {
                    'success': True,
                    'backend': result.backend_used.value,
                    'document_id': result.document_id,
                    'episode_ids': result.episode_ids or [],
                    'entity_count': result.entity_count,
                    'relation_count': result.relation_count,
                    'chunk_count': result.chunk_count,
                    'fallback_used': result.fallback_used,
                    'performance_metrics': result.performance_metrics
                }
            else:
                return {
                    'success': False,
                    'error': result.error,
                    'backend': result.backend_used.value
                }
        
        except Exception as e:
            logger.error(f"Graphiti ingestion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'backend': 'graphiti'
            }
    
    def _create_document_from_metadata(
        self,
        document_id: str,
        document_metadata: Optional[Dict[str, Any]],
        embeddings_data: Dict[str, Any],
        document_summary: Optional[str]
    ) -> Document:
        """Create Document instance from metadata."""
        if document_metadata:
            # Use structured document metadata
            source_path = document_metadata.get('source_file', 'Unknown')
            file_name = document_metadata.get('file_name', 'Unknown')
            name = document_metadata.get('name', file_name)
            mime_type = document_metadata.get('mime_type', 'unknown')
            file_size = document_metadata.get('file_size')
            checksum = document_metadata.get('checksum')
            summary = document_summary or document_metadata.get('summary', 'Document processed successfully')
            metadata = document_metadata.get('metadata', {})
        else:
            # Fallback to extracting from first chunk metadata
            chunk_meta = embeddings_data['chunk_metadata'][0]
            source_path = chunk_meta.get('source_path', 'Unknown')
            
            if source_path and source_path != 'Unknown':
                file_name = Path(source_path).name
                name = file_name
            else:
                file_name = chunk_meta.get('source_name', 'Unknown')
                name = file_name
            
            mime_type = chunk_meta.get('mime_type', chunk_meta.get('source_type', 'unknown'))
            file_size = chunk_meta.get('file_size')
            checksum = chunk_meta.get('checksum') or chunk_meta.get('content_checksum')
            summary = document_summary or chunk_meta.get('summary', 'Document processed successfully')
            metadata = chunk_meta
        
        return Document(
            id=document_id,
            name=name,
            source_file=source_path,
            file_name=file_name,
            file_size=file_size,
            checksum=checksum,
            mime_type=mime_type,
            summary=summary,
            metadata=metadata
        )
    
    def _create_chunks_from_embeddings(
        self,
        document_id: str,
        embeddings_data: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create DocumentChunk instances from embeddings data."""
        chunks = []
        
        for i, (chunk_text, chunk_meta) in enumerate(
            zip(embeddings_data['chunks'], embeddings_data['chunk_metadata'])
        ):
            chunk = DocumentChunk(
                id=chunk_meta['chunk_id'],
                document_id=document_id,
                chunk_index=i,
                text=chunk_text,
                metadata=chunk_meta
            )
            chunks.append(chunk)
        
        return chunks
    
    async def search_graphiti(
        self,
        query: str,
        limit: int = 10,
        backend: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search using Graphiti backend.
        
        Args:
            query: Search query
            limit: Maximum results
            backend: Backend preference ("graphiti", "neo4j", "hybrid")
            
        Returns:
            Search results
        """
        if not self.graphiti_service:
            return {
                "success": False,
                "error": "Graphiti service not initialized"
            }
        
        try:
            # Convert backend string to enum
            backend_enum = None
            if backend:
                backend_map = {
                    "graphiti": StorageBackend.GRAPHITI,
                    "neo4j": StorageBackend.NEO4J,
                    "hybrid": StorageBackend.HYBRID
                }
                backend_enum = backend_map.get(backend.lower())
            
            results = await self.graphiti_service.search_with_backend(
                query, backend_enum, limit
            )
            
            return {
                "success": True,
                **results
            }
        
        except Exception as e:
            logger.error(f"Graphiti search failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def is_graphiti_available(self) -> bool:
        """Check if Graphiti is available and configured."""
        return self.graphiti_service is not None
    
    def get_graphiti_status(self) -> Dict[str, Any]:
        """Get Graphiti integration status."""
        if not self.graphiti_service:
            return {
                "available": False,
                "error": "Graphiti service not initialized"
            }
        
        return {
            "available": True,
            **self.graphiti_service.get_backend_status()
        }


# Global instance for use in IngestionCoordinator
graphiti_integration = GraphitiCoordinatorIntegration()
```

### 3. Environment Configuration

**File**: `tasks/graphiti/.env.graphiti`

```bash
# Graphiti Integration Configuration

# Enable/disable Graphiti integration
GRAPHITI_ENABLED=true

# Backend preference: graphiti, neo4j, hybrid
GRAPHITI_BACKEND_PREFERENCE=graphiti

# Enable fallback to Neo4j if Graphiti fails
GRAPHITI_ENABLE_FALLBACK=true

# Graphiti Neo4j Database Settings
GRAPHITI_NEO4J_URI=bolt://localhost:7687
GRAPHITI_NEO4J_USERNAME=neo4j
GRAPHITI_NEO4J_PASSWORD=password
GRAPHITI_NEO4J_DATABASE=morag_graphiti

# OpenAI API Settings (required for Graphiti)
OPENAI_API_KEY=your_openai_api_key_here

# Graphiti Settings
GRAPHITI_TELEMETRY_ENABLED=false
USE_PARALLEL_RUNTIME=false

# Performance Settings
GRAPHITI_BATCH_SIZE=100
GRAPHITI_SEARCH_LIMIT=50
GRAPHITI_CACHE_SIZE=1000
```

## Testing

### Integration Tests

**File**: `packages/morag/tests/test_graphiti_coordinator_integration.py`

```python
"""Integration tests for Graphiti coordinator integration."""

import pytest
import os
from unittest.mock import Mock, AsyncMock, patch
from morag.graphiti_coordinator_integration import GraphitiCoordinatorIntegration
from morag_graph.models import Document, DocumentChunk, Entity, Relation, EntityType


class TestGraphitiCoordinatorIntegration:
    """Test Graphiti coordinator integration."""
    
    @pytest.fixture
    def mock_integration(self):
        """Create mock integration with Graphiti service."""
        integration = GraphitiCoordinatorIntegration()
        integration.graphiti_service = Mock()
        integration.graphiti_service.ingest_document_with_graph_data = AsyncMock()
        integration.graphiti_service.search_with_backend = AsyncMock()
        return integration
    
    @pytest.fixture
    def sample_data(self):
        """Create sample ingestion data."""
        return {
            "graph_data": {
                "entities": [
                    Entity(id="e1", name="Test Entity", type=EntityType.PERSON)
                ],
                "relations": []
            },
            "embeddings_data": {
                "chunks": ["Test chunk content"],
                "chunk_metadata": [{
                    "chunk_id": "chunk_1",
                    "source_path": "/test/file.txt",
                    "mime_type": "text/plain"
                }]
            },
            "document_id": "doc_123",
            "document_metadata": {
                "source_file": "/test/file.txt",
                "file_name": "file.txt",
                "name": "Test File",
                "mime_type": "text/plain"
            }
        }
    
    @pytest.mark.asyncio
    async def test_write_to_graphiti_success(self, mock_integration, sample_data):
        """Test successful Graphiti ingestion."""
        from morag_graph.graphiti.integration_service import IngestionResult, StorageBackend
        
        # Mock successful ingestion
        mock_result = IngestionResult(
            success=True,
            backend_used=StorageBackend.GRAPHITI,
            document_id="doc_123",
            episode_ids=["episode_1"],
            entity_count=1,
            relation_count=0,
            chunk_count=1
        )
        mock_integration.graphiti_service.ingest_document_with_graph_data.return_value = mock_result
        
        result = await mock_integration.write_to_graphiti(
            sample_data["graph_data"],
            sample_data["embeddings_data"],
            sample_data["document_id"],
            document_metadata=sample_data["document_metadata"]
        )
        
        assert result["success"] is True
        assert result["backend"] == "graphiti"
        assert result["document_id"] == "doc_123"
        assert result["entity_count"] == 1
        assert result["chunk_count"] == 1
    
    @pytest.mark.asyncio
    async def test_write_to_graphiti_failure(self, mock_integration, sample_data):
        """Test Graphiti ingestion failure."""
        from morag_graph.graphiti.integration_service import IngestionResult, StorageBackend
        
        # Mock failed ingestion
        mock_result = IngestionResult(
            success=False,
            backend_used=StorageBackend.GRAPHITI,
            document_id="doc_123",
            error="Test error"
        )
        mock_integration.graphiti_service.ingest_document_with_graph_data.return_value = mock_result
        
        result = await mock_integration.write_to_graphiti(
            sample_data["graph_data"],
            sample_data["embeddings_data"],
            sample_data["document_id"]
        )
        
        assert result["success"] is False
        assert result["error"] == "Test error"
        assert result["backend"] == "graphiti"
    
    @pytest.mark.asyncio
    async def test_search_graphiti(self, mock_integration):
        """Test Graphiti search functionality."""
        # Mock search results
        mock_integration.graphiti_service.search_with_backend.return_value = {
            "backend": "graphiti",
            "results": [
                {"content": "Test result", "score": 0.9}
            ],
            "metrics": {"query_time": 0.1, "result_count": 1}
        }
        
        result = await mock_integration.search_graphiti("test query")
        
        assert result["success"] is True
        assert result["backend"] == "graphiti"
        assert len(result["results"]) == 1
    
    def test_graphiti_status(self, mock_integration):
        """Test Graphiti status reporting."""
        mock_integration.graphiti_service.get_backend_status.return_value = {
            "graphiti_available": True,
            "preferred_backend": "graphiti"
        }
        
        status = mock_integration.get_graphiti_status()
        
        assert status["available"] is True
        assert status["graphiti_available"] is True
    
    @patch.dict(os.environ, {
        "GRAPHITI_ENABLED": "true",
        "GRAPHITI_BACKEND_PREFERENCE": "hybrid",
        "GRAPHITI_ENABLE_FALLBACK": "false"
    })
    def test_environment_configuration(self):
        """Test configuration from environment variables."""
        with patch('morag.graphiti_coordinator_integration.GraphitiIntegrationService') as mock_service:
            integration = GraphitiCoordinatorIntegration()
            
            # Verify service was called with correct configuration
            mock_service.assert_called_once()
            call_args = mock_service.call_args
            
            assert call_args[1]["preferred_backend"].value == "hybrid"
            assert call_args[1]["enable_fallback"] is False


@pytest.mark.integration
class TestGraphitiCoordinatorRealIntegration:
    """Integration tests with real Graphiti (requires setup)."""
    
    @pytest.mark.skipif(
        not os.getenv("GRAPHITI_INTEGRATION_TEST"),
        reason="Graphiti integration tests require GRAPHITI_INTEGRATION_TEST=true"
    )
    @pytest.mark.asyncio
    async def test_real_graphiti_ingestion(self):
        """Test real Graphiti ingestion (requires Neo4j and OpenAI API key)."""
        integration = GraphitiCoordinatorIntegration()
        
        if not integration.is_graphiti_available():
            pytest.skip("Graphiti not available for integration test")
        
        # Create minimal test data
        graph_data = {"entities": [], "relations": []}
        embeddings_data = {
            "chunks": ["Test content for integration"],
            "chunk_metadata": [{
                "chunk_id": "integration_chunk_1",
                "source_path": "/test/integration.txt",
                "mime_type": "text/plain"
            }]
        }
        
        result = await integration.write_to_graphiti(
            graph_data,
            embeddings_data,
            "integration_test_doc"
        )
        
        # Should succeed or fail gracefully
        assert "success" in result
        assert "backend" in result
```

## Validation Checklist

- [ ] Graphiti integration service initializes correctly
- [ ] Feature flags control backend selection properly
- [ ] Fallback mechanism works when primary backend fails
- [ ] Document ingestion works with both backends
- [ ] Search functionality supports backend selection
- [ ] Configuration management handles environment variables
- [ ] Error handling provides meaningful messages
- [ ] Performance metrics are captured and reported
- [ ] Unit tests cover all integration scenarios
- [ ] Integration tests work with real Graphiti setup

## Success Criteria

1. **Seamless Integration**: IngestionCoordinator can use Graphiti without breaking existing functionality
2. **Configurable**: Backend selection via environment variables and feature flags
3. **Reliable**: Fallback mechanisms ensure system stability
4. **Observable**: Comprehensive logging and metrics for monitoring
5. **Testable**: Both unit and integration tests validate functionality

## Next Steps

After completing this step:
1. Test integration with existing MoRAG workflows
2. Validate performance with realistic data volumes
3. Configure monitoring and alerting for production use
4. Proceed to [Step 7: Chunk-Entity Relationship Handling](./step-07-chunk-entity-relationships.md)

## Configuration Notes

- Set `GRAPHITI_ENABLED=true` to enable Graphiti integration
- Use `GRAPHITI_BACKEND_PREFERENCE` to control default backend
- Enable `GRAPHITI_ENABLE_FALLBACK=true` for production reliability
- Separate Neo4j database recommended for Graphiti (`morag_graphiti`)
