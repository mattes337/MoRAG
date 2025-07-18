"""Tests for Graphiti integration service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from morag_core.models import Document, DocumentChunk, DocumentMetadata, DocumentType
from morag_graph.models import Entity, Relation
from morag_graph.graphiti.integration_service import (
    GraphitiIntegrationService, StorageBackend, IngestionResult,
    create_integration_service
)
from morag_graph.graphiti.ingestion_service import (
    GraphitiIngestionService, create_ingestion_service
)
from morag_graph.graphiti import GraphitiConfig


class TestStorageBackend:
    """Test StorageBackend enum."""
    
    def test_enum_values(self):
        """Test enum values."""
        assert StorageBackend.GRAPHITI.value == "graphiti"
        assert StorageBackend.NEO4J.value == "neo4j"
        assert StorageBackend.HYBRID.value == "hybrid"


class TestIngestionResult:
    """Test IngestionResult dataclass."""
    
    def test_init_success(self):
        """Test successful result initialization."""
        result = IngestionResult(
            success=True,
            backend_used=StorageBackend.GRAPHITI,
            document_id="test-doc-123",
            episode_ids=["episode-1", "episode-2"],
            entity_count=5,
            relation_count=3,
            chunk_count=10
        )
        
        assert result.success is True
        assert result.backend_used == StorageBackend.GRAPHITI
        assert result.document_id == "test-doc-123"
        assert result.episode_ids == ["episode-1", "episode-2"]
        assert result.entity_count == 5
        assert result.relation_count == 3
        assert result.chunk_count == 10
        assert result.error is None
        assert result.fallback_used is False
        assert result.performance_metrics == {}
    
    def test_init_failure(self):
        """Test failure result initialization."""
        result = IngestionResult(
            success=False,
            backend_used=StorageBackend.GRAPHITI,
            document_id="test-doc-123",
            error="Ingestion failed"
        )
        
        assert result.success is False
        assert result.error == "Ingestion failed"
        assert result.episode_ids == []
        assert result.performance_metrics == {}


class TestGraphitiIngestionService:
    """Test GraphitiIngestionService functionality."""
    
    def create_test_document(self):
        """Create a test document."""
        metadata = DocumentMetadata(
            title="Test Document",
            source_path="/test/document.pdf",
            mime_type="application/pdf"
        )
        
        chunks = [
            DocumentChunk(
                id="chunk_1",
                content="First chunk content",
                document_id="test-doc-123"
            ),
            DocumentChunk(
                id="chunk_2",
                content="Second chunk content", 
                document_id="test-doc-123"
            )
        ]
        
        return Document(
            id="test-doc-123",
            content="First chunk content Second chunk content",
            metadata=metadata,
            chunks=chunks,
            document_type=DocumentType.PDF
        )
    
    def create_test_entities(self):
        """Create test entities."""
        return [
            Entity(
                id="entity-1",
                name="Test Entity 1",
                type="PERSON",
                description="First test entity"
            ),
            Entity(
                id="entity-2", 
                name="Test Entity 2",
                type="ORGANIZATION",
                description="Second test entity"
            )
        ]
    
    def create_test_relations(self):
        """Create test relations."""
        return [
            Relation(
                id="relation-1",
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                type="WORKS_FOR",
                description="Test relation"
            )
        ]
    
    def test_init(self):
        """Test service initialization."""
        service = GraphitiIngestionService()
        
        assert service.config is None
        assert service.episode_mapper is not None
        assert service.entity_storage is not None
    
    def test_init_with_config(self):
        """Test service initialization with config."""
        config = GraphitiConfig(openai_api_key="test-key")
        service = GraphitiIngestionService(config)
        
        assert service.config == config
    
    @pytest.mark.asyncio
    async def test_ingest_document_no_graphiti(self):
        """Test document ingestion when Graphiti is not available."""
        service = GraphitiIngestionService()
        service.graphiti = None  # Simulate unavailable Graphiti
        
        document = self.create_test_document()
        entities = self.create_test_entities()
        relations = self.create_test_relations()
        
        result = await service.ingest_document_as_episode(
            document, document.chunks, entities, relations
        )
        
        assert result["success"] is False
        assert "Graphiti instance not available" in result["error"]
        assert result["episode_ids"] == []
    
    @pytest.mark.asyncio
    async def test_get_ingestion_stats(self):
        """Test ingestion statistics."""
        service = GraphitiIngestionService()
        
        stats = await service.get_ingestion_stats()
        
        assert "graphiti_available" in stats
        assert "episode_mapper_stats" in stats
        assert "entity_storage_stats" in stats
    
    @pytest.mark.asyncio
    async def test_validate_ingestion_setup(self):
        """Test ingestion setup validation."""
        service = GraphitiIngestionService()
        
        validation = await service.validate_ingestion_setup()
        
        assert "graphiti_available" in validation
        assert "episode_mapper_available" in validation
        assert "entity_storage_available" in validation
        assert "overall_ready" in validation
        assert "issues" in validation


class TestGraphitiIntegrationService:
    """Test GraphitiIntegrationService functionality."""
    
    def create_test_document(self):
        """Create a test document."""
        metadata = DocumentMetadata(
            title="Test Document",
            source_path="/test/document.pdf",
            mime_type="application/pdf"
        )
        
        chunks = [
            DocumentChunk(
                id="chunk_1",
                content="First chunk content",
                document_id="test-doc-123"
            )
        ]
        
        return Document(
            id="test-doc-123",
            content="First chunk content",
            metadata=metadata,
            chunks=chunks,
            document_type=DocumentType.PDF
        )
    
    def create_test_entities(self):
        """Create test entities."""
        return [
            Entity(
                id="entity-1",
                name="Test Entity",
                type="PERSON",
                description="Test entity"
            )
        ]
    
    def create_test_relations(self):
        """Create test relations."""
        return [
            Relation(
                id="relation-1",
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                type="RELATED_TO",
                description="Test relation"
            )
        ]
    
    def test_init(self):
        """Test service initialization."""
        service = GraphitiIntegrationService()
        
        assert service.preferred_backend == StorageBackend.GRAPHITI
        assert service.enable_fallback is True
        assert service.graphiti_ingestion is not None
        assert service.graphiti_entity_storage is not None
        assert service.graphiti_search is not None
    
    def test_init_with_config(self):
        """Test service initialization with config."""
        config = GraphitiConfig(openai_api_key="test-key")
        service = GraphitiIntegrationService(
            graphiti_config=config,
            preferred_backend=StorageBackend.NEO4J,
            enable_fallback=False
        )
        
        assert service.graphiti_config == config
        assert service.preferred_backend == StorageBackend.NEO4J
        assert service.enable_fallback is False
    
    @pytest.mark.asyncio
    async def test_ingest_document_fallback(self):
        """Test document ingestion with fallback to Neo4j."""
        service = GraphitiIntegrationService(enable_fallback=True)
        service.graphiti_available = False  # Simulate Graphiti unavailable
        
        document = self.create_test_document()
        entities = self.create_test_entities()
        relations = self.create_test_relations()
        
        result = await service.ingest_document_with_graph_data(
            document, document.chunks, entities, relations
        )
        
        assert result.success is True
        assert result.backend_used == StorageBackend.NEO4J
        assert result.document_id == document.id
    
    def test_is_graphiti_available(self):
        """Test Graphiti availability check."""
        service = GraphitiIntegrationService()
        
        # Should return the availability status
        assert isinstance(service.is_graphiti_available(), bool)
    
    def test_get_backend_status(self):
        """Test backend status retrieval."""
        service = GraphitiIntegrationService()
        
        status = service.get_backend_status()
        
        assert "preferred_backend" in status
        assert "graphiti_available" in status
        assert "neo4j_available" in status
        assert "fallback_enabled" in status
    
    @pytest.mark.asyncio
    async def test_get_integration_stats(self):
        """Test integration statistics."""
        service = GraphitiIntegrationService()
        
        stats = await service.get_integration_stats()
        
        assert "backend_status" in stats
        assert "graphiti_stats" in stats


class TestCreateFunctions:
    """Test creation functions."""
    
    def test_create_ingestion_service(self):
        """Test ingestion service creation."""
        service = create_ingestion_service()
        
        assert isinstance(service, GraphitiIngestionService)
        assert service.config is None
    
    def test_create_ingestion_service_with_config(self):
        """Test ingestion service creation with config."""
        config = GraphitiConfig(openai_api_key="test-key")
        service = create_ingestion_service(config)
        
        assert isinstance(service, GraphitiIngestionService)
        assert service.config == config
    
    def test_create_integration_service(self):
        """Test integration service creation."""
        service = create_integration_service()
        
        assert isinstance(service, GraphitiIntegrationService)
        assert service.preferred_backend == StorageBackend.GRAPHITI
        assert service.enable_fallback is True
    
    def test_create_integration_service_with_params(self):
        """Test integration service creation with parameters."""
        config = GraphitiConfig(openai_api_key="test-key")
        service = create_integration_service(
            config, 
            StorageBackend.NEO4J, 
            enable_fallback=False
        )
        
        assert isinstance(service, GraphitiIntegrationService)
        assert service.graphiti_config == config
        assert service.preferred_backend == StorageBackend.NEO4J
        assert service.enable_fallback is False
