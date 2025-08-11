"""Tests for GraphBuilder."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any
import asyncio

from morag_graph.builders import GraphBuilder, GraphBuildResult, GraphBuildError
# Import LLMConfig from morag-reasoning package
try:
    from morag_reasoning.llm import LLMConfig
except ImportError:
    # Fallback LLMConfig for compatibility
    from pydantic import BaseModel
    class LLMConfig(BaseModel):
        provider: str = "gemini"
        model: str = "gemini-1.5-flash"
        api_key: str = None
        temperature: float = 0.1
        max_tokens: int = 2000
from morag_graph.models import Entity, Relation, DocumentChunk
from morag_graph.storage.base import BaseStorage


class MockStorage(BaseStorage):
    """Mock storage for testing."""
    
    def __init__(self):
        self.entities = []
        self.relations = []
        self.closed = False
        self.connected = False
    
    async def connect(self):
        self.connected = True
    
    async def disconnect(self):
        self.connected = False
        self.closed = True
    
    async def store_entity(self, entity):
        self.entities.append(entity)
        return entity.id
    
    async def store_entities(self, entities):
        self.entities.extend(entities)
        return [entity.id for entity in entities]
    
    async def get_entity(self, entity_id):
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None
    
    async def get_entities(self, entity_ids):
        return [entity for entity in self.entities if entity.id in entity_ids]
    
    async def search_entities(self, query, entity_type=None, limit=10):
        results = []
        for entity in self.entities:
            if query.lower() in entity.name.lower():
                if entity_type is None or entity.type == entity_type:
                    results.append(entity)
                    if len(results) >= limit:
                        break
        return results
    
    async def update_entity(self, entity):
        for i, existing in enumerate(self.entities):
            if existing.id == entity.id:
                self.entities[i] = entity
                return True
        return False
    
    async def delete_entity(self, entity_id):
        for i, entity in enumerate(self.entities):
            if entity.id == entity_id:
                del self.entities[i]
                # Remove related relations
                self.relations = [r for r in self.relations 
                                if r.source_entity_id != entity_id and r.target_entity_id != entity_id]
                return True
        return False
    
    async def store_relation(self, relation):
        self.relations.append(relation)
        return relation.id
    
    async def store_relations(self, relations):
        self.relations.extend(relations)
        return [relation.id for relation in relations]
    
    async def get_relation(self, relation_id):
        for relation in self.relations:
            if relation.id == relation_id:
                return relation
        return None
    
    async def get_relations(self, relation_ids):
        return [relation for relation in self.relations if relation.id in relation_ids]
    
    async def get_entity_relations(self, entity_id, relation_type=None, direction="both"):
        results = []
        for relation in self.relations:
            if direction in ["out", "both"] and relation.source_entity_id == entity_id:
                if relation_type is None or relation.type == relation_type:
                    results.append(relation)
            if direction in ["in", "both"] and relation.target_entity_id == entity_id:
                if relation_type is None or relation.type == relation_type:
                    results.append(relation)
        return results
    
    async def update_relation(self, relation):
        for i, existing in enumerate(self.relations):
            if existing.id == relation.id:
                self.relations[i] = relation
                return True
        return False
    
    async def delete_relation(self, relation_id):
        for i, relation in enumerate(self.relations):
            if relation.id == relation_id:
                del self.relations[i]
                return True
        return False
    
    async def get_neighbors(self, entity_id, relation_type=None, max_depth=1):
        # Simple implementation for testing
        neighbors = set()
        for relation in self.relations:
            if relation.source_entity_id == entity_id:
                if relation_type is None or relation.type == relation_type:
                    neighbor = await self.get_entity(relation.target_entity_id)
                    if neighbor:
                        neighbors.add(neighbor)
            if relation.target_entity_id == entity_id:
                if relation_type is None or relation.type == relation_type:
                    neighbor = await self.get_entity(relation.source_entity_id)
                    if neighbor:
                        neighbors.add(neighbor)
        return list(neighbors)
    
    async def find_path(self, source_entity_id, target_entity_id, max_depth=3):
        # Simple implementation for testing
        return [[source_entity_id, target_entity_id]]  # Mock path
    
    async def store_graph(self, graph):
        await self.store_entities(graph.entities)
        await self.store_relations(graph.relations)
    
    async def get_graph(self, entity_ids=None):
        from morag_graph.models import Graph
        entities = self.entities if entity_ids is None else await self.get_entities(entity_ids)
        return Graph(entities=entities, relations=self.relations)
    
    async def clear(self):
        self.entities.clear()
        self.relations.clear()
    
    async def get_statistics(self):
        return {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations)
        }
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False
    
    async def close(self):
        await self.disconnect()


@pytest.fixture
def mock_storage():
    return MockStorage()


@pytest.fixture
def llm_config():
    return LLMConfig(
        provider="gemini",
        api_key="test-key",
        model="gemini-pro"
    )


@pytest.fixture
def graph_builder(mock_storage, llm_config):
    return GraphBuilder(
        storage=mock_storage,
        llm_config=llm_config
    )


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [
        Entity(
            id="ent_entity1",
            name="Test Entity 1",
            type="Person",
            attributes={"age": 30}
        ),
        Entity(
            id="ent_entity2",
            name="Test Entity 2",
            type="Organization",
            attributes={"founded": 2020}
        )
    ]


@pytest.fixture
def sample_relations(sample_entities):
    return [
        Relation(
            id="rel_relation1",
            source_entity_id=sample_entities[0].id,
            target_entity_id=sample_entities[1].id,
            type="WORKS_FOR",
            description="Person works for organization",
            metadata={"source": "test"}
        )
    ]


class TestGraphBuilder:
    """Test cases for GraphBuilder."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_storage, llm_config):
        """Test GraphBuilder initialization."""
        builder = GraphBuilder(
            storage=mock_storage,
            llm_config=llm_config
        )
        
        assert builder.storage == mock_storage
        assert builder.entity_extractor is not None
        assert builder.relation_extractor is not None
    
    @pytest.mark.asyncio
    async def test_process_document_success(self, mock_storage, llm_config, sample_entities, sample_relations):
        """Test successful document processing."""
        print("=== TEST STARTED ===")
        print(f"Sample entities length: {len(sample_entities)}")
        print(f"Sample relations length: {len(sample_relations)}")
        
        # Mock the extractors at class level before instantiation
        with patch('morag_graph.builders.graph_builder.EntityExtractor') as mock_entity_extractor_class, \
             patch('morag_graph.builders.graph_builder.RelationExtractor') as mock_relation_extractor_class:
            
            # Create mock instances
            mock_entity_extractor = AsyncMock()
            mock_relation_extractor = AsyncMock()
            
            # Configure the class mocks to return our instances
            mock_entity_extractor_class.return_value = mock_entity_extractor
            mock_relation_extractor_class.return_value = mock_relation_extractor
            
            # Configure the extract methods with AsyncMock return values
            mock_entity_extractor.extract = AsyncMock(return_value=sample_entities)
            mock_relation_extractor.extract = AsyncMock(return_value=sample_relations)
            
            print(f"Mock entity extractor configured: {mock_entity_extractor.extract}")
            print(f"Mock relation extractor configured: {mock_relation_extractor.extract}")
            
            # Now create the GraphBuilder
            graph_builder = GraphBuilder(
                storage=mock_storage,
                llm_config=llm_config
            )
            
            print(f"Sample entities: {len(sample_entities)}")
            print(f"Sample relations: {len(sample_relations)}")
            
            result = await graph_builder.process_document(
                content="Test content",
                document_id="test_doc_1",
                metadata={"test": "metadata"}
            )
            
            print(f"Result entities_created: {result.entities_created}")
            print(f"Result relations_created: {result.relations_created}")
            print(f"Storage entities: {len(mock_storage.entities)}")
            print(f"Storage relations: {len(mock_storage.relations)}")
            
            print(f"Result: entities_created={result.entities_created}, relations_created={result.relations_created}")
            
            # Verify result
            assert isinstance(result, GraphBuildResult)
            assert result.document_id == "test_doc_1"
            assert result.entities_created == 2
            assert result.relations_created == 1
            assert len(result.entity_ids) == 2
            assert len(result.relation_ids) == 1
            assert result.processing_time > 0
            assert len(result.errors) == 0
            
            # Verify extractors were called
            mock_entity_extractor.extract.assert_called_once_with(
                "Test content",
                source_doc_id="test_doc_1"
            )
            mock_relation_extractor.extract.assert_called_once_with(
                "Test content",
                entities=sample_entities
            )
            
            # Verify entities and relations were stored
            assert len(graph_builder.storage.entities) == 2
            assert len(graph_builder.storage.relations) == 1
    
    @pytest.mark.asyncio
    async def test_process_document_with_extraction_error(self, graph_builder):
        """Test document processing with extraction error."""
        # Mock the extractors to raise an exception
        with patch.object(graph_builder.entity_extractor, 'extract', new_callable=AsyncMock) as mock_entity_extract:
            mock_entity_extract.side_effect = Exception("Extraction failed")
            
            result = await graph_builder.process_document(
                content="Test document content",
                document_id="test_doc_1"
            )
            
            # Verify error handling
            assert isinstance(result, GraphBuildResult)
            assert result.document_id == "test_doc_1"
            assert result.entities_created == 0
            assert result.relations_created == 0
            assert len(result.errors) == 1
            assert "Extraction failed" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_process_document_chunks(self, graph_builder, sample_entities, sample_relations):
        """Test processing document chunks."""
        chunks = [
            DocumentChunk(
                id="doc_test1:chunk:0",
                document_id="doc_test1",
                chunk_index=0,
                text="Test content 1",
                metadata={"page": 1}
            ),
            DocumentChunk(
                id="doc_test1:chunk:1",
                document_id="doc_test1",
                chunk_index=1,
                text="Test content 2",
                metadata={"page": 2}
            )
        ]
        
        # Mock the extractors with proper method signatures
        async def mock_entity_extract(content, source_doc_id=None, **kwargs):
            return sample_entities[:1]  # One entity per chunk
        
        async def mock_relation_extract(content, entities=None, **kwargs):
            return sample_relations[:1]  # One relation per chunk
        
        graph_builder.entity_extractor.extract = AsyncMock(side_effect=mock_entity_extract)
        graph_builder.relation_extractor.extract = AsyncMock(side_effect=mock_relation_extract)
        
        result = await graph_builder.process_document_chunks(
            chunks=chunks,
            document_id="test_doc_1",
            metadata={"test": "metadata"}
        )
        
        # Verify result
        assert isinstance(result, GraphBuildResult)
        assert result.document_id == "test_doc_1"
        assert result.entities_created == 2  # One per chunk
        assert result.relations_created == 2  # One per chunk
        assert result.chunks_processed == 2
        assert len(result.errors) == 0
        
        # Verify extractors were called for each chunk
        assert graph_builder.entity_extractor.extract.call_count == 2
        assert graph_builder.relation_extractor.extract.call_count == 2
    
    @pytest.mark.asyncio
    async def test_process_documents_batch(self, graph_builder, sample_entities, sample_relations):
        """Test batch processing of documents."""
        documents = [
            ("Content 1", "doc1", {"meta": "1"}),
            ("Content 2", "doc2", {"meta": "2"}),
            ("Content 3", "doc3", {"meta": "3"})
        ]
        
        # Mock the extractors with proper method signatures
        async def mock_entity_extract(content, source_doc_id=None, **kwargs):
            return sample_entities[:1]
        
        async def mock_relation_extract(content, entities=None, **kwargs):
            return sample_relations[:1]
        
        graph_builder.entity_extractor.extract = AsyncMock(side_effect=mock_entity_extract)
        graph_builder.relation_extractor.extract = AsyncMock(side_effect=mock_relation_extract)
        
        results = await graph_builder.process_documents_batch(documents)
        
        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, GraphBuildResult)
            assert result.document_id == f"doc{i+1}"
            assert result.entities_created == 1
            assert result.relations_created == 1
    
    @pytest.mark.asyncio
    async def test_close(self, graph_builder):
        """Test closing the graph builder."""
        await graph_builder.close()
        assert graph_builder.storage.closed
    
    @pytest.mark.asyncio
    async def test_store_entities_and_relations_error(self, graph_builder, sample_entities, sample_relations):
        """Test error handling in storage operations."""
        # Mock storage to raise an exception
        with patch.object(graph_builder.storage, 'store_entities', new_callable=AsyncMock) as mock_store:
            mock_store.side_effect = Exception("Storage failed")
            
            with pytest.raises(GraphBuildError, match="Failed to store entities and relations"):
                await graph_builder._store_entities_and_relations(
                    sample_entities, sample_relations, "test_doc"
                )
    
    def test_graph_build_result_dataclass(self):
        """Test GraphBuildResult dataclass."""
        result = GraphBuildResult(
            document_id="test_doc",
            entities_created=5,
            relations_created=3,
            entity_ids=["e1", "e2", "e3", "e4", "e5"],
            relation_ids=["r1", "r2", "r3"]
        )
        
        assert result.document_id == "test_doc"
        assert result.entities_created == 5
        assert result.relations_created == 3
        assert len(result.entity_ids) == 5
        assert len(result.relation_ids) == 3
        assert result.processing_time == 0.0  # Default value
        assert result.chunks_processed == 0  # Default value
        assert result.errors == []  # Default value
        assert result.metadata == {}  # Default value
    
    def test_graph_build_error(self):
        """Test GraphBuildError exception."""
        error = GraphBuildError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)