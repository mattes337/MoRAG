# Step 7: Chunk-Entity Relationship Handling

**Duration**: 2-3 days  
**Phase**: Core Integration  
**Prerequisites**: Steps 1-6 completed, coordinator integration working

## Objective

Implement proper chunk-to-entity relationship mapping using Graphiti's episode model, ensuring that entity mentions within document chunks are accurately tracked and searchable.

## Deliverables

1. Chunk-entity relationship mapping service
2. Entity mention tracking within episodes
3. Chunk-based entity search functionality
4. Relationship integrity validation
5. Performance optimization for large documents

## Implementation

### 1. Create Chunk-Entity Relationship Service

**File**: `packages/morag-graph/src/morag_graph/graphiti/chunk_entity_service.py`

```python
"""Service for managing chunk-entity relationships in Graphiti."""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .entity_storage import GraphitiEntityStorage
from .search_service import GraphitiSearchService
from morag_graph.models import Entity, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkEntityMapping:
    """Mapping between chunks and entities."""
    chunk_id: str
    entity_ids: List[str]
    confidence_scores: Dict[str, float]
    mention_contexts: Dict[str, str]  # entity_id -> context text


@dataclass
class EntityMentionResult:
    """Result of entity mention analysis."""
    entity_id: str
    chunk_id: str
    mention_text: str
    context: str
    confidence: float
    start_position: int
    end_position: int


class ChunkEntityRelationshipService:
    """Service for managing chunk-entity relationships in Graphiti."""
    
    def __init__(
        self, 
        entity_storage: GraphitiEntityStorage,
        search_service: GraphitiSearchService
    ):
        self.entity_storage = entity_storage
        self.search_service = search_service
        self.graphiti = entity_storage.graphiti
    
    async def create_chunk_entity_mappings(
        self,
        chunks: List[DocumentChunk],
        entities: List[Entity],
        document_id: str
    ) -> List[ChunkEntityMapping]:
        """Create mappings between chunks and entities.
        
        Args:
            chunks: List of document chunks
            entities: List of extracted entities
            document_id: Document ID
            
        Returns:
            List of chunk-entity mappings
        """
        mappings = []
        
        for chunk in chunks:
            # Find entities mentioned in this chunk
            chunk_entities = await self._find_entities_in_chunk(chunk, entities)
            
            if chunk_entities:
                # Calculate confidence scores and extract contexts
                confidence_scores = {}
                mention_contexts = {}
                
                for entity in chunk_entities:
                    confidence, context = self._analyze_entity_mention(chunk, entity)
                    confidence_scores[entity.id] = confidence
                    mention_contexts[entity.id] = context
                
                mapping = ChunkEntityMapping(
                    chunk_id=chunk.id,
                    entity_ids=[e.id for e in chunk_entities],
                    confidence_scores=confidence_scores,
                    mention_contexts=mention_contexts
                )
                mappings.append(mapping)
        
        # Store mappings in episode metadata
        await self._store_chunk_entity_mappings(mappings, document_id)
        
        return mappings
    
    async def _find_entities_in_chunk(
        self, 
        chunk: DocumentChunk, 
        entities: List[Entity]
    ) -> List[Entity]:
        """Find entities that are mentioned in a specific chunk.
        
        Args:
            chunk: Document chunk to analyze
            entities: List of entities to check
            
        Returns:
            List of entities found in the chunk
        """
        chunk_text_lower = chunk.text.lower()
        found_entities = []
        
        for entity in entities:
            # Simple text-based matching (can be enhanced with NLP)
            if entity.name.lower() in chunk_text_lower:
                found_entities.append(entity)
                continue
            
            # Check entity attributes for alternative names
            if hasattr(entity, 'attributes') and entity.attributes:
                alt_names = entity.attributes.get('alternative_names', [])
                if isinstance(alt_names, list):
                    for alt_name in alt_names:
                        if alt_name.lower() in chunk_text_lower:
                            found_entities.append(entity)
                            break
        
        return found_entities
    
    def _analyze_entity_mention(
        self, 
        chunk: DocumentChunk, 
        entity: Entity
    ) -> Tuple[float, str]:
        """Analyze how an entity is mentioned in a chunk.
        
        Args:
            chunk: Document chunk
            entity: Entity to analyze
            
        Returns:
            Tuple of (confidence_score, context_text)
        """
        chunk_text = chunk.text
        entity_name = entity.name.lower()
        chunk_text_lower = chunk_text.lower()
        
        # Find entity mention position
        mention_pos = chunk_text_lower.find(entity_name)
        if mention_pos == -1:
            return 0.0, ""
        
        # Extract context around the mention
        context_window = 100  # characters before and after
        start_pos = max(0, mention_pos - context_window)
        end_pos = min(len(chunk_text), mention_pos + len(entity_name) + context_window)
        context = chunk_text[start_pos:end_pos]
        
        # Calculate confidence based on various factors
        confidence = 0.5  # Base confidence
        
        # Boost confidence if entity name appears multiple times
        mention_count = chunk_text_lower.count(entity_name)
        confidence += min(0.3, mention_count * 0.1)
        
        # Boost confidence if entity appears at sentence boundaries
        if mention_pos == 0 or chunk_text[mention_pos - 1] in ' \n\t.!?':
            confidence += 0.1
        
        # Boost confidence based on entity type context
        if entity.type and str(entity.type).lower() in chunk_text_lower:
            confidence += 0.1
        
        return min(1.0, confidence), context
    
    async def _store_chunk_entity_mappings(
        self, 
        mappings: List[ChunkEntityMapping], 
        document_id: str
    ) -> None:
        """Store chunk-entity mappings in episode metadata.
        
        Args:
            mappings: List of chunk-entity mappings
            document_id: Document ID
        """
        # Create a mapping episode that contains all chunk-entity relationships
        mapping_data = {
            "document_id": document_id,
            "chunk_entity_mappings": [
                {
                    "chunk_id": mapping.chunk_id,
                    "entity_ids": mapping.entity_ids,
                    "confidence_scores": mapping.confidence_scores,
                    "mention_contexts": mapping.mention_contexts
                }
                for mapping in mappings
            ],
            "total_mappings": len(mappings),
            "adapter_type": "chunk_entity_mapping",
            "morag_integration": True
        }
        
        # Create episode for the mappings
        episode_id = await self.graphiti.add_episode(
            name=f"Chunk-Entity Mappings: {document_id}",
            episode_body=f"Chunk-entity relationship mappings for document {document_id}",
            source_description=f"MoRAG Chunk-Entity Mapping | Document: {document_id}",
            episode_type="json",
            metadata=mapping_data
        )
        
        logger.info(f"Stored chunk-entity mappings in episode {episode_id} for document {document_id}")
    
    async def get_entities_by_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """Get all entities mentioned in a specific chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            List of entity information with mention details
        """
        # Search for mapping episodes containing this chunk
        search_results = await self.graphiti.search(
            query=f"chunk_id:{chunk_id} adapter_type:chunk_entity_mapping",
            limit=10
        )
        
        entities = []
        
        for result in search_results:
            metadata = getattr(result, 'metadata', {})
            mappings = metadata.get('chunk_entity_mappings', [])
            
            for mapping in mappings:
                if mapping['chunk_id'] == chunk_id:
                    for entity_id in mapping['entity_ids']:
                        entity_info = {
                            'entity_id': entity_id,
                            'confidence': mapping['confidence_scores'].get(entity_id, 0.0),
                            'context': mapping['mention_contexts'].get(entity_id, ''),
                            'chunk_id': chunk_id
                        }
                        entities.append(entity_info)
        
        return entities
    
    async def get_chunks_by_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all chunks where an entity is mentioned.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            List of chunk information with mention details
        """
        # Search for mapping episodes containing this entity
        search_results = await self.graphiti.search(
            query=f"entity_ids:{entity_id} adapter_type:chunk_entity_mapping",
            limit=50
        )
        
        chunks = []
        
        for result in search_results:
            metadata = getattr(result, 'metadata', {})
            mappings = metadata.get('chunk_entity_mappings', [])
            
            for mapping in mappings:
                if entity_id in mapping['entity_ids']:
                    chunk_info = {
                        'chunk_id': mapping['chunk_id'],
                        'confidence': mapping['confidence_scores'].get(entity_id, 0.0),
                        'context': mapping['mention_contexts'].get(entity_id, ''),
                        'entity_id': entity_id
                    }
                    chunks.append(chunk_info)
        
        return chunks
    
    async def find_related_entities(
        self, 
        entity_id: str, 
        max_distance: int = 2
    ) -> List[Dict[str, Any]]:
        """Find entities related to a given entity through chunk co-occurrence.
        
        Args:
            entity_id: Source entity ID
            max_distance: Maximum relationship distance
            
        Returns:
            List of related entities with relationship information
        """
        # Get chunks where the source entity appears
        source_chunks = await self.get_chunks_by_entity(entity_id)
        
        # Find other entities in the same chunks
        related_entities = defaultdict(list)
        
        for chunk_info in source_chunks:
            chunk_id = chunk_info['chunk_id']
            chunk_entities = await self.get_entities_by_chunk(chunk_id)
            
            for entity_info in chunk_entities:
                other_entity_id = entity_info['entity_id']
                if other_entity_id != entity_id:
                    related_entities[other_entity_id].append({
                        'chunk_id': chunk_id,
                        'confidence': entity_info['confidence'],
                        'context': entity_info['context']
                    })
        
        # Format results with relationship strength
        results = []
        for related_entity_id, occurrences in related_entities.items():
            relationship_strength = len(occurrences)
            avg_confidence = sum(occ['confidence'] for occ in occurrences) / len(occurrences)
            
            results.append({
                'entity_id': related_entity_id,
                'relationship_strength': relationship_strength,
                'average_confidence': avg_confidence,
                'co_occurrences': occurrences,
                'distance': 1  # Direct co-occurrence
            })
        
        # Sort by relationship strength
        results.sort(key=lambda x: x['relationship_strength'], reverse=True)
        
        return results
    
    async def validate_chunk_entity_integrity(self, document_id: str) -> Dict[str, Any]:
        """Validate the integrity of chunk-entity relationships for a document.
        
        Args:
            document_id: Document ID to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            "document_id": document_id,
            "total_mappings": 0,
            "valid_mappings": 0,
            "invalid_mappings": 0,
            "missing_entities": [],
            "missing_chunks": [],
            "errors": []
        }
        
        try:
            # Find mapping episodes for this document
            search_results = await self.graphiti.search(
                query=f"document_id:{document_id} adapter_type:chunk_entity_mapping",
                limit=10
            )
            
            for result in search_results:
                metadata = getattr(result, 'metadata', {})
                mappings = metadata.get('chunk_entity_mappings', [])
                validation_results["total_mappings"] += len(mappings)
                
                for mapping in mappings:
                    try:
                        # Validate chunk exists
                        chunk_id = mapping['chunk_id']
                        # Note: In a full implementation, we would check if chunk episode exists
                        
                        # Validate entities exist
                        for entity_id in mapping['entity_ids']:
                            entity_exists = await self.entity_storage._entity_exists(entity_id)
                            if not entity_exists:
                                validation_results["missing_entities"].append(entity_id)
                        
                        validation_results["valid_mappings"] += 1
                        
                    except Exception as e:
                        validation_results["invalid_mappings"] += 1
                        validation_results["errors"].append(str(e))
        
        except Exception as e:
            validation_results["errors"].append(f"Validation failed: {str(e)}")
        
        return validation_results


class ChunkEntitySearchService:
    """Enhanced search service that leverages chunk-entity relationships."""
    
    def __init__(self, relationship_service: ChunkEntityRelationshipService):
        self.relationship_service = relationship_service
        self.search_service = relationship_service.search_service
    
    async def search_by_entity_context(
        self, 
        entity_id: str, 
        context_query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for content related to an entity with specific context.
        
        Args:
            entity_id: Entity ID to search for
            context_query: Additional context to search for
            limit: Maximum results
            
        Returns:
            List of search results with entity context
        """
        # Get chunks where the entity appears
        entity_chunks = await self.relationship_service.get_chunks_by_entity(entity_id)
        
        # Search within those chunks for the context
        results = []
        
        for chunk_info in entity_chunks:
            chunk_id = chunk_info['chunk_id']
            
            # Search for the context query within this chunk's content
            chunk_results = await self.search_service.search(
                query=f"chunk_id:{chunk_id} {context_query}",
                limit=5
            )
            
            for search_result in chunk_results[0]:  # search returns (results, metrics)
                result_data = {
                    'content': search_result.content,
                    'score': search_result.score,
                    'chunk_id': chunk_id,
                    'entity_id': entity_id,
                    'entity_confidence': chunk_info['confidence'],
                    'entity_context': chunk_info['context']
                }
                results.append(result_data)
        
        # Sort by combined score
        results.sort(key=lambda x: x['score'] * x['entity_confidence'], reverse=True)
        
        return results[:limit]
    
    async def search_entity_relationships(
        self, 
        entity_id: str, 
        relationship_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for entities related to a given entity.
        
        Args:
            entity_id: Source entity ID
            relationship_type: Optional relationship type filter
            limit: Maximum results
            
        Returns:
            List of related entities with relationship details
        """
        related_entities = await self.relationship_service.find_related_entities(entity_id)
        
        # Filter by relationship type if specified
        if relationship_type:
            # This would require more sophisticated relationship type detection
            # For now, we return all related entities
            pass
        
        return related_entities[:limit]
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_chunk_entity_relationships.py`

```python
"""Unit tests for chunk-entity relationship handling."""

import pytest
from unittest.mock import Mock, AsyncMock
from morag_graph.models import Entity, DocumentChunk, EntityType
from morag_graph.graphiti.chunk_entity_service import ChunkEntityRelationshipService, ChunkEntityMapping


class TestChunkEntityRelationshipService:
    """Test chunk-entity relationship service."""
    
    @pytest.fixture
    def mock_service(self):
        """Create mock relationship service."""
        mock_entity_storage = Mock()
        mock_search_service = Mock()
        mock_graphiti = Mock()
        mock_graphiti.add_episode = AsyncMock()
        mock_graphiti.search = AsyncMock()
        
        service = ChunkEntityRelationshipService(mock_entity_storage, mock_search_service)
        service.graphiti = mock_graphiti
        return service
    
    @pytest.fixture
    def sample_data(self):
        """Create sample chunks and entities."""
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_1",
                chunk_index=0,
                text="John Doe works at Microsoft Corporation in Seattle."
            ),
            DocumentChunk(
                id="chunk_2",
                document_id="doc_1",
                chunk_index=1,
                text="The company Microsoft was founded by Bill Gates."
            )
        ]
        
        entities = [
            Entity(id="e1", name="John Doe", type=EntityType.PERSON),
            Entity(id="e2", name="Microsoft Corporation", type=EntityType.ORGANIZATION),
            Entity(id="e3", name="Seattle", type=EntityType.LOCATION),
            Entity(id="e4", name="Bill Gates", type=EntityType.PERSON)
        ]
        
        return chunks, entities
    
    @pytest.mark.asyncio
    async def test_create_chunk_entity_mappings(self, mock_service, sample_data):
        """Test creation of chunk-entity mappings."""
        chunks, entities = sample_data
        
        mock_service.graphiti.add_episode.return_value = "mapping_episode_1"
        
        mappings = await mock_service.create_chunk_entity_mappings(chunks, entities, "doc_1")
        
        assert len(mappings) == 2  # Two chunks with entities
        
        # Check first chunk mapping
        chunk1_mapping = next(m for m in mappings if m.chunk_id == "chunk_1")
        assert "e1" in chunk1_mapping.entity_ids  # John Doe
        assert "e2" in chunk1_mapping.entity_ids  # Microsoft Corporation
        assert "e3" in chunk1_mapping.entity_ids  # Seattle
        
        # Check second chunk mapping
        chunk2_mapping = next(m for m in mappings if m.chunk_id == "chunk_2")
        assert "e2" in chunk2_mapping.entity_ids  # Microsoft Corporation
        assert "e4" in chunk2_mapping.entity_ids  # Bill Gates
        
        # Verify episode was created
        mock_service.graphiti.add_episode.assert_called_once()
    
    def test_analyze_entity_mention(self, mock_service, sample_data):
        """Test entity mention analysis."""
        chunks, entities = sample_data
        chunk = chunks[0]  # "John Doe works at Microsoft Corporation in Seattle."
        entity = entities[0]  # John Doe
        
        confidence, context = mock_service._analyze_entity_mention(chunk, entity)
        
        assert confidence > 0.5  # Should have reasonable confidence
        assert "John Doe" in context
        assert len(context) > 0
    
    @pytest.mark.asyncio
    async def test_find_entities_in_chunk(self, mock_service, sample_data):
        """Test finding entities in a chunk."""
        chunks, entities = sample_data
        chunk = chunks[0]  # "John Doe works at Microsoft Corporation in Seattle."
        
        found_entities = await mock_service._find_entities_in_chunk(chunk, entities)
        
        # Should find John Doe, Microsoft Corporation, and Seattle
        found_names = [e.name for e in found_entities]
        assert "John Doe" in found_names
        assert "Microsoft Corporation" in found_names
        assert "Seattle" in found_names
        assert "Bill Gates" not in found_names  # Not in this chunk
    
    @pytest.mark.asyncio
    async def test_get_entities_by_chunk(self, mock_service):
        """Test getting entities by chunk ID."""
        # Mock search results
        mock_result = Mock()
        mock_result.metadata = {
            'chunk_entity_mappings': [
                {
                    'chunk_id': 'chunk_1',
                    'entity_ids': ['e1', 'e2'],
                    'confidence_scores': {'e1': 0.9, 'e2': 0.8},
                    'mention_contexts': {'e1': 'John Doe context', 'e2': 'Microsoft context'}
                }
            ]
        }
        
        mock_service.graphiti.search.return_value = [mock_result]
        
        entities = await mock_service.get_entities_by_chunk('chunk_1')
        
        assert len(entities) == 2
        assert entities[0]['entity_id'] in ['e1', 'e2']
        assert entities[0]['confidence'] > 0
        assert len(entities[0]['context']) > 0
    
    @pytest.mark.asyncio
    async def test_find_related_entities(self, mock_service):
        """Test finding related entities through co-occurrence."""
        # Mock chunks for entity e1
        mock_service.get_chunks_by_entity = AsyncMock(return_value=[
            {'chunk_id': 'chunk_1', 'confidence': 0.9, 'context': 'context1'}
        ])
        
        # Mock entities in chunk_1
        mock_service.get_entities_by_chunk = AsyncMock(return_value=[
            {'entity_id': 'e1', 'confidence': 0.9, 'context': 'context1'},
            {'entity_id': 'e2', 'confidence': 0.8, 'context': 'context2'}
        ])
        
        related = await mock_service.find_related_entities('e1')
        
        assert len(related) == 1
        assert related[0]['entity_id'] == 'e2'
        assert related[0]['relationship_strength'] == 1
        assert related[0]['distance'] == 1


class TestChunkEntitySearchService:
    """Test chunk-entity search service."""
    
    @pytest.fixture
    def mock_search_service(self):
        """Create mock search service."""
        mock_relationship_service = Mock()
        mock_relationship_service.get_chunks_by_entity = AsyncMock()
        mock_relationship_service.search_service = Mock()
        mock_relationship_service.search_service.search = AsyncMock()
        
        from morag_graph.graphiti.chunk_entity_service import ChunkEntitySearchService
        return ChunkEntitySearchService(mock_relationship_service)
    
    @pytest.mark.asyncio
    async def test_search_by_entity_context(self, mock_search_service):
        """Test searching by entity context."""
        # Mock entity chunks
        mock_search_service.relationship_service.get_chunks_by_entity.return_value = [
            {'chunk_id': 'chunk_1', 'confidence': 0.9, 'context': 'entity context'}
        ]
        
        # Mock search results
        from morag_graph.graphiti.search_service import SearchResult, SearchMetrics
        mock_result = SearchResult("Test content", 0.8, "doc_1", "chunk_1")
        mock_metrics = SearchMetrics(0.1, 1, 1, "test")
        
        mock_search_service.relationship_service.search_service.search.return_value = (
            [mock_result], mock_metrics
        )
        
        results = await mock_search_service.search_by_entity_context('e1', 'test query')
        
        assert len(results) == 1
        assert results[0]['entity_id'] == 'e1'
        assert results[0]['chunk_id'] == 'chunk_1'
        assert results[0]['content'] == "Test content"
```

## Validation Checklist

- [ ] Chunk-entity mappings are created accurately
- [ ] Entity mentions are detected with appropriate confidence scores
- [ ] Context extraction provides meaningful surrounding text
- [ ] Relationship storage in episode metadata works correctly
- [ ] Entity-by-chunk and chunk-by-entity queries function properly
- [ ] Related entity discovery through co-occurrence works
- [ ] Search functionality leverages chunk-entity relationships
- [ ] Validation methods detect integrity issues
- [ ] Performance is acceptable for large documents
- [ ] Unit tests cover all major functionality

## Success Criteria

1. **Accurate Mapping**: Chunk-entity relationships are correctly identified and stored
2. **Searchable**: Entity mentions can be found through chunk-based queries
3. **Contextual**: Entity mentions include surrounding context for relevance
4. **Relational**: Related entities can be discovered through co-occurrence
5. **Performant**: Operations scale well with document and entity count

## Next Steps

After completing this step:
1. Test with documents containing many entities
2. Validate relationship accuracy with manual review
3. Optimize performance for large-scale processing
4. Proceed to [Step 8: Temporal Query Implementation](./step-08-temporal-queries.md)

## Performance Considerations

- Batch processing for large documents with many entities
- Caching of frequently accessed chunk-entity mappings
- Efficient search indexing for relationship queries
- Memory management for large context windows
