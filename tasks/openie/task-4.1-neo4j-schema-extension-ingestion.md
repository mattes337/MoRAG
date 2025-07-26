# Task 4.1: Neo4j Schema Extension and Triplet Ingestion

## Objective
Extend the Neo4j graph schema to support OpenIE-extracted triplets and implement efficient batch ingestion pipeline for relationship data with proper provenance tracking.

## Scope
- Extend Neo4j schema for OpenIE relationships
- Implement batch ingestion pipeline for triplets
- Add provenance tracking and metadata storage
- Create relationship indexing and optimization
- **MANDATORY**: Test thoroughly before proceeding to Task 4.2

## Implementation Details

### 1. Schema Extension

**File**: `packages/morag-graph/src/morag_graph/schema/openie_schema.py`

```python
"""Neo4j schema extensions for OpenIE relationships."""

from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class OpenIESchema:
    """Schema definitions for OpenIE relationships in Neo4j."""
    
    def __init__(self):
        """Initialize OpenIE schema definitions."""
        self.relationship_types = self._define_relationship_types()
        self.node_labels = self._define_node_labels()
        self.indexes = self._define_indexes()
        self.constraints = self._define_constraints()
    
    def _define_relationship_types(self) -> Dict[str, Dict[str, Any]]:
        """Define OpenIE relationship types and their properties."""
        return {
            'WORKS_AT': {
                'description': 'Employment relationship',
                'properties': {
                    'confidence': 'float',
                    'extraction_method': 'string',
                    'source_sentence': 'string',
                    'document_id': 'string',
                    'sentence_index': 'integer',
                    'created_at': 'datetime',
                    'original_predicate': 'string',
                    'normalization_confidence': 'float'
                }
            },
            'OWNS': {
                'description': 'Ownership relationship',
                'properties': {
                    'confidence': 'float',
                    'extraction_method': 'string',
                    'source_sentence': 'string',
                    'document_id': 'string',
                    'sentence_index': 'integer',
                    'created_at': 'datetime',
                    'original_predicate': 'string',
                    'normalization_confidence': 'float'
                }
            },
            'LOCATED_IN': {
                'description': 'Location relationship',
                'properties': {
                    'confidence': 'float',
                    'extraction_method': 'string',
                    'source_sentence': 'string',
                    'document_id': 'string',
                    'sentence_index': 'integer',
                    'created_at': 'datetime',
                    'original_predicate': 'string',
                    'normalization_confidence': 'float'
                }
            },
            'LEADS': {
                'description': 'Leadership relationship',
                'properties': {
                    'confidence': 'float',
                    'extraction_method': 'string',
                    'source_sentence': 'string',
                    'document_id': 'string',
                    'sentence_index': 'integer',
                    'created_at': 'datetime',
                    'original_predicate': 'string',
                    'normalization_confidence': 'float'
                }
            },
            'MEMBER_OF': {
                'description': 'Membership relationship',
                'properties': {
                    'confidence': 'float',
                    'extraction_method': 'string',
                    'source_sentence': 'string',
                    'document_id': 'string',
                    'sentence_index': 'integer',
                    'created_at': 'datetime',
                    'original_predicate': 'string',
                    'normalization_confidence': 'float'
                }
            },
            'CREATED': {
                'description': 'Creation relationship',
                'properties': {
                    'confidence': 'float',
                    'extraction_method': 'string',
                    'source_sentence': 'string',
                    'document_id': 'string',
                    'sentence_index': 'integer',
                    'created_at': 'datetime',
                    'original_predicate': 'string',
                    'normalization_confidence': 'float'
                }
            },
            'STUDIED_AT': {
                'description': 'Education relationship',
                'properties': {
                    'confidence': 'float',
                    'extraction_method': 'string',
                    'source_sentence': 'string',
                    'document_id': 'string',
                    'sentence_index': 'integer',
                    'created_at': 'datetime',
                    'original_predicate': 'string',
                    'normalization_confidence': 'float'
                }
            },
            'RELATED_TO': {
                'description': 'Family/personal relationship',
                'properties': {
                    'confidence': 'float',
                    'extraction_method': 'string',
                    'source_sentence': 'string',
                    'document_id': 'string',
                    'sentence_index': 'integer',
                    'created_at': 'datetime',
                    'original_predicate': 'string',
                    'normalization_confidence': 'float',
                    'relationship_type': 'string'  # spouse, parent, child, etc.
                }
            },
            'OPENIE_RELATION': {
                'description': 'Generic OpenIE relationship for unmapped predicates',
                'properties': {
                    'predicate': 'string',
                    'confidence': 'float',
                    'extraction_method': 'string',
                    'source_sentence': 'string',
                    'document_id': 'string',
                    'sentence_index': 'integer',
                    'created_at': 'datetime',
                    'original_predicate': 'string',
                    'normalization_confidence': 'float'
                }
            }
        }
    
    def _define_node_labels(self) -> Dict[str, Dict[str, Any]]:
        """Define additional node labels for OpenIE entities."""
        return {
            'OpenIEEntity': {
                'description': 'Entity extracted via OpenIE',
                'properties': {
                    'text': 'string',
                    'normalized_text': 'string',
                    'entity_type': 'string',
                    'confidence': 'float',
                    'extraction_method': 'string',
                    'document_id': 'string',
                    'created_at': 'datetime',
                    'variations': 'list[string]',
                    'canonical_form': 'string'
                }
            },
            'OpenIETriplet': {
                'description': 'Metadata node for OpenIE triplets',
                'properties': {
                    'triplet_id': 'string',
                    'subject_text': 'string',
                    'predicate_text': 'string',
                    'object_text': 'string',
                    'confidence': 'float',
                    'quality_score': 'float',
                    'source_sentence': 'string',
                    'document_id': 'string',
                    'sentence_index': 'integer',
                    'created_at': 'datetime',
                    'extraction_method': 'string'
                }
            }
        }
    
    def _define_indexes(self) -> List[Dict[str, Any]]:
        """Define indexes for OpenIE data."""
        return [
            {
                'type': 'btree',
                'label': 'OpenIEEntity',
                'property': 'normalized_text',
                'name': 'idx_openie_entity_normalized_text'
            },
            {
                'type': 'btree',
                'label': 'OpenIEEntity',
                'property': 'canonical_form',
                'name': 'idx_openie_entity_canonical_form'
            },
            {
                'type': 'btree',
                'label': 'OpenIETriplet',
                'property': 'triplet_id',
                'name': 'idx_openie_triplet_id'
            },
            {
                'type': 'btree',
                'label': 'OpenIETriplet',
                'property': 'document_id',
                'name': 'idx_openie_triplet_document_id'
            },
            {
                'type': 'btree',
                'property': 'confidence',
                'name': 'idx_openie_relationship_confidence'
            },
            {
                'type': 'btree',
                'property': 'extraction_method',
                'name': 'idx_openie_extraction_method'
            }
        ]
    
    def _define_constraints(self) -> List[Dict[str, Any]]:
        """Define constraints for OpenIE data."""
        return [
            {
                'type': 'unique',
                'label': 'OpenIETriplet',
                'property': 'triplet_id',
                'name': 'constraint_openie_triplet_id_unique'
            }
        ]
    
    def get_schema_creation_queries(self) -> List[str]:
        """Get Cypher queries to create schema."""
        queries = []
        
        # Create indexes
        for index in self.indexes:
            if index['type'] == 'btree':
                if 'label' in index:
                    query = f"CREATE INDEX {index['name']} IF NOT EXISTS FOR (n:{index['label']}) ON (n.{index['property']})"
                else:
                    query = f"CREATE INDEX {index['name']} IF NOT EXISTS FOR ()-[r]-() ON (r.{index['property']})"
                queries.append(query)
        
        # Create constraints
        for constraint in self.constraints:
            if constraint['type'] == 'unique':
                query = f"CREATE CONSTRAINT {constraint['name']} IF NOT EXISTS FOR (n:{constraint['label']}) REQUIRE n.{constraint['property']} IS UNIQUE"
                queries.append(query)
        
        return queries
```

### 2. Triplet Ingestion Service

**File**: `packages/morag-graph/src/morag_graph/services/openie_ingestion_service.py`

```python
"""OpenIE triplet ingestion service for Neo4j."""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import structlog
from neo4j import AsyncGraphDatabase

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from morag_graph.schema.openie_schema import OpenIESchema

logger = structlog.get_logger(__name__)

class OpenIEIngestionService:
    """Service for ingesting OpenIE triplets into Neo4j."""
    
    def __init__(self, neo4j_driver, config: Optional[Dict[str, Any]] = None):
        """Initialize ingestion service.
        
        Args:
            neo4j_driver: Neo4j async driver instance
            config: Optional configuration overrides
        """
        self.driver = neo4j_driver
        self.settings = get_settings()
        self.config = config or {}
        self.schema = OpenIESchema()
        
        # Batch configuration
        self.batch_size = self.config.get('batch_size', 1000)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
    
    async def initialize_schema(self) -> None:
        """Initialize Neo4j schema for OpenIE data."""
        try:
            schema_queries = self.schema.get_schema_creation_queries()
            
            async with self.driver.session() as session:
                for query in schema_queries:
                    await session.run(query)
                    logger.debug("Executed schema query", query=query)
            
            logger.info("OpenIE schema initialized", queries_executed=len(schema_queries))
            
        except Exception as e:
            logger.error("Schema initialization failed", error=str(e))
            raise ProcessingError(f"Schema initialization failed: {str(e)}")
    
    async def ingest_triplets(
        self, 
        triplets: List[Dict[str, Any]], 
        document_id: str
    ) -> Dict[str, Any]:
        """Ingest OpenIE triplets into Neo4j.
        
        Args:
            triplets: List of processed triplets
            document_id: Source document identifier
            
        Returns:
            Ingestion results and statistics
            
        Raises:
            ProcessingError: If ingestion fails
        """
        if not triplets:
            return {'triplets_ingested': 0, 'relationships_created': 0, 'nodes_created': 0}
        
        try:
            # Process triplets in batches
            total_ingested = 0
            total_relationships = 0
            total_nodes = 0
            
            for i in range(0, len(triplets), self.batch_size):
                batch = triplets[i:i + self.batch_size]
                batch_results = await self._ingest_batch(batch, document_id)
                
                total_ingested += batch_results['triplets_processed']
                total_relationships += batch_results['relationships_created']
                total_nodes += batch_results['nodes_created']
                
                logger.debug(
                    "Batch ingested",
                    batch_size=len(batch),
                    batch_index=i // self.batch_size + 1,
                    total_batches=(len(triplets) + self.batch_size - 1) // self.batch_size
                )
            
            results = {
                'triplets_ingested': total_ingested,
                'relationships_created': total_relationships,
                'nodes_created': total_nodes,
                'document_id': document_id
            }
            
            logger.info("Triplet ingestion completed", **results)
            return results
            
        except Exception as e:
            logger.error("Triplet ingestion failed", error=str(e), document_id=document_id)
            raise ProcessingError(f"Triplet ingestion failed: {str(e)}")
    
    async def _ingest_batch(self, batch: List[Dict[str, Any]], document_id: str) -> Dict[str, Any]:
        """Ingest a batch of triplets.
        
        Args:
            batch: Batch of triplets to ingest
            document_id: Source document identifier
            
        Returns:
            Batch ingestion results
        """
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                async with self.driver.session() as session:
                    # Prepare batch data
                    batch_data = await self._prepare_batch_data(batch, document_id)
                    
                    # Execute batch ingestion
                    result = await session.execute_write(
                        self._execute_batch_ingestion, 
                        batch_data
                    )
                    
                    return result
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(
                        "Batch ingestion failed after retries",
                        error=str(e),
                        retry_count=retry_count,
                        batch_size=len(batch)
                    )
                    raise
                
                logger.warning(
                    "Batch ingestion failed, retrying",
                    error=str(e),
                    retry_count=retry_count,
                    max_retries=self.max_retries
                )
                await asyncio.sleep(self.retry_delay * retry_count)
    
    async def _prepare_batch_data(self, batch: List[Dict[str, Any]], document_id: str) -> Dict[str, Any]:
        """Prepare batch data for ingestion.
        
        Args:
            batch: Batch of triplets
            document_id: Source document identifier
            
        Returns:
            Prepared batch data
        """
        current_time = datetime.utcnow().isoformat()
        
        triplet_data = []
        entity_data = []
        
        for triplet in batch:
            # Prepare triplet metadata
            triplet_info = {
                'triplet_id': triplet.get('triplet_id', ''),
                'subject_text': triplet.get('subject', ''),
                'predicate_text': triplet.get('predicate', ''),
                'object_text': triplet.get('object', ''),
                'confidence': triplet.get('confidence', 0.0),
                'quality_score': triplet.get('quality_score', 0.0),
                'source_sentence': triplet.get('source_sentence', ''),
                'document_id': document_id,
                'sentence_index': triplet.get('sentence_index', 0),
                'created_at': current_time,
                'extraction_method': triplet.get('extraction_method', 'stanford_openie'),
                'original_predicate': triplet.get('original_predicate', ''),
                'normalization_confidence': triplet.get('predicate_normalization', {}).get('confidence', 0.0)
            }
            
            triplet_data.append(triplet_info)
            
            # Prepare entity data
            for entity_key in ['subject_entity', 'object_entity']:
                if entity_key in triplet:
                    entity = triplet[entity_key]
                    entity_info = {
                        'text': entity.get('text', ''),
                        'normalized_text': entity.get('normalized_text', ''),
                        'entity_type': entity.get('label', 'UNKNOWN'),
                        'confidence': entity.get('match_confidence', 0.0),
                        'extraction_method': 'spacy_ner',
                        'document_id': document_id,
                        'created_at': current_time,
                        'variations': entity.get('all_variations', []),
                        'canonical_form': entity.get('canonical_form', '')
                    }
                    entity_data.append(entity_info)
        
        return {
            'triplets': triplet_data,
            'entities': entity_data,
            'document_id': document_id
        }
    
    async def _execute_batch_ingestion(self, tx, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute batch ingestion transaction.
        
        Args:
            tx: Neo4j transaction
            batch_data: Prepared batch data
            
        Returns:
            Ingestion results
        """
        nodes_created = 0
        relationships_created = 0
        
        # 1. Create/merge entities
        if batch_data['entities']:
            entity_query = """
            UNWIND $entities AS entity
            MERGE (e:Entity:OpenIEEntity {canonical_form: entity.canonical_form})
            ON CREATE SET 
                e.text = entity.text,
                e.normalized_text = entity.normalized_text,
                e.entity_type = entity.entity_type,
                e.confidence = entity.confidence,
                e.extraction_method = entity.extraction_method,
                e.document_id = entity.document_id,
                e.created_at = entity.created_at,
                e.variations = entity.variations
            ON MATCH SET
                e.confidence = CASE WHEN entity.confidence > e.confidence THEN entity.confidence ELSE e.confidence END
            RETURN count(e) as entities_processed
            """
            
            result = await tx.run(entity_query, entities=batch_data['entities'])
            record = await result.single()
            if record:
                nodes_created += record['entities_processed']
        
        # 2. Create triplet metadata nodes
        triplet_query = """
        UNWIND $triplets AS triplet
        CREATE (t:OpenIETriplet)
        SET t = triplet
        RETURN count(t) as triplets_created
        """
        
        result = await tx.run(triplet_query, triplets=batch_data['triplets'])
        record = await result.single()
        if record:
            nodes_created += record['triplets_created']
        
        # 3. Create relationships
        for triplet in batch_data['triplets']:
            relationship_type = triplet['predicate_text']
            
            # Use generic OPENIE_RELATION if not a standard type
            if relationship_type not in self.schema.relationship_types:
                relationship_type = 'OPENIE_RELATION'
            
            rel_query = f"""
            MATCH (s:Entity {{canonical_form: $subject_canonical}})
            MATCH (o:Entity {{canonical_form: $object_canonical}})
            MATCH (t:OpenIETriplet {{triplet_id: $triplet_id}})
            CREATE (s)-[r:{relationship_type}]->(o)
            SET r.confidence = $confidence,
                r.extraction_method = $extraction_method,
                r.source_sentence = $source_sentence,
                r.document_id = $document_id,
                r.sentence_index = $sentence_index,
                r.created_at = $created_at,
                r.original_predicate = $original_predicate,
                r.normalization_confidence = $normalization_confidence
            CREATE (t)-[:REPRESENTS]->(r)
            RETURN count(r) as relationships_created
            """
            
            # Get canonical forms for subject and object
            subject_canonical = triplet['subject_text']  # Would be enhanced with entity linking
            object_canonical = triplet['object_text']    # Would be enhanced with entity linking
            
            result = await tx.run(
                rel_query,
                subject_canonical=subject_canonical,
                object_canonical=object_canonical,
                triplet_id=triplet['triplet_id'],
                confidence=triplet['confidence'],
                extraction_method=triplet['extraction_method'],
                source_sentence=triplet['source_sentence'],
                document_id=triplet['document_id'],
                sentence_index=triplet['sentence_index'],
                created_at=triplet['created_at'],
                original_predicate=triplet['original_predicate'],
                normalization_confidence=triplet['normalization_confidence']
            )
            
            record = await result.single()
            if record:
                relationships_created += record['relationships_created']
        
        return {
            'triplets_processed': len(batch_data['triplets']),
            'nodes_created': nodes_created,
            'relationships_created': relationships_created
        }
    
    async def get_ingestion_statistics(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Get ingestion statistics.
        
        Args:
            document_id: Optional document ID to filter by
            
        Returns:
            Ingestion statistics
        """
        try:
            async with self.driver.session() as session:
                if document_id:
                    query = """
                    MATCH (t:OpenIETriplet {document_id: $document_id})
                    OPTIONAL MATCH ()-[r {document_id: $document_id}]->()
                    RETURN 
                        count(DISTINCT t) as triplet_count,
                        count(DISTINCT r) as relationship_count,
                        avg(t.confidence) as avg_confidence,
                        avg(t.quality_score) as avg_quality
                    """
                    result = await session.run(query, document_id=document_id)
                else:
                    query = """
                    MATCH (t:OpenIETriplet)
                    OPTIONAL MATCH ()-[r]->() WHERE r.extraction_method = 'stanford_openie'
                    RETURN 
                        count(DISTINCT t) as triplet_count,
                        count(DISTINCT r) as relationship_count,
                        avg(t.confidence) as avg_confidence,
                        avg(t.quality_score) as avg_quality
                    """
                    result = await session.run(query)
                
                record = await result.single()
                if record:
                    return {
                        'triplet_count': record['triplet_count'],
                        'relationship_count': record['relationship_count'],
                        'avg_confidence': record['avg_confidence'] or 0.0,
                        'avg_quality': record['avg_quality'] or 0.0,
                        'document_id': document_id
                    }
                
                return {
                    'triplet_count': 0,
                    'relationship_count': 0,
                    'avg_confidence': 0.0,
                    'avg_quality': 0.0,
                    'document_id': document_id
                }
                
        except Exception as e:
            logger.error("Failed to get ingestion statistics", error=str(e))
            raise ProcessingError(f"Failed to get ingestion statistics: {str(e)}")
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_openie_ingestion_service.py`

```python
"""Tests for OpenIE ingestion service."""

import pytest
from unittest.mock import AsyncMock, Mock
from morag_graph.services.openie_ingestion_service import OpenIEIngestionService

class TestOpenIEIngestionService:
    
    @pytest.fixture
    def mock_driver(self):
        """Mock Neo4j driver."""
        driver = AsyncMock()
        session = AsyncMock()
        driver.session.return_value.__aenter__.return_value = session
        return driver
    
    def test_initialization(self, mock_driver):
        """Test service initialization."""
        service = OpenIEIngestionService(mock_driver)
        assert service.driver == mock_driver
        assert service.batch_size == 1000
    
    @pytest.mark.asyncio
    async def test_prepare_batch_data(self, mock_driver):
        """Test batch data preparation."""
        service = OpenIEIngestionService(mock_driver)
        
        triplets = [{
            'triplet_id': 'test_123',
            'subject': 'John',
            'predicate': 'WORKS_AT',
            'object': 'Microsoft',
            'confidence': 0.8,
            'quality_score': 0.9,
            'source_sentence': 'John works at Microsoft.'
        }]
        
        batch_data = await service._prepare_batch_data(triplets, 'doc_001')
        
        assert len(batch_data['triplets']) == 1
        assert batch_data['triplets'][0]['subject_text'] == 'John'
        assert batch_data['triplets'][0]['document_id'] == 'doc_001'
    
    @pytest.mark.asyncio
    async def test_ingest_triplets(self, mock_driver):
        """Test triplet ingestion."""
        service = OpenIEIngestionService(mock_driver)
        
        # Mock the batch ingestion
        service._ingest_batch = AsyncMock(return_value={
            'triplets_processed': 1,
            'relationships_created': 1,
            'nodes_created': 2
        })
        
        triplets = [{
            'triplet_id': 'test_123',
            'subject': 'John',
            'predicate': 'WORKS_AT',
            'object': 'Microsoft',
            'confidence': 0.8
        }]
        
        result = await service.ingest_triplets(triplets, 'doc_001')
        
        assert result['triplets_ingested'] == 1
        assert result['relationships_created'] == 1
        assert result['nodes_created'] == 2
```

## Acceptance Criteria

- [ ] OpenIESchema class with comprehensive relationship type definitions
- [ ] OpenIEIngestionService with batch processing capabilities
- [ ] Neo4j schema extension with proper indexes and constraints
- [ ] Provenance tracking for all ingested relationships
- [ ] Error handling and retry mechanisms for ingestion
- [ ] Performance optimization for large triplet sets
- [ ] Comprehensive unit tests with >90% coverage
- [ ] Integration with existing graph infrastructure
- [ ] Statistics and monitoring capabilities
- [ ] Proper logging and error handling

## Dependencies
- Task 1.3: Basic Triplet Extraction and Validation
- Task 2.1: Entity Linking Between OpenIE and spaCy NER
- Task 2.2: Entity Normalization and Canonical Mapping
- Task 3.1: Predicate Normalization and Standardization

## Estimated Effort
- **Development**: 10-12 hours
- **Testing**: 5-6 hours
- **Integration**: 4-5 hours
- **Total**: 19-23 hours

## Notes
- Focus on data integrity and consistency in the graph
- Implement proper transaction handling for batch operations
- Consider graph query performance implications
- Plan for schema evolution and migration strategies
