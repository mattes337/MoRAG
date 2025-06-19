# Task 2.1: Graph Construction Pipeline

**Phase**: 2 - Core Graph Features  
**Priority**: Critical  
**Estimated Time**: 6-8 days total  
**Dependencies**: Task 1.2 (Core Graph Package), Task 1.3 (LLM-Based Entity and Relation Extraction)

## Overview

This task implements the graph construction pipeline that processes documents and builds the knowledge graph by extracting entities and relations, then storing them in the graph database. It includes integration with the existing document processing pipeline and support for incremental updates.

## Subtasks

### 2.2.1: Document Processing Integration
**Estimated Time**: 4-5 days  
**Priority**: Critical

#### Implementation Steps

1. **Graph Builder Service**
   ```python
   # src/morag_graph/builders/graph_builder.py
   from typing import List, Dict, Any
   from morag_core.models import ProcessedDocument
   from morag_nlp.processors import NLPProcessor
   from morag_graph.storage.base import BaseGraphStorage
   from morag_graph.models import Entity, Relation
   
   class GraphBuilder:
       def __init__(self, graph_storage: BaseGraphStorage, nlp_processor: NLPProcessor):
           self.graph_storage = graph_storage
           self.nlp_processor = nlp_processor
           self.logger = logging.getLogger(__name__)
       
       async def process_document(self, document: ProcessedDocument) -> GraphBuildResult:
           """Process a document and build graph entities and relations."""
           try:
               # Extract entities and relations
               entities = await self.nlp_processor.extract_entities(document.content)
               relations = await self.nlp_processor.extract_relations(document.content, entities)
               
               # Store in graph
               result = await self._store_entities_and_relations(entities, relations, document)
               
               self.logger.info(f"Processed document {document.id}: {len(entities)} entities, {len(relations)} relations")
               return result
               
           except Exception as e:
               self.logger.error(f"Error processing document {document.id}: {str(e)}")
               raise GraphBuildError(f"Failed to process document: {str(e)}")
       
       async def _store_entities_and_relations(self, entities: List[Entity], relations: List[Relation], document: ProcessedDocument) -> GraphBuildResult:
           """Store extracted entities and relations in the graph database."""
           stored_entities = []
           stored_relations = []
           
           # Store entities
           for entity in entities:
               entity.source_documents.append(document.id)
               entity_id = await self.graph_storage.create_entity(entity)
               stored_entities.append(entity_id)
           
           # Store relations
           for relation in relations:
               relation.source_document = document.id
               relation_id = await self.graph_storage.create_relation(relation)
               stored_relations.append(relation_id)
           
           return GraphBuildResult(
               document_id=document.id,
               entities_created=len(stored_entities),
               relations_created=len(stored_relations),
               entity_ids=stored_entities,
               relation_ids=stored_relations
           )
   
   @dataclass
   class GraphBuildResult:
       document_id: str
       entities_created: int
       relations_created: int
       entity_ids: List[str]
       relation_ids: List[str]
       errors: List[str] = field(default_factory=list)
       skipped: bool = False
   ```

2. **Integration with Existing Pipeline**
   ```python
   # Modify packages/morag-services/src/morag_services/tasks/ingest_tasks.py
   from morag_graph.builders import GraphBuilder
   
   @celery_app.task(bind=True, base=BaseTask)
   def build_graph_task(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
       """Build graph from processed document."""
       try:
           document = ProcessedDocument.from_dict(document_data)
           
           # Initialize graph builder
           graph_storage = get_graph_storage()
           nlp_processor = get_nlp_processor()
           graph_builder = GraphBuilder(graph_storage, nlp_processor)
           
           # Build graph
           result = asyncio.run(graph_builder.process_document(document))
           
           return {
               "status": "success",
               "document_id": document.id,
               "entities_created": result.entities_created,
               "relations_created": result.relations_created
           }
           
       except Exception as e:
           self.logger.error(f"Graph building failed: {str(e)}")
           raise self.retry(exc=e, countdown=60, max_retries=3)
   ```

3. **Parallel Processing Support**
   ```python
   class ParallelGraphBuilder(GraphBuilder):
       async def process_documents_batch(self, documents: List[ProcessedDocument]) -> List[GraphBuildResult]:
           """Process multiple documents in parallel."""
           tasks = [self.process_document(doc) for doc in documents]
           results = await asyncio.gather(*tasks, return_exceptions=True)
           
           # Handle exceptions
           successful_results = []
           for i, result in enumerate(results):
               if isinstance(result, Exception):
                   self.logger.error(f"Failed to process document {documents[i].id}: {str(result)}")
               else:
                   successful_results.append(result)
           
           return successful_results
   ```

#### Deliverables
- Graph builder service implementation
- Integration with existing document processing pipeline
- Parallel processing support
- Error handling and logging

### 2.2.2: Checksum-Based Document Updates
**Estimated Time**: 2-3 days  
**Priority**: High

#### Implementation Steps

1. **Document Checksum Manager**
   ```python
   # src/morag_graph/updates/checksum_manager.py
   import hashlib
   from typing import Optional
   from morag_core.models import ProcessedDocument
   from morag_graph.storage.base import BaseGraphStorage
   
   class DocumentChecksumManager:
       def __init__(self, graph_storage: BaseGraphStorage):
           self.graph_storage = graph_storage
           self.logger = logging.getLogger(__name__)
       
       def calculate_document_checksum(self, document: ProcessedDocument) -> str:
           """Calculate SHA-256 checksum of document content."""
           content = f"{document.content}{document.metadata}"
           return hashlib.sha256(content.encode('utf-8')).hexdigest()
       
       async def get_stored_checksum(self, document_id: str) -> Optional[str]:
           """Get stored checksum for a document from graph database."""
           return await self.graph_storage.get_document_checksum(document_id)
       
       async def store_document_checksum(self, document_id: str, checksum: str) -> None:
           """Store document checksum in graph database."""
           await self.graph_storage.store_document_checksum(document_id, checksum)
       
       async def needs_update(self, document: ProcessedDocument) -> bool:
           """Check if document needs to be updated based on checksum comparison."""
           current_checksum = self.calculate_document_checksum(document)
           stored_checksum = await self.get_stored_checksum(document.id)
           
           if stored_checksum is None:
               self.logger.info(f"Document {document.id} not found in graph, needs processing")
               return True
           
           if current_checksum != stored_checksum:
               self.logger.info(f"Document {document.id} checksum changed, needs reprocessing")
               return True
           
           self.logger.info(f"Document {document.id} unchanged, skipping")
           return False
   ```

2. **Document Cleanup Manager**
   ```python
   # src/morag_graph/updates/cleanup_manager.py
   from typing import List
   from morag_graph.storage.base import BaseGraphStorage
   
   class DocumentCleanupManager:
       def __init__(self, graph_storage: BaseGraphStorage):
           self.graph_storage = graph_storage
           self.logger = logging.getLogger(__name__)
       
       async def cleanup_document_data(self, document_id: str) -> CleanupResult:
           """Remove all entities and relations associated with a document."""
           try:
               # Find all entities linked to this document
               entities = await self.graph_storage.find_entities_by_document(document_id)
               entity_ids = [entity.id for entity in entities]
               
               # Find all relations linked to this document
               relations = await self.graph_storage.find_relations_by_document(document_id)
               relation_ids = [relation.id for relation in relations]
               
               # Delete relations first (to maintain referential integrity)
               if relation_ids:
                   await self.graph_storage.delete_relations(relation_ids)
               
               # Delete entities
               if entity_ids:
                   await self.graph_storage.delete_entities(entity_ids)
               
               # Remove document checksum
               await self.graph_storage.delete_document_checksum(document_id)
               
               self.logger.info(f"Cleaned up document {document_id}: {len(entity_ids)} entities, {len(relation_ids)} relations")
               
               return CleanupResult(
                   document_id=document_id,
                   entities_deleted=len(entity_ids),
                   relations_deleted=len(relation_ids)
               )
               
           except Exception as e:
               self.logger.error(f"Error cleaning up document {document_id}: {str(e)}")
               raise
   
   @dataclass
   class CleanupResult:
       document_id: str
       entities_deleted: int
       relations_deleted: int
   ```

3. **Updated Graph Builder with Checksum Logic**
   ```python
   # Updated GraphBuilder class
   class GraphBuilder:
       def __init__(self, graph_storage: BaseGraphStorage, nlp_processor: NLPProcessor):
           self.graph_storage = graph_storage
           self.nlp_processor = nlp_processor
           self.checksum_manager = DocumentChecksumManager(graph_storage)
           self.cleanup_manager = DocumentCleanupManager(graph_storage)
           self.logger = logging.getLogger(__name__)
       
       async def process_document(self, document: ProcessedDocument) -> GraphBuildResult:
           """Process a document with checksum-based change detection."""
           try:
               # Check if document needs processing
               if not await self.checksum_manager.needs_update(document):
                   return GraphBuildResult(
                       document_id=document.id,
                       entities_created=0,
                       relations_created=0,
                       entity_ids=[],
                       relation_ids=[],
                       skipped=True
                   )
               
               # Clean up existing data for this document
               cleanup_result = await self.cleanup_manager.cleanup_document_data(document.id)
               self.logger.info(f"Cleaned up existing data: {cleanup_result.entities_deleted} entities, {cleanup_result.relations_deleted} relations")
               
               # Extract entities and relations
               entities = await self.nlp_processor.extract_entities(document.content)
               relations = await self.nlp_processor.extract_relations(document.content, entities)
               
               # Store in graph
               result = await self._store_entities_and_relations(entities, relations, document)
               
               # Store new checksum
               checksum = self.checksum_manager.calculate_document_checksum(document)
               await self.checksum_manager.store_document_checksum(document.id, checksum)
               
               self.logger.info(f"Processed document {document.id}: {len(entities)} entities, {len(relations)} relations")
               return result
               
           except Exception as e:
               self.logger.error(f"Error processing document {document.id}: {str(e)}")
               raise GraphBuildError(f"Failed to process document: {str(e)}")
   ```

#### Deliverables
- Document checksum management system
- Document cleanup and deletion mechanisms
- Efficient skip logic for unchanged documents
- Performance monitoring and logging

## Testing Requirements

### Unit Tests
```python
# tests/test_graph_construction.py
import pytest
from morag_graph.builders import GraphBuilder
from morag_graph.updates import GraphUpdateManager

class TestGraphBuilder:
    @pytest.mark.asyncio
    async def test_process_document(self, mock_graph_storage, mock_nlp_processor, sample_document):
        builder = GraphBuilder(mock_graph_storage, mock_nlp_processor)
        result = await builder.process_document(sample_document)
        
        assert result.entities_created > 0
        assert result.relations_created > 0
        assert result.document_id == sample_document.id
    
    @pytest.mark.asyncio
    async def test_parallel_processing(self, mock_graph_storage, mock_nlp_processor, sample_documents):
        builder = ParallelGraphBuilder(mock_graph_storage, mock_nlp_processor)
        results = await builder.process_documents_batch(sample_documents)
        
        assert len(results) == len(sample_documents)
        assert all(isinstance(r, GraphBuildResult) for r in results)

class TestDocumentChecksumManager:
    @pytest.mark.asyncio
    async def test_checksum_calculation(self, checksum_manager, sample_document):
        checksum = checksum_manager.calculate_document_checksum(sample_document)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex digest length
    
    @pytest.mark.asyncio
    async def test_needs_update_new_document(self, checksum_manager, sample_document):
        # Mock no existing checksum
        checksum_manager.graph_storage.get_document_checksum = AsyncMock(return_value=None)
        needs_update = await checksum_manager.needs_update(sample_document)
        assert needs_update is True
    
    @pytest.mark.asyncio
    async def test_needs_update_unchanged_document(self, checksum_manager, sample_document):
        # Mock existing matching checksum
        current_checksum = checksum_manager.calculate_document_checksum(sample_document)
        checksum_manager.graph_storage.get_document_checksum = AsyncMock(return_value=current_checksum)
        needs_update = await checksum_manager.needs_update(sample_document)
        assert needs_update is False

class TestDocumentCleanupManager:
    @pytest.mark.asyncio
    async def test_cleanup_document_data(self, cleanup_manager, sample_document_id):
        # Mock entities and relations
        mock_entities = [Mock(id="entity1"), Mock(id="entity2")]
        mock_relations = [Mock(id="relation1")]
        
        cleanup_manager.graph_storage.find_entities_by_document = AsyncMock(return_value=mock_entities)
        cleanup_manager.graph_storage.find_relations_by_document = AsyncMock(return_value=mock_relations)
        cleanup_manager.graph_storage.delete_relations = AsyncMock()
        cleanup_manager.graph_storage.delete_entities = AsyncMock()
        cleanup_manager.graph_storage.delete_document_checksum = AsyncMock()
        
        result = await cleanup_manager.cleanup_document_data(sample_document_id)
        
        assert result.entities_deleted == 2
        assert result.relations_deleted == 1
        assert result.document_id == sample_document_id
```

### Integration Tests
```python
# tests/integration/test_graph_pipeline.py
class TestGraphPipeline:
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, graph_storage, nlp_processor):
        # Test complete pipeline from document to graph
        document = create_test_document()
        builder = GraphBuilder(graph_storage, nlp_processor)
        
        result = await builder.process_document(document)
        
        # Verify entities were created
        entities = await graph_storage.find_entities_by_document(document.id)
        assert len(entities) > 0
        
        # Verify relations were created
        relations = await graph_storage.find_relations_by_document(document.id)
        assert len(relations) > 0
```

## Success Criteria

- [ ] Graph builder successfully processes documents and creates entities/relations
- [ ] Integration with existing document processing pipeline works seamlessly
- [ ] Checksum-based change detection correctly identifies unchanged documents
- [ ] Document cleanup properly removes all associated entities and relations
- [ ] Performance meets requirements (< 5 seconds per document, < 1 second for unchanged documents)
- [ ] Error handling gracefully manages failures
- [ ] Unit test coverage > 90%
- [ ] Integration tests pass

## Performance Targets

- **Document Processing**: < 5 seconds per new/changed document
- **Unchanged Document Detection**: < 1 second per document
- **Batch Processing**: > 100 documents per minute
- **Document Cleanup**: < 3 seconds per document
- **Memory Usage**: < 1GB for processing 1000 documents

## Next Steps

After completing this task:
1. Proceed to **Task 2.3**: Basic Graph Traversal
2. Integrate with **Task 3.1**: Hybrid Retrieval System
3. Implement monitoring and analytics for graph construction metrics

## Dependencies

**Requires**:
- Task 1.2.2: Graph Storage Implementation
- Task 2.1.2: ML-Based Relation Extraction

**Enables**:
- Task 2.3: Basic Graph Traversal
- Task 3.1: Hybrid Retrieval System
- Task 4.1: Multi-Hop Reasoning