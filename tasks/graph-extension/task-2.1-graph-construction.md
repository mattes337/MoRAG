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

### 2.2.2: Incremental Graph Updates
**Estimated Time**: 3-4 days  
**Priority**: High

#### Implementation Steps

1. **Change Detection System**
   ```python
   # src/morag_graph/updates/change_detector.py
   from typing import Set, List, Tuple
   from morag_graph.models import Entity, Relation, Graph
   
   class GraphUpdateManager:
       def __init__(self, graph_storage: BaseGraphStorage):
           self.graph_storage = graph_storage
           self.logger = logging.getLogger(__name__)
       
       async def detect_changes(self, new_document: ProcessedDocument, existing_entities: List[Entity]) -> ChangeSet:
           """Detect changes between new document and existing graph."""
           # Extract entities from new document
           new_entities = await self.nlp_processor.extract_entities(new_document.content)
           
           # Compare with existing entities
           changes = ChangeSet()
           
           for new_entity in new_entities:
               existing_match = self._find_matching_entity(new_entity, existing_entities)
               
               if existing_match:
                   if self._entity_changed(new_entity, existing_match):
                       changes.modified_entities.append((existing_match.id, new_entity))
               else:
                   changes.new_entities.append(new_entity)
           
           return changes
       
       async def apply_changes(self, changes: ChangeSet) -> UpdateResult:
           """Apply incremental updates to graph."""
           result = UpdateResult()
           
           # Create new entities
           for entity in changes.new_entities:
               entity_id = await self.graph_storage.create_entity(entity)
               result.entities_created.append(entity_id)
           
           # Update modified entities
           for entity_id, updated_entity in changes.modified_entities:
               await self.graph_storage.update_entity(entity_id, updated_entity)
               result.entities_updated.append(entity_id)
           
           # Handle relation updates
           await self._update_relations(changes, result)
           
           return result
       
       def _find_matching_entity(self, new_entity: Entity, existing_entities: List[Entity]) -> Optional[Entity]:
           """Find matching entity using similarity metrics."""
           best_match = None
           best_score = 0.0
           
           for existing_entity in existing_entities:
               score = self._calculate_similarity(new_entity, existing_entity)
               if score > 0.8 and score > best_score:  # Threshold for matching
                   best_match = existing_entity
                   best_score = score
           
           return best_match
       
       def _calculate_similarity(self, entity1: Entity, entity2: Entity) -> float:
           """Calculate similarity between two entities."""
           # Name similarity
           name_sim = self._string_similarity(entity1.name, entity2.name)
           
           # Type similarity
           type_sim = 1.0 if entity1.type == entity2.type else 0.0
           
           # Embedding similarity (if available)
           embedding_sim = 0.0
           if entity1.embedding and entity2.embedding:
               embedding_sim = self._cosine_similarity(entity1.embedding, entity2.embedding)
           
           # Weighted combination
           return 0.4 * name_sim + 0.3 * type_sim + 0.3 * embedding_sim
   
   @dataclass
   class ChangeSet:
       new_entities: List[Entity] = field(default_factory=list)
       modified_entities: List[Tuple[str, Entity]] = field(default_factory=list)
       new_relations: List[Relation] = field(default_factory=list)
       modified_relations: List[Tuple[str, Relation]] = field(default_factory=list)
   
   @dataclass
   class UpdateResult:
       entities_created: List[str] = field(default_factory=list)
       entities_updated: List[str] = field(default_factory=list)
       relations_created: List[str] = field(default_factory=list)
       relations_updated: List[str] = field(default_factory=list)
   ```

2. **Conflict Resolution**
   ```python
   class ConflictResolver:
       def resolve_entity_conflict(self, existing: Entity, new: Entity) -> Entity:
           """Resolve conflicts between existing and new entity data."""
           resolved = Entity(
               id=existing.id,
               name=self._resolve_name_conflict(existing.name, new.name),
               type=existing.type,  # Keep existing type
               summary=self._merge_summaries(existing.summary, new.summary),
               embedding=new.embedding if new.embedding else existing.embedding,
               metadata=self._merge_metadata(existing.metadata, new.metadata),
               source_documents=list(set(existing.source_documents + new.source_documents)),
               created_at=existing.created_at,
               updated_at=datetime.utcnow()
           )
           return resolved
       
       def _resolve_name_conflict(self, existing_name: str, new_name: str) -> str:
           """Choose the better name based on length and completeness."""
           if len(new_name) > len(existing_name) and new_name.lower() not in existing_name.lower():
               return new_name
           return existing_name
   ```

3. **Performance Optimization**
   ```python
   class BatchUpdateManager:
       async def batch_update_entities(self, updates: List[Tuple[str, Entity]]) -> List[str]:
           """Perform batch updates for better performance."""
           batch_size = 100
           updated_ids = []
           
           for i in range(0, len(updates), batch_size):
               batch = updates[i:i + batch_size]
               batch_ids = await self.graph_storage.batch_update_entities(batch)
               updated_ids.extend(batch_ids)
           
           return updated_ids
   ```

#### Deliverables
- Change detection system
- Conflict resolution mechanisms
- Batch update optimization
- Performance monitoring

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

class TestGraphUpdateManager:
    @pytest.mark.asyncio
    async def test_detect_changes(self, update_manager, new_document, existing_entities):
        changes = await update_manager.detect_changes(new_document, existing_entities)
        
        assert isinstance(changes, ChangeSet)
        assert len(changes.new_entities) >= 0
        assert len(changes.modified_entities) >= 0
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
- [ ] Incremental updates detect and apply changes correctly
- [ ] Conflict resolution maintains data integrity
- [ ] Performance meets requirements (< 5 seconds per document)
- [ ] Error handling gracefully manages failures
- [ ] Unit test coverage > 90%
- [ ] Integration tests pass

## Performance Targets

- **Document Processing**: < 5 seconds per document
- **Batch Processing**: > 100 documents per minute
- **Update Detection**: < 2 seconds per document
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