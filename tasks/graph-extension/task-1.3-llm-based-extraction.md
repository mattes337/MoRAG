# Task 1.3: LLM-Based Entity and Relation Extraction

**Phase**: 1 - Foundation Infrastructure  
**Priority**: Critical  
**Total Estimated Time**: 8-10 days  
**Dependencies**: Task 1.1 (Graph Database Setup), Task 1.2 (Core Graph Package)

## Overview

This task implements a comprehensive LLM-based entity and relation extraction system that replaces traditional NLP pipelines. The system uses Large Language Models to dynamically identify entities and relationships without requiring pre-defined schemas or domain-specific training data.

## Rationale

Traditional NLP approaches require:
- Pre-trained models for specific domains
- Extensive rule-based patterns
- Manual schema definition
- Domain expertise for configuration

LLM-based approach provides:
- Domain-agnostic entity recognition
- Dynamic relation discovery
- Self-evolving knowledge graphs
- Reduced maintenance overhead
- Better handling of complex, nuanced relationships

## Subtasks

### Task 1.4.1: LLM Entity Extraction Service
**Priority**: Critical  
**Estimated Time**: 3-4 days  
**Dependencies**: 1.1.1, 1.2.1

#### Implementation Steps

1. **LLM Client Integration**
   - Create unified LLM client interface
   - Support multiple LLM providers (OpenAI, Anthropic, local models)
   - Implement retry logic and error handling

2. **Entity Extraction Pipeline**
   - Design entity extraction prompts
   - Implement response parsing and validation
   - Create entity deduplication logic

3. **Entity Management**
   - Create entity storage interface
   - Implement entity linking and merging
   - Design confidence scoring system

#### Code Examples

**LLM Entity Extraction Service**:
```python
# morag-graph/src/morag_graph/services/llm_entity_extractor.py
from typing import List, Dict, Any, Optional
import asyncio
import json
import uuid
from datetime import datetime
import hashlib

from morag_core.services.llm_client import LLMClient
from morag_graph.storage.entity_storage import EntityStorage
from morag_graph.models.entity import Entity

class LLMEntityExtractor:
    """LLM-based entity extraction service"""
    
    def __init__(self, llm_client: LLMClient, entity_storage: EntityStorage):
        self.llm_client = llm_client
        self.entity_storage = entity_storage
        
    async def extract_entities(self, text_chunk: str, 
                             chunk_id: str, 
                             document_id: str) -> List[Dict[str, Any]]:
        """Extract entities from a text chunk using LLM"""
        
        # Create entity extraction prompt
        prompt = self._create_entity_extraction_prompt(text_chunk)
        
        # Call LLM
        response = await self.llm_client.generate_text(prompt)
        
        # Parse entities from LLM response
        raw_entities = self._parse_entities_from_llm(response)
        
        # Process and validate entities
        processed_entities = await self._process_entities(
            raw_entities, text_chunk, chunk_id, document_id
        )
        
        return processed_entities
    
    def _create_entity_extraction_prompt(self, text_chunk: str) -> str:
        """Create prompt for entity extraction"""
        prompt = f"""Extract all significant entities from the following text. Focus on:
- People (names, roles, titles)
- Organizations (companies, institutions, groups)
- Locations (places, addresses, geographical features)
- Concepts (ideas, theories, methodologies)
- Objects (products, tools, substances)
- Events (meetings, processes, incidents)
- Temporal references (dates, periods, durations)

Text: "{text_chunk}"

Instructions:
1. Identify all entities that are meaningful and could be referenced in other contexts.
2. For each entity, determine its type and provide a brief description.
3. Include aliases or alternative names if mentioned.
4. Format your response as a JSON array with the following structure:
   [{{"name": "EntityName", "type": "ENTITY_TYPE", "description": "Brief description", "aliases": ["alias1", "alias2"], "context": "How it appears in the text"}}]

Entity types to use: PERSON, ORGANIZATION, LOCATION, CONCEPT, OBJECT, EVENT, TEMPORAL, OTHER

Response:"""
        
        return prompt
    
    def _parse_entities_from_llm(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse entities from LLM response"""
        try:
            # Extract JSON from response
            json_start = llm_response.find('[')
            json_end = llm_response.rfind(']') + 1
            if json_start == -1 or json_end == 0:
                return []
                
            json_str = llm_response[json_start:json_end]
            entities_data = json.loads(json_str)
            
            # Validate entity structure
            validated_entities = []
            for entity in entities_data:
                if isinstance(entity, dict) and "name" in entity and "type" in entity:
                    validated_entities.append({
                        "name": entity["name"].strip(),
                        "type": entity.get("type", "OTHER").upper(),
                        "description": entity.get("description", ""),
                        "aliases": entity.get("aliases", []),
                        "context": entity.get("context", "")
                    })
            
            return validated_entities
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return []
    
    async def _process_entities(self, raw_entities: List[Dict[str, Any]], 
                              text_chunk: str, chunk_id: str, 
                              document_id: str) -> List[Dict[str, Any]]:
        """Process and validate extracted entities"""
        processed_entities = []
        
        for entity_data in raw_entities:
            # Generate entity ID based on normalized name
            normalized_name = self._normalize_entity_name(entity_data["name"])
            entity_id = self._generate_entity_id(normalized_name, entity_data["type"])
            
            # Check if entity already exists
            existing_entity = await self.entity_storage.get_entity_by_id(entity_id)
            
            if existing_entity:
                # Update existing entity
                await self._update_existing_entity(
                    existing_entity, entity_data, chunk_id, document_id
                )
                processed_entities.append(existing_entity)
            else:
                # Create new entity
                new_entity = await self._create_new_entity(
                    entity_id, entity_data, text_chunk, chunk_id, document_id
                )
                processed_entities.append(new_entity)
        
        return processed_entities
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for consistent identification"""
        return name.strip().lower().replace("  ", " ")
    
    def _generate_entity_id(self, normalized_name: str, entity_type: str) -> str:
        """Generate consistent entity ID"""
        content = f"{normalized_name}:{entity_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def _create_new_entity(self, entity_id: str, entity_data: Dict[str, Any],
                               text_chunk: str, chunk_id: str, 
                               document_id: str) -> Dict[str, Any]:
        """Create a new entity"""
        entity = {
            "id": entity_id,
            "name": entity_data["name"],
            "type": entity_data["type"],
            "description": entity_data["description"],
            "aliases": entity_data["aliases"],
            "confidence": 0.85,  # Default confidence for LLM extraction
            "source_chunks": [chunk_id],
            "source_documents": [document_id],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "extraction_method": "llm"
        }
        
        # Store entity
        await self.entity_storage.create_entity(entity)
        return entity
    
    async def _update_existing_entity(self, existing_entity: Dict[str, Any],
                                    new_data: Dict[str, Any], 
                                    chunk_id: str, document_id: str):
        """Update an existing entity with new information"""
        # Add new source references
        if chunk_id not in existing_entity.get("source_chunks", []):
            existing_entity.setdefault("source_chunks", []).append(chunk_id)
        
        if document_id not in existing_entity.get("source_documents", []):
            existing_entity.setdefault("source_documents", []).append(document_id)
        
        # Merge aliases
        existing_aliases = set(existing_entity.get("aliases", []))
        new_aliases = set(new_data.get("aliases", []))
        existing_entity["aliases"] = list(existing_aliases.union(new_aliases))
        
        # Update description if new one is more detailed
        if len(new_data.get("description", "")) > len(existing_entity.get("description", "")):
            existing_entity["description"] = new_data["description"]
        
        existing_entity["updated_at"] = datetime.now().isoformat()
        
        # Update entity in storage
        await self.entity_storage.update_entity(existing_entity)
```

### Task 1.4.2: Integrated LLM Extraction Pipeline
**Priority**: Critical  
**Estimated Time**: 3-4 days  
**Dependencies**: 1.4.1, 1.1.3

#### Implementation Steps

1. **Pipeline Orchestration**
   - Create extraction pipeline coordinator
   - Implement chunk processing workflow
   - Design batch processing capabilities

2. **Entity-Relation Integration**
   - Combine entity and relation extraction
   - Implement cross-validation between entities and relations
   - Create consistency checking mechanisms

3. **Performance Optimization**
   - Implement parallel processing
   - Create caching mechanisms
   - Design rate limiting for LLM calls

#### Code Examples

**Integrated Extraction Pipeline**:
```python
# morag-graph/src/morag_graph/services/llm_extraction_pipeline.py
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from morag_graph.services.llm_entity_extractor import LLMEntityExtractor
from morag_graph.services.llm_relation_discovery import LLMRelationDiscoveryService
from morag_graph.storage.graph_storage import GraphStorage
from morag_core.services.rate_limiter import RateLimiter

class LLMExtractionPipeline:
    """Integrated LLM-based entity and relation extraction pipeline"""
    
    def __init__(self, 
                 entity_extractor: LLMEntityExtractor,
                 relation_discovery: LLMRelationDiscoveryService,
                 graph_storage: GraphStorage,
                 rate_limiter: RateLimiter,
                 max_concurrent_chunks: int = 5):
        self.entity_extractor = entity_extractor
        self.relation_discovery = relation_discovery
        self.graph_storage = graph_storage
        self.rate_limiter = rate_limiter
        self.max_concurrent_chunks = max_concurrent_chunks
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_chunks)
    
    async def process_document(self, document_id: str, 
                             chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a complete document through the extraction pipeline"""
        start_time = datetime.now()
        
        # Process chunks in batches to respect rate limits
        all_entities = []
        all_relations = []
        
        # Process chunks concurrently with rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent_chunks)
        
        async def process_chunk_with_semaphore(chunk):
            async with semaphore:
                await self.rate_limiter.acquire()
                return await self._process_single_chunk(chunk, document_id)
        
        # Process all chunks
        chunk_results = await asyncio.gather(*[
            process_chunk_with_semaphore(chunk) for chunk in chunks
        ])
        
        # Aggregate results
        for chunk_entities, chunk_relations in chunk_results:
            all_entities.extend(chunk_entities)
            all_relations.extend(chunk_relations)
        
        # Deduplicate entities across chunks
        deduplicated_entities = await self._deduplicate_entities(all_entities)
        
        # Validate and filter relations
        validated_relations = await self._validate_relations(
            all_relations, deduplicated_entities
        )
        
        # Store in graph database
        await self._store_extraction_results(
            document_id, deduplicated_entities, validated_relations
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "document_id": document_id,
            "entities_count": len(deduplicated_entities),
            "relations_count": len(validated_relations),
            "chunks_processed": len(chunks),
            "processing_time_seconds": processing_time,
            "entities": deduplicated_entities,
            "relations": validated_relations
        }
    
    async def _process_single_chunk(self, chunk: Dict[str, Any], 
                                   document_id: str) -> tuple:
        """Process a single chunk for entities and relations"""
        chunk_id = chunk["id"]
        text_content = chunk["content"]
        
        # Extract entities
        entities = await self.entity_extractor.extract_entities(
            text_content, chunk_id, document_id
        )
        
        # Extract relations if we have multiple entities
        relations = []
        if len(entities) >= 2:
            relations = await self.relation_discovery.discover_relations(
                text_content, entities
            )
        
        return entities, relations
    
    async def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities across chunks"""
        entity_map = {}
        
        for entity in entities:
            entity_id = entity["id"]
            if entity_id in entity_map:
                # Merge entity information
                existing = entity_map[entity_id]
                
                # Merge source chunks and documents
                existing["source_chunks"] = list(set(
                    existing.get("source_chunks", []) + 
                    entity.get("source_chunks", [])
                ))
                existing["source_documents"] = list(set(
                    existing.get("source_documents", []) + 
                    entity.get("source_documents", [])
                ))
                
                # Merge aliases
                existing["aliases"] = list(set(
                    existing.get("aliases", []) + 
                    entity.get("aliases", [])
                ))
                
                # Use more detailed description
                if len(entity.get("description", "")) > len(existing.get("description", "")):
                    existing["description"] = entity["description"]
                
                # Update confidence (average)
                existing["confidence"] = (
                    existing.get("confidence", 0.5) + 
                    entity.get("confidence", 0.5)
                ) / 2
                
            else:
                entity_map[entity_id] = entity
        
        return list(entity_map.values())
    
    async def _validate_relations(self, relations: List[Dict[str, Any]], 
                                entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate relations against extracted entities"""
        entity_ids = {entity["id"] for entity in entities}
        validated_relations = []
        
        for relation in relations:
            # Check if both source and target entities exist
            if (relation["source_id"] in entity_ids and 
                relation["target_id"] in entity_ids):
                validated_relations.append(relation)
        
        return validated_relations
    
    async def _store_extraction_results(self, document_id: str,
                                      entities: List[Dict[str, Any]],
                                      relations: List[Dict[str, Any]]):
        """Store extraction results in graph database"""
        # Store entities
        for entity in entities:
            await self.graph_storage.upsert_entity(entity)
        
        # Store relations
        for relation in relations:
            await self.graph_storage.create_relation(relation)
        
        # Update document processing status
        await self.graph_storage.update_document_status(
            document_id, "extracted", {
                "entities_count": len(entities),
                "relations_count": len(relations),
                "extraction_method": "llm",
                "processed_at": datetime.now().isoformat()
            }
        )
```

### Task 1.4.3: Configuration and Optimization
**Priority**: High  
**Estimated Time**: 2 days  
**Dependencies**: 1.4.2

#### Implementation Steps

1. **Configuration Management**
   - Create LLM extraction configuration
   - Implement prompt templates management
   - Design model selection logic

2. **Performance Optimization**
   - Implement result caching
   - Create batch processing optimizations
   - Design cost monitoring

3. **Quality Assurance**
   - Implement extraction quality metrics
   - Create validation workflows
   - Design feedback mechanisms

## Benefits of LLM-Based Approach

### Advantages over Traditional NLP

1. **Domain Agnostic**: Works across any domain without training
2. **Dynamic Schema**: Discovers new entity types and relations automatically
3. **Contextual Understanding**: Better handling of nuanced relationships
4. **Reduced Maintenance**: No need for rule updates or model retraining
5. **Multilingual Support**: Works with multiple languages out of the box
6. **Complex Reasoning**: Can identify implicit relationships

### Cost Considerations

1. **LLM API Costs**: Higher per-document processing cost
2. **Caching Strategy**: Reduce costs through intelligent caching
3. **Batch Processing**: Optimize API usage patterns
4. **Quality vs Cost**: Balance extraction quality with cost constraints

## Testing Requirements

### Unit Tests
```python
# tests/test_llm_extraction.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from morag_graph.services.llm_entity_extractor import LLMEntityExtractor
from morag_graph.services.llm_extraction_pipeline import LLMExtractionPipeline

@pytest.mark.asyncio
async def test_entity_extraction():
    """Test LLM entity extraction"""
    # Mock LLM client
    llm_client = Mock()
    llm_client.generate_text = AsyncMock(return_value='''
    [{"name": "Apple Inc.", "type": "ORGANIZATION", "description": "Technology company", "aliases": ["Apple"], "context": "Apple Inc. announced"}]
    ''')
    
    # Mock entity storage
    entity_storage = Mock()
    entity_storage.get_entity_by_id = AsyncMock(return_value=None)
    entity_storage.create_entity = AsyncMock()
    
    extractor = LLMEntityExtractor(llm_client, entity_storage)
    
    entities = await extractor.extract_entities(
        "Apple Inc. announced new products today.",
        "chunk_1",
        "doc_1"
    )
    
    assert len(entities) == 1
    assert entities[0]["name"] == "Apple Inc."
    assert entities[0]["type"] == "ORGANIZATION"

@pytest.mark.asyncio
async def test_extraction_pipeline():
    """Test complete extraction pipeline"""
    # Test pipeline integration
    pass
```

### Integration Tests
```python
# tests/integration/test_llm_extraction_integration.py
import pytest
import asyncio

@pytest.mark.asyncio
async def test_document_processing_pipeline():
    """Test complete document processing through LLM extraction"""
    # Test end-to-end document processing
    pass

@pytest.mark.asyncio
async def test_graph_construction():
    """Test graph construction from LLM extraction results"""
    # Test graph building from extracted entities and relations
    pass
```

## Success Criteria

- [ ] LLM entity extraction service implemented and tested
- [ ] LLM relation discovery service integrated
- [ ] Complete extraction pipeline operational
- [ ] Dynamic schema evolution working
- [ ] Performance optimization implemented
- [ ] Cost monitoring and control mechanisms in place
- [ ] Quality metrics and validation workflows established
- [ ] Integration tests passing
- [ ] Documentation complete

## Next Steps

After completing this task:
1. Integrate with document ingestion pipeline
2. Implement graph-based retrieval system
3. Create monitoring and analytics for extraction quality
4. Optimize for production deployment

---

**Status**: ⏳ Not Started  
**Assignee**: TBD  
**Last Updated**: December 2024