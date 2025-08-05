"""Enhanced fact processing service with entity creation and relationship management."""

import asyncio
from typing import List, Dict, Any, Set, Tuple
import structlog

from ..models.fact import Fact
from ..models.entity import Entity
from ..models.relation import Relation
from ..storage.neo4j_storage import Neo4jStorage
from ..utils.id_generation import UnifiedIDGenerator


class EnhancedFactProcessingService:
    """Service for enhanced fact processing with entity creation and relationship management."""
    
    def __init__(self, neo4j_storage: Neo4jStorage):
        """Initialize the enhanced fact processing service.
        
        Args:
            neo4j_storage: Neo4j storage instance
        """
        self.neo4j_storage = neo4j_storage
        self.logger = structlog.get_logger(__name__)
    
    async def process_facts_with_entities(
        self,
        facts: List[Fact],
        create_keyword_entities: bool = True,
        create_mandatory_relations: bool = True
    ) -> Dict[str, Any]:
        """Process facts and create associated entities and relationships.
        
        Args:
            facts: List of facts to process
            create_keyword_entities: Whether to create entities from keywords
            create_mandatory_relations: Whether to create mandatory fact-entity relations
            
        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"Processing {len(facts)} facts with entity creation")
            
            # Step 1: Create entities from fact components
            created_entities = await self._create_entities_from_facts(facts)
            
            # Step 2: Create keyword entities if requested
            keyword_entities = []
            if create_keyword_entities:
                keyword_entities = await self._create_keyword_entities(facts)
            
            # Step 3: Create mandatory relationships between facts and entities
            created_relations = []
            if create_mandatory_relations:
                created_relations = await self._create_fact_entity_relations(
                    facts, created_entities + keyword_entities
                )
            
            # Step 4: Store all facts
            stored_facts = await self._store_facts(facts)
            
            result = {
                'facts_processed': len(facts),
                'facts_stored': len(stored_facts),
                'entities_created': len(created_entities),
                'keyword_entities_created': len(keyword_entities),
                'relations_created': len(created_relations),
                'entities': created_entities + keyword_entities,
                'relations': created_relations
            }
            
            self.logger.info(
                "Enhanced fact processing completed",
                **{k: v for k, v in result.items() if isinstance(v, (int, str))}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced fact processing failed: {e}")
            raise
    
    async def _create_entities_from_facts(self, facts: List[Fact]) -> List[Entity]:
        """Create entities from fact subjects and objects.
        
        Args:
            facts: List of facts to extract entities from
            
        Returns:
            List of created entities
        """
        entities_to_create = {}  # Use dict to deduplicate by name
        
        for fact in facts:
            # Create entity from subject
            if fact.subject and fact.subject.strip():
                subject_name = fact.subject.strip()
                normalized_name = subject_name.lower()
                
                if normalized_name not in entities_to_create:
                    entities_to_create[normalized_name] = {
                        'name': subject_name,
                        'type': 'SUBJECT',
                        'confidence': fact.extraction_confidence,
                        'source_doc_id': fact.source_document_id,
                        'domain': fact.domain,
                        'language': fact.language
                    }
            
            # Create entity from object
            if fact.object and fact.object.strip():
                object_name = fact.object.strip()
                normalized_name = object_name.lower()
                
                if normalized_name not in entities_to_create:
                    entities_to_create[normalized_name] = {
                        'name': object_name,
                        'type': 'OBJECT',
                        'confidence': fact.extraction_confidence,
                        'source_doc_id': fact.source_document_id,
                        'domain': fact.domain,
                        'language': fact.language
                    }
        
        # Create Entity objects
        created_entities = []
        for entity_data in entities_to_create.values():
            try:
                entity = Entity(
                    name=entity_data['name'],
                    type=entity_data['type'],
                    confidence=entity_data['confidence'],
                    source_doc_id=entity_data['source_doc_id'],
                    attributes={
                        'domain': entity_data['domain'],
                        'language': entity_data['language'],
                        'created_from': 'fact_processing',
                        'entity_source': 'subject_object_extraction'
                    }
                )
                
                # Store entity (with automatic deduplication by name)
                await self.neo4j_storage.store_entity(entity)
                created_entities.append(entity)
                
            except Exception as e:
                self.logger.warning(f"Failed to create entity from fact: {e}")
                continue
        
        self.logger.info(f"Created {len(created_entities)} entities from fact subjects/objects")
        return created_entities
    
    async def _create_keyword_entities(self, facts: List[Fact]) -> List[Entity]:
        """Create entities from fact keywords.
        
        Args:
            facts: List of facts to extract keyword entities from
            
        Returns:
            List of created keyword entities
        """
        keyword_entities_to_create = {}  # Use dict to deduplicate by name
        
        for fact in facts:
            if not fact.keywords:
                continue
            
            for keyword in fact.keywords:
                if not keyword or not keyword.strip():
                    continue
                
                keyword_name = keyword.strip()
                normalized_name = keyword_name.lower()
                
                # Skip very short or generic keywords
                if len(keyword_name) < 3 or keyword_name.lower() in ['the', 'and', 'for', 'with']:
                    continue
                
                if normalized_name not in keyword_entities_to_create:
                    keyword_entities_to_create[normalized_name] = {
                        'name': keyword_name,
                        'confidence': fact.extraction_confidence * 0.8,  # Slightly lower confidence
                        'source_doc_id': fact.source_document_id,
                        'domain': fact.domain,
                        'language': fact.language
                    }
        
        # Create keyword Entity objects
        created_keyword_entities = []
        for keyword_data in keyword_entities_to_create.values():
            try:
                entity = Entity(
                    name=keyword_data['name'],
                    type='KEYWORD',
                    confidence=keyword_data['confidence'],
                    source_doc_id=keyword_data['source_doc_id'],
                    attributes={
                        'domain': keyword_data['domain'],
                        'language': keyword_data['language'],
                        'created_from': 'fact_processing',
                        'entity_source': 'keyword_extraction'
                    }
                )
                
                # Store entity (with automatic deduplication by name)
                await self.neo4j_storage.store_entity(entity)
                created_keyword_entities.append(entity)
                
            except Exception as e:
                self.logger.warning(f"Failed to create keyword entity: {e}")
                continue
        
        self.logger.info(f"Created {len(created_keyword_entities)} keyword entities")
        return created_keyword_entities
    
    async def _create_fact_entity_relations(
        self,
        facts: List[Fact],
        entities: List[Entity]
    ) -> List[Relation]:
        """Create mandatory relationships between facts and related entities.
        
        Args:
            facts: List of facts
            entities: List of entities to relate to facts
            
        Returns:
            List of created relations
        """
        created_relations = []
        
        # Create entity name lookup for faster matching
        entity_lookup = {}
        for entity in entities:
            normalized_name = entity.name.lower().strip()
            entity_lookup[normalized_name] = entity
        
        for fact in facts:
            try:
                # Create relation from fact to subject entity
                if fact.subject and fact.subject.strip():
                    subject_normalized = fact.subject.lower().strip()
                    if subject_normalized in entity_lookup:
                        subject_entity = entity_lookup[subject_normalized]
                        relation = await self._create_fact_entity_relation(
                            fact, subject_entity, "ABOUT_SUBJECT"
                        )
                        if relation:
                            created_relations.append(relation)
                
                # Create relation from fact to object entity
                if fact.object and fact.object.strip():
                    object_normalized = fact.object.lower().strip()
                    if object_normalized in entity_lookup:
                        object_entity = entity_lookup[object_normalized]
                        relation = await self._create_fact_entity_relation(
                            fact, object_entity, "ADDRESSES_OBJECT"
                        )
                        if relation:
                            created_relations.append(relation)
                
                # Create relations from fact to keyword entities
                if fact.keywords:
                    for keyword in fact.keywords:
                        if keyword and keyword.strip():
                            keyword_normalized = keyword.lower().strip()
                            if keyword_normalized in entity_lookup:
                                keyword_entity = entity_lookup[keyword_normalized]
                                relation = await self._create_fact_entity_relation(
                                    fact, keyword_entity, "TAGGED_WITH"
                                )
                                if relation:
                                    created_relations.append(relation)
                
            except Exception as e:
                self.logger.warning(f"Failed to create relations for fact {fact.id}: {e}")
                continue
        
        self.logger.info(f"Created {len(created_relations)} fact-entity relations")
        return created_relations
    
    async def _create_fact_entity_relation(
        self,
        fact: Fact,
        entity: Entity,
        relation_type: str
    ) -> Relation:
        """Create a single fact-entity relation.
        
        Args:
            fact: Source fact
            entity: Target entity
            relation_type: Type of relation
            
        Returns:
            Created relation or None if failed
        """
        try:
            relation = Relation(
                source_entity_id=f"fact_{fact.id}",  # Treat fact as source entity
                target_entity_id=entity.id,
                relation_type=relation_type,
                confidence=min(fact.extraction_confidence, entity.confidence),
                attributes={
                    'fact_id': fact.id,
                    'entity_name': entity.name,
                    'created_from': 'enhanced_fact_processing',
                    'domain': fact.domain
                }
            )
            
            # Store relation
            await self.neo4j_storage.store_relation(relation)
            return relation
            
        except Exception as e:
            self.logger.warning(f"Failed to create fact-entity relation: {e}")
            return None
    
    async def _store_facts(self, facts: List[Fact]) -> List[Fact]:
        """Store facts in the database.
        
        Args:
            facts: List of facts to store
            
        Returns:
            List of successfully stored facts
        """
        stored_facts = []
        
        for fact in facts:
            try:
                await self.neo4j_storage.store_fact(fact)
                stored_facts.append(fact)
            except Exception as e:
                self.logger.warning(f"Failed to store fact {fact.id}: {e}")
                continue
        
        self.logger.info(f"Stored {len(stored_facts)} facts")
        return stored_facts
