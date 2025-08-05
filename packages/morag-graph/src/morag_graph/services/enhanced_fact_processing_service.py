"""Enhanced fact processing service with entity creation and relationship management."""

import asyncio
from typing import List, Dict, Any, Set, Tuple, Optional
import structlog

from ..models.fact import Fact
from ..models.entity import Entity
from ..models.relation import Relation
from ..storage.neo4j_storage import Neo4jStorage
from ..utils.id_generation import UnifiedIDGenerator
from ..extraction.entity_normalizer import LLMEntityNormalizer


class EnhancedFactProcessingService:
    """Service for enhanced fact processing with entity creation and relationship management."""
    
    def __init__(self, neo4j_storage: Neo4jStorage, entity_normalizer: Optional[LLMEntityNormalizer] = None):
        """Initialize the enhanced fact processing service.

        Args:
            neo4j_storage: Neo4j storage instance
            entity_normalizer: Optional entity normalizer for canonical entity names
        """
        self.neo4j_storage = neo4j_storage
        self.entity_normalizer = entity_normalizer
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
            
            # Step 3: Store facts in Neo4j first
            await self._store_facts(facts)

            # Step 4: Create fact-chunk relationships
            chunk_relations_created = await self._create_fact_chunk_relations(facts)

            # Step 5: Create mandatory relationships between facts and entities
            created_relations = []
            if create_mandatory_relations:
                created_relations = await self._create_fact_entity_relations(
                    facts, created_entities + keyword_entities
                )
            
            # Step 4: Store all facts
            stored_facts = await self._store_facts(facts)

            # Step 5: Create chunk-entity relationships
            chunk_entity_relations = 0
            if created_entities or keyword_entities:
                chunk_entity_relations = await self._create_chunk_entity_relations(
                    facts, created_entities + keyword_entities
                )

            result = {
                'facts_processed': len(facts),
                'facts_stored': len(stored_facts),
                'entities_created': len(created_entities),
                'keyword_entities_created': len(keyword_entities),
                'relations_created': len(created_relations),
                'chunk_relations_created': chunk_relations_created,
                'chunk_entity_relations': chunk_entity_relations,
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
        """Create normalized entities from fact subjects and objects.

        All entities use generic 'ENTITY' label and are normalized to canonical forms.
        Entities are unique by normalized name globally across all documents.

        Args:
            facts: List of facts to extract entities from

        Returns:
            List of created entities
        """
        entities_to_create = {}  # Use dict to deduplicate by normalized name

        for fact in facts:
            # Process subject entity
            if fact.subject and fact.subject.strip():
                await self._process_entity_from_fact_component(
                    fact.subject.strip(), fact, entities_to_create, 'subject'
                )

            # Process object entity
            if fact.object and fact.object.strip():
                await self._process_entity_from_fact_component(
                    fact.object.strip(), fact, entities_to_create, 'object'
                )

        # Create Entity objects with normalized names
        created_entities = []
        for entity_data in entities_to_create.values():
            try:
                entity = Entity(
                    name=entity_data['normalized_name'],  # Use normalized name
                    type='ENTITY',  # Generic label for all entities
                    confidence=entity_data['confidence'],
                    source_doc_id=entity_data['source_doc_id'],
                    attributes={
                        'original_name': entity_data['original_name'],
                        'domain': entity_data['domain'],
                        'language': entity_data['language'],
                        'created_from': 'fact_processing',
                        'entity_source': entity_data['source_component'],
                        'normalization_confidence': entity_data.get('normalization_confidence', 1.0)
                    }
                )

                # Store entity (with automatic deduplication by name)
                await self.neo4j_storage.store_entity(entity)
                created_entities.append(entity)

            except Exception as e:
                self.logger.warning(f"Failed to create entity from fact: {e}")
                continue

        self.logger.info(f"Created {len(created_entities)} normalized entities from fact subjects/objects")
        return created_entities

    async def _process_entity_from_fact_component(
        self,
        entity_name: str,
        fact: Fact,
        entities_dict: Dict[str, Any],
        component_type: str
    ) -> None:
        """Process and normalize an entity from a fact component.

        Args:
            entity_name: Original entity name
            fact: Source fact
            entities_dict: Dictionary to store entity data
            component_type: Type of component ('subject' or 'object')
        """
        # Normalize entity name to canonical form
        if self.entity_normalizer:
            try:
                normalization_result = await self.entity_normalizer.normalize_entity(entity_name)
                normalized_name = normalization_result.normalized
                normalization_confidence = normalization_result.confidence
            except Exception as e:
                self.logger.warning(f"Entity normalization failed for '{entity_name}': {e}")
                normalized_name = entity_name.strip()
                normalization_confidence = 0.5
        else:
            # Basic normalization if no normalizer available
            normalized_name = entity_name.strip()
            normalization_confidence = 1.0

        # Skip empty normalized names
        if not normalized_name:
            return

        # Use normalized name as key for global uniqueness
        if normalized_name not in entities_dict:
            entities_dict[normalized_name] = {
                'original_name': entity_name,
                'normalized_name': normalized_name,
                'confidence': fact.extraction_confidence or 0.8,
                'source_doc_id': fact.source_document_id,
                'domain': fact.domain,
                'language': fact.language,
                'source_component': component_type,
                'normalization_confidence': normalization_confidence
            }
    
    async def _create_keyword_entities(self, facts: List[Fact]) -> List[Entity]:
        """Create normalized entities from fact keywords.

        All entities use generic 'ENTITY' label and are normalized to canonical forms.

        Args:
            facts: List of facts to extract keyword entities from

        Returns:
            List of created keyword entities
        """
        keyword_entities_to_create = {}  # Use dict to deduplicate by normalized name

        for fact in facts:
            if not fact.keywords:
                continue

            for keyword in fact.keywords:
                if not keyword or not keyword.strip():
                    continue

                keyword_name = keyword.strip()

                # Skip very short or generic keywords
                if len(keyword_name) < 3 or keyword_name.lower() in ['the', 'and', 'for', 'with', 'der', 'die', 'das']:
                    continue

                # Normalize keyword to canonical form
                if self.entity_normalizer:
                    try:
                        normalization_result = await self.entity_normalizer.normalize_entity(keyword_name)
                        normalized_name = normalization_result.normalized
                        normalization_confidence = normalization_result.confidence
                    except Exception as e:
                        self.logger.warning(f"Keyword normalization failed for '{keyword_name}': {e}")
                        normalized_name = keyword_name
                        normalization_confidence = 0.5
                else:
                    normalized_name = keyword_name
                    normalization_confidence = 1.0

                # Skip empty normalized names
                if not normalized_name:
                    continue

                # Use normalized name as key for global uniqueness
                if normalized_name not in keyword_entities_to_create:
                    keyword_entities_to_create[normalized_name] = {
                        'original_name': keyword_name,
                        'normalized_name': normalized_name,
                        'confidence': (fact.extraction_confidence or 0.8) * 0.8,  # Slightly lower confidence
                        'source_doc_id': fact.source_document_id,
                        'domain': fact.domain,
                        'language': fact.language,
                        'normalization_confidence': normalization_confidence
                    }

        # Create keyword Entity objects with normalized names
        created_keyword_entities = []
        for keyword_data in keyword_entities_to_create.values():
            try:
                entity = Entity(
                    name=keyword_data['normalized_name'],  # Use normalized name
                    type='ENTITY',  # Generic label for all entities
                    confidence=keyword_data['confidence'],
                    source_doc_id=keyword_data['source_doc_id'],
                    attributes={
                        'original_name': keyword_data['original_name'],
                        'domain': keyword_data['domain'],
                        'language': keyword_data['language'],
                        'created_from': 'fact_processing',
                        'entity_source': 'keyword_extraction',
                        'normalization_confidence': keyword_data['normalization_confidence']
                    }
                )

                # Store entity (with automatic deduplication by name)
                await self.neo4j_storage.store_entity(entity)
                created_keyword_entities.append(entity)

            except Exception as e:
                self.logger.warning(f"Failed to create keyword entity: {e}")
                continue

        self.logger.info(f"Created {len(created_keyword_entities)} normalized keyword entities")
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
        
        # Create entity name lookup for faster matching using normalized names
        entity_lookup = {}
        for entity in entities:
            # Entity names are already normalized, so use them directly
            entity_lookup[entity.name] = entity
        
        for fact in facts:
            try:
                # Create semantic relation from fact to subject entity
                if fact.subject and fact.subject.strip():
                    # Normalize subject name to find entity
                    subject_normalized = await self._normalize_entity_name_for_lookup(fact.subject.strip())
                    if subject_normalized in entity_lookup:
                        subject_entity = entity_lookup[subject_normalized]
                        relation_type = self._determine_subject_relation_type(fact)
                        success = await self._create_fact_entity_relation(
                            fact, subject_entity, relation_type
                        )
                        if success:
                            created_relations.append(f"fact_{fact.id}_to_{subject_entity.id}_{relation_type}")

                # Create semantic relation from fact to object entity
                if fact.object and fact.object.strip():
                    # Normalize object name to find entity
                    object_normalized = await self._normalize_entity_name_for_lookup(fact.object.strip())
                    if object_normalized in entity_lookup:
                        object_entity = entity_lookup[object_normalized]
                        relation_type = self._determine_object_relation_type(fact)
                        success = await self._create_fact_entity_relation(
                            fact, object_entity, relation_type
                        )
                        if success:
                            created_relations.append(f"fact_{fact.id}_to_{object_entity.id}_{relation_type}")

                # Create semantic relations from fact to keyword entities
                if fact.keywords:
                    for keyword in fact.keywords:
                        if keyword and keyword.strip():
                            # Normalize keyword name to find entity
                            keyword_normalized = await self._normalize_entity_name_for_lookup(keyword.strip())
                            if keyword_normalized in entity_lookup:
                                keyword_entity = entity_lookup[keyword_normalized]
                                relation_type = self._determine_keyword_relation_type(fact, keyword)
                                success = await self._create_fact_entity_relation(
                                    fact, keyword_entity, relation_type
                                )
                                if success:
                                    created_relations.append(f"fact_{fact.id}_to_{keyword_entity.id}_{relation_type}")

            except Exception as e:
                self.logger.warning(f"Failed to create relations for fact {fact.id}: {e}")
                continue
        
        self.logger.info(f"Created {len(created_relations)} fact-entity relations")
        return created_relations

    async def _normalize_entity_name_for_lookup(self, entity_name: str) -> str:
        """Normalize entity name for lookup in entity dictionary.

        Args:
            entity_name: Original entity name

        Returns:
            Normalized entity name for lookup
        """
        if self.entity_normalizer:
            try:
                normalization_result = await self.entity_normalizer.normalize_entity(entity_name)
                return normalization_result.normalized
            except Exception as e:
                self.logger.warning(f"Entity normalization failed for lookup '{entity_name}': {e}")
                return entity_name.strip()
        else:
            return entity_name.strip()

    def _determine_subject_relation_type(self, fact: Fact) -> str:
        """Determine semantic relationship type for subject entity.

        Args:
            fact: The fact containing the subject

        Returns:
            Semantic relationship type
        """
        # Analyze fact content to determine relationship type
        fact_text = fact.get_search_text().lower() if fact.get_search_text() else ""

        # Check for different semantic patterns
        if any(word in fact_text for word in ['causes', 'leads to', 'results in', 'triggers', 'verursacht', 'führt zu']):
            return 'CAUSES'
        elif any(word in fact_text for word in ['treats', 'heals', 'cures', 'helps', 'behandelt', 'heilt', 'hilft']):
            return 'TREATS'
        elif any(word in fact_text for word in ['prevents', 'blocks', 'inhibits', 'verhindert', 'blockiert']):
            return 'PREVENTS'
        elif any(word in fact_text for word in ['contains', 'includes', 'has', 'enthält', 'beinhaltet']):
            return 'CONTAINS'
        elif any(word in fact_text for word in ['affects', 'influences', 'impacts', 'beeinflusst', 'wirkt auf']):
            return 'AFFECTS'
        elif any(word in fact_text for word in ['produces', 'creates', 'generates', 'produziert', 'erzeugt']):
            return 'PRODUCES'
        else:
            return 'INVOLVES'  # Default semantic relationship

    def _determine_object_relation_type(self, fact: Fact) -> str:
        """Determine semantic relationship type for object entity.

        Args:
            fact: The fact containing the object

        Returns:
            Semantic relationship type
        """
        # Analyze fact content to determine relationship type
        fact_text = fact.get_search_text().lower() if fact.get_search_text() else ""

        # Check for different semantic patterns for objects
        if any(word in fact_text for word in ['solves', 'addresses', 'fixes', 'löst', 'behebt']):
            return 'SOLVES'
        elif any(word in fact_text for word in ['targets', 'focuses on', 'zielt auf', 'fokussiert auf']):
            return 'TARGETS'
        elif any(word in fact_text for word in ['improves', 'enhances', 'boosts', 'verbessert', 'steigert']):
            return 'IMPROVES'
        elif any(word in fact_text for word in ['reduces', 'decreases', 'lowers', 'reduziert', 'senkt']):
            return 'REDUCES'
        elif any(word in fact_text for word in ['supports', 'aids', 'assists', 'unterstützt', 'hilft']):
            return 'SUPPORTS'
        elif any(word in fact_text for word in ['requires', 'needs', 'depends on', 'benötigt', 'braucht']):
            return 'REQUIRES'
        else:
            return 'RELATES_TO'  # Default semantic relationship

    def _determine_keyword_relation_type(self, fact: Fact, keyword: str) -> str:
        """Determine semantic relationship type for keyword entity.

        Args:
            fact: The fact containing the keyword
            keyword: The specific keyword

        Returns:
            Semantic relationship type
        """
        # Keywords typically represent topics, categories, or tags
        fact_text = fact.get_search_text().lower() if fact.get_search_text() else ""
        keyword_lower = keyword.lower()

        # Check if keyword represents a domain or category
        if any(domain in keyword_lower for domain in ['health', 'medical', 'nutrition', 'gesundheit', 'medizin', 'ernährung']):
            return 'CATEGORIZED_AS'
        elif any(symptom in keyword_lower for symptom in ['symptom', 'condition', 'disease', 'disorder']):
            return 'ADDRESSES'
        elif any(method in keyword_lower for method in ['method', 'technique', 'approach', 'methode', 'technik']):
            return 'USES_METHOD'
        else:
            return 'TAGGED_WITH'  # Default for general keywords

    async def _store_facts(self, facts: List[Fact]) -> List[str]:
        """Store facts in Neo4j database.

        Args:
            facts: List of facts to store

        Returns:
            List of stored fact IDs
        """
        stored_fact_ids = []

        for fact in facts:
            try:
                # Store fact using Neo4j storage
                fact_id = await self.neo4j_storage.store_fact(fact)
                stored_fact_ids.append(fact_id)
                self.logger.debug(f"Stored fact: {fact_id}")

            except Exception as e:
                self.logger.warning(f"Failed to store fact {fact.id}: {e}")
                continue

        self.logger.info(f"Stored {len(stored_fact_ids)} facts in Neo4j")
        return stored_fact_ids

    async def _create_fact_chunk_relations(self, facts: List[Fact]) -> int:
        """Create relationships between facts and their source chunks.

        Args:
            facts: List of facts with source chunk information

        Returns:
            Number of fact-chunk relations created
        """
        created_relations = 0

        for fact in facts:
            if not fact.source_chunk_id:
                self.logger.warning(f"Fact {fact.id} has no source_chunk_id")
                continue

            try:
                # Create CONTAINS relationship: chunk -> fact
                await self.neo4j_storage.create_chunk_contains_fact_relation(
                    fact.source_chunk_id,
                    fact.id,
                    context=f"Fact extracted from chunk content"
                )
                created_relations += 1
                self.logger.debug(f"Created chunk-fact relation: {fact.source_chunk_id} -> {fact.id}")

            except Exception as e:
                self.logger.warning(f"Failed to create chunk-fact relation for fact {fact.id}: {e}")
                continue

        self.logger.info(f"Created {created_relations} fact-chunk relations")
        return created_relations
    
    async def _create_fact_entity_relation(
        self,
        fact: Fact,
        entity: Entity,
        relation_type: str
    ) -> bool:
        """Create a direct relationship between a fact and an entity in Neo4j.

        Args:
            fact: Source fact
            entity: Target entity
            relation_type: Type of relation

        Returns:
            True if relation was created successfully, False otherwise
        """
        try:
            # Create direct relationship between Fact node and Entity node
            query = """
            MATCH (f:Fact {id: $fact_id})
            MATCH (e:ENTITY {id: $entity_id})
            MERGE (f)-[r:RELATES_TO {type: $relation_type}]->(e)
            SET r.confidence = $confidence,
                r.created_from = 'enhanced_fact_processing',
                r.domain = $domain,
                r.created_at = datetime()
            RETURN r
            """

            result = await self.neo4j_storage._connection_ops._execute_query(query, {
                'fact_id': fact.id,
                'entity_id': entity.id,
                'relation_type': relation_type,
                'confidence': min(fact.extraction_confidence or 0.8, entity.confidence or 0.8),
                'domain': fact.domain or 'general'
            })

            if result and len(result) > 0:
                self.logger.debug(f"Created {relation_type} relation between fact {fact.id} and entity {entity.id}")
                return True
            else:
                self.logger.warning(f"Failed to create relation between fact {fact.id} and entity {entity.id}")
                return False

        except Exception as e:
            self.logger.warning(f"Failed to create fact-entity relation: {e}")
            return False

    async def _create_chunk_entity_relations(self, facts: List[Fact], entities: List[Entity]) -> int:
        """Create relationships between document chunks and entities derived from facts.

        Args:
            facts: List of facts with source chunk information
            entities: List of entities to relate to chunks

        Returns:
            Number of chunk-entity relations created
        """
        created_relations = 0

        # Create entity name lookup for faster matching
        entity_lookup = {}
        for entity in entities:
            normalized_name = entity.name.lower().strip()
            entity_lookup[normalized_name] = entity

        # Group facts by source chunk
        chunk_facts = {}
        for fact in facts:
            chunk_id = fact.source_chunk_id
            if chunk_id:
                if chunk_id not in chunk_facts:
                    chunk_facts[chunk_id] = []
                chunk_facts[chunk_id].append(fact)

        # Create chunk-entity relationships
        for chunk_id, chunk_fact_list in chunk_facts.items():
            try:
                # Collect all entities mentioned in this chunk's facts
                chunk_entities = set()

                for fact in chunk_fact_list:
                    # Add subject entity
                    if fact.subject and fact.subject.strip():
                        subject_normalized = fact.subject.lower().strip()
                        if subject_normalized in entity_lookup:
                            chunk_entities.add(entity_lookup[subject_normalized])

                    # Add object entity
                    if fact.object and fact.object.strip():
                        object_normalized = fact.object.lower().strip()
                        if object_normalized in entity_lookup:
                            chunk_entities.add(entity_lookup[object_normalized])

                    # Add keyword entities
                    if fact.keywords:
                        for keyword in fact.keywords:
                            if keyword and keyword.strip():
                                keyword_normalized = keyword.lower().strip()
                                if keyword_normalized in entity_lookup:
                                    chunk_entities.add(entity_lookup[keyword_normalized])

                # Create relationships between chunk and entities
                for entity in chunk_entities:
                    query = """
                    MATCH (c:DocumentChunk {id: $chunk_id})
                    MATCH (e:ENTITY {id: $entity_id})
                    MERGE (c)-[r:MENTIONS]->(e)
                    SET r.created_from = 'enhanced_fact_processing',
                        r.created_at = datetime()
                    RETURN r
                    """

                    result = await self.neo4j_storage._connection_ops._execute_query(query, {
                        'chunk_id': chunk_id,
                        'entity_id': entity.id
                    })

                    if result and len(result) > 0:
                        created_relations += 1
                        self.logger.debug(f"Created MENTIONS relation between chunk {chunk_id} and entity {entity.id}")

            except Exception as e:
                self.logger.warning(f"Failed to create chunk-entity relations for chunk {chunk_id}: {e}")
                continue

        self.logger.info(f"Created {created_relations} chunk-entity relations")
        return created_relations

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
