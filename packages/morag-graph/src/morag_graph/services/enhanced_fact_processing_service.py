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
from .entity_categorization_service import EntityCategorizationService


class EnhancedFactProcessingService:
    """Service for enhanced fact processing with entity creation and relationship management."""
    
    def __init__(self, neo4j_storage: Neo4jStorage, entity_normalizer: Optional[LLMEntityNormalizer] = None,
                 llm_service=None):
        """Initialize the enhanced fact processing service.

        Args:
            neo4j_storage: Neo4j storage instance
            entity_normalizer: Optional entity normalizer for canonical entity names
            llm_service: Optional LLM service for advanced processing
        """
        self.neo4j_storage = neo4j_storage
        self.entity_normalizer = entity_normalizer
        self.llm_service = llm_service
        self.categorization_service = EntityCategorizationService(llm_service=llm_service)
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
            
            # Step 3: Check which facts already exist and only store new ones
            existing_facts, new_facts = await self._identify_existing_and_new_facts(facts)
            stored_facts = []

            if new_facts:
                stored_facts = await self._store_facts(new_facts)
                self.logger.info(f"Stored {len(stored_facts)} new facts, {len(existing_facts)} facts already existed")
            else:
                self.logger.info(f"All {len(existing_facts)} facts already exist in database")
                stored_facts = [f.id for f in existing_facts]

            # Step 4: Create fact-chunk relationships for ALL facts (existing and new)
            # This ensures that facts connected to entities also get connected to chunks
            chunk_relations_created = await self._create_fact_chunk_relations(facts)

            # Step 5: Create mandatory relationships between facts and entities
            created_relations = []
            if create_mandatory_relations:
                created_relations = await self._create_fact_entity_relations(
                    facts, created_entities + keyword_entities
                )

            # Step 6: Create chunk-entity relationships
            chunk_entity_relations = 0
            if created_entities or keyword_entities:
                chunk_entity_relations = await self._create_chunk_entity_relations(
                    facts, created_entities + keyword_entities
                )

            # Step 7: Generate and store embeddings for entities and facts
            embeddings_stored = await self._generate_and_store_embeddings(
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
                'embeddings_stored': embeddings_stored,
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
            # Process primary entities
            for entity_name in fact.structured_metadata.primary_entities:
                if entity_name and entity_name.strip():
                    await self._process_entity_from_fact_component(
                        entity_name.strip(), fact, entities_to_create, 'primary_entity'
                    )

            # Process domain concepts as entities
            for concept_name in fact.structured_metadata.domain_concepts:
                if concept_name and concept_name.strip():
                    await self._process_entity_from_fact_component(
                        concept_name.strip(), fact, entities_to_create, 'domain_concept'
                    )

        # Create Entity objects with normalized names
        created_entities = []
        entities_for_categorization = []

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
                entities_for_categorization.append(entity)

            except Exception as e:
                self.logger.warning(f"Failed to create entity from fact: {e}")
                continue

        # Apply semantic categorization to entities
        if entities_for_categorization:
            try:
                domain = entities_for_categorization[0].attributes.get('domain', 'general')
                categorized_entities = await self.categorization_service.categorize_entities(
                    entities_for_categorization, domain
                )

                # Store categorized entities
                for entity in categorized_entities:
                    await self.neo4j_storage.store_entity(entity)
                    created_entities.append(entity)

            except Exception as e:
                self.logger.warning(f"Entity categorization failed, storing without categories: {e}")
                # Fallback: store entities without categorization
                for entity in entities_for_categorization:
                    await self.neo4j_storage.store_entity(entity)
                    created_entities.append(entity)

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
                # Create semantic relations from fact to primary entities
                for entity_name in fact.structured_metadata.primary_entities:
                    if entity_name and entity_name.strip():
                        # Normalize entity name to find entity
                        entity_normalized = await self._normalize_entity_name_for_lookup(entity_name.strip())
                        if entity_normalized in entity_lookup:
                            entity = entity_lookup[entity_normalized]
                            relation_type = await self._determine_entity_relation_type(fact, entity_name)
                            success = await self._create_fact_entity_relation(
                                fact, entity, relation_type
                            )
                            if success:
                                created_relations.append(f"fact_{fact.id}_to_{entity.id}_{relation_type}")

                # Create semantic relations from fact to domain concept entities
                for concept_name in fact.structured_metadata.domain_concepts:
                    if concept_name and concept_name.strip():
                        # Normalize concept name to find entity
                        concept_normalized = await self._normalize_entity_name_for_lookup(concept_name.strip())
                        if concept_normalized in entity_lookup:
                            concept_entity = entity_lookup[concept_normalized]
                            relation_type = "RELATES_TO_CONCEPT"
                            success = await self._create_fact_entity_relation(
                                fact, concept_entity, relation_type
                            )
                            if success:
                                created_relations.append(f"fact_{fact.id}_to_{concept_entity.id}_{relation_type}")

                # Create semantic relations from fact to keyword entities
                if fact.keywords:
                    for keyword in fact.keywords:
                        if keyword and keyword.strip():
                            # Normalize keyword name to find entity
                            keyword_normalized = await self._normalize_entity_name_for_lookup(keyword.strip())
                            if keyword_normalized in entity_lookup:
                                keyword_entity = entity_lookup[keyword_normalized]
                                relation_type = await self._determine_keyword_relation_type(fact, keyword)
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

    async def _determine_entity_relation_type(self, fact: Fact, entity_name: str) -> str:
        """Determine semantic relationship type for entity using LLM analysis.

        Args:
            fact: The fact containing the entity
            entity_name: Name of the entity

        Returns:
            Semantic relationship type determined by LLM
        """
        fact_text = fact.fact_text

        # Use LLM to determine the semantic relationship
        prompt = f"""Analyze how the entity "{entity_name}" relates to the content in this fact:

"{fact_text}"

Determine the most appropriate semantic relationship type for the entity. Consider what action or relationship the entity has in this context. You can choose from common types or create a more specific domain-appropriate type:

Common relationship types:
- CAUSES, TREATS, PREVENTS, CURES, HEALS
- CONTAINS, INCLUDES, COMPRISES, CONSISTS_OF
- AFFECTS, INFLUENCES, IMPACTS, MODIFIES
- PRODUCES, CREATES, GENERATES, MAKES
- REDUCES, DECREASES, MINIMIZES, LOWERS
- INCREASES, ENHANCES, IMPROVES, BOOSTS
- REQUIRES, NEEDS, DEPENDS_ON, USES
- SUPPORTS, ENABLES, FACILITATES, HELPS
- SOLVES, ADDRESSES, HANDLES, MANAGES
- INVOLVES, PARTICIPATES_IN, ENGAGES_IN
- DEFINES, DESCRIBES, CHARACTERIZES, REPRESENTS

Feel free to create domain-specific relationship types that better capture the semantic meaning.

Return only the relationship type in uppercase, no explanation."""

        try:
            if hasattr(self, 'llm_service') and self.llm_service:
                response = await self.llm_service.generate(prompt, max_tokens=50)
                relation_type = response.strip().upper()

                # Validate the response is a reasonable relation type
                if relation_type and len(relation_type) < 50 and relation_type.replace('_', '').isalpha():
                    return relation_type

        except Exception as e:
            self.logger.warning(f"LLM entity relation type determination failed: {e}")

        # Fallback to simple pattern matching if LLM fails
        fact_text_lower = fact_text.lower()
        if any(word in fact_text_lower for word in ['causes', 'leads to', 'results in', 'triggers', 'creates']):
            return 'CAUSES'
        elif any(word in fact_text_lower for word in ['treats', 'heals', 'cures', 'helps', 'fixes']):
            return 'TREATS'
        elif any(word in fact_text_lower for word in ['prevents', 'blocks', 'stops', 'avoids']):
            return 'PREVENTS'
        elif any(word in fact_text_lower for word in ['improves', 'enhances', 'boosts', 'increases']):
            return 'IMPROVES'
        elif any(word in fact_text_lower for word in ['contains', 'includes', 'has', 'comprises']):
            return 'CONTAINS'
        else:
            return 'INVOLVES'  # Default semantic relationship



    async def _determine_keyword_relation_type(self, fact: Fact, keyword: str) -> str:
        """Determine semantic relationship type for keyword entity using LLM analysis.

        Args:
            fact: The fact containing the keyword
            keyword: The specific keyword

        Returns:
            Semantic relationship type determined by LLM
        """
        fact_text = fact.get_search_text() if fact.get_search_text() else ""

        # Use LLM to determine the semantic relationship
        prompt = f"""Analyze how the keyword "{keyword}" relates to the content in this fact:

"{fact_text}"

Determine the most appropriate semantic relationship type for the keyword. Consider what role the keyword plays in relation to the fact content. Choose from these common types or suggest a more specific one:

- CATEGORIZED_AS (if keyword represents a domain/category/field)
- ADDRESSES (if keyword represents a problem/condition being addressed)
- USES_METHOD (if keyword represents a method/technique/approach)
- USES_TOOL (if keyword represents a tool/instrument/device)
- BASED_ON (if keyword represents a concept/theory/principle)
- OCCURS_IN (if keyword represents a location/place/context)
- OCCURS_DURING (if keyword represents a time/period/phase)
- INVOLVES (if keyword represents an entity involved in the process)
- PRODUCES (if keyword represents something produced/created)
- REQUIRES (if keyword represents something needed/required)
- AFFECTS (if keyword represents something affected/influenced)
- RELATES_TO (general semantic relationship)

Return only the relationship type in uppercase, no explanation."""

        try:
            if hasattr(self, 'llm_service') and self.llm_service:
                response = await self.llm_service.generate(prompt, max_tokens=50)
                relation_type = response.strip().upper()

                # Validate the response is a reasonable relation type
                if relation_type and len(relation_type) < 50 and relation_type.replace('_', '').isalpha():
                    return relation_type

        except Exception as e:
            self.logger.warning(f"LLM keyword relation type determination failed: {e}")

        # Fallback to simple semantic analysis if LLM fails
        return self._fallback_keyword_relation_type(fact, keyword)

    def _fallback_keyword_relation_type(self, fact: Fact, keyword: str) -> str:
        """Fallback method for keyword relation type determination when LLM fails.

        Args:
            fact: The fact containing the keyword
            keyword: The specific keyword

        Returns:
            Semantic relationship type based on simple heuristics
        """
        fact_text = fact.get_search_text().lower() if fact.get_search_text() else ""
        keyword_lower = keyword.lower()

        # Simple pattern-based fallback (domain-agnostic)
        if any(word in fact_text for word in ['method', 'technique', 'approach', 'way', 'process']):
            return 'USES_METHOD'
        elif any(word in fact_text for word in ['tool', 'instrument', 'device', 'equipment']):
            return 'USES_TOOL'
        elif any(word in fact_text for word in ['location', 'place', 'where', 'in', 'at']):
            return 'OCCURS_IN'
        elif any(word in fact_text for word in ['time', 'when', 'during', 'period']):
            return 'OCCURS_DURING'
        elif any(word in fact_text for word in ['cause', 'effect', 'result', 'impact']):
            return 'AFFECTS'
        else:
            return 'RELATES_TO'  # Generic semantic relationship

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

    async def _identify_existing_and_new_facts(self, facts: List[Fact]) -> tuple[List[Fact], List[Fact]]:
        """Identify which facts already exist in the database and which are new.

        Args:
            facts: List of facts to check

        Returns:
            Tuple of (existing_facts, new_facts)
        """
        existing_facts = []
        new_facts = []

        for fact in facts:
            try:
                # Check if fact exists in Neo4j
                query = "MATCH (f:Fact {id: $fact_id}) RETURN f.id as id LIMIT 1"
                result = await self.neo4j_storage._connection_ops._execute_query(
                    query, {"fact_id": fact.id}
                )

                if result:
                    existing_facts.append(fact)
                    self.logger.debug(f"Fact {fact.id} already exists in database")
                else:
                    new_facts.append(fact)
                    self.logger.debug(f"Fact {fact.id} is new and will be stored")

            except Exception as e:
                self.logger.warning(f"Error checking existence of fact {fact.id}: {e}")
                # If we can't check, assume it's new to be safe
                new_facts.append(fact)

        self.logger.info(f"Fact analysis: {len(existing_facts)} existing, {len(new_facts)} new")
        return existing_facts, new_facts

    async def _create_fact_chunk_relations(self, facts: List[Fact]) -> int:
        """Create relationships between facts and their source chunks.

        Uses MERGE to ensure no duplicate relationships are created.

        Args:
            facts: List of facts with source chunk information

        Returns:
            Number of fact-chunk relations processed (created or already existing)
        """
        processed_relations = 0

        for fact in facts:
            if not fact.source_chunk_id:
                self.logger.warning(f"Fact {fact.id} has no source_chunk_id")
                continue

            try:
                # Create CONTAINS relationship: chunk -> fact (uses MERGE internally)
                await self.neo4j_storage.create_chunk_contains_fact_relation(
                    fact.source_chunk_id,
                    fact.id,
                    context=f"Fact extracted from chunk content"
                )
                processed_relations += 1
                self.logger.debug(f"Processed chunk-fact relation: {fact.source_chunk_id} -> {fact.id}")

            except Exception as e:
                self.logger.warning(f"Failed to create chunk-fact relation for fact {fact.id}: {e}")
                continue

        self.logger.info(f"Processed {processed_relations} fact-chunk relations (created or updated)")
        return processed_relations

    async def _generate_and_store_embeddings(self, facts: List[Fact], entities: List[Entity]) -> dict:
        """Generate and store embeddings for facts and entities in Neo4j.

        Args:
            facts: List of facts to generate embeddings for
            entities: List of entities to generate embeddings for

        Returns:
            Dictionary with embedding statistics
        """
        try:
            from morag_graph.services.entity_embedding_service import EntityEmbeddingService
            from morag_graph.services.fact_embedding_service import FactEmbeddingService
            from morag_services.embedding import GeminiEmbeddingService
            import os

            # Initialize embedding services
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                self.logger.warning("GEMINI_API_KEY not found - skipping embedding generation")
                return {'entities_embedded': 0, 'facts_embedded': 0}

            gemini_service = GeminiEmbeddingService(api_key=api_key)
            entity_embedding_service = EntityEmbeddingService(self.neo4j_storage, gemini_service)
            fact_embedding_service = FactEmbeddingService(self.neo4j_storage, gemini_service)

            # Generate and store entity embeddings
            entities_embedded = 0
            for entity in entities:
                try:
                    embedding = await entity_embedding_service.generate_entity_embedding(entity.id)
                    if embedding:
                        success = await entity_embedding_service.store_entity_embedding(entity.id, embedding)
                        if success:
                            entities_embedded += 1
                            self.logger.debug(f"Stored embedding for entity {entity.id}")
                except Exception as e:
                    self.logger.warning(f"Failed to generate/store embedding for entity {entity.id}: {e}")
                    continue

            # Generate and store fact embeddings
            facts_embedded = 0
            for fact in facts:
                try:
                    embedding = await fact_embedding_service.generate_fact_embedding(fact.id)
                    if embedding:
                        success = await fact_embedding_service.store_fact_embedding(fact.id, embedding)
                        if success:
                            facts_embedded += 1
                            self.logger.debug(f"Stored embedding for fact {fact.id}")
                except Exception as e:
                    self.logger.warning(f"Failed to generate/store embedding for fact {fact.id}: {e}")
                    continue

            self.logger.info(f"Generated and stored embeddings: {entities_embedded} entities, {facts_embedded} facts")
            return {'entities_embedded': entities_embedded, 'facts_embedded': facts_embedded}

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            return {'entities_embedded': 0, 'facts_embedded': 0}
    
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
            # Use the dynamic relation_type as the actual relationship type
            query = f"""
            MATCH (f:Fact {{id: $fact_id}})
            MATCH (e:Entity {{id: $entity_id}})
            MERGE (f)-[r:{relation_type}]->(e)
            SET r.confidence = $confidence,
                r.created_from = 'enhanced_fact_processing',
                r.domain = $domain,
                r.created_at = datetime()
            RETURN r
            """

            result = await self.neo4j_storage._connection_ops._execute_query(query, {
                'fact_id': fact.id,
                'entity_id': entity.id,
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
                    # Add primary entities from structured metadata
                    if fact.structured_metadata and fact.structured_metadata.primary_entities:
                        for entity_name in fact.structured_metadata.primary_entities:
                            if entity_name and entity_name.strip():
                                entity_normalized = entity_name.lower().strip()
                                if entity_normalized in entity_lookup:
                                    chunk_entities.add(entity_lookup[entity_normalized])

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
                    MATCH (e:Entity {id: $entity_id})
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
