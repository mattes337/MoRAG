"""Convert facts to entities and relationships for graph storage."""

import hashlib
import re
from typing import Dict, List, Optional, Tuple

import structlog

from ..models.entity import Entity
from ..models.fact import Fact
from ..models.relation import Relation


class FactEntityConverter:
    """Convert facts to entities and relationships for graph storage."""

    def __init__(self, domain: str = "general"):
        """Initialize the converter.

        Args:
            domain: Domain context for entity type inference
        """
        self.domain = domain
        self.logger = structlog.get_logger(__name__)

        # Generic entity names to skip
        self.generic_names = {
            "it",
            "this",
            "that",
            "they",
            "them",
            "these",
            "those",
            "something",
            "anything",
            "everything",
            "nothing",
            "someone",
            "anyone",
            "everyone",
            "nobody",
            "thing",
            "things",
            "stuff",
            "item",
            "items",
            "person",
            "people",
            "individual",
            "individuals",
            "method",
            "approach",
            "way",
            "manner",
            "means",
            "result",
            "outcome",
            "effect",
            "consequence",
            "study",
            "research",
            "analysis",
            "investigation",
        }

    def convert_facts_to_entities(
        self, facts: List[Fact]
    ) -> Tuple[List[Entity], List[Relation]]:
        """Convert facts to entities and relationships.

        Args:
            facts: List of facts to convert

        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []
        entity_cache = {}  # Cache to avoid duplicate entities

        self.logger.info(f"Converting {len(facts)} facts to entities and relationships")

        for fact in facts:
            # Create entities from fact components
            fact_entities, fact_relations = self._convert_single_fact(
                fact, entity_cache
            )
            entities.extend(fact_entities)
            relationships.extend(fact_relations)

        # Remove duplicates while preserving order
        unique_entities = self._deduplicate_entities(entities)
        unique_relationships = self._deduplicate_relationships(relationships)

        self.logger.info(
            f"Converted facts to {len(unique_entities)} entities and {len(unique_relationships)} relationships"
        )

        return unique_entities, unique_relationships

    def _convert_single_fact(
        self, fact: Fact, entity_cache: Dict[str, Entity]
    ) -> Tuple[List[Entity], List[Relation]]:
        """Convert a single fact to entities and relationships using hybrid approach.

        Args:
            fact: Fact to convert
            entity_cache: Cache of existing entities

        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []

        # Create entities from structured metadata
        metadata = fact.structured_metadata

        # Create entities from primary entities
        for entity_text in metadata.primary_entities:
            entity = self._create_entity_from_text(
                entity_text, "PRIMARY", fact, entity_cache
            )
            if entity:
                entities.append(entity)
                # Create relationship: Entity -> MENTIONED_IN -> Fact
                relationships.append(
                    self._create_fact_relationship(
                        entity.id,
                        fact.id,
                        "MENTIONED_IN",
                        f"Entity '{entity_text}' is mentioned in this fact",
                    )
                )

        # Create entities from domain concepts
        for concept_text in metadata.domain_concepts:
            entity = self._create_entity_from_text(
                concept_text, "CONCEPT", fact, entity_cache
            )
            if entity:
                entities.append(entity)
                # Create relationship: Concept -> RELATES_TO -> Fact
                relationships.append(
                    self._create_fact_relationship(
                        entity.id,
                        fact.id,
                        "RELATES_TO",
                        f"Concept '{concept_text}' relates to this fact",
                    )
                )

        # Create entities from keywords
        for keyword in fact.keywords:
            keyword_entity = self._create_entity_from_text(
                keyword, "KEYWORD", fact, entity_cache
            )
            if keyword_entity:
                entities.append(keyword_entity)
                # Create relationship: Keyword -> TAGGED_IN -> Fact
                relationships.append(
                    self._create_fact_relationship(
                        keyword_entity.id,
                        fact.id,
                        "TAGGED_IN",
                        f"Keyword '{keyword}' tags this fact",
                    )
                )

        # Create relationships between entities based on metadata relationships
        primary_entities_objs = [e for e in entities if e.type == "PRIMARY"]
        for i, entity1 in enumerate(primary_entities_objs):
            for entity2 in primary_entities_objs[i + 1 :]:
                if entity1.id != entity2.id:
                    # Use relationships from metadata to determine connection type
                    relation_type = "RELATES_TO"
                    if metadata.relationships:
                        # Use first relationship as the connection type
                        relation_type = (
                            metadata.relationships[0].upper().replace(" ", "_")
                        )

                    relationships.append(
                        self._create_entity_relationship(
                            entity1.id,
                            entity2.id,
                            relation_type,
                            f"Entities are related through fact {fact.id}",
                        )
                    )

        return entities, relationships

    def _create_entity_from_text(
        self, text: str, entity_type: str, fact: Fact, entity_cache: Dict[str, Entity]
    ) -> Optional[Entity]:
        """Create an entity from text content.

        Args:
            text: Text content for the entity
            entity_type: Type of entity (SUBJECT, OBJECT, KEYWORD)
            fact: Source fact
            entity_cache: Cache of existing entities

        Returns:
            Entity object or None if text is generic
        """
        if not text or not text.strip():
            return None

        # Clean and normalize text
        cleaned_text = self._clean_entity_text(text)
        if not cleaned_text or self._is_generic_entity_name(cleaned_text):
            return None

        # Use normalized name as the primary cache key for deduplication
        # This ensures the same entity (e.g., "Ginkgo Biloba") gets the same ID
        # regardless of whether it appears as subject, object, or keyword
        normalized_name = cleaned_text.lower().strip()

        # Check if we already have this entity (by normalized name, not by type)
        existing_entity = None
        for cached_entity in entity_cache.values():
            cached_normalized = cached_entity.name.lower().strip()
            # Check for exact match or if one is a subset of the other (for compound names)
            if (
                cached_normalized == normalized_name
                or (len(normalized_name) > 5 and normalized_name in cached_normalized)
                or (len(cached_normalized) > 5 and cached_normalized in normalized_name)
            ):
                existing_entity = cached_entity
                break

        if existing_entity:
            # Return the existing entity to ensure deduplication
            return existing_entity

        # Infer more specific entity type based on domain and content
        specific_type = self._infer_entity_type(
            cleaned_text, entity_type, fact.domain or self.domain
        )

        # Generate entity ID based on normalized name only (not type)
        # This ensures the same entity gets the same ID regardless of context
        entity_id = self._generate_entity_id(
            cleaned_text, "ENTITY"
        )  # Use generic type for ID generation

        # Create entity
        entity = Entity(
            id=entity_id,
            name=cleaned_text,
            type=specific_type,
            confidence=0.8,  # Medium confidence for fact-derived entities
            source_doc_id=fact.source_document_id,
            attributes={
                "derived_from_fact": fact.id,
                "fact_component": entity_type.lower(),
                "domain": fact.domain or self.domain,
                "source_chunk_id": fact.source_chunk_id,
                "normalized_name": normalized_name,
            },
        )

        # Cache the entity using normalized name as key
        cache_key = normalized_name
        entity_cache[cache_key] = entity

        return entity

    def _clean_entity_text(self, text: str) -> str:
        """Clean and normalize entity text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace and normalize
        cleaned = re.sub(r"\s+", " ", text.strip())

        # Remove common prefixes/suffixes that don't add meaning
        prefixes_to_remove = ["the ", "a ", "an ", "some ", "any ", "this ", "that "]
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :]
                break

        # Remove common suffixes that don't change the core entity
        suffixes_to_remove = [
            " treatment",
            " therapy",
            " medication",
            " supplement",
            " extract",
            " dose",
            " dosage",
        ]
        for suffix in suffixes_to_remove:
            if cleaned.lower().endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
                break

        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned

    def _is_generic_entity_name(self, name: str) -> bool:
        """Check if entity name is too generic.

        Args:
            name: Entity name to check

        Returns:
            True if name is generic
        """
        name_lower = name.lower().strip()

        # Check against generic names
        if name_lower in self.generic_names:
            return True

        # Check if it's too short
        if len(name_lower) < 2:
            return True

        # Check if it's just punctuation or numbers
        if re.match(r"^[^\w\s]*$", name_lower) or re.match(r"^\d+$", name_lower):
            return True

        return False

    def _infer_entity_type(self, text: str, base_type: str, domain: str) -> str:
        """Infer more specific entity type based on content and domain.

        Args:
            text: Entity text
            base_type: Base type (SUBJECT, OBJECT, KEYWORD)
            domain: Domain context

        Returns:
            Specific entity type
        """
        text_lower = text.lower()

        # Domain-specific type inference
        if domain == "medical" or domain == "health":
            if any(
                term in text_lower
                for term in ["vitamin", "mineral", "supplement", "herb", "extract"]
            ):
                return "SUPPLEMENT"
            elif any(
                term in text_lower
                for term in ["toxin", "heavy metal", "mercury", "aluminum", "lead"]
            ):
                return "TOXIN"
            elif any(
                term in text_lower
                for term in ["thyroid", "hormone", "enzyme", "protein"]
            ):
                return "BIOLOGICAL_COMPOUND"
            elif any(
                term in text_lower
                for term in ["treatment", "therapy", "medication", "drug"]
            ):
                return "TREATMENT"
            elif any(
                term in text_lower
                for term in ["symptom", "condition", "disease", "disorder"]
            ):
                return "MEDICAL_CONDITION"

        # General type inference
        if base_type == "KEYWORD":
            return "KEYWORD"
        elif any(
            term in text_lower
            for term in ["method", "approach", "technique", "procedure"]
        ):
            return "METHOD"
        elif any(
            term in text_lower
            for term in ["result", "outcome", "effect", "consequence"]
        ):
            return "OUTCOME"
        elif any(
            term in text_lower
            for term in ["study", "research", "analysis", "investigation"]
        ):
            return "RESEARCH"

        return base_type

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a unique entity ID.

        Args:
            name: Entity name
            entity_type: Entity type

        Returns:
            Unique entity ID
        """
        # Create deterministic ID based on name and type
        content_for_hash = f"{name.lower()}_{entity_type.lower()}"
        content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:12]
        return f"ent_{content_hash}"

    def _create_fact_relationship(
        self, entity_id: str, fact_id: str, relation_type: str, description: str
    ) -> Relation:
        """Create a relationship between an entity and a fact.

        Args:
            entity_id: Source entity ID
            fact_id: Target fact ID
            relation_type: Type of relationship
            description: Description of the relationship

        Returns:
            Relation object
        """
        # Generate proper relation ID
        content_for_hash = f"{entity_id}_{relation_type}_{fact_id}"
        content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:12]
        rel_id = f"rel_{content_hash}"

        return Relation(
            id=rel_id,
            source_entity_id=entity_id,
            target_entity_id=fact_id,
            type=relation_type,
            description=description,
            confidence=0.9,
            attributes={"relationship_category": "fact_entity"},
        )

    def _create_entity_relationship(
        self, source_id: str, target_id: str, relation_type: str, description: str
    ) -> Relation:
        """Create a relationship between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relationship
            description: Description of the relationship

        Returns:
            Relation object
        """
        # Generate proper relation ID
        content_for_hash = f"{source_id}_{relation_type}_{target_id}"
        content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:12]
        rel_id = f"rel_{content_hash}"

        return Relation(
            id=rel_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            type=relation_type,
            description=description,
            confidence=0.7,
            attributes={"relationship_category": "entity_entity"},
        )

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities while preserving order.

        Args:
            entities: List of entities

        Returns:
            Deduplicated list of entities
        """
        seen_ids = set()
        unique_entities = []

        for entity in entities:
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                unique_entities.append(entity)

        return unique_entities

    def _deduplicate_relationships(
        self, relationships: List[Relation]
    ) -> List[Relation]:
        """Remove duplicate relationships while preserving order.

        Args:
            relationships: List of relationships

        Returns:
            Deduplicated list of relationships
        """
        seen_ids = set()
        unique_relationships = []

        for relationship in relationships:
            if relationship.id not in seen_ids:
                seen_ids.add(relationship.id)
                unique_relationships.append(relationship)

        return unique_relationships
