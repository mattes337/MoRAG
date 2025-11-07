"""
Graph extractor wrapper for MoRAG ingestion system.

This module provides a unified interface for extracting entities and relations
from text content using the morag-graph package.
"""

import os
from typing import Any, Dict, List, Optional

import structlog
from morag_graph.extraction import EntityExtractor, RelationExtractor

logger = structlog.get_logger(__name__)


class GraphExtractor:
    """Unified graph extractor that combines entity and relation extraction."""

    def __init__(self):
        """Initialize the graph extractor."""
        self.entity_extractor = None
        self.relation_extractor = None
        self._initialized = False

    async def initialize(self):
        """Initialize the extractors with LangExtract configuration."""
        if self._initialized:
            return

        # Initialize LangExtract-based extractors
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
        model_id = os.getenv("MORAG_GEMINI_MODEL", "gemini-2.0-flash")

        self.entity_extractor = EntityExtractor(
            api_key=api_key, model_id=model_id, dynamic_types=True, domain="general"
        )
        self.relation_extractor = RelationExtractor(
            api_key=api_key, model_id=model_id, dynamic_types=True, domain="general"
        )
        self._initialized = True

        logger.info("Graph extractor initialized")

    async def extract_entities_and_relations(
        self, content: str, source_path: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract entities and relations from text content.

        Args:
            content: Text content to extract from
            source_path: Source file path or identifier
            language: Language code for processing (e.g., 'en', 'de', 'fr')

        Returns:
            Dictionary containing entities, relations, and metadata
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use extractors with language parameter if provided
            entity_extractor = self.entity_extractor
            relation_extractor = self.relation_extractor

            if language:
                # Create new extractors with language parameter
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv(
                    "LANGEXTRACT_API_KEY"
                )
                model_id = os.getenv("MORAG_GEMINI_MODEL", "gemini-2.0-flash")

                entity_extractor = EntityExtractor(
                    api_key=api_key,
                    model_id=model_id,
                    dynamic_types=True,
                    language=language,
                    domain="general",
                )
                relation_extractor = RelationExtractor(
                    api_key=api_key,
                    model_id=model_id,
                    dynamic_types=True,
                    language=language,
                    domain="general",
                )

            # Extract entities
            logger.info(
                "Extracting entities from content",
                content_length=len(content),
                source_path=source_path,
                language=language,
            )

            entities = await entity_extractor.extract(
                text=content, source_doc_id=source_path
            )

            logger.info("Entities extracted", count=len(entities))

            # Extract relations using extract method to get missing entities
            logger.info("Extracting relations from content")

            relations = await relation_extractor.extract(
                text=content, entities=entities, source_doc_id=source_path
            )

            logger.info("Relations extracted", count=len(relations))

            # Check for auto-created entities in relations and add them to entities list
            auto_created_entities = self._extract_auto_created_entities_from_relations(
                relations, source_path
            )
            if auto_created_entities:
                logger.info(
                    "Found auto-created entities in relations",
                    count=len(auto_created_entities),
                )
                entities.extend(auto_created_entities)

            # Convert to serializable format
            entities_data = []
            for entity in entities:
                entity_data = {
                    "id": entity.id,
                    "name": entity.name,
                    "type": str(entity.type),  # Handle both enum and string types
                    "description": entity.description
                    if hasattr(entity, "description")
                    else "",
                    "attributes": entity.attributes or {},
                    "confidence": entity.confidence,
                    "source_doc_id": entity.source_doc_id,
                }
                entities_data.append(entity_data)

            relations_data = []
            for relation in relations:
                relation_data = {
                    "id": relation.id,
                    "source_entity_id": relation.source_entity_id,
                    "target_entity_id": relation.target_entity_id,
                    "relation_type": str(
                        relation.type
                    ),  # Handle both enum and string types
                    "description": relation.description
                    if hasattr(relation, "description")
                    else "",
                    "context": relation.context if hasattr(relation, "context") else "",
                    "attributes": relation.attributes or {},
                    "confidence": relation.confidence,
                    "source_doc_id": relation.source_doc_id,
                }
                relations_data.append(relation_data)

            return {
                "entities": entities_data,
                "relations": relations_data,
                "metadata": {
                    "entity_count": len(entities),
                    "relation_count": len(relations),
                    "source_path": source_path,
                    "content_length": len(content),
                },
            }

        except Exception as e:
            logger.error(
                "Failed to extract graph data", error=str(e), source_path=source_path
            )
            return {
                "entities": [],
                "relations": [],
                "metadata": {
                    "error": str(e),
                    "entity_count": 0,
                    "relation_count": 0,
                    "source_path": source_path,
                    "content_length": len(content),
                },
            }

    def _extract_auto_created_entities_from_relations(
        self, relations: List, source_path: str
    ) -> List:
        """Extract auto-created entities from relations and create Entity objects.

        Args:
            relations: List of relation objects
            source_path: Source document path

        Returns:
            List of Entity objects for auto-created entities
        """
        from morag_graph.models.entity import Entity

        auto_created_entities = []
        seen_entity_ids = set()

        for relation in relations:
            # Check if relation has auto-created entities in its attributes
            source_name = relation.attributes.get("source_entity_name", "")
            target_name = relation.attributes.get("target_entity_name", "")

            # Check source entity
            if (
                source_name
                and relation.source_entity_id
                and relation.source_entity_id not in seen_entity_ids
                and relation.source_entity_id.startswith("ent_")
            ):
                # Check if this looks like an auto-created entity (has document hash suffix)
                if (
                    "_" in relation.source_entity_id
                    and len(relation.source_entity_id.split("_")[-1]) >= 8
                ):
                    entity = Entity(
                        id=relation.source_entity_id,
                        name=source_name,
                        type="CUSTOM",
                        confidence=0.5,
                        source_doc_id=source_path,
                        attributes={
                            "auto_created": True,
                            "creation_reason": "missing_entity_for_relation",
                        },
                    )
                    auto_created_entities.append(entity)
                    seen_entity_ids.add(relation.source_entity_id)

            # Check target entity
            if (
                target_name
                and relation.target_entity_id
                and relation.target_entity_id not in seen_entity_ids
                and relation.target_entity_id.startswith("ent_")
            ):
                # Check if this looks like an auto-created entity (has document hash suffix)
                if (
                    "_" in relation.target_entity_id
                    and len(relation.target_entity_id.split("_")[-1]) >= 8
                ):
                    entity = Entity(
                        id=relation.target_entity_id,
                        name=target_name,
                        type="CUSTOM",
                        confidence=0.5,
                        source_doc_id=source_path,
                        attributes={
                            "auto_created": True,
                            "creation_reason": "missing_entity_for_relation",
                        },
                    )
                    auto_created_entities.append(entity)
                    seen_entity_ids.add(relation.target_entity_id)

        return auto_created_entities
