"""Document cleanup management for removing outdated graph data."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..models.entity import EntityId
from ..models.relation import RelationId
from ..storage.base import BaseStorage


@dataclass
class CleanupResult:
    """Result of document cleanup operation."""

    document_id: str
    entities_deleted: int = 0
    relations_deleted: int = 0
    entity_ids_deleted: List[EntityId] = field(default_factory=list)
    relation_ids_deleted: List[RelationId] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.entity_ids_deleted is None:
            self.entity_ids_deleted = []
        if self.relation_ids_deleted is None:
            self.relation_ids_deleted = []
        if self.errors is None:
            self.errors = []


class DocumentCleanupManager:
    """Manages cleanup of document-related graph data.

    This class handles the removal of all entities and relations
    associated with a document when its content changes.
    """

    def __init__(self, graph_storage: BaseStorage):
        """Initialize the cleanup manager.

        Args:
            graph_storage: Graph storage backend
        """
        self.graph_storage = graph_storage
        self.logger = logging.getLogger(__name__)

    async def cleanup_document_data(self, document_id: str) -> CleanupResult:
        """Remove all entities and relations associated with a document.

        Args:
            document_id: Document identifier

        Returns:
            CleanupResult with details of what was removed
        """
        result = CleanupResult(document_id=document_id)

        try:
            self.logger.info(f"Starting cleanup for document {document_id}")

            # Get all entities for this document
            document_entities = await self._get_document_entities(document_id)

            if not document_entities:
                self.logger.info(f"No entities found for document {document_id}")
                return result

            # Get all relations involving these entities
            document_relations = await self._get_document_relations(document_entities)

            # Delete relations first (to maintain referential integrity)
            for relation_id in document_relations:
                try:
                    deleted = await self.graph_storage.delete_relation(relation_id)
                    if deleted:
                        result.relations_deleted += 1
                        result.relation_ids_deleted.append(relation_id)
                except Exception as e:
                    error_msg = f"Failed to delete relation {relation_id}: {str(e)}"
                    self.logger.error(error_msg)
                    result.errors.append(error_msg)

            # Delete entities
            for entity_id in document_entities:
                try:
                    deleted = await self.graph_storage.delete_entity(entity_id)
                    if deleted:
                        result.entities_deleted += 1
                        result.entity_ids_deleted.append(entity_id)
                except Exception as e:
                    error_msg = f"Failed to delete entity {entity_id}: {str(e)}"
                    self.logger.error(error_msg)
                    result.errors.append(error_msg)

            # Remove document checksum
            await self._remove_document_checksum(document_id)

            self.logger.info(
                f"Cleanup completed for document {document_id}: "
                f"{result.entities_deleted} entities, {result.relations_deleted} relations deleted"
            )

            return result

        except Exception as e:
            error_msg = f"Failed to cleanup document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)
            return result

    async def _get_document_entities(self, document_id: str) -> List[EntityId]:
        """Get all entity IDs associated with a document.

        Args:
            document_id: Document identifier

        Returns:
            List of entity IDs from this document
        """
        try:
            # This method needs to be implemented in storage backends
            if hasattr(self.graph_storage, "get_entities_by_document"):
                entities = await self.graph_storage.get_entities_by_document(
                    document_id
                )
                return [entity.id for entity in entities]
            else:
                self.logger.warning(
                    "Storage backend does not support document-based entity queries"
                )
                return []
        except Exception as e:
            self.logger.error(
                f"Error getting entities for document {document_id}: {str(e)}"
            )
            return []

    async def _get_document_relations(
        self, entity_ids: List[EntityId]
    ) -> List[RelationId]:
        """Get all relation IDs involving the given entities.

        Args:
            entity_ids: List of entity IDs

        Returns:
            List of relation IDs involving these entities
        """
        relation_ids = set()

        for entity_id in entity_ids:
            try:
                # Get all relations for this entity
                relations = await self.graph_storage.get_entity_relations(entity_id)
                for relation in relations:
                    relation_ids.add(relation.id)
            except Exception as e:
                self.logger.error(
                    f"Error getting relations for entity {entity_id}: {str(e)}"
                )

        return list(relation_ids)

    async def _remove_document_checksum(self, document_id: str) -> None:
        """Remove stored checksum for a document.

        Args:
            document_id: Document identifier
        """
        try:
            if hasattr(self.graph_storage, "delete_document_checksum"):
                await self.graph_storage.delete_document_checksum(document_id)
        except Exception as e:
            self.logger.error(
                f"Error removing checksum for document {document_id}: {str(e)}"
            )
