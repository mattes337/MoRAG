"""JSON file storage backend for graph data."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiofiles
from pydantic import BaseModel

from ..models import Entity, Graph, Relation
from ..models.types import EntityId, RelationId
from .base import BaseStorage

logger = logging.getLogger(__name__)


class JsonConfig(BaseModel):
    """Configuration for JSON file storage."""

    storage_path: str = "./graph_data"
    entities_file: str = "entities.json"
    relations_file: str = "relations.json"
    auto_save: bool = True
    backup_count: int = 3


class JsonStorage(BaseStorage):
    """JSON file storage backend for graph data.

    This class implements the BaseStorage interface using JSON files as the backend.
    It's useful for development, testing, and small-scale deployments.
    """

    def __init__(self, config: JsonConfig):
        """Initialize JSON storage.

        Args:
            config: JSON storage configuration
        """
        self.config = config
        self.storage_path = Path(config.storage_path)
        self.entities_file = self.storage_path / config.entities_file
        self.relations_file = self.storage_path / config.relations_file
        self.checksums_file = self.storage_path / "document_checksums.json"

        # In-memory cache
        self._entities: Dict[EntityId, Entity] = {}
        self._relations: Dict[RelationId, Relation] = {}
        self._document_checksums: Dict[str, str] = {}
        self._connected = False

    async def connect(self) -> None:
        """Connect to JSON storage (create directories and load data)."""
        try:
            # Create storage directory if it doesn't exist
            self.storage_path.mkdir(parents=True, exist_ok=True)

            # Load existing data
            await self._load_entities()
            await self._load_relations()
            await self._load_checksums()

            self._connected = True
            logger.info(f"Connected to JSON storage at {self.storage_path}")

        except Exception as e:
            logger.error(f"Failed to connect to JSON storage: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from JSON storage (save data if auto_save is enabled)."""
        if self._connected and self.config.auto_save:
            await self._save_entities()
            await self._save_relations()
            await self._save_checksums()

        self._connected = False
        logger.info("Disconnected from JSON storage")

    async def _load_entities(self) -> None:
        """Load entities from JSON file."""
        if not self.entities_file.exists():
            self._entities = {}
            return

        try:
            async with aiofiles.open(self.entities_file, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)

                self._entities = {}
                for entity_data in data.get("entities", []):
                    entity = Entity(**entity_data)
                    self._entities[entity.id] = entity

                logger.info(
                    f"Loaded {len(self._entities)} entities from {self.entities_file}"
                )

        except Exception as e:
            logger.warning(f"Failed to load entities: {e}")
            self._entities = {}

    async def _save_entities(self) -> None:
        """Save entities to JSON file."""
        try:
            # Create backup if file exists
            if self.entities_file.exists() and self.config.backup_count > 0:
                await self._create_backup(self.entities_file)

            data = {
                "entities": [entity.to_dict() for entity in self._entities.values()]
            }

            async with aiofiles.open(self.entities_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))

            logger.debug(
                f"Saved {len(self._entities)} entities to {self.entities_file}"
            )

        except Exception as e:
            logger.error(f"Failed to save entities: {e}")
            raise

    async def _load_relations(self) -> None:
        """Load relations from JSON file."""
        if not self.relations_file.exists():
            self._relations = {}
            return

        try:
            async with aiofiles.open(self.relations_file, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)

                self._relations = {}
                for relation_data in data.get("relations", []):
                    relation = Relation(**relation_data)
                    self._relations[relation.id] = relation

                logger.info(
                    f"Loaded {len(self._relations)} relations from {self.relations_file}"
                )

        except Exception as e:
            logger.warning(f"Failed to load relations: {e}")
            self._relations = {}

    async def _save_relations(self) -> None:
        """Save relations to JSON file."""
        try:
            # Create backup if file exists
            if self.relations_file.exists() and self.config.backup_count > 0:
                await self._create_backup(self.relations_file)

            data = {
                "relations": [
                    relation.to_dict() for relation in self._relations.values()
                ]
            }

            async with aiofiles.open(self.relations_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))

            logger.debug(
                f"Saved {len(self._relations)} relations to {self.relations_file}"
            )

        except Exception as e:
            logger.error(f"Failed to save relations: {e}")
            raise

    async def _create_backup(self, file_path: Path) -> None:
        """Create a backup of the given file."""
        try:
            backup_dir = file_path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)

            # Rotate existing backups
            for i in range(self.config.backup_count - 1, 0, -1):
                old_backup = backup_dir / f"{file_path.stem}.{i}.json"
                new_backup = backup_dir / f"{file_path.stem}.{i + 1}.json"

                if old_backup.exists():
                    if new_backup.exists():
                        new_backup.unlink()
                    old_backup.rename(new_backup)

            # Create new backup
            backup_file = backup_dir / f"{file_path.stem}.1.json"
            if backup_file.exists():
                backup_file.unlink()

            async with aiofiles.open(file_path, "rb") as src:
                async with aiofiles.open(backup_file, "wb") as dst:
                    await dst.write(await src.read())

        except Exception as e:
            logger.warning(f"Failed to create backup for {file_path}: {e}")

    def _ensure_connected(self) -> None:
        """Ensure storage is connected."""
        if not self._connected:
            raise RuntimeError("JSON storage is not connected")

    async def store_entity(self, entity: Entity) -> EntityId:
        """Store an entity.

        Args:
            entity: Entity to store

        Returns:
            ID of the stored entity
        """
        self._ensure_connected()

        self._entities[entity.id] = entity

        if self.config.auto_save:
            await self._save_entities()

        return entity.id

    async def store_entities(self, entities: List[Entity]) -> List[EntityId]:
        """Store multiple entities.

        Args:
            entities: List of entities to store

        Returns:
            List of IDs of the stored entities
        """
        self._ensure_connected()

        entity_ids = []
        for entity in entities:
            self._entities[entity.id] = entity
            entity_ids.append(entity.id)

        if self.config.auto_save:
            await self._save_entities()

        return entity_ids

    async def get_entity(self, entity_id: EntityId) -> Optional[Entity]:
        """Get an entity by ID.

        Args:
            entity_id: ID of the entity to get

        Returns:
            Entity if found, None otherwise
        """
        self._ensure_connected()
        return self._entities.get(entity_id)

    async def get_entities(self, entity_ids: List[EntityId]) -> List[Entity]:
        """Get multiple entities by IDs.

        Args:
            entity_ids: List of entity IDs to get

        Returns:
            List of entities
        """
        self._ensure_connected()

        entities = []
        for entity_id in entity_ids:
            entity = self._entities.get(entity_id)
            if entity:
                entities.append(entity)

        return entities

    async def get_all_entities(self) -> List[Entity]:
        """Get all entities.

        Returns:
            List of all entities
        """
        self._ensure_connected()
        return list(self._entities.values())

    async def get_all_relations(self) -> List[Relation]:
        """Get all relations.

        Returns:
            List of all relations
        """
        self._ensure_connected()
        return list(self._relations.values())

    async def search_entities(
        self, query: str, entity_type: Optional[str] = None, limit: int = 10
    ) -> List[Entity]:
        """Search for entities by name or attributes.

        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum number of results

        Returns:
            List of matching entities
        """
        self._ensure_connected()

        query_lower = query.lower()
        matches = []

        for entity in self._entities.values():
            # Check entity type filter
            if entity_type and entity.type.value != entity_type:
                continue

            # Check name match
            if query_lower in entity.name.lower():
                matches.append((entity, 2))  # Higher score for name match
                continue

            # Check attributes match
            for key, value in entity.attributes.items():
                if query_lower in str(value).lower():
                    matches.append((entity, 1))  # Lower score for attribute match
                    break

        # Sort by score (descending) and name (ascending)
        matches.sort(key=lambda x: (-x[1], x[0].name))

        return [match[0] for match in matches[:limit]]

    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity.

        Args:
            entity: Entity with updated data

        Returns:
            True if entity was updated, False if not found
        """
        self._ensure_connected()

        if entity.id not in self._entities:
            return False

        self._entities[entity.id] = entity

        if self.config.auto_save:
            await self._save_entities()

        return True

    async def delete_entity(self, entity_id: EntityId) -> bool:
        """Delete an entity and all its relations.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if entity was deleted, False if not found
        """
        self._ensure_connected()

        if entity_id not in self._entities:
            return False

        # Remove entity
        del self._entities[entity_id]

        # Remove all relations involving this entity
        relations_to_remove = []
        for relation_id, relation in self._relations.items():
            if (
                relation.source_entity_id == entity_id
                or relation.target_entity_id == entity_id
            ):
                relations_to_remove.append(relation_id)

        for relation_id in relations_to_remove:
            del self._relations[relation_id]

        if self.config.auto_save:
            await self._save_entities()
            await self._save_relations()

        return True

    async def store_relation(self, relation: Relation) -> RelationId:
        """Store a relation.

        Args:
            relation: Relation to store

        Returns:
            ID of the stored relation
        """
        self._ensure_connected()

        # Validate that both entities exist
        if relation.source_entity_id not in self._entities:
            raise ValueError(f"Source entity {relation.source_entity_id} not found")
        if relation.target_entity_id not in self._entities:
            raise ValueError(f"Target entity {relation.target_entity_id} not found")

        self._relations[relation.id] = relation

        if self.config.auto_save:
            await self._save_relations()

        return relation.id

    async def store_relations(self, relations: List[Relation]) -> List[RelationId]:
        """Store multiple relations.

        Args:
            relations: List of relations to store

        Returns:
            List of IDs of the stored relations
        """
        self._ensure_connected()

        relation_ids = []
        for relation in relations:
            # Validate that both entities exist
            if relation.source_entity_id not in self._entities:
                logger.warning(
                    f"Source entity {relation.source_entity_id} not found, skipping relation"
                )
                continue
            if relation.target_entity_id not in self._entities:
                logger.warning(
                    f"Target entity {relation.target_entity_id} not found, skipping relation"
                )
                continue

            self._relations[relation.id] = relation
            relation_ids.append(relation.id)

        if self.config.auto_save:
            await self._save_relations()

        return relation_ids

    async def get_relation(self, relation_id: RelationId) -> Optional[Relation]:
        """Get a relation by ID.

        Args:
            relation_id: ID of the relation to get

        Returns:
            Relation if found, None otherwise
        """
        self._ensure_connected()
        return self._relations.get(relation_id)

    async def get_relations(self, relation_ids: List[RelationId]) -> List[Relation]:
        """Get multiple relations by IDs.

        Args:
            relation_ids: List of relation IDs to get

        Returns:
            List of relations
        """
        self._ensure_connected()

        relations = []
        for relation_id in relation_ids:
            relation = self._relations.get(relation_id)
            if relation:
                relations.append(relation)

        return relations

    async def get_entity_relations(
        self,
        entity_id: EntityId,
        relation_type: Optional[str] = None,
        direction: str = "both",
    ) -> List[Relation]:
        """Get all relations for an entity.

        Args:
            entity_id: ID of the entity
            relation_type: Optional relation type filter
            direction: Direction of relations ("in", "out", "both")

        Returns:
            List of relations involving the entity
        """
        self._ensure_connected()

        relations = []

        for relation in self._relations.values():
            # Check direction
            if direction == "out" and relation.source_entity_id != entity_id:
                continue
            elif direction == "in" and relation.target_entity_id != entity_id:
                continue
            elif direction == "both" and entity_id not in [
                relation.source_entity_id,
                relation.target_entity_id,
            ]:
                continue

            # Check relation type
            if relation_type and relation.type.value != relation_type:
                continue

            relations.append(relation)

        return relations

    async def update_relation(self, relation: Relation) -> bool:
        """Update an existing relation.

        Args:
            relation: Relation with updated data

        Returns:
            True if relation was updated, False if not found
        """
        self._ensure_connected()

        if relation.id not in self._relations:
            return False

        self._relations[relation.id] = relation

        if self.config.auto_save:
            await self._save_relations()

        return True

    async def delete_relation(self, relation_id: RelationId) -> bool:
        """Delete a relation.

        Args:
            relation_id: ID of the relation to delete

        Returns:
            True if relation was deleted, False if not found
        """
        self._ensure_connected()

        if relation_id not in self._relations:
            return False

        del self._relations[relation_id]

        if self.config.auto_save:
            await self._save_relations()

        return True

    async def get_neighbors(
        self,
        entity_id: EntityId,
        relation_type: Optional[str] = None,
        max_depth: int = 1,
    ) -> List[Entity]:
        """Get neighboring entities.

        Args:
            entity_id: ID of the entity
            relation_type: Optional relation type filter
            max_depth: Maximum traversal depth

        Returns:
            List of neighboring entities
        """
        self._ensure_connected()

        visited = set()
        neighbors = set()
        current_level = {entity_id}

        for depth in range(max_depth):
            next_level = set()

            for current_entity_id in current_level:
                if current_entity_id in visited:
                    continue

                visited.add(current_entity_id)

                # Find connected entities
                for relation in self._relations.values():
                    if relation_type and relation.type.value != relation_type:
                        continue

                    connected_entity_id = None
                    if relation.source_entity_id == current_entity_id:
                        connected_entity_id = relation.target_entity_id
                    elif relation.target_entity_id == current_entity_id:
                        connected_entity_id = relation.source_entity_id

                    if connected_entity_id and connected_entity_id != entity_id:
                        neighbors.add(connected_entity_id)
                        if depth < max_depth - 1:
                            next_level.add(connected_entity_id)

            current_level = next_level
            if not current_level:
                break

        # Convert entity IDs to entities
        result = []
        for neighbor_id in neighbors:
            entity = self._entities.get(neighbor_id)
            if entity:
                result.append(entity)

        return result

    async def find_path(
        self, source_entity_id: EntityId, target_entity_id: EntityId, max_depth: int = 3
    ) -> List[List[EntityId]]:
        """Find paths between two entities.

        Args:
            source_entity_id: ID of the source entity
            target_entity_id: ID of the target entity
            max_depth: Maximum path length

        Returns:
            List of paths (each path is a list of entity IDs)
        """
        self._ensure_connected()

        if (
            source_entity_id not in self._entities
            or target_entity_id not in self._entities
        ):
            return []

        paths = []

        def dfs(current_path: List[EntityId], visited: Set[EntityId], depth: int):
            if depth > max_depth:
                return

            current_entity_id = current_path[-1]

            if current_entity_id == target_entity_id and len(current_path) > 1:
                paths.append(current_path.copy())
                return

            if depth == max_depth:
                return

            # Find connected entities
            for relation in self._relations.values():
                connected_entity_id = None
                if relation.source_entity_id == current_entity_id:
                    connected_entity_id = relation.target_entity_id
                elif relation.target_entity_id == current_entity_id:
                    connected_entity_id = relation.source_entity_id

                if connected_entity_id and connected_entity_id not in visited:
                    current_path.append(connected_entity_id)
                    visited.add(connected_entity_id)
                    dfs(current_path, visited, depth + 1)
                    current_path.pop()
                    visited.remove(connected_entity_id)

        # Start DFS
        dfs([source_entity_id], {source_entity_id}, 0)

        return paths[:10]  # Limit to 10 paths

    async def store_graph(self, graph: Graph) -> None:
        """Store an entire graph.

        Args:
            graph: Graph to store
        """
        self._ensure_connected()

        # Store entities
        for entity in graph.entities.values():
            self._entities[entity.id] = entity

        # Store relations
        for relation in graph.relations.values():
            self._relations[relation.id] = relation

        if self.config.auto_save:
            await self._save_entities()
            await self._save_relations()

    async def get_graph(self, entity_ids: Optional[List[EntityId]] = None) -> Graph:
        """Get a graph or subgraph.

        Args:
            entity_ids: Optional list of entity IDs to include (None for all)

        Returns:
            Graph containing the requested entities and their relations
        """
        self._ensure_connected()

        graph = Graph()

        # Add entities
        if entity_ids:
            for entity_id in entity_ids:
                entity = self._entities.get(entity_id)
                if entity:
                    graph.add_entity(entity)
        else:
            for entity in self._entities.values():
                graph.add_entity(entity)

        # Add relations between included entities
        included_entity_ids = set(graph.entities.keys())
        for relation in self._relations.values():
            if (
                relation.source_entity_id in included_entity_ids
                and relation.target_entity_id in included_entity_ids
            ):
                graph.add_relation(relation)

        return graph

    async def clear(self) -> None:
        """Clear all data from the storage."""
        self._ensure_connected()

        self._entities.clear()
        self._relations.clear()

        if self.config.auto_save:
            await self._save_entities()
            await self._save_relations()

        logger.info("Cleared all data from JSON storage")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary containing statistics
        """
        self._ensure_connected()

        # Count entities by type
        entity_types = {}
        for entity in self._entities.values():
            entity_type = entity.type.value
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        # Count relations by type
        relation_types = {}
        for relation in self._relations.values():
            relation_type = relation.type.value
            relation_types[relation_type] = relation_types.get(relation_type, 0) + 1

        return {
            "entity_count": len(self._entities),
            "relation_count": len(self._relations),
            "entity_types": [
                {"type": entity_type, "count": count}
                for entity_type, count in sorted(
                    entity_types.items(), key=lambda x: -x[1]
                )
            ],
            "relation_types": [
                {"type": relation_type, "count": count}
                for relation_type, count in sorted(
                    relation_types.items(), key=lambda x: -x[1]
                )
            ],
            "storage_path": str(self.storage_path),
            "auto_save": self.config.auto_save,
        }

    # Checksum management methods

    async def get_document_checksum(self, document_id: str) -> Optional[str]:
        """Get stored checksum for a document.

        Args:
            document_id: Document identifier

        Returns:
            Stored checksum if found, None otherwise
        """
        self._ensure_connected()
        return self._document_checksums.get(document_id)

    async def store_document_checksum(self, document_id: str, checksum: str) -> None:
        """Store document checksum.

        Args:
            document_id: Document identifier
            checksum: Document checksum to store
        """
        self._ensure_connected()
        self._document_checksums[document_id] = checksum

        if self.config.auto_save:
            await self._save_checksums()

    async def delete_document_checksum(self, document_id: str) -> None:
        """Remove stored checksum for a document.

        Args:
            document_id: Document identifier
        """
        self._ensure_connected()
        if document_id in self._document_checksums:
            del self._document_checksums[document_id]

            if self.config.auto_save:
                await self._save_checksums()

    async def get_entities_by_document(self, document_id: str) -> List[Entity]:
        """Get all entities associated with a document.

        Args:
            document_id: Document identifier

        Returns:
            List of entities from this document
        """
        self._ensure_connected()

        entities = []
        for entity in self._entities.values():
            if hasattr(entity, "source_doc_id") and entity.source_doc_id == document_id:
                entities.append(entity)

        return entities

    async def _save_checksums(self) -> None:
        """Save document checksums to file."""
        try:
            async with aiofiles.open(self.checksums_file, "w") as f:
                await f.write(json.dumps(self._document_checksums, indent=2))
        except Exception as e:
            logger.error(f"Failed to save checksums: {e}")
            raise

    async def _load_checksums(self) -> None:
        """Load document checksums from file."""
        if not self.checksums_file.exists():
            self._document_checksums = {}
            return

        try:
            async with aiofiles.open(self.checksums_file, "r") as f:
                content = await f.read()
                self._document_checksums = (
                    json.loads(content) if content.strip() else {}
                )
        except Exception as e:
            logger.warning(f"Failed to load checksums, starting with empty cache: {e}")
            self._document_checksums = {}
