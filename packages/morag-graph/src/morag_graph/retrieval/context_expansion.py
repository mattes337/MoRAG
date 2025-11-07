"""Context expansion engine for graph-guided retrieval."""

import logging
from typing import Dict, List, Optional, Tuple

from ..models import Entity
from ..operations import GraphTraversal
from ..operations.traversal import GraphPath
from ..query.models import QueryAnalysis
from ..storage.base import BaseStorage
from .models import ContextExpansionConfig, ExpandedContext

logger = logging.getLogger(__name__)


class ContextExpansionEngine:
    """Engine for expanding query context using graph traversal."""

    def __init__(
        self,
        graph_storage: BaseStorage,
        config: Optional[ContextExpansionConfig] = None,
    ):
        """Initialize the context expansion engine.

        Args:
            graph_storage: Graph storage backend
            config: Configuration for context expansion
        """
        self.graph_storage = graph_storage
        self.config = config or ContextExpansionConfig()
        self.graph_traversal = GraphTraversal(graph_storage)
        self.logger = logging.getLogger(__name__)

    async def expand_context(self, query_analysis: QueryAnalysis) -> ExpandedContext:
        """Expand context based on query entities and intent.

        Args:
            query_analysis: Analysis of the user query

        Returns:
            ExpandedContext with related entities and paths
        """
        try:
            # Get linked entities from query
            linked_entities = [
                e.linked_entity_id
                for e in query_analysis.entities
                if e.linked_entity_id
            ]

            if not linked_entities:
                return ExpandedContext(
                    original_entities=[],
                    expanded_entities=[],
                    expansion_paths=[],
                    context_score=0.0,
                    expansion_reasoning="No linked entities found in query",
                )

            self.logger.info(
                f"Expanding context for {len(linked_entities)} linked entities"
            )

            # Choose expansion strategy based on query type and intent
            strategy = self._select_expansion_strategy(query_analysis)

            # Perform context expansion
            expanded_entities, expansion_paths = await self._execute_expansion_strategy(
                linked_entities, strategy, query_analysis
            )

            # Calculate context relevance score
            context_score = self._calculate_context_score(
                query_analysis, expanded_entities, expansion_paths
            )

            self.logger.info(
                f"Context expansion complete: {len(expanded_entities)} entities, score: {context_score:.3f}"
            )

            return ExpandedContext(
                original_entities=linked_entities,
                expanded_entities=expanded_entities,
                expansion_paths=expansion_paths,
                context_score=context_score,
                expansion_reasoning=f"Used {strategy} strategy based on {query_analysis.query_type} query",
            )

        except Exception as e:
            self.logger.error(f"Error expanding context: {str(e)}")
            # Return empty context rather than failing
            return ExpandedContext(
                original_entities=[],
                expanded_entities=[],
                expansion_paths=[],
                context_score=0.0,
                expansion_reasoning=f"Context expansion failed: {str(e)}",
            )

    def _select_expansion_strategy(self, query_analysis: QueryAnalysis) -> str:
        """Select appropriate expansion strategy based on query characteristics.

        Args:
            query_analysis: Analysis of the user query

        Returns:
            Strategy name
        """
        if query_analysis.query_type == "single_entity":
            if query_analysis.intent == "factual":
                return "direct_neighbors"
            elif query_analysis.intent == "exploratory":
                return "breadth_first"
        elif query_analysis.query_type == "entity_relationship":
            return "shortest_path"
        elif query_analysis.query_type == "multi_entity":
            return "subgraph_extraction"
        else:
            return "adaptive"

    async def _execute_expansion_strategy(
        self, entity_ids: List[str], strategy: str, query_analysis: QueryAnalysis
    ) -> Tuple[List[Entity], List[GraphPath]]:
        """Execute the selected expansion strategy.

        Args:
            entity_ids: List of entity IDs to expand from
            strategy: Expansion strategy to use
            query_analysis: Original query analysis

        Returns:
            Tuple of (expanded entities, expansion paths)
        """
        expanded_entities = []
        expansion_paths = []

        try:
            if strategy == "direct_neighbors":
                (
                    expanded_entities,
                    expansion_paths,
                ) = await self._expand_direct_neighbors(entity_ids)
            elif strategy == "breadth_first":
                expanded_entities, expansion_paths = await self._expand_breadth_first(
                    entity_ids
                )
            elif strategy == "shortest_path":
                expanded_entities, expansion_paths = await self._expand_shortest_paths(
                    entity_ids
                )
            elif strategy == "subgraph_extraction":
                expanded_entities, expansion_paths = await self._extract_subgraph(
                    entity_ids
                )
            elif strategy == "adaptive":
                expanded_entities, expansion_paths = await self._adaptive_expansion(
                    entity_ids, query_analysis
                )
            else:
                self.logger.warning(
                    f"Unknown expansion strategy: {strategy}, using direct_neighbors"
                )
                (
                    expanded_entities,
                    expansion_paths,
                ) = await self._expand_direct_neighbors(entity_ids)

        except Exception as e:
            self.logger.error(f"Error executing expansion strategy {strategy}: {e}")
            # Return empty results rather than failing
            return [], []

        return expanded_entities, expansion_paths

    async def _expand_direct_neighbors(
        self, entity_ids: List[str]
    ) -> Tuple[List[Entity], List[GraphPath]]:
        """Expand to direct neighbors of given entities.

        Args:
            entity_ids: List of entity IDs to expand from

        Returns:
            Tuple of (neighboring entities, paths to neighbors)
        """
        all_entities = []
        all_paths = []

        for entity_id in entity_ids:
            try:
                # Find neighbors using graph traversal
                neighbors = await self.graph_traversal.find_neighbors(
                    entity_id, max_distance=1
                )

                # Limit results per entity
                neighbors = neighbors[: self.config.max_entities_per_hop]
                all_entities.extend(neighbors)

                # Create simple paths for direct neighbors
                for neighbor in neighbors:
                    path = GraphPath(entities=[neighbor], relations=[])
                    all_paths.append(path)

            except Exception as e:
                self.logger.warning(
                    f"Failed to expand neighbors for entity {entity_id}: {e}"
                )
                continue

        # Remove duplicates while preserving order
        unique_entities = self._deduplicate_entities(all_entities)

        return unique_entities, all_paths

    async def _expand_breadth_first(
        self, entity_ids: List[str]
    ) -> Tuple[List[Entity], List[GraphPath]]:
        """Expand using breadth-first traversal.

        Args:
            entity_ids: List of entity IDs to expand from

        Returns:
            Tuple of (expanded entities, expansion paths)
        """
        all_entities = []
        all_paths = []

        for entity_id in entity_ids:
            try:
                # Find neighbors at multiple depths
                neighbors = await self.graph_traversal.find_neighbors(
                    entity_id, max_distance=self.config.max_expansion_depth
                )

                # Limit results
                neighbors = neighbors[: self.config.max_entities_per_hop * 2]
                all_entities.extend(neighbors)

                # Create paths (simplified - in real implementation would track actual paths)
                for neighbor in neighbors:
                    path = GraphPath(entities=[neighbor], relations=[])
                    all_paths.append(path)

            except Exception as e:
                self.logger.warning(
                    f"Failed breadth-first expansion for entity {entity_id}: {e}"
                )
                continue

        return self._deduplicate_entities(all_entities), all_paths

    async def _expand_shortest_paths(
        self, entity_ids: List[str]
    ) -> Tuple[List[Entity], List[GraphPath]]:
        """Find shortest paths between all pairs of entities.

        Args:
            entity_ids: List of entity IDs

        Returns:
            Tuple of (entities on paths, shortest paths)
        """
        all_entities = []
        all_paths = []

        # Find paths between all pairs
        for i, entity1 in enumerate(entity_ids):
            for entity2 in entity_ids[i + 1 :]:
                try:
                    path = await self.graph_traversal.find_shortest_path(
                        entity1, entity2
                    )

                    if path:
                        all_entities.extend(path.entities)
                        all_paths.append(path)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to find path between {entity1} and {entity2}: {e}"
                    )
                    continue

        return self._deduplicate_entities(all_entities), all_paths

    async def _extract_subgraph(
        self, entity_ids: List[str]
    ) -> Tuple[List[Entity], List[GraphPath]]:
        """Extract subgraph around given entities.

        Args:
            entity_ids: List of entity IDs

        Returns:
            Tuple of (subgraph entities, paths in subgraph)
        """
        # For now, use breadth-first expansion as a simplified subgraph extraction
        return await self._expand_breadth_first(entity_ids)

    async def _adaptive_expansion(
        self, entity_ids: List[str], query_analysis: QueryAnalysis
    ) -> Tuple[List[Entity], List[GraphPath]]:
        """Adaptively choose expansion based on query characteristics.

        Args:
            entity_ids: List of entity IDs
            query_analysis: Original query analysis

        Returns:
            Tuple of (expanded entities, expansion paths)
        """
        # Simple adaptive strategy - choose based on complexity
        if query_analysis.complexity_score > 0.7:
            return await self._expand_breadth_first(entity_ids)
        else:
            return await self._expand_direct_neighbors(entity_ids)

    def _calculate_context_score(
        self,
        query_analysis: QueryAnalysis,
        entities: List[Entity],
        paths: List[GraphPath],
    ) -> float:
        """Calculate relevance score for expanded context.

        Args:
            query_analysis: Original query analysis
            entities: Expanded entities
            paths: Expansion paths

        Returns:
            Context relevance score between 0.0 and 1.0
        """
        if not entities:
            return 0.0

        # Base score from number of entities
        entity_score = min(len(entities) / 20.0, 1.0)

        # Path quality score (simplified since GraphPath doesn't have total_weight)
        if paths:
            avg_path_length = sum(len(p.entities) for p in paths) / len(paths)
            path_score = 1.0 / (1.0 + avg_path_length)
        else:
            path_score = 0.0

        # Entity type diversity
        entity_types = set(str(e.type) for e in entities)
        diversity_score = min(len(entity_types) / 5.0, 1.0)

        # Query complexity bonus
        complexity_bonus = query_analysis.complexity_score * 0.2

        return min(
            (entity_score + path_score + diversity_score + complexity_bonus) / 3.0, 1.0
        )

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities while preserving order.

        Args:
            entities: List of entities that may contain duplicates

        Returns:
            List of unique entities
        """
        unique_entities = []
        seen_ids = set()

        for entity in entities:
            if entity.id not in seen_ids:
                unique_entities.append(entity)
                seen_ids.add(entity.id)

        return unique_entities
