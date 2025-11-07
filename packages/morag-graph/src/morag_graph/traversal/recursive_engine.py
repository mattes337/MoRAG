"""Recursive graph traversal engine with intelligent path discovery."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import structlog
from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

from ..models import Entity, Relation
from ..operations.traversal import GraphPath, GraphTraversal
from .path_selector import (
    LLMPathSelector,
    PathRelevanceScore,
    QueryContext,
    TraversalStrategy,
)

logger = structlog.get_logger(__name__)


@dataclass
class TraversalState:
    """Represents the current state of traversal with enhanced context preservation."""

    current_entity: Entity
    visited_entities: Set[str]
    path_entities: List[Entity]
    path_relations: List[Relation]
    depth: int
    accumulated_score: float
    metadata: Dict[str, Any]

    # Enhanced context preservation fields
    relationship_chain: List[str] = None  # Semantic chain of relationships
    context_summary: str = ""  # Running summary of path context
    semantic_coherence: float = 1.0  # Measure of semantic coherence
    query_relevance_history: List[float] = None  # Relevance at each hop
    source_documents: Set[str] = None  # Documents contributing to this path

    def __post_init__(self):
        """Initialize default values for new fields."""
        if self.relationship_chain is None:
            self.relationship_chain = []
        if self.query_relevance_history is None:
            self.query_relevance_history = []
        if self.source_documents is None:
            self.source_documents = set()

    def add_hop(
        self,
        entity: Entity,
        relation: Relation,
        relevance_score: float,
        context_description: str = "",
    ) -> "TraversalState":
        """Create new state with added hop and preserved context."""
        new_visited = self.visited_entities.copy()
        new_visited.add(entity.id)

        new_chain = self.relationship_chain.copy()
        if relation:
            new_chain.append(f"{relation.type}: {context_description}")

        new_relevance_history = self.query_relevance_history.copy()
        new_relevance_history.append(relevance_score)

        # Update context summary
        new_context = self.context_summary
        if context_description:
            new_context += f" -> {context_description}"

        # Calculate semantic coherence decay
        coherence_decay = 0.9 if relevance_score > 0.7 else 0.8
        new_coherence = self.semantic_coherence * coherence_decay

        return TraversalState(
            current_entity=entity,
            visited_entities=new_visited,
            path_entities=self.path_entities + [entity],
            path_relations=self.path_relations + ([relation] if relation else []),
            depth=self.depth + 1,
            accumulated_score=self.accumulated_score * relevance_score,
            metadata=self.metadata.copy(),
            relationship_chain=new_chain,
            context_summary=new_context,
            semantic_coherence=new_coherence,
            query_relevance_history=new_relevance_history,
            source_documents=self.source_documents.copy(),
        )


@dataclass
class TraversalResult:
    """Result of recursive graph traversal."""

    discovered_paths: List[PathRelevanceScore]
    total_entities_visited: int
    total_relations_traversed: int
    traversal_time: float
    strategy_used: TraversalStrategy
    query_context: QueryContext
    metadata: Dict[str, Any]


class CycleDetectionStrategy(Enum):
    """Strategies for detecting and handling cycles."""

    STRICT = "strict"  # Never revisit any entity
    RELAXED = "relaxed"  # Allow revisiting entities at different depths
    SMART = "smart"  # Use heuristics to determine when to allow revisits


class RecursiveTraversalEngine:
    """Intelligent recursive graph traversal engine."""

    def __init__(
        self,
        graph_traversal: GraphTraversal,
        path_selector: Optional[LLMPathSelector] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the recursive traversal engine.

        Args:
            graph_traversal: Graph traversal operations
            path_selector: Optional LLM path selector
            config: Optional configuration dictionary
        """
        self.graph_traversal = graph_traversal
        self.path_selector = path_selector or LLMPathSelector()
        self.config = config or {}
        self.settings = get_settings()

        # Traversal parameters
        self.max_depth = self.config.get("max_depth", 4)
        self.max_paths = self.config.get("max_paths", 100)
        self.max_entities_per_level = self.config.get("max_entities_per_level", 20)
        self.max_traversal_time = self.config.get("max_traversal_time", 30.0)  # seconds

        # Cycle detection
        self.cycle_detection = CycleDetectionStrategy(
            self.config.get("cycle_detection", "smart")
        )
        self.allow_entity_revisit_depth = self.config.get(
            "allow_entity_revisit_depth", 2
        )

        # Performance settings
        self.enable_parallel_traversal = self.config.get(
            "enable_parallel_traversal", True
        )
        self.batch_size = self.config.get("batch_size", 10)
        self.enable_early_termination = self.config.get(
            "enable_early_termination", True
        )

        # Scoring and filtering
        self.min_path_score = self.config.get("min_path_score", 0.2)
        self.score_decay_factor = self.config.get("score_decay_factor", 0.9)

        logger.info(
            "Recursive traversal engine initialized",
            max_depth=self.max_depth,
            max_paths=self.max_paths,
            cycle_detection=self.cycle_detection.value,
            enable_parallel_traversal=self.enable_parallel_traversal,
        )

    async def traverse(
        self,
        starting_entities: List[Entity],
        query_context: QueryContext,
        strategy: TraversalStrategy = TraversalStrategy.ADAPTIVE,
        max_depth: Optional[int] = None,
    ) -> TraversalResult:
        """Perform recursive graph traversal.

        Args:
            starting_entities: Starting entities for traversal
            query_context: Query context for guided traversal
            strategy: Traversal strategy to use
            max_depth: Optional override for maximum depth

        Returns:
            TraversalResult with discovered paths and metadata
        """
        if not starting_entities:
            return TraversalResult(
                discovered_paths=[],
                total_entities_visited=0,
                total_relations_traversed=0,
                traversal_time=0.0,
                strategy_used=strategy,
                query_context=query_context,
                metadata={"error": "No starting entities provided"},
            )

        start_time = time.time()
        max_depth = max_depth or self.max_depth

        try:
            logger.info(
                "Starting recursive traversal",
                starting_entities=[e.name for e in starting_entities],
                query=query_context.query,
                strategy=strategy.value,
                max_depth=max_depth,
            )

            # Initialize traversal state
            all_discovered_paths = []
            total_entities_visited = 0
            total_relations_traversed = 0

            # Choose traversal method based on strategy
            if strategy == TraversalStrategy.BREADTH_FIRST:
                paths, stats = await self._traverse_breadth_first(
                    starting_entities, query_context, max_depth
                )
            elif strategy == TraversalStrategy.DEPTH_FIRST:
                paths, stats = await self._traverse_depth_first(
                    starting_entities, query_context, max_depth
                )
            elif strategy == TraversalStrategy.RELEVANCE_GUIDED:
                paths, stats = await self._traverse_relevance_guided(
                    starting_entities, query_context, max_depth
                )
            else:  # ADAPTIVE or SHORTEST_PATH
                paths, stats = await self._traverse_adaptive(
                    starting_entities, query_context, max_depth
                )

            all_discovered_paths.extend(paths)
            total_entities_visited += stats["entities_visited"]
            total_relations_traversed += stats["relations_traversed"]

            # Score and filter paths
            if all_discovered_paths:
                scored_paths = await self.path_selector.select_paths(
                    query_context.query,
                    starting_entities,
                    [p.path for p in all_discovered_paths],
                    query_context,
                    strategy,
                )
            else:
                scored_paths = []

            traversal_time = time.time() - start_time

            logger.info(
                "Recursive traversal completed",
                discovered_paths=len(all_discovered_paths),
                scored_paths=len(scored_paths),
                entities_visited=total_entities_visited,
                relations_traversed=total_relations_traversed,
                traversal_time=traversal_time,
            )

            return TraversalResult(
                discovered_paths=scored_paths,
                total_entities_visited=total_entities_visited,
                total_relations_traversed=total_relations_traversed,
                traversal_time=traversal_time,
                strategy_used=strategy,
                query_context=query_context,
                metadata={
                    "max_depth_reached": max_depth,
                    "cycle_detection": self.cycle_detection.value,
                    "early_termination_used": self.enable_early_termination,
                },
            )

        except Exception as e:
            logger.error(f"Recursive traversal failed: {e}")
            return TraversalResult(
                discovered_paths=[],
                total_entities_visited=0,
                total_relations_traversed=0,
                traversal_time=time.time() - start_time,
                strategy_used=strategy,
                query_context=query_context,
                metadata={"error": str(e)},
            )

    async def _traverse_breadth_first(
        self,
        starting_entities: List[Entity],
        query_context: QueryContext,
        max_depth: int,
    ) -> Tuple[List[PathRelevanceScore], Dict[str, int]]:
        """Perform breadth-first traversal."""
        discovered_paths = []
        entities_visited = 0
        relations_traversed = 0

        # Initialize queue with starting states
        queue = deque()
        for entity in starting_entities:
            initial_state = TraversalState(
                current_entity=entity,
                visited_entities={entity.id},
                path_entities=[entity],
                path_relations=[],
                depth=0,
                accumulated_score=1.0,
                metadata={
                    "query": query_context.query
                    if hasattr(query_context, "query")
                    else ""
                },
                relationship_chain=[],
                context_summary=f"Starting from {entity.name}",
                semantic_coherence=1.0,
                query_relevance_history=[1.0],
                source_documents=set(),
            )
            queue.append(initial_state)

        while queue and len(discovered_paths) < self.max_paths:
            # Check time limit
            if time.time() - time.time() > self.max_traversal_time:
                logger.warning("Traversal time limit exceeded")
                break

            current_state = queue.popleft()
            entities_visited += 1

            # Check depth limit
            if current_state.depth >= max_depth:
                # Create path from current state
                if len(current_state.path_entities) > 1:
                    path = GraphPath(
                        entities=current_state.path_entities,
                        relations=current_state.path_relations,
                    )
                    path_score = PathRelevanceScore(
                        path=path,
                        relevance_score=current_state.accumulated_score,
                        confidence=min(0.9, current_state.semantic_coherence),
                        reasoning=f"Breadth-first path at depth {current_state.depth}: {current_state.context_summary}",
                        metadata={
                            "traversal_method": "breadth_first",
                            "depth": current_state.depth,
                            "semantic_coherence": current_state.semantic_coherence,
                        },
                        relationship_chain=current_state.relationship_chain.copy(),
                        context_summary=current_state.context_summary,
                        semantic_coherence=current_state.semantic_coherence,
                        query_alignment=self._calculate_query_alignment(
                            current_state, query_context
                        ),
                        source_documents=current_state.source_documents.copy(),
                        hop_relevances=current_state.query_relevance_history.copy(),
                    )
                    discovered_paths.append(path_score)
                continue

            # Get neighbors
            try:
                neighbors = await self.graph_traversal.find_neighbors(
                    current_state.current_entity.id, max_distance=1
                )

                # Limit neighbors per level
                neighbors = neighbors[: self.max_entities_per_level]

                for neighbor in neighbors:
                    # Check cycle detection
                    if not self._should_visit_entity(neighbor, current_state):
                        continue

                    # Get relation between current entity and neighbor
                    relations = await self._get_relations_between(
                        current_state.current_entity, neighbor
                    )

                    for relation in relations:
                        relations_traversed += 1

                        # Calculate new score
                        new_score = (
                            current_state.accumulated_score * self.score_decay_factor
                        )

                        # Skip if score too low
                        if new_score < self.min_path_score:
                            continue

                        # Create new state
                        new_visited = current_state.visited_entities.copy()
                        new_visited.add(neighbor.id)

                        new_state = TraversalState(
                            current_entity=neighbor,
                            visited_entities=new_visited,
                            path_entities=current_state.path_entities + [neighbor],
                            path_relations=current_state.path_relations + [relation],
                            depth=current_state.depth + 1,
                            accumulated_score=new_score,
                            metadata={},
                        )

                        queue.append(new_state)

            except Exception as e:
                logger.warning(
                    f"Failed to get neighbors for {current_state.current_entity.id}: {e}"
                )
                continue

        return discovered_paths, {
            "entities_visited": entities_visited,
            "relations_traversed": relations_traversed,
        }

    async def _traverse_depth_first(
        self,
        starting_entities: List[Entity],
        query_context: QueryContext,
        max_depth: int,
    ) -> Tuple[List[PathRelevanceScore], Dict[str, int]]:
        """Perform depth-first traversal."""
        discovered_paths = []
        entities_visited = 0
        relations_traversed = 0

        async def dfs_recursive(
            current_entity: Entity,
            visited: Set[str],
            path_entities: List[Entity],
            path_relations: List[Relation],
            depth: int,
            accumulated_score: float,
        ):
            nonlocal entities_visited, relations_traversed

            entities_visited += 1

            # Check limits
            if (
                depth >= max_depth
                or len(discovered_paths) >= self.max_paths
                or accumulated_score < self.min_path_score
            ):
                # Create path if valid
                if len(path_entities) > 1:
                    path = GraphPath(entities=path_entities, relations=path_relations)
                    path_score = PathRelevanceScore(
                        path=path,
                        relevance_score=accumulated_score,
                        confidence=0.8,
                        reasoning=f"Depth-first path at depth {depth}",
                        metadata={"traversal_method": "depth_first"},
                    )
                    discovered_paths.append(path_score)
                return

            # Get neighbors
            try:
                neighbors = await self.graph_traversal.find_neighbors(
                    current_entity.id, max_distance=1
                )

                for neighbor in neighbors[: self.max_entities_per_level]:
                    if neighbor.id in visited:
                        continue

                    # Get relations
                    relations = await self._get_relations_between(
                        current_entity, neighbor
                    )

                    for relation in relations:
                        relations_traversed += 1

                        new_score = accumulated_score * self.score_decay_factor
                        new_visited = visited.copy()
                        new_visited.add(neighbor.id)

                        await dfs_recursive(
                            neighbor,
                            new_visited,
                            path_entities + [neighbor],
                            path_relations + [relation],
                            depth + 1,
                            new_score,
                        )

            except Exception as e:
                logger.warning(f"DFS failed for {current_entity.id}: {e}")

        # Start DFS from each starting entity
        for entity in starting_entities:
            await dfs_recursive(entity, {entity.id}, [entity], [], 0, 1.0)

        return discovered_paths, {
            "entities_visited": entities_visited,
            "relations_traversed": relations_traversed,
        }

    async def _traverse_relevance_guided(
        self,
        starting_entities: List[Entity],
        query_context: QueryContext,
        max_depth: int,
    ) -> Tuple[List[PathRelevanceScore], Dict[str, int]]:
        """Perform relevance-guided traversal using LLM scoring."""
        discovered_paths = []
        entities_visited = 0
        relations_traversed = 0

        # Use priority queue for relevance-guided exploration
        from heapq import heappop, heappush

        # Priority queue: (-relevance_score, depth, path)
        priority_queue = []

        # Initialize with starting entities
        for entity in starting_entities:
            initial_path = PathRelevanceScore(
                path=GraphPath(entities=[entity], relations=[]),
                relevance_score=1.0,  # Starting entities get max relevance
                confidence=0.9,
                reasoning="Starting entity",
                metadata={"depth": 0},
            )
            heappush(priority_queue, (-1.0, 0, initial_path))

        visited_paths = set()

        while priority_queue and len(discovered_paths) < self.max_paths_per_depth:
            neg_relevance, depth, current_path = heappop(priority_queue)
            relevance = -neg_relevance

            # Skip if we've reached max depth
            if depth >= max_depth:
                discovered_paths.append(current_path)
                continue

            # Skip if we've seen this path before
            path_signature = self._get_path_signature(current_path.path)
            if path_signature in visited_paths:
                continue
            visited_paths.add(path_signature)

            # Add current path to results
            discovered_paths.append(current_path)
            entities_visited += 1

            # Get neighbors of the last entity in the path
            last_entity = current_path.path.entities[-1]

            try:
                neighbors = await self.graph_operations.find_neighbors(
                    last_entity.id, max_distance=1
                )

                # Score and filter neighbors using path selector if available
                if hasattr(self, "path_selector") and self.path_selector:
                    neighbor_paths = []
                    for neighbor in neighbors:
                        # Create extended path
                        extended_path = GraphPath(
                            entities=current_path.path.entities + [neighbor],
                            relations=current_path.path.relations
                            + [
                                # Simplified relation - in practice, get actual relation
                                type(
                                    "Relation",
                                    (),
                                    {
                                        "type": "RELATED_TO",
                                        "id": f"rel_{last_entity.id}_{neighbor.id}",
                                    },
                                )()
                            ],
                        )
                        neighbor_paths.append(extended_path)

                    # Score paths with LLM
                    scored_neighbor_paths = await self.path_selector.select_paths(
                        query_context.query if hasattr(query_context, "query") else "",
                        starting_entities,
                        neighbor_paths,
                        query_context,
                    )

                    # Add scored paths to priority queue
                    for scored_path in scored_neighbor_paths:
                        if (
                            scored_path.relevance_score > 0.3
                        ):  # Threshold for exploration
                            heappush(
                                priority_queue,
                                (-scored_path.relevance_score, depth + 1, scored_path),
                            )
                            relations_traversed += 1

            except Exception as e:
                logger.warning(
                    f"Failed to get neighbors for entity {last_entity.id}: {e}"
                )
                continue

        return discovered_paths, {
            "entities_visited": entities_visited,
            "relations_traversed": relations_traversed,
        }

    def _get_path_signature(self, path: GraphPath) -> str:
        """Generate a unique signature for a path to detect duplicates."""
        entity_ids = [entity.id for entity in path.entities]
        return "->".join(entity_ids)

    def _calculate_query_alignment(
        self, state: TraversalState, query_context: QueryContext
    ) -> float:
        """Calculate how well the current path aligns with the query."""
        try:
            # Get query terms
            query_terms = set()
            if hasattr(query_context, "query"):
                query_terms.update(query_context.query.lower().split())
            if hasattr(query_context, "keywords"):
                query_terms.update([k.lower() for k in query_context.keywords])

            if not query_terms:
                return 0.5  # Default alignment if no query terms

            # Check alignment with path entities
            path_terms = set()
            for entity in state.path_entities:
                path_terms.update(entity.name.lower().split())
                if hasattr(entity, "type") and entity.type:
                    path_terms.add(entity.type.lower())

            # Check alignment with relationship chain
            for rel_desc in state.relationship_chain:
                path_terms.update(rel_desc.lower().split())

            # Calculate overlap
            if not path_terms:
                return 0.3

            overlap = len(query_terms.intersection(path_terms))
            total_terms = len(query_terms.union(path_terms))

            return overlap / total_terms if total_terms > 0 else 0.0

        except Exception as e:
            logger.warning(f"Failed to calculate query alignment: {e}")
            return 0.5

    async def _traverse_adaptive(
        self,
        starting_entities: List[Entity],
        query_context: QueryContext,
        max_depth: int,
    ) -> Tuple[List[PathRelevanceScore], Dict[str, int]]:
        """Perform adaptive traversal based on query characteristics."""
        # Choose strategy based on query complexity
        if query_context.complexity_score > 0.7:
            return await self._traverse_breadth_first(
                starting_entities, query_context, max_depth
            )
        else:
            return await self._traverse_depth_first(
                starting_entities, query_context, max_depth
            )

    def _should_visit_entity(
        self, entity: Entity, current_state: TraversalState
    ) -> bool:
        """Determine if an entity should be visited based on cycle detection strategy."""
        if self.cycle_detection == CycleDetectionStrategy.STRICT:
            return entity.id not in current_state.visited_entities

        elif self.cycle_detection == CycleDetectionStrategy.RELAXED:
            # Allow revisiting if we're at a different depth
            return (
                entity.id not in current_state.visited_entities
                or current_state.depth >= self.allow_entity_revisit_depth
            )

        else:  # SMART
            # Use heuristics: allow revisiting important entities
            if entity.id not in current_state.visited_entities:
                return True

            # Allow revisiting if entity appears in query
            query_lower = current_state.metadata.get("query", "").lower()
            return entity.name.lower() in query_lower

    async def _get_relations_between(
        self, entity1: Entity, entity2: Entity
    ) -> List[Relation]:
        """Get relations between two entities."""
        try:
            # This is a simplified implementation
            # In practice, you'd query the graph storage for actual relations
            return [
                Relation(
                    id=f"rel_{entity1.id}_{entity2.id}",
                    type="RELATED_TO",
                    source_entity_id=entity1.id,
                    target_entity_id=entity2.id,
                    confidence=0.8,
                )
            ]
        except Exception as e:
            logger.warning(
                f"Failed to get relations between {entity1.id} and {entity2.id}: {e}"
            )
            return []
