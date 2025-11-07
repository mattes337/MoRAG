"""Query entity extraction and linking for graph-guided retrieval."""

import difflib
import logging
from typing import Any, Dict, List, Optional

from ..extraction import EntityExtractor
from ..models import Entity
from ..storage.base import BaseStorage
from .models import QueryAnalysis, QueryEntity, QueryProcessingError

logger = logging.getLogger(__name__)


class QueryEntityExtractor:
    """Extract and link entities from user queries to the knowledge graph."""

    def __init__(
        self,
        entity_extractor: EntityExtractor,
        graph_storage: BaseStorage,
        similarity_threshold: float = 0.8,
    ):
        """Initialize the query entity extractor.

        Args:
            entity_extractor: Entity extractor for NLP processing
            graph_storage: Graph storage backend
            similarity_threshold: Minimum similarity score for entity linking
        """
        self.entity_extractor = entity_extractor
        self.graph_storage = graph_storage
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)

    async def extract_and_link_entities(self, query: str) -> QueryAnalysis:
        """Extract entities from query and link to knowledge graph.

        Args:
            query: User query text

        Returns:
            QueryAnalysis with extracted and linked entities

        Raises:
            QueryProcessingError: If entity extraction fails
        """
        try:
            self.logger.info(f"Processing query: {query}")

            # Extract entities using NLP pipeline
            extracted_entities = await self.entity_extractor.extract(query)

            # Convert to QueryEntity objects and link to graph
            query_entities = []
            for entity in extracted_entities:
                query_entity = QueryEntity(
                    text=entity.name,
                    entity_type=entity.type if hasattr(entity, "type") else "UNKNOWN",
                    confidence=entity.confidence
                    if hasattr(entity, "confidence")
                    else 1.0,
                    start_pos=0,  # Could be enhanced with NER position info
                    end_pos=len(entity.name),
                )

                # Attempt to link to existing entities in graph
                linked_entity = await self._link_to_graph_entity(query_entity)
                if linked_entity:
                    query_entity.linked_entity_id = linked_entity.id
                    query_entity.linked_entity = linked_entity
                    self.logger.debug(
                        f"Linked '{query_entity.text}' to entity {linked_entity.id}"
                    )

                query_entities.append(query_entity)

            # Analyze query intent and type
            intent = self._analyze_query_intent(query, query_entities)
            query_type = self._classify_query_type(query, query_entities)
            complexity = self._calculate_complexity_score(query, query_entities)

            self.logger.info(
                f"Query analysis complete: {len(query_entities)} entities, intent={intent}, type={query_type}"
            )

            return QueryAnalysis(
                original_query=query,
                entities=query_entities,
                intent=intent,
                query_type=query_type,
                complexity_score=complexity,
            )

        except Exception as e:
            self.logger.error(
                f"Error extracting entities from query '{query}': {str(e)}"
            )
            raise QueryProcessingError(f"Failed to extract entities: {str(e)}")

    async def _link_to_graph_entity(
        self, query_entity: QueryEntity
    ) -> Optional[Entity]:
        """Link query entity to existing entity in knowledge graph.

        Args:
            query_entity: Entity extracted from query

        Returns:
            Linked Entity if found, None otherwise
        """
        try:
            # Search for entities with similar names
            candidates = await self.graph_storage.search_entities(
                query_entity.text,
                entity_type=query_entity.entity_type,
                limit=20,  # Get more candidates for better matching
            )

            if not candidates:
                self.logger.debug(
                    f"No candidates found for entity '{query_entity.text}'"
                )
                return None

            # Find best match using similarity scoring
            best_match = None
            best_score = 0.0

            for candidate in candidates:
                similarity = self._calculate_entity_similarity(query_entity, candidate)
                if similarity > best_score and similarity >= self.similarity_threshold:
                    best_score = similarity
                    best_match = candidate

            if best_match:
                self.logger.debug(
                    f"Found match for '{query_entity.text}': {best_match.name} (score: {best_score:.3f})"
                )

            return best_match

        except Exception as e:
            self.logger.warning(f"Error linking entity '{query_entity.text}': {e}")
            return None

    def _calculate_entity_similarity(
        self, query_entity: QueryEntity, graph_entity: Entity
    ) -> float:
        """Calculate similarity between query entity and graph entity.

        Args:
            query_entity: Entity from query
            graph_entity: Entity from knowledge graph

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Type compatibility
        type_match = query_entity.entity_type == str(graph_entity.type)
        type_score = 1.0 if type_match else 0.5

        # Text similarity using sequence matcher
        text_similarity = self._calculate_text_similarity(
            query_entity.text.lower(), graph_entity.name.lower()
        )

        # Exact name match with type consideration
        if query_entity.text.lower() == graph_entity.name.lower():
            return 0.9 + (0.1 * type_score)  # Slight bonus for type match

        # Alias matching (if entity has aliases)
        alias_score = 0.0
        if hasattr(graph_entity, "attributes") and graph_entity.attributes:
            aliases = graph_entity.attributes.get("aliases", [])
            if aliases:
                for alias in aliases:
                    alias_sim = self._calculate_text_similarity(
                        query_entity.text.lower(), str(alias).lower()
                    )
                    alias_score = max(alias_score, alias_sim)

        # Combine scores with weights
        final_score = 0.4 * text_similarity + 0.3 * type_score + 0.3 * alias_score

        return final_score

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using Levenshtein distance.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def _analyze_query_intent(self, query: str, entities: List[QueryEntity]) -> str:
        """Analyze the intent of the query.

        Args:
            query: Original query text
            entities: Extracted entities

        Returns:
            Intent classification string
        """
        query_lower = query.lower()

        # Intent keywords
        if any(
            word in query_lower for word in ["what", "who", "where", "when", "which"]
        ):
            return "factual"
        elif any(word in query_lower for word in ["how", "why", "explain"]):
            return "explanatory"
        elif any(
            word in query_lower for word in ["compare", "difference", "versus", "vs"]
        ):
            return "comparative"
        elif any(word in query_lower for word in ["find", "search", "show", "list"]):
            return "exploratory"
        else:
            return "general"

    def _classify_query_type(self, query: str, entities: List[QueryEntity]) -> str:
        """Classify the type of query based on structure and entities.

        Args:
            query: Original query text
            entities: Extracted entities

        Returns:
            Query type classification string
        """
        if len(entities) == 0:
            return "general"
        elif len(entities) == 1:
            return "single_entity"
        elif len(entities) == 2:
            return "entity_relationship"
        else:
            return "multi_entity"

    def _calculate_complexity_score(
        self, query: str, entities: List[QueryEntity]
    ) -> float:
        """Calculate query complexity score.

        Args:
            query: Original query text
            entities: Extracted entities

        Returns:
            Complexity score between 0.0 and 1.0
        """
        # Base complexity from query length
        length_score = min(len(query.split()) / 20.0, 1.0)

        # Entity complexity
        entity_score = min(len(entities) / 5.0, 1.0)

        # Linked entities bonus
        linked_entities = sum(1 for e in entities if e.linked_entity_id)
        linked_score = min(linked_entities / len(entities) if entities else 0, 1.0)

        return (length_score + entity_score + linked_score) / 3.0
