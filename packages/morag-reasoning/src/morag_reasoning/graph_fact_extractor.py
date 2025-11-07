"""Enhanced graph-based fact extraction from traversal results."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from morag_core.config import get_settings

from .llm import LLMClient

logger = structlog.get_logger(__name__)

# Import traversal components
try:
    from morag_graph.models import Entity, Relation
    from morag_graph.operations.traversal import GraphPath
    from morag_graph.traversal import QueryContext, TraversalResult

    GRAPH_COMPONENTS_AVAILABLE = True
except ImportError:
    GRAPH_COMPONENTS_AVAILABLE = False
    logger.warning("Graph traversal components not available")


class FactType(Enum):
    """Types of facts that can be extracted."""

    DIRECT = "direct"  # Single entity-relation-entity triplet
    CHAIN = "chain"  # Multi-hop relationship chain
    CONTEXTUAL = "contextual"  # Facts requiring document context
    INFERRED = "inferred"  # Facts derived from multiple sources
    TEMPORAL = "temporal"  # Facts with time-dependent information


@dataclass
class ExtractedFact:
    """Represents a fact extracted from graph traversal."""

    fact_id: str
    content: str
    fact_type: FactType
    confidence: float
    source_entities: List[str]
    source_relations: List[str]
    source_documents: List[str]
    extraction_path: List[str]
    context: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class FactExtractionContext:
    """Context for fact extraction from traversal results."""

    query: str
    query_intent: str
    target_entities: List[str]
    traversal_depth: int
    relationship_chains: List[List[str]]
    document_context: Dict[str, Any]


class GraphFactExtractor:
    """Enhanced fact extraction from graph traversal results."""

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        """Initialize the graph fact extractor.

        Args:
            llm_client: LLM client for fact extraction
            config: Optional configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.settings = get_settings()

        # Extraction parameters
        self.max_facts_per_path = self.config.get("max_facts_per_path", 10)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.3)
        self.enable_chain_facts = self.config.get("enable_chain_facts", True)
        self.enable_contextual_facts = self.config.get("enable_contextual_facts", True)
        self.enable_inferred_facts = self.config.get("enable_inferred_facts", True)

        # Performance settings
        self.batch_size = self.config.get("batch_size", 5)
        self.max_extraction_time = self.config.get("max_extraction_time", 30.0)

        logger.info(
            "Graph fact extractor initialized",
            max_facts_per_path=self.max_facts_per_path,
            min_confidence_threshold=self.min_confidence_threshold,
            enable_chain_facts=self.enable_chain_facts,
            enable_contextual_facts=self.enable_contextual_facts,
        )

    async def extract_facts(
        self, traversal_result: "TraversalResult", query_context: "QueryContext"
    ) -> List[ExtractedFact]:
        """Extract facts from graph traversal results.

        Args:
            traversal_result: Results from recursive graph traversal
            query_context: Context information for the query

        Returns:
            List of extracted facts with full context
        """
        if not GRAPH_COMPONENTS_AVAILABLE:
            logger.warning("Graph components not available, using fallback extraction")
            return await self._extract_facts_fallback(traversal_result, query_context)

        if not traversal_result.discovered_paths:
            return []

        start_time = time.time()

        try:
            logger.info(
                "Starting graph fact extraction",
                query=query_context.query,
                num_paths=len(traversal_result.discovered_paths),
                strategy=traversal_result.strategy_used.value,
            )

            # Create extraction context
            extraction_context = self._create_extraction_context(
                traversal_result, query_context
            )

            # Extract facts from each path
            all_facts = []

            # Process paths in batches
            paths = [p.path for p in traversal_result.discovered_paths]
            for i in range(0, len(paths), self.batch_size):
                batch = paths[i : i + self.batch_size]
                batch_facts = await self._extract_facts_from_path_batch(
                    batch, extraction_context
                )
                all_facts.extend(batch_facts)

                # Check time limit
                if time.time() - start_time > self.max_extraction_time:
                    logger.warning("Fact extraction time limit exceeded")
                    break

            # Extract chain facts if enabled
            if self.enable_chain_facts:
                chain_facts = await self._extract_chain_facts(paths, extraction_context)
                all_facts.extend(chain_facts)

            # Extract contextual facts if enabled
            if self.enable_contextual_facts:
                contextual_facts = await self._extract_contextual_facts(
                    traversal_result, extraction_context
                )
                all_facts.extend(contextual_facts)

            # Filter by confidence threshold
            filtered_facts = [
                f for f in all_facts if f.confidence >= self.min_confidence_threshold
            ]

            # Deduplicate facts
            unique_facts = self._deduplicate_facts(filtered_facts)

            extraction_time = time.time() - start_time

            logger.info(
                "Graph fact extraction completed",
                total_facts=len(all_facts),
                filtered_facts=len(filtered_facts),
                unique_facts=len(unique_facts),
                extraction_time=extraction_time,
            )

            return unique_facts

        except Exception as e:
            logger.error(f"Graph fact extraction failed: {e}")
            return []

    def _create_extraction_context(
        self, traversal_result: "TraversalResult", query_context: "QueryContext"
    ) -> FactExtractionContext:
        """Create context for fact extraction."""
        # Extract relationship chains from paths
        relationship_chains = []
        for path_score in traversal_result.discovered_paths:
            path = path_score.path
            if hasattr(path, "relations") and path.relations:
                chain = [rel.type for rel in path.relations]
                relationship_chains.append(chain)

        return FactExtractionContext(
            query=query_context.query,
            query_intent=query_context.intent,
            target_entities=query_context.entities,
            traversal_depth=traversal_result.metadata.get("max_depth_reached", 0),
            relationship_chains=relationship_chains,
            document_context={},
        )

    async def _extract_facts_from_path_batch(
        self, paths: List["GraphPath"], context: FactExtractionContext
    ) -> List[ExtractedFact]:
        """Extract facts from a batch of paths."""
        facts = []

        for path in paths:
            try:
                path_facts = await self._extract_facts_from_single_path(path, context)
                facts.extend(path_facts)
            except Exception as e:
                logger.warning(f"Failed to extract facts from path: {e}")
                continue

        return facts

    async def _extract_facts_from_single_path(
        self, path: "GraphPath", context: FactExtractionContext
    ) -> List[ExtractedFact]:
        """Extract facts from a single graph path with enhanced context."""
        facts = []

        if not hasattr(path, "entities") or not hasattr(path, "relations"):
            return facts

        # Extract direct facts (entity-relation-entity triplets)
        for i, relation in enumerate(path.relations):
            if i + 1 < len(path.entities):
                source_entity = path.entities[i]
                target_entity = path.entities[i + 1]

                # Enhanced fact content with context
                fact_content = self._generate_enhanced_fact_content(
                    source_entity, relation, target_entity, context, i
                )

                # Calculate enhanced confidence
                enhanced_confidence = self._calculate_enhanced_confidence(
                    relation, source_entity, target_entity, context
                )

                fact = ExtractedFact(
                    fact_id=f"fact_{source_entity.id}_{relation.id}_{target_entity.id}",
                    content=fact_content,
                    fact_type=FactType.DIRECT,
                    confidence=enhanced_confidence,
                    source_entities=[source_entity.id, target_entity.id],
                    source_relations=[relation.id],
                    source_documents=self._get_source_documents(
                        source_entity, target_entity
                    ),
                    extraction_path=[entity.id for entity in path.entities],
                    context={
                        "relation_type": relation.type,
                        "source_entity_type": getattr(source_entity, "type", "Unknown"),
                        "target_entity_type": getattr(target_entity, "type", "Unknown"),
                        "hop_position": i,
                        "total_hops": len(path.relations),
                        "query_context": context.query
                        if hasattr(context, "query")
                        else "",
                        "semantic_context": self._extract_semantic_context(path, i),
                    },
                    metadata={
                        "extraction_method": "direct_triplet_enhanced",
                        "path_length": len(path.entities),
                        "relation_strength": getattr(relation, "weight", 1.0),
                        "extraction_timestamp": time.time(),
                    },
                )
                facts.append(fact)

        # Extract chain facts if enabled and path is multi-hop
        if self.enable_chain_facts and len(path.entities) > 2:
            chain_fact = await self._extract_chain_fact_from_path(path, context)
            if chain_fact:
                facts.append(chain_fact)

        return facts

    def _generate_enhanced_fact_content(
        self,
        source_entity: "Entity",
        relation: "Relation",
        target_entity: "Entity",
        context: FactExtractionContext,
        hop_position: int,
    ) -> str:
        """Generate enhanced fact content with context."""
        # Basic fact content
        relation_text = relation.type.replace("_", " ").lower()
        basic_content = f"{source_entity.name} {relation_text} {target_entity.name}"

        # Add contextual information if available
        if hasattr(relation, "properties") and relation.properties:
            # Add relevant properties
            props = []
            for key, value in relation.properties.items():
                if key in ["confidence", "strength", "weight"] and isinstance(
                    value, (int, float)
                ):
                    props.append(f"{key}: {value:.2f}")
                elif key in ["date", "time", "year"] and value:
                    props.append(f"({value})")

            if props:
                basic_content += f" [{', '.join(props)}]"

        # Add hop context for multi-hop paths
        if hop_position > 0:
            basic_content = f"Through {hop_position} hop(s): {basic_content}"

        return basic_content

    def _calculate_enhanced_confidence(
        self,
        relation: "Relation",
        source_entity: "Entity",
        target_entity: "Entity",
        context: FactExtractionContext,
    ) -> float:
        """Calculate enhanced confidence score."""
        base_confidence = getattr(relation, "confidence", 0.8)

        # Adjust based on entity types
        entity_confidence = 1.0
        if hasattr(source_entity, "type") and hasattr(target_entity, "type"):
            # Higher confidence for well-defined entity types
            known_types = {"PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"}
            if source_entity.type in known_types and target_entity.type in known_types:
                entity_confidence = 1.1

        # Adjust based on relation strength
        relation_confidence = 1.0
        if hasattr(relation, "weight") and relation.weight:
            relation_confidence = min(1.2, 1.0 + (relation.weight - 1.0) * 0.2)

        # Combine confidences
        final_confidence = base_confidence * entity_confidence * relation_confidence
        return min(1.0, final_confidence)

    def _extract_semantic_context(self, path: "GraphPath", hop_position: int) -> str:
        """Extract semantic context around a specific hop."""
        context_parts = []

        # Add previous hop context
        if hop_position > 0:
            prev_entity = path.entities[hop_position - 1]
            context_parts.append(f"from {prev_entity.name}")

        # Add next hop context
        if hop_position + 2 < len(path.entities):
            next_entity = path.entities[hop_position + 2]
            context_parts.append(f"leading to {next_entity.name}")

        return " ".join(context_parts) if context_parts else ""

    async def _extract_chain_fact_from_path(
        self, path: "GraphPath", context: FactExtractionContext
    ) -> Optional[ExtractedFact]:
        """Extract a chain fact representing the entire path."""
        if len(path.entities) < 3:
            return None

        try:
            # Create chain description
            chain_description = self._create_chain_description(path)

            # Calculate chain confidence
            chain_confidence = self._calculate_chain_confidence(path)

            return ExtractedFact(
                fact_id=f"chain_{path.entities[0].id}_to_{path.entities[-1].id}",
                content=chain_description,
                fact_type=FactType.CHAIN,
                confidence=chain_confidence,
                source_entities=[entity.id for entity in path.entities],
                source_relations=[relation.id for relation in path.relations],
                source_documents=self._get_all_source_documents(path),
                extraction_path=[entity.id for entity in path.entities],
                context={
                    "chain_length": len(path.entities),
                    "start_entity": path.entities[0].name,
                    "end_entity": path.entities[-1].name,
                    "intermediate_entities": [e.name for e in path.entities[1:-1]],
                },
                metadata={
                    "extraction_method": "chain_fact",
                    "path_length": len(path.entities),
                    "extraction_timestamp": time.time(),
                },
            )

        except Exception as e:
            logger.warning(f"Failed to extract chain fact: {e}")
            return None

    def _create_chain_description(self, path: "GraphPath") -> str:
        """Create a human-readable description of the relationship chain."""
        if len(path.entities) < 2:
            return ""

        parts = []
        for i, relation in enumerate(path.relations):
            if i + 1 < len(path.entities):
                source = path.entities[i]
                target = path.entities[i + 1]
                relation_text = relation.type.replace("_", " ").lower()
                parts.append(f"{source.name} {relation_text} {target.name}")

        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]}, which {parts[1]}"
        else:
            return f"{parts[0]}, through {len(parts)-1} connections, ultimately {parts[-1]}"

    def _calculate_chain_confidence(self, path: "GraphPath") -> float:
        """Calculate confidence for a chain fact."""
        if not path.relations:
            return 0.5

        # Average confidence of all relations with decay for length
        relation_confidences = [
            getattr(rel, "confidence", 0.8) for rel in path.relations
        ]
        avg_confidence = sum(relation_confidences) / len(relation_confidences)

        # Apply decay for longer chains
        length_penalty = 0.9 ** (len(path.relations) - 1)

        return avg_confidence * length_penalty

    async def _extract_chain_facts(
        self, paths: List["GraphPath"], context: FactExtractionContext
    ) -> List[ExtractedFact]:
        """Extract chain facts from multi-hop paths."""
        chain_facts = []

        for path in paths:
            if not hasattr(path, "entities") or len(path.entities) < 3:
                continue

            # Create chain fact for the entire path
            start_entity = path.entities[0]
            end_entity = path.entities[-1]

            # Build chain description
            chain_description = self._build_chain_description(path)

            chain_fact = ExtractedFact(
                fact_id=f"chain_{start_entity.id}_to_{end_entity.id}",
                content=chain_description,
                fact_type=FactType.CHAIN,
                confidence=0.7,  # Chain facts have moderate confidence
                source_entities=[entity.id for entity in path.entities],
                source_relations=[rel.id for rel in path.relations]
                if hasattr(path, "relations")
                else [],
                source_documents=self._get_source_documents(*path.entities),
                extraction_path=[entity.id for entity in path.entities],
                context={
                    "chain_length": len(path.entities),
                    "relationship_types": [rel.type for rel in path.relations]
                    if hasattr(path, "relations")
                    else [],
                },
                metadata={
                    "extraction_method": "relationship_chain",
                    "path_complexity": len(path.entities) + len(path.relations)
                    if hasattr(path, "relations")
                    else len(path.entities),
                },
            )
            chain_facts.append(chain_fact)

        return chain_facts

    async def _extract_contextual_facts(
        self, traversal_result: "TraversalResult", context: FactExtractionContext
    ) -> List[ExtractedFact]:
        """Extract contextual facts that require document context."""
        # This would typically involve analyzing document chunks
        # For now, return empty list as this requires document storage integration
        logger.debug("Contextual fact extraction not yet implemented")
        return []

    def _build_chain_description(self, path: "GraphPath") -> str:
        """Build a human-readable description of a relationship chain."""
        if not hasattr(path, "entities") or not hasattr(path, "relations"):
            return "Unknown relationship chain"

        if len(path.entities) < 2:
            return "Single entity"

        description_parts = [path.entities[0].name]

        for i, relation in enumerate(path.relations):
            if i + 1 < len(path.entities):
                relation_text = relation.type.replace("_", " ").lower()
                target_entity = path.entities[i + 1]
                description_parts.append(f" {relation_text} {target_entity.name}")

        return "".join(description_parts)

    def _get_source_documents(self, *entities) -> List[str]:
        """Get source document IDs for entities."""
        documents = []
        for entity in entities:
            if hasattr(entity, "source_doc_id") and entity.source_doc_id:
                documents.append(entity.source_doc_id)
            # Also check for document references in metadata
            if hasattr(entity, "metadata") and entity.metadata:
                if "source_documents" in entity.metadata:
                    if isinstance(entity.metadata["source_documents"], list):
                        documents.extend(entity.metadata["source_documents"])
                    else:
                        documents.append(str(entity.metadata["source_documents"]))
        return list(set(documents))  # Remove duplicates

    def _get_all_source_documents(self, path: "GraphPath") -> List[str]:
        """Get all source documents for entities and relations in a path."""
        documents = []

        # Get documents from entities
        if hasattr(path, "entities"):
            documents.extend(self._get_source_documents(*path.entities))

        # Get documents from relations
        if hasattr(path, "relations"):
            for relation in path.relations:
                if hasattr(relation, "source_doc_id") and relation.source_doc_id:
                    documents.append(relation.source_doc_id)
                if hasattr(relation, "metadata") and relation.metadata:
                    if "source_documents" in relation.metadata:
                        if isinstance(relation.metadata["source_documents"], list):
                            documents.extend(relation.metadata["source_documents"])
                        else:
                            documents.append(str(relation.metadata["source_documents"]))

        return list(set(documents))  # Remove duplicates

    def _deduplicate_facts(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """Remove duplicate facts based on content similarity."""
        unique_facts = []
        seen_contents = set()

        for fact in facts:
            # Simple deduplication based on content
            content_key = fact.content.lower().strip()
            if content_key not in seen_contents:
                unique_facts.append(fact)
                seen_contents.add(content_key)

        return unique_facts

    async def _extract_facts_fallback(
        self, traversal_result: Any, query_context: Any
    ) -> List[ExtractedFact]:
        """Fallback fact extraction when graph components are not available."""
        logger.info("Using fallback fact extraction")

        # Create a simple fact from the query context
        if hasattr(query_context, "query"):
            fallback_fact = ExtractedFact(
                fact_id="fallback_001",
                content=f"Query: {query_context.query}",
                fact_type=FactType.CONTEXTUAL,
                confidence=0.5,
                source_entities=[],
                source_relations=[],
                source_documents=[],
                extraction_path=[],
                context={"fallback": True},
                metadata={"extraction_method": "fallback"},
            )
            return [fallback_fact]

        return []
