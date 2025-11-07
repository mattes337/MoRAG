"""Graph construction utilities and helper functions."""

from typing import Any, Dict, List

import structlog

from ..models.fact import Fact, FactRelation
from ..models.graph import Graph


class GraphUtilities:
    """Utility functions for graph construction and manipulation."""

    def __init__(self):
        """Initialize graph utilities."""
        self.logger = structlog.get_logger(__name__)

    def build_graph_structure(
        self, facts: List[Fact], relationships: List[FactRelation]
    ) -> Graph:
        """Build graph structure from facts and relationships.

        Args:
            facts: List of facts to include as nodes
            relationships: List of relationships to include as edges

        Returns:
            Graph object with nodes and edges
        """
        try:
            # Create nodes from facts
            nodes = []
            for i, fact in enumerate(facts):
                node = {
                    "id": f"fact_{i}",
                    "type": "fact",
                    "data": {
                        "subject": fact.subject,
                        "object": fact.object,
                        "approach": fact.approach,
                        "solution": fact.solution,
                        "confidence": fact.confidence,
                        "context": fact.context,
                        "citations": getattr(fact, "citations", []),
                    },
                }
                nodes.append(node)

            # Create edges from relationships
            edges = []
            for i, relationship in enumerate(relationships):
                # Find source and target fact indices
                source_idx = self._find_fact_index(relationship.source_fact, facts)
                target_idx = self._find_fact_index(relationship.target_fact, facts)

                if source_idx is None or target_idx is None:
                    self.logger.warning(
                        "Could not find fact indices for relationship",
                        relationship_id=i,
                    )
                    continue

                edge = {
                    "id": f"rel_{i}",
                    "source": f"fact_{source_idx}",
                    "target": f"fact_{target_idx}",
                    "type": relationship.relation_type.value
                    if hasattr(relationship.relation_type, "value")
                    else str(relationship.relation_type),
                    "data": {
                        "confidence": relationship.confidence,
                        "explanation": relationship.explanation,
                        "context": relationship.context,
                    },
                }
                edges.append(edge)

            # Create and return graph
            graph = Graph(nodes=nodes, edges=edges)

            self.logger.info(
                "Graph structure built successfully",
                num_nodes=len(nodes),
                num_edges=len(edges),
            )

            return graph

        except Exception as e:
            self.logger.error(
                "Error building graph structure",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Return empty graph on error
            return Graph(nodes=[], edges=[])

    def _find_fact_index(self, target_fact: Fact, facts: List[Fact]) -> int:
        """Find the index of a fact in the facts list.

        Args:
            target_fact: Fact to find
            facts: List of facts to search in

        Returns:
            Index of the fact, or None if not found
        """
        for i, fact in enumerate(facts):
            if self._facts_equal(fact, target_fact):
                return i
        return None

    def _facts_equal(self, fact1: Fact, fact2: Fact) -> bool:
        """Check if two facts are equal based on their content.

        Args:
            fact1: First fact
            fact2: Second fact

        Returns:
            True if facts are considered equal
        """
        return (
            fact1.subject == fact2.subject
            and fact1.object == fact2.object
            and fact1.approach == fact2.approach
            and fact1.solution == fact2.solution
        )

    def validate_graph(self, graph: Graph) -> Dict[str, Any]:
        """Validate graph structure and return statistics.

        Args:
            graph: Graph to validate

        Returns:
            Dictionary with validation results and statistics
        """
        try:
            stats = {
                "valid": True,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "issues": [],
            }

            # Check for duplicate node IDs
            node_ids = [node.get("id") for node in graph.nodes if node.get("id")]
            if len(node_ids) != len(set(node_ids)):
                stats["issues"].append("Duplicate node IDs found")
                stats["valid"] = False

            # Check for duplicate edge IDs
            edge_ids = [edge.get("id") for edge in graph.edges if edge.get("id")]
            if len(edge_ids) != len(set(edge_ids)):
                stats["issues"].append("Duplicate edge IDs found")
                stats["valid"] = False

            # Check edge references
            valid_node_ids = set(node_ids)
            for edge in graph.edges:
                source = edge.get("source")
                target = edge.get("target")

                if source not in valid_node_ids:
                    stats["issues"].append(
                        f"Edge references invalid source node: {source}"
                    )
                    stats["valid"] = False

                if target not in valid_node_ids:
                    stats["issues"].append(
                        f"Edge references invalid target node: {target}"
                    )
                    stats["valid"] = False

            return stats

        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "node_count": 0,
                "edge_count": 0,
                "issues": [f"Validation error: {e}"],
            }

    def get_graph_statistics(self, graph: Graph) -> Dict[str, Any]:
        """Get detailed statistics about the graph.

        Args:
            graph: Graph to analyze

        Returns:
            Dictionary with graph statistics
        """
        stats = {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "node_types": {},
            "edge_types": {},
            "avg_confidence": 0.0,
        }

        # Count node types
        for node in graph.nodes:
            node_type = node.get("type", "unknown")
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1

        # Count edge types and calculate average confidence
        confidences = []
        for edge in graph.edges:
            edge_type = edge.get("type", "unknown")
            stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1

            # Get confidence from edge data
            confidence = edge.get("data", {}).get("confidence")
            if confidence is not None:
                confidences.append(float(confidence))

        if confidences:
            stats["avg_confidence"] = sum(confidences) / len(confidences)

        return stats
