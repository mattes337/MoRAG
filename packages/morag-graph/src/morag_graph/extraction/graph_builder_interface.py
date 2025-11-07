"""Graph builder interface and main coordination logic."""

import asyncio
import traceback
from typing import List, Optional
from morag_core.utils.logging import get_logger

from morag_reasoning.llm import LLMClient, LLMConfig
from ..models.fact import Fact
from ..models.graph import Graph
from .fact_graph_operations import FactGraphOperations
from .graph_utilities import GraphUtilities


class FactGraphBuilder:
    """Build knowledge graph from extracted facts."""

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        min_relation_confidence: float = 0.6,
        max_relations_per_fact: int = 5,
        language: str = "en",
        processing_timeout: int = 300
    ):
        """Initialize fact graph builder.

        Args:
            model_id: LLM model for relationship extraction
            api_key: API key for LLM service
            min_relation_confidence: Minimum confidence for relationships
            max_relations_per_fact: Maximum relationships per fact
            language: Language for relationship extraction
            processing_timeout: Timeout for graph building operations in seconds
        """
        self.model_id = model_id
        self.api_key = api_key
        self.min_relation_confidence = min_relation_confidence
        self.max_relations_per_fact = max_relations_per_fact
        self.language = language
        self.processing_timeout = processing_timeout

        self.logger = get_logger(__name__)

        # Initialize LLM client
        self.llm_config = LLMConfig(
            provider="gemini",
            model=model_id,
            api_key=api_key,
            temperature=0.1,
            max_tokens=2000
        )
        self.llm_client = LLMClient(self.llm_config)

        # Initialize helper components
        self.operations = FactGraphOperations(self.llm_client, self.logger)
        self.utilities = GraphUtilities()

    async def build_fact_graph(self, facts: List[Fact]) -> Graph:
        """Build knowledge graph with proper resource management."""
        if not facts:
            return Graph(nodes=[], edges=[])

        relationships = []
        try:
            # Create relationships and build graph with timeout
            async with asyncio.timeout(self.processing_timeout):
                relationships = await self.operations.create_fact_relationships(
                    facts,
                    self.min_relation_confidence,
                    self.max_relations_per_fact
                )

                # Build graph structure
                graph = self.utilities.build_graph_structure(facts, relationships)

            # Index with separate error handling (outside timeout)
            try:
                await self._index_facts(facts)
            except Exception as index_error:
                self.logger.warning("Fact indexing failed", error=str(index_error))

            return graph

        except asyncio.TimeoutError:
            self.logger.error("Graph building timed out", num_facts=len(facts))
            return self.utilities.build_graph_structure(facts, relationships)
        except Exception as e:
            self.logger.error("Graph building failed", error=str(e), traceback=traceback.format_exc())
            raise  # Let caller handle - don't hide errors

    async def _index_facts(self, facts: List[Fact]):
        """Index facts for efficient retrieval (placeholder for now)."""
        # TODO: Implement fact indexing if needed

    async def close(self):
        """Close the graph builder and clean up resources."""
        try:
            if hasattr(self.llm_client, 'close'):
                await self.llm_client.close()
        except Exception as e:
            self.logger.warning("Error closing LLM client", error=str(e))

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()
