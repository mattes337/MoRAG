"""Graph builder interface and main coordination logic."""

import asyncio
from typing import List, Optional
from morag_core.utils.logging import get_logger

from morag_reasoning.llm import LLMClient, LLMConfig
from ..models.fact import Fact, FactRelation
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
        language: str = "en"
    ):
        """Initialize fact graph builder.

        Args:
            model_id: LLM model for relationship extraction
            api_key: API key for LLM service
            min_relation_confidence: Minimum confidence for relationships
            max_relations_per_fact: Maximum relationships per fact
            language: Language for relationship extraction
        """
        self.model_id = model_id
        self.api_key = api_key
        self.min_relation_confidence = min_relation_confidence
        self.max_relations_per_fact = max_relations_per_fact
        self.language = language
        
        self.logger = get_logger(__name__)
        
        # Initialize LLM client
        llm_config = LLMConfig(
            provider="gemini",
            model=model_id,
            api_key=api_key,
            temperature=0.1,
            max_tokens=2000
        )
        self.llm_client = LLMClient(llm_config)
        
        # Initialize helper components
        self.operations = FactGraphOperations(self.llm_client, self.logger)
        self.utilities = GraphUtilities()
    
    async def build_fact_graph(self, facts: List[Fact]) -> Graph:
        """Build knowledge graph from extracted facts.
        
        Args:
            facts: List of facts to build graph from
            
        Returns:
            Graph object containing facts and their relationships
        """
        if not facts:
            return Graph(nodes=[], edges=[])
        
        self.logger.info(
            "Starting fact graph building",
            num_facts=len(facts)
        )
        
        try:
            # Create fact relationships using operations component
            relationships = await self.operations.create_fact_relationships(
                facts, 
                self.min_relation_confidence,
                self.max_relations_per_fact
            )
            
            # Build graph structure using utilities
            graph = self.utilities.build_graph_structure(facts, relationships)
            
            # Index facts for efficient retrieval
            await self._index_facts(facts)
            
            self.logger.info(
                "Fact graph building completed",
                num_facts=len(facts),
                num_relationships=len(relationships)
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(
                "Fact graph building failed",
                error=str(e),
                error_type=type(e).__name__
            )
            # Add detailed debugging for the string join error
            if "expected str instance" in str(e):
                self.logger.error(
                    "String join error detected - checking relationship types"
                )
            import traceback
            self.logger.error("Full traceback:", traceback=traceback.format_exc())
            # Return empty graph on failure
            return Graph(nodes=[], edges=[])

    async def _index_facts(self, facts: List[Fact]):
        """Index facts for efficient retrieval (placeholder for now)."""
        # TODO: Implement fact indexing if needed
        pass