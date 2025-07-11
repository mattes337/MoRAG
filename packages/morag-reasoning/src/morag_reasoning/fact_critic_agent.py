"""FactCriticAgent for evaluating relevance of raw facts and assigning scores."""

import json
import structlog
from typing import Optional
from pydantic_ai import Agent
from pydantic import BaseModel, Field

from morag_reasoning.llm import LLMClient
from morag_reasoning.recursive_fact_models import RawFact, ScoredFact, FCAResponse


class FactCriticAgent:
    """Agent responsible for evaluating the relevance of raw facts and assigning scores."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the FactCriticAgent.
        
        Args:
            llm_client: LLM client for AI operations
        """
        self.llm_client = llm_client
        self.logger = structlog.get_logger(__name__)
        
        # Create PydanticAI agent for fact criticism
        self.agent = Agent(
            model=llm_client.get_model(),
            result_type=FCAResponse,
            system_prompt=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the FactCriticAgent."""
        return """You are a FactCriticAgent (FCA) responsible for evaluating the relevance and quality of extracted facts.

Your role is to:
1. Analyze a raw fact in the context of the original user query
2. Assign a relevance score from 0.0 to 1.0
3. Generate a user-friendly source description

SCORING GUIDELINES:
- 1.0: Directly answers the user query or provides crucial information
- 0.8-0.9: Highly relevant, provides important context or supporting information
- 0.6-0.7: Moderately relevant, provides useful background or related information
- 0.4-0.5: Somewhat relevant, tangentially related to the query
- 0.2-0.3: Minimally relevant, provides general context but not specific to query
- 0.0-0.1: Not relevant or irrelevant to the query

SOURCE DESCRIPTION GUIDELINES:
- Create a clear, user-friendly description of where the fact came from
- Include document name, node type, or content source as appropriate
- Examples: "From the Wikipedia page for Neo4j", "From a research paper on graph algorithms", "From company profile data"
- Be specific but concise

RESPONSE FORMAT:
Return the original raw fact enhanced with:
- score: Float between 0.0 and 1.0 indicating relevance
- source_description: User-friendly description of the source

Be objective and consistent in your scoring. Focus on how well the fact helps answer the user's specific question."""

    async def evaluate_fact(
        self,
        user_query: str,
        raw_fact: RawFact
    ) -> ScoredFact:
        """Evaluate a raw fact and assign a relevance score.
        
        Args:
            user_query: Original user query
            raw_fact: Raw fact to evaluate
            
        Returns:
            ScoredFact with relevance score and source description
        """
        self.logger.debug(
            "Evaluating fact",
            query=user_query,
            fact=raw_fact.fact_text[:100]
        )
        
        try:
            # Create evaluation prompt
            prompt = self._create_evaluation_prompt(user_query, raw_fact)
            
            # Call LLM for fact evaluation
            result = await self.agent.run(prompt)
            response = result.data
            
            # Create scored fact
            scored_fact = response.scored_fact
            
            self.logger.debug(
                "Fact evaluation completed",
                fact=raw_fact.fact_text[:100],
                score=scored_fact.score,
                source=scored_fact.source_description
            )
            
            return scored_fact
            
        except Exception as e:
            self.logger.error(
                "Error evaluating fact",
                fact=raw_fact.fact_text[:100],
                error=str(e)
            )
            # Return fact with low score as fallback
            return ScoredFact(
                fact_text=raw_fact.fact_text,
                source_node_id=raw_fact.source_node_id,
                source_property=raw_fact.source_property,
                source_qdrant_chunk_id=raw_fact.source_qdrant_chunk_id,
                extracted_from_depth=raw_fact.extracted_from_depth,
                score=0.1,  # Low score for error cases
                source_description=f"Error evaluating fact from node {raw_fact.source_node_id}"
            )
    
    def _create_evaluation_prompt(self, user_query: str, raw_fact: RawFact) -> str:
        """Create the prompt for fact evaluation."""
        
        # Determine source type for better description
        source_info = []
        if raw_fact.source_property:
            source_info.append(f"Property: {raw_fact.source_property}")
        if raw_fact.source_qdrant_chunk_id:
            source_info.append(f"Content Chunk: {raw_fact.source_qdrant_chunk_id}")
        
        source_context = f"Node ID: {raw_fact.source_node_id}"
        if source_info:
            source_context += f" ({', '.join(source_info)})"
        
        prompt = f"""FACT EVALUATION TASK

User Query: "{user_query}"

Raw Fact to Evaluate:
Fact Text: "{raw_fact.fact_text}"
Source: {source_context}
Extraction Depth: {raw_fact.extracted_from_depth}

Your task:
1. Evaluate how relevant this fact is to answering the user query
2. Assign a score from 0.0 (not relevant) to 1.0 (highly relevant)
3. Create a user-friendly description of the source

Consider:
- Does this fact directly answer the query?
- Does it provide important context or supporting information?
- How specific is it to the user's question?
- Is it factual and verifiable?

Return the fact with an assigned score and source description."""

        return prompt
    
    async def batch_evaluate_facts(
        self,
        user_query: str,
        raw_facts: list[RawFact],
        batch_size: int = 10
    ) -> list[ScoredFact]:
        """Evaluate multiple facts in batches for efficiency.
        
        Args:
            user_query: Original user query
            raw_facts: List of raw facts to evaluate
            batch_size: Number of facts to process in each batch
            
        Returns:
            List of scored facts
        """
        if not raw_facts:
            return []
        
        self.logger.info(
            "Starting batch fact evaluation",
            total_facts=len(raw_facts),
            batch_size=batch_size
        )
        
        scored_facts = []
        
        # Process facts in batches
        for i in range(0, len(raw_facts), batch_size):
            batch = raw_facts[i:i + batch_size]
            self.logger.debug(f"Processing batch {i//batch_size + 1}")
            
            # Evaluate each fact in the batch
            batch_results = []
            for fact in batch:
                try:
                    scored_fact = await self.evaluate_fact(user_query, fact)
                    batch_results.append(scored_fact)
                except Exception as e:
                    self.logger.warning(
                        "Failed to evaluate fact in batch",
                        fact=fact.fact_text[:100],
                        error=str(e)
                    )
                    # Add with low score as fallback
                    batch_results.append(ScoredFact(
                        fact_text=fact.fact_text,
                        source_node_id=fact.source_node_id,
                        source_property=fact.source_property,
                        source_qdrant_chunk_id=fact.source_qdrant_chunk_id,
                        extracted_from_depth=fact.extracted_from_depth,
                        score=0.1,
                        source_description=f"Batch evaluation error for node {fact.source_node_id}"
                    ))
            
            scored_facts.extend(batch_results)
        
        self.logger.info(
            "Batch fact evaluation completed",
            total_evaluated=len(scored_facts)
        )
        
        return scored_facts
    
    def apply_relevance_decay(
        self,
        scored_facts: list[ScoredFact],
        decay_rate: float = 0.2
    ) -> list[ScoredFact]:
        """Apply depth-based relevance decay to scored facts.
        
        Args:
            scored_facts: List of scored facts
            decay_rate: Rate of decay per depth level
            
        Returns:
            List of facts with final decayed scores
        """
        if not scored_facts:
            return []
        
        self.logger.info(
            "Applying relevance decay",
            total_facts=len(scored_facts),
            decay_rate=decay_rate
        )
        
        from morag_reasoning.recursive_fact_models import FinalFact
        
        final_facts = []
        for fact in scored_facts:
            # Calculate decay factor: score_at_depth = initial_score * (1 - decay_rate * depth)
            decay_factor = max(0.0, 1.0 - decay_rate * fact.extracted_from_depth)
            final_decayed_score = max(0.0, fact.score * decay_factor)
            
            final_fact = FinalFact(
                fact_text=fact.fact_text,
                source_node_id=fact.source_node_id,
                source_property=fact.source_property,
                source_qdrant_chunk_id=fact.source_qdrant_chunk_id,
                extracted_from_depth=fact.extracted_from_depth,
                score=fact.score,
                source_description=fact.source_description,
                final_decayed_score=final_decayed_score
            )
            final_facts.append(final_fact)
        
        # Sort by final decayed score, highest first
        final_facts.sort(key=lambda x: x.final_decayed_score, reverse=True)
        
        self.logger.info(
            "Relevance decay applied",
            facts_processed=len(final_facts)
        )
        
        return final_facts
