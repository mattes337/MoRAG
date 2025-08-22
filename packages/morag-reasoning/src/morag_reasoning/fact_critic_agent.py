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
            output_type=FCAResponse,
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
- Reference entities by their meaningful names (e.g., "ADHS", "Ernährung") rather than technical IDs
- Include specific document details: document name, chunk/page number, section if available
- For audio/video: include timestamp information
- Examples: "From document 'Research_Paper.pdf', page 5, section 'Methodology'", "From video 'Training_Session.mp4' at 15:30", "From entity 'ADHS' in document 'Treatment_Guide.pdf'"
- Be specific and include location details that help users find the source
- Focus on meaningful entity names and content sources, not database metadata

RESPONSE FORMAT:
Return the original raw fact enhanced with:
- score: Float between 0.0 and 1.0 indicating relevance
- source_description: User-friendly description of the source

Be objective and consistent in your scoring. Focus on how well the fact helps answer the user's specific question."""

    async def evaluate_fact(
        self,
        user_query: str,
        raw_fact: RawFact,
        language: Optional[str] = None
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
            prompt = self._create_evaluation_prompt(user_query, raw_fact, language)

            # Call LLM for fact evaluation with retry logic for malformed function calls
            try:
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

            except Exception as agent_error:
                # Check if this is a malformed function call error
                error_str = str(agent_error)
                if "MALFORMED_FUNCTION_CALL" in error_str or "Content field missing" in error_str:
                    self.logger.warning(
                        "PydanticAI agent failed with malformed function call, using fallback evaluation",
                        fact=raw_fact.fact_text[:100],
                        error=error_str
                    )
                    # Use fallback evaluation method
                    return await self._fallback_evaluation(user_query, raw_fact, language)
                else:
                    # Re-raise other errors
                    raise agent_error

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
                source_metadata=raw_fact.source_metadata,
                extracted_from_depth=raw_fact.extracted_from_depth,
                score=0.1,  # Low score for error cases
                source_description=f"Document: {raw_fact.source_metadata.document_name or 'Unknown'}, Chunk: {raw_fact.source_metadata.chunk_index or 0}"
            )
    
    def _create_evaluation_prompt(self, user_query: str, raw_fact: RawFact, language: Optional[str] = None) -> str:
        """Create the prompt for fact evaluation."""

        # Build comprehensive source context focusing on meaningful information
        source_info = []
        if raw_fact.source_property:
            source_info.append(f"Property: {raw_fact.source_property}")
        if raw_fact.source_qdrant_chunk_id:
            source_info.append(f"Content Chunk")

        # Add detailed source metadata
        metadata_info = []
        entity_name = None

        # Extract entity name from metadata if available
        if raw_fact.source_metadata.additional_metadata:
            entity_name = raw_fact.source_metadata.additional_metadata.get("entity_name")

        if raw_fact.source_metadata.document_name:
            metadata_info.append(f"Document: {raw_fact.source_metadata.document_name}")
        if raw_fact.source_metadata.chunk_index is not None:
            metadata_info.append(f"Chunk: {raw_fact.source_metadata.chunk_index}")
        if raw_fact.source_metadata.page_number:
            metadata_info.append(f"Page: {raw_fact.source_metadata.page_number}")
        if raw_fact.source_metadata.section:
            metadata_info.append(f"Section: {raw_fact.source_metadata.section}")
        if raw_fact.source_metadata.timestamp:
            metadata_info.append(f"Timestamp: {raw_fact.source_metadata.timestamp}")

        # Create user-friendly source context - only use document information
        if metadata_info:
            source_context = ", ".join(metadata_info)
        else:
            # If no document metadata, this fact should be filtered out
            source_context = "No document source available"
        
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

IMPORTANT for source description:
- Use meaningful entity names (e.g., "ADHS", "Ernährung") rather than technical IDs
- Focus on document names, sections, and entity relationships
- Make it clear and helpful for users to understand where the information comes from

Return the fact with an assigned score and source description."""

        # Add language instruction if specified
        if language:
            language_names = {
                'en': 'English',
                'de': 'German',
                'fr': 'French',
                'es': 'Spanish',
                'it': 'Italian',
                'pt': 'Portuguese',
                'nl': 'Dutch',
                'ru': 'Russian',
                'zh': 'Chinese',
                'ja': 'Japanese',
                'ko': 'Korean'
            }
            language_name = language_names.get(language, language)
            prompt += f"\n\nIMPORTANT: Provide source description in {language_name} ({language})."

        return prompt
    
    async def batch_evaluate_facts(
        self,
        user_query: str,
        raw_facts: list[RawFact],
        batch_size: int = 10,
        language: Optional[str] = None
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
                    scored_fact = await self.evaluate_fact(user_query, fact, language)
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
                        source_metadata=fact.source_metadata,
                        extracted_from_depth=fact.extracted_from_depth,
                        score=0.1,
                        source_description=f"Document: {fact.source_metadata.document_name or 'Unknown'}, Chunk: {fact.source_metadata.chunk_index or 0}"
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
    ) -> list["FinalFact"]:
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
                source_metadata=fact.source_metadata,
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

    async def _fallback_evaluation(
        self,
        user_query: str,
        raw_fact: RawFact,
        language: Optional[str] = None
    ) -> ScoredFact:
        """Fallback evaluation method when PydanticAI agent fails.

        This method uses the direct LLM client to get a simple text response
        and parses it manually to extract score and source description.

        Args:
            user_query: Original user query
            raw_fact: Raw fact to evaluate
            language: Optional language specification

        Returns:
            ScoredFact with relevance score and source description
        """
        try:
            # Create a simpler prompt for direct text response
            prompt = f"""Evaluate the relevance of this fact to the user query and provide a score and source description.

USER QUERY: {user_query}

FACT TO EVALUATE: {raw_fact.fact_text}

SOURCE CONTEXT:
- Document: {raw_fact.source_metadata.document_name or 'Unknown'}
- Chunk: {raw_fact.source_metadata.chunk_index or 0}
- Property: {raw_fact.source_property or 'content'}

Please respond with exactly this format:
SCORE: [number between 0.0 and 1.0]
SOURCE: [user-friendly description of the source]

Example:
SCORE: 0.8
SOURCE: Document 'Medical Guide', Chapter 3, Page 45"""

            # Use direct LLM call instead of PydanticAI agent
            response_text = await self.llm_client.generate(
                prompt,
                max_tokens=200,
                temperature=0.1
            )

            # Parse the response
            score = 0.5  # Default score
            source_description = f"Document: {raw_fact.source_metadata.document_name or 'Unknown'}, Chunk: {raw_fact.source_metadata.chunk_index or 0}"

            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('SCORE:'):
                    try:
                        score_str = line.replace('SCORE:', '').strip()
                        score = float(score_str)
                        score = max(0.0, min(1.0, score))  # Clamp to valid range
                    except ValueError:
                        self.logger.warning(f"Could not parse score from: {line}")
                elif line.startswith('SOURCE:'):
                    source_description = line.replace('SOURCE:', '').strip()

            self.logger.debug(
                "Fallback fact evaluation completed",
                fact=raw_fact.fact_text[:100],
                score=score,
                source=source_description
            )

            return ScoredFact(
                fact_text=raw_fact.fact_text,
                source_node_id=raw_fact.source_node_id,
                source_property=raw_fact.source_property,
                source_qdrant_chunk_id=raw_fact.source_qdrant_chunk_id,
                source_metadata=raw_fact.source_metadata,
                extracted_from_depth=raw_fact.extracted_from_depth,
                score=score,
                source_description=source_description
            )

        except Exception as e:
            self.logger.error(
                "Fallback evaluation also failed",
                fact=raw_fact.fact_text[:100],
                error=str(e)
            )
            # Return with default low score
            return ScoredFact(
                fact_text=raw_fact.fact_text,
                source_node_id=raw_fact.source_node_id,
                source_property=raw_fact.source_property,
                source_qdrant_chunk_id=raw_fact.source_qdrant_chunk_id,
                source_metadata=raw_fact.source_metadata,
                extracted_from_depth=raw_fact.extracted_from_depth,
                score=0.1,
                source_description=f"Document: {raw_fact.source_metadata.document_name or 'Unknown'}, Chunk: {raw_fact.source_metadata.chunk_index or 0}"
            )
