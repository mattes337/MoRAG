"""Key fact extraction service for intelligent retrieval."""

from typing import Any, Dict, List, Optional

import structlog
from morag_reasoning.intelligent_retrieval_models import (
    FactExtractionRequest,
    KeyFact,
    SourceInfo,
)
from morag_reasoning.llm import LLMClient
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class ExtractedFacts(BaseModel):
    """Result of fact extraction from chunks."""

    facts: List[KeyFact] = Field(..., description="Extracted key facts")


class FactSourceMapping(BaseModel):
    """Mapping of facts to their supporting chunks."""

    fact_index: int = Field(..., description="Index of the fact in the facts list")
    supporting_chunk_indices: List[int] = Field(
        ..., description="Indices of chunks that support this fact"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the source mapping"
    )
    reasoning: str = Field(
        ..., description="Reasoning for why these chunks support the fact"
    )


class SourceMappingResult(BaseModel):
    """Result of source mapping analysis."""

    mappings: List[FactSourceMapping] = Field(
        ..., description="Fact to source mappings"
    )


class FactExtractionService:
    """Service for extracting key facts from retrieved chunks."""

    def __init__(
        self, llm_client: LLMClient, min_confidence: float = 0.5, max_facts: int = 10000
    ):
        """Initialize the fact extraction service.

        Args:
            llm_client: LLM client for fact extraction
            min_confidence: Minimum confidence threshold
            max_facts: Maximum facts to extract
        """
        self.llm_client = llm_client
        self.min_confidence = min_confidence
        self.max_facts = max_facts
        self.logger = structlog.get_logger(__name__)

        # Create PydanticAI agent for fact extraction
        self.agent = Agent(
            model=llm_client.get_model(),
            result_type=ExtractedFacts,
            system_prompt=self._get_system_prompt(),
        )

        # Create PydanticAI agent for source mapping
        self.source_mapping_agent = Agent(
            model=llm_client.get_model(),
            result_type=SourceMappingResult,
            system_prompt=self._get_source_mapping_prompt(),
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for fact extraction."""
        return """You are an expert fact extraction system. Your task is to extract key facts from document chunks that are relevant to answering a user's query.

Guidelines:
1. Focus on facts that directly help answer the user's query
2. Extract specific, actionable information rather than general statements
3. Assign appropriate fact types (definition, relationship, statistic, process, etc.)
4. Provide confidence scores based on how well-supported the fact is in the text
5. Rate relevance to the query (how much this fact helps answer the question)
6. Identify supporting entities mentioned in the fact
7. Ensure facts are self-contained and understandable without additional context

Fact Types to Consider:
- definition: Explanations of what something is
- relationship: How entities relate to each other
- statistic: Numerical data, measurements, quantities
- process: Step-by-step procedures or workflows
- characteristic: Properties or attributes of entities
- comparison: How things differ or are similar
- causation: Cause and effect relationships
- temporal: Time-based information, sequences, history
- location: Geographic or spatial information
- requirement: Prerequisites, conditions, constraints

Quality Criteria:
- Accuracy: Fact is well-supported by the source text
- Relevance: Fact helps answer the user's query
- Specificity: Fact provides concrete, actionable information
- Completeness: Fact is self-contained and clear

Return only the most relevant and well-supported facts."""

    async def extract_facts(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_facts: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> List[KeyFact]:
        """Extract key facts from retrieved chunks.

        Args:
            query: Original user query
            chunks: Retrieved chunks with metadata
            max_facts: Maximum facts to extract (overrides instance default)
            min_confidence: Minimum confidence threshold (overrides instance default)

        Returns:
            List of extracted key facts
        """
        if not chunks:
            return []

        max_facts_limit = max_facts or self.max_facts
        confidence_threshold = min_confidence or self.min_confidence

        self.logger.info(
            "Starting fact extraction",
            query=query,
            num_chunks=len(chunks),
            max_facts=max_facts_limit,
        )

        try:
            # Prepare chunks for processing
            chunk_texts = []
            chunk_metadata = []

            for i, chunk in enumerate(chunks):
                # Handle both string chunks and dict chunks
                if isinstance(chunk, str):
                    chunk_text = chunk
                    # Create minimal metadata for string chunks
                    chunk_dict = {
                        "text": chunk,
                        "content": chunk,
                        "chunk_id": f"chunk_{i}",
                        "document_name": f"Document {i+1}",
                    }
                else:
                    chunk_text = chunk.get("text", chunk.get("content", ""))
                    chunk_dict = chunk

                if chunk_text:
                    chunk_texts.append(f"[Chunk {i+1}] {chunk_text}")
                    chunk_metadata.append(chunk_dict)

            if not chunk_texts:
                self.logger.warning("No valid chunk texts found")
                return []

            # Create extraction prompt
            prompt = self._create_extraction_prompt(query, chunk_texts)

            # Extract facts using LLM
            result = await self.agent.run(prompt)
            extracted_facts = result.data.facts

            # Process and enhance facts with source information
            enhanced_facts = []
            for fact in extracted_facts:
                if fact.confidence >= confidence_threshold:
                    enhanced_facts.append(fact)

            # Map facts to their supporting sources using LLM
            if enhanced_facts:
                enhanced_facts = await self._map_facts_to_sources(
                    enhanced_facts, chunk_metadata, query
                )

            # Sort by relevance and confidence, then limit
            enhanced_facts.sort(
                key=lambda f: (f.relevance_to_query, f.confidence), reverse=True
            )
            enhanced_facts = enhanced_facts[:max_facts_limit]

            self.logger.info(
                "Fact extraction completed",
                total_facts=len(enhanced_facts),
                query=query,
            )

            return enhanced_facts

        except Exception as e:
            self.logger.error(
                "Fact extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                query=query,
            )
            raise

    def _create_extraction_prompt(self, query: str, chunk_texts: List[str]) -> str:
        """Create prompt for fact extraction.

        Args:
            query: Original user query
            chunk_texts: List of chunk texts with identifiers

        Returns:
            Extraction prompt for LLM
        """
        chunks_text = "\n\n".join(chunk_texts)

        return f"""Extract key facts from the following document chunks that help answer the user's query.

USER QUERY: "{query}"

DOCUMENT CHUNKS:
{chunks_text}

TASK:
Extract the most relevant and well-supported facts that help answer the user's query. For each fact:
1. Provide the specific fact statement
2. Assign a confidence score (0.0-1.0) based on how well-supported it is
3. Rate relevance to the query (0.0-1.0)
4. Classify the fact type (definition, relationship, statistic, etc.)
5. List supporting entities mentioned in the fact

Focus on facts that are:
- Directly relevant to answering the query
- Well-supported by the source text
- Specific and actionable
- Self-contained and clear

Limit to the most important facts (typically 5-15 facts)."""

    def _get_source_mapping_prompt(self) -> str:
        """Get the system prompt for source mapping."""
        return """You are an expert at analyzing which text chunks support specific facts.

Your task is to determine which chunks from the provided text support each extracted fact.

For each fact, analyze:
1. Which chunks contain information that directly supports or validates the fact
2. Which chunks provide context or background that strengthens the fact
3. The confidence level of the mapping (how certain you are that the chunk supports the fact)

Guidelines:
- A chunk supports a fact if it contains direct evidence, data, or statements that validate the fact
- A chunk may partially support a fact if it provides relevant context or background
- Be conservative - only map chunks that genuinely support the fact
- Provide clear reasoning for each mapping
- A fact may be supported by multiple chunks
- Some facts may not be supported by any specific chunk (if they are inferred or synthesized)

Return mappings with high confidence only for clear, direct support."""

    async def _map_facts_to_sources(
        self, facts: List[KeyFact], chunk_metadata: List[Dict[str, Any]], query: str
    ) -> List[KeyFact]:
        """Map facts to their supporting source chunks using LLM analysis.

        Args:
            facts: List of extracted facts
            chunk_metadata: Metadata for all chunks
            query: Original user query

        Returns:
            List of facts with proper source mappings
        """
        if not facts or not chunk_metadata:
            return facts

        try:
            # Create prompt for source mapping
            facts_text = "\n".join(
                [
                    f"Fact {i}: {fact.fact} (Type: {fact.fact_type}, Confidence: {fact.confidence:.2f})"
                    for i, fact in enumerate(facts)
                ]
            )

            chunks_text = "\n".join(
                [
                    f"Chunk {i}: {chunk.get('text', chunk.get('content', ''))[:500]}..."
                    for i, chunk in enumerate(chunk_metadata)
                ]
            )

            prompt = f"""Original Query: {query}

EXTRACTED FACTS:
{facts_text}

AVAILABLE CHUNKS:
{chunks_text}

Analyze which chunks support each fact. For each fact, identify the chunk indices that provide evidence or support for that fact."""

            # Get source mappings from LLM
            result = await self.source_mapping_agent.run(prompt)
            mappings = result.data.mappings

            # Apply mappings to facts
            for mapping in mappings:
                if 0 <= mapping.fact_index < len(facts):
                    fact = facts[mapping.fact_index]
                    supporting_sources = []

                    for chunk_idx in mapping.supporting_chunk_indices:
                        if 0 <= chunk_idx < len(chunk_metadata):
                            chunk = chunk_metadata[chunk_idx]
                            source = SourceInfo(
                                document_id=chunk.get(
                                    "document_id", f"doc_{chunk_idx}"
                                ),
                                chunk_id=chunk.get(
                                    "chunk_id", chunk.get("id", f"chunk_{chunk_idx}")
                                ),
                                document_name=chunk.get(
                                    "document_name",
                                    chunk.get("source", f"Document {chunk_idx+1}"),
                                ),
                                chunk_text=chunk.get("text", chunk.get("content", "")),
                                relevance_score=chunk.get(
                                    "score", chunk.get("relevance_score", 0.5)
                                ),
                                page_number=chunk.get("page_number"),
                                section=chunk.get("section"),
                                metadata=chunk.get("metadata", {}),
                            )
                            supporting_sources.append(source)

                    fact.sources = supporting_sources

            # For facts without mapped sources, provide fallback
            for fact in facts:
                if not hasattr(fact, "sources") or not fact.sources:
                    fact.sources = []

            return facts

        except Exception as e:
            self.logger.warning(
                "Source mapping failed, falling back to basic source info", error=str(e)
            )
            # Fallback to original method
            for fact in facts:
                fact.sources = self._create_source_info(fact, chunk_metadata)
            return facts

    def _create_source_info(
        self, fact: KeyFact, chunk_metadata: List[Dict[str, Any]]
    ) -> List[SourceInfo]:
        """Create source information for a fact.

        Args:
            fact: Extracted fact
            chunk_metadata: Metadata for all chunks

        Returns:
            List of source information
        """
        sources = []

        # For now, we'll associate the fact with all chunks
        # In a more sophisticated implementation, we could determine
        # which specific chunks support each fact
        for i, chunk in enumerate(chunk_metadata):
            source = SourceInfo(
                document_id=chunk.get("document_id", f"doc_{i}"),
                chunk_id=chunk.get("chunk_id", chunk.get("id", f"chunk_{i}")),
                document_name=chunk.get(
                    "document_name", chunk.get("source", f"Document {i+1}")
                ),
                chunk_text=chunk.get("content", chunk.get("text", "")),
                relevance_score=chunk.get("score", chunk.get("relevance_score", 0.5)),
                page_number=chunk.get("page_number"),
                section=chunk.get("section"),
                metadata=chunk.get("metadata", {}),
            )
            sources.append(source)

        return sources

    async def extract_facts_from_request(
        self, request: FactExtractionRequest
    ) -> List[KeyFact]:
        """Extract facts from a fact extraction request.

        Args:
            request: Fact extraction request

        Returns:
            List of extracted key facts
        """
        return await self.extract_facts(
            query=request.query,
            chunks=request.chunks,
            max_facts=request.max_facts,
            min_confidence=request.min_confidence,
        )
