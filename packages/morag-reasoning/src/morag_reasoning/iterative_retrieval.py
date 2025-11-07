"""Iterative context refinement for multi-hop reasoning."""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from morag_graph.operations import GraphPath
from .llm import LLMClient

# Import agents framework
from agents import get_agent

logger = logging.getLogger(__name__)


@dataclass
class ContextGap:
    """Represents a gap in the current context."""
    gap_type: str  # "missing_entity", "missing_relation", "insufficient_detail"
    description: str
    entities_needed: List[str] = field(default_factory=list)
    relations_needed: List[str] = field(default_factory=list)
    priority: float = 1.0


@dataclass
class ContextAnalysis:
    """Analysis of context sufficiency."""
    is_sufficient: bool
    confidence: float
    gaps: List[ContextGap]
    reasoning: str
    suggested_queries: List[str] = field(default_factory=list)


@dataclass
class RetrievalContext:
    """Context for retrieval operations."""
    entities: Dict[str, Any] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    documents: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[GraphPath] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IterativeRetriever:
    """Iterative context refinement system."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        graph_engine=None,
        vector_retriever=None,
        max_iterations: int = 5,
        sufficiency_threshold: float = 0.8
    ):
        """Initialize the iterative retriever.

        Args:
            llm_client: LLM client for context analysis (deprecated, kept for compatibility)
            graph_engine: Graph engine for additional retrieval
            vector_retriever: Vector retriever for document search
            max_iterations: Maximum refinement iterations
            sufficiency_threshold: Confidence threshold for stopping
        """
        # ALWAYS use agents framework
        self.reasoning_agent = get_agent("reasoning")
        self.context_agent = get_agent("context_analysis")

        self.llm_client = llm_client  # Keep for backward compatibility
        self.graph_engine = graph_engine
        self.vector_retriever = vector_retriever
        self.max_iterations = max_iterations
        self.sufficiency_threshold = sufficiency_threshold
        self.logger = logging.getLogger(__name__)

    async def refine_context(
        self,
        query: str,
        initial_context: RetrievalContext
    ) -> RetrievalContext:
        """Iteratively refine context until sufficient for answering the query.

        Args:
            query: Query to answer
            initial_context: Initial retrieval context

        Returns:
            Refined context with additional information
        """
        current_context = initial_context
        iteration_count = 0

        self.logger.info(f"Starting iterative context refinement for query: {query}")

        while iteration_count < self.max_iterations:
            iteration_count += 1
            self.logger.info(f"Iteration {iteration_count}/{self.max_iterations}")

            # Analyze current context
            analysis = await self._analyze_context(query, current_context)

            self.logger.info(
                f"Context analysis - Sufficient: {analysis.is_sufficient}, "
                f"Confidence: {analysis.confidence:.2f}, Gaps: {len(analysis.gaps)}"
            )

            # Check if context is sufficient
            if analysis.is_sufficient and analysis.confidence >= self.sufficiency_threshold:
                self.logger.info("Context deemed sufficient, stopping refinement")
                break

            # Retrieve additional information based on gaps
            additional_context = await self._retrieve_additional(
                query, analysis.gaps, current_context
            )

            # Merge contexts
            current_context = self._merge_context(current_context, additional_context)

            # Log progress
            self.logger.info(
                f"Added {len(additional_context.entities)} entities, "
                f"{len(additional_context.relations)} relations, "
                f"{len(additional_context.documents)} documents"
            )

        # Final analysis
        final_analysis = await self._analyze_context(query, current_context)
        current_context.metadata['final_analysis'] = final_analysis
        current_context.metadata['iterations_used'] = iteration_count

        self.logger.info(
            f"Context refinement completed after {iteration_count} iterations. "
            f"Final confidence: {final_analysis.confidence:.2f}"
        )

        return current_context

    async def _analyze_context(self, query: str, context: RetrievalContext) -> ContextAnalysis:
        """Analyze current context to determine if it's sufficient."""
        try:
            # Use context analysis agent - ALWAYS
            context_summary = self._format_context_for_agent(context)

            result = await self.context_agent.execute(
                f"Query: {query}\n\nContext: {context_summary}",
                query=query,
                context_data=context
            )

            # Convert agent result to ContextAnalysis
            analysis = ContextAnalysis(
                is_sufficient=result.relevance_scores.get("sufficient", 0.0) > 0.7,
                confidence=result.confidence.value if hasattr(result.confidence, 'value') else 0.8,
                gaps=result.context_gaps,
                reasoning=result.context_summary
            )
            return analysis

        except Exception as e:
            self.logger.error(f"Error in context analysis: {str(e)}")
            # Fallback analysis
            return ContextAnalysis(
                is_sufficient=len(context.entities) > 0 and len(context.documents) > 0,
                confidence=0.5,
                gaps=[],
                reasoning="Fallback analysis due to agent error"
            )

    def _format_context_for_agent(self, context: RetrievalContext) -> str:
        """Format context for agent analysis."""
        context_parts = []

        if context.entities:
            entities_str = ", ".join([f"{name} ({info.get('type', 'unknown')})"
                                    for name, info in context.entities.items()])
            context_parts.append(f"Entities: {entities_str}")

        if context.documents:
            docs_str = f"{len(context.documents)} documents available"
            context_parts.append(f"Documents: {docs_str}")

        if context.paths:
            paths_str = f"{len(context.paths)} reasoning paths"
            context_parts.append(f"Paths: {paths_str}")

        return "; ".join(context_parts) if context_parts else "No context available"

    def _create_analysis_prompt(self, query: str, context: RetrievalContext) -> str:
        """Create prompt for context analysis."""
        prompt = f"""Analyze whether the provided context is sufficient to answer the given query.

Query: {query}

Current Context:

Entities ({len(context.entities)}):
"""

        # Add entity information
        for entity_id, entity_data in list(context.entities.items())[:10]:  # Limit for token efficiency
            entity_type = entity_data.get('type', 'Unknown') if isinstance(entity_data, dict) else 'Unknown'
            prompt += f"- {entity_id}: {entity_type}\n"

        prompt += f"\nRelations ({len(context.relations)}):\n"

        # Add relation information
        for relation in context.relations[:10]:
            subject = relation.get('subject', '?')
            predicate = relation.get('predicate', '?')
            obj = relation.get('object', '?')
            prompt += f"- {subject} --[{predicate}]--> {obj}\n"

        prompt += f"\nDocuments ({len(context.documents)}):\n"

        # Add document information
        for doc in context.documents[:5]:
            content = doc.get('content', '')
            content_preview = content[:200] + "..." if len(content) > 200 else content
            doc_id = doc.get('id', 'Unknown')
            prompt += f"- {doc_id}: {content_preview}\n"

        prompt += """
Analyze this context and provide:
1. Is the context sufficient to answer the query? (true/false)
2. Confidence level (0-10)
3. What gaps exist in the context?
4. Suggested additional queries to fill gaps

Format as JSON:
{
  "is_sufficient": false,
  "confidence": 6.5,
  "reasoning": "Context provides basic information but lacks specific details about...",
  "gaps": [
    {
      "gap_type": "missing_entity",
      "description": "Need more information about entity X",
      "entities_needed": ["entity_name"],
      "priority": 0.8
    }
  ],
  "suggested_queries": ["What is the relationship between X and Y?"]
}"""

        return prompt

    def _parse_context_analysis(self, response: str) -> ContextAnalysis:
        """Parse LLM response into ContextAnalysis object."""
        try:
            # Log the raw response for debugging
            self.logger.debug(f"Raw LLM response: {response}")

            # Try to extract JSON from response if it's wrapped in markdown or other text
            response_clean = response.strip()

            # Look for JSON block in markdown
            if "```json" in response_clean:
                start = response_clean.find("```json") + 7
                end = response_clean.find("```", start)
                if end != -1:
                    response_clean = response_clean[start:end].strip()
            elif "```" in response_clean:
                start = response_clean.find("```") + 3
                end = response_clean.find("```", start)
                if end != -1:
                    response_clean = response_clean[start:end].strip()

            # Try to find JSON object boundaries
            if not response_clean.startswith("{"):
                start = response_clean.find("{")
                if start != -1:
                    response_clean = response_clean[start:]

            if not response_clean.endswith("}"):
                end = response_clean.rfind("}")
                if end != -1:
                    response_clean = response_clean[:end+1]

            self.logger.debug(f"Cleaned response: {response_clean}")

            data = json.loads(response_clean)

            gaps = []
            for gap_data in data.get("gaps", []):
                gap = ContextGap(
                    gap_type=gap_data.get("gap_type", "unknown"),
                    description=gap_data.get("description", ""),
                    entities_needed=gap_data.get("entities_needed", []),
                    relations_needed=gap_data.get("relations_needed", []),
                    priority=float(gap_data.get("priority", 1.0))
                )
                gaps.append(gap)

            return ContextAnalysis(
                is_sufficient=bool(data.get("is_sufficient", False)),
                confidence=float(data.get("confidence", 0)) / 10.0,  # Normalize to 0-1
                gaps=gaps,
                reasoning=data.get("reasoning", ""),
                suggested_queries=data.get("suggested_queries", [])
            )

        except Exception as e:
            self.logger.error(f"Error parsing context analysis: {str(e)}")
            self.logger.error(f"Raw response was: {response}")
            print(f"Error parsing context analysis: {str(e)}")
            return ContextAnalysis(
                is_sufficient=False,
                confidence=0.3,
                gaps=[],
                reasoning="Failed to parse LLM analysis"
            )

    async def _retrieve_additional(
        self,
        query: str,
        gaps: List[ContextGap],
        current_context: RetrievalContext
    ) -> RetrievalContext:
        """Retrieve additional information to fill context gaps."""
        additional_context = RetrievalContext()

        # Sort gaps by priority
        sorted_gaps = sorted(gaps, key=lambda g: g.priority, reverse=True)

        for gap in sorted_gaps[:3]:  # Process top 3 gaps to avoid overwhelming
            try:
                if gap.gap_type == "missing_entity":
                    # Retrieve entity information
                    for entity_name in gap.entities_needed:
                        if hasattr(self.graph_engine, 'get_entity_details'):
                            entity_info = await self.graph_engine.get_entity_details(entity_name)
                            if entity_info:
                                additional_context.entities[entity_name] = entity_info
                        elif hasattr(self.graph_engine, 'get_entity'):
                            entity_info = await self.graph_engine.get_entity(entity_name)
                            if entity_info:
                                additional_context.entities[entity_name] = entity_info.to_dict()

                elif gap.gap_type == "missing_relation":
                    # Retrieve relation information
                    for relation_name in gap.relations_needed:
                        if hasattr(self.graph_engine, 'get_relations_by_type'):
                            relations = await self.graph_engine.get_relations_by_type(relation_name)
                            additional_context.relations.extend(relations)

                elif gap.gap_type in ["insufficient_detail", "missing_information"]:
                    # Perform additional vector search
                    search_query = f"{query} {gap.description}"
                    if hasattr(self.vector_retriever, 'search'):
                        vector_results = await self.vector_retriever.search(
                            search_query, limit=5
                        )
                        additional_context.documents.extend(vector_results)
                    elif hasattr(self.vector_retriever, 'retrieve'):
                        vector_results = await self.vector_retriever.retrieve(
                            search_query, max_results=5
                        )
                        additional_context.documents.extend(vector_results)

            except Exception as e:
                self.logger.error(f"Error retrieving additional info for gap {gap.gap_type}: {str(e)}")
                continue

        return additional_context

    def _merge_context(
        self,
        current_context: RetrievalContext,
        additional_context: RetrievalContext
    ) -> RetrievalContext:
        """Merge additional context into current context."""
        # Merge entities (avoid duplicates)
        for entity_id, entity_data in additional_context.entities.items():
            if entity_id not in current_context.entities:
                current_context.entities[entity_id] = entity_data

        # Merge relations (avoid duplicates)
        existing_relations = set(
            (r.get('subject'), r.get('predicate'), r.get('object'))
            for r in current_context.relations
        )

        for relation in additional_context.relations:
            relation_tuple = (relation.get('subject'), relation.get('predicate'), relation.get('object'))
            if relation_tuple not in existing_relations:
                current_context.relations.append(relation)
                existing_relations.add(relation_tuple)

        # Merge documents (avoid duplicates)
        existing_doc_ids = set(doc.get('id') for doc in current_context.documents)
        for doc in additional_context.documents:
            doc_id = doc.get('id')
            if doc_id not in existing_doc_ids:
                current_context.documents.append(doc)
                existing_doc_ids.add(doc_id)

        # Merge paths
        current_context.paths.extend(additional_context.paths)

        # Update metadata
        current_context.metadata.update(additional_context.metadata)

        return current_context
