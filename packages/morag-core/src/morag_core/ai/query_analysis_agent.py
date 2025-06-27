"""PydanticAI agent for query analysis and intent detection."""

import asyncio
from typing import Type, List, Optional, Dict, Any
import structlog

from .base_agent import MoRAGBaseAgent
from .models import QueryAnalysisResult, ConfidenceLevel

logger = structlog.get_logger(__name__)


class QueryAnalysisAgent(MoRAGBaseAgent[QueryAnalysisResult]):
    """PydanticAI agent for analyzing user queries and detecting intent."""
    
    def __init__(self, **kwargs):
        """Initialize the query analysis agent.
        
        Args:
            **kwargs: Additional arguments passed to base agent
        """
        super().__init__(**kwargs)
        self.logger = logger.bind(agent="query_analysis")
    
    def get_result_type(self) -> Type[QueryAnalysisResult]:
        return QueryAnalysisResult
    
    def get_system_prompt(self) -> str:
        return """You are an expert query analysis agent. Your task is to analyze user queries and extract meaningful information about their intent, entities, and requirements.

For each query, provide:
1. intent: The primary intent or purpose of the query (search, question, request, command, etc.)
2. entities: Important entities mentioned in the query (people, places, concepts, products, etc.)
3. keywords: Key terms and phrases that are important for understanding the query
4. query_type: The type of query (factual, analytical, procedural, comparative, etc.)
5. complexity: The complexity level (simple, medium, complex)
6. confidence: Your confidence in the analysis (low, medium, high, very_high)

Query Intent Categories:
- SEARCH: Looking for specific information or documents
- QUESTION: Asking for explanations, definitions, or answers
- COMPARISON: Comparing different items, concepts, or options
- ANALYSIS: Requesting analysis, insights, or interpretation
- PROCEDURE: Asking for step-by-step instructions or how-to information
- RECOMMENDATION: Seeking suggestions or recommendations
- CLARIFICATION: Asking for clarification or more details
- SUMMARY: Requesting a summary or overview
- CREATION: Asking for content creation or generation
- TROUBLESHOOTING: Seeking help with problems or issues

Query Types:
- FACTUAL: Seeking specific facts or information
- ANALYTICAL: Requiring analysis or interpretation
- PROCEDURAL: Asking for processes or procedures
- COMPARATIVE: Comparing multiple items
- TEMPORAL: Related to time, dates, or sequences
- SPATIAL: Related to location or geography
- CAUSAL: About causes and effects
- HYPOTHETICAL: About possibilities or scenarios

Complexity Levels:
- SIMPLE: Single concept, direct question, clear intent
- MEDIUM: Multiple concepts, some ambiguity, moderate complexity
- COMPLEX: Multiple interrelated concepts, high ambiguity, requires deep analysis

Guidelines:
- Extract all relevant entities, including proper nouns, technical terms, and key concepts
- Identify the most important keywords that capture the query's essence
- Consider context and implied meaning, not just literal text
- Be precise about intent classification
- Assess complexity based on the number of concepts and required reasoning"""
    
    async def analyze_query(
        self,
        query: str,
        context: Optional[str] = None,
        user_history: Optional[List[str]] = None
    ) -> QueryAnalysisResult:
        """Analyze a user query to extract intent and entities.
        
        Args:
            query: The user query to analyze
            context: Optional context about the query session
            user_history: Optional list of previous queries from the user
            
        Returns:
            QueryAnalysisResult with analysis details
        """
        if not query or not query.strip():
            return QueryAnalysisResult(
                intent="unknown",
                entities=[],
                keywords=[],
                query_type="unknown",
                complexity="simple",
                confidence=ConfidenceLevel.LOW,
                metadata={"error": "Empty query"}
            )
        
        self.logger.info(
            "Starting query analysis",
            query_length=len(query),
            has_context=context is not None,
            has_history=user_history is not None and len(user_history) > 0
        )
        
        try:
            # Build the analysis prompt
            prompt = self._build_analysis_prompt(query, context, user_history)
            
            # Generate analysis using the agent
            result = await self.run(prompt)
            
            # Add metadata
            result.metadata.update({
                "original_query": query,
                "query_length": len(query),
                "word_count": len(query.split()),
                "has_context": context is not None,
                "history_length": len(user_history) if user_history else 0
            })
            
            self.logger.info(
                "Query analysis completed",
                intent=result.intent,
                entity_count=len(result.entities),
                keyword_count=len(result.keywords),
                query_type=result.query_type,
                complexity=result.complexity,
                confidence=result.confidence
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Query analysis failed", error=str(e), error_type=type(e).__name__)
            raise
    
    async def analyze_batch_queries(
        self,
        queries: List[str],
        context: Optional[str] = None
    ) -> List[QueryAnalysisResult]:
        """Analyze multiple queries in batch.
        
        Args:
            queries: List of queries to analyze
            context: Optional shared context for all queries
            
        Returns:
            List of QueryAnalysisResult objects
        """
        if not queries:
            return []
        
        self.logger.info(f"Starting batch query analysis for {len(queries)} queries")
        
        # Process queries concurrently with limited concurrency
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests
        
        async def analyze_single_query(query: str) -> QueryAnalysisResult:
            async with semaphore:
                try:
                    return await self.analyze_query(query, context=context)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze query: {query[:50]}...", error=str(e))
                    return QueryAnalysisResult(
                        intent="unknown",
                        entities=[],
                        keywords=[],
                        query_type="unknown",
                        complexity="simple",
                        confidence=ConfidenceLevel.LOW,
                        metadata={"error": str(e), "original_query": query}
                    )
        
        results = await asyncio.gather(
            *[analyze_single_query(query) for query in queries],
            return_exceptions=True
        )
        
        # Filter out exceptions and return valid results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Query {i} failed with exception: {result}")
                # Create a fallback result
                valid_results.append(QueryAnalysisResult(
                    intent="unknown",
                    entities=[],
                    keywords=[],
                    query_type="unknown",
                    complexity="simple",
                    confidence=ConfidenceLevel.LOW,
                    metadata={"error": str(result), "original_query": queries[i] if i < len(queries) else ""}
                ))
            else:
                valid_results.append(result)
        
        self.logger.info(f"Batch query analysis completed for {len(valid_results)} queries")
        return valid_results
    
    async def extract_search_terms(
        self,
        query: str,
        expand_terms: bool = True
    ) -> List[str]:
        """Extract optimized search terms from a query.
        
        Args:
            query: The user query
            expand_terms: Whether to expand terms with synonyms
            
        Returns:
            List of optimized search terms
        """
        analysis = await self.analyze_query(query)
        
        # Combine entities and keywords
        search_terms = list(set(analysis.entities + analysis.keywords))
        
        # Remove very short terms (less than 3 characters)
        search_terms = [term for term in search_terms if len(term) >= 3]
        
        # If expansion is requested, add the original query words
        if expand_terms:
            query_words = [word.strip().lower() for word in query.split() if len(word.strip()) >= 3]
            search_terms.extend(query_words)
            search_terms = list(set(search_terms))  # Remove duplicates
        
        return search_terms
    
    def _build_analysis_prompt(
        self,
        query: str,
        context: Optional[str],
        user_history: Optional[List[str]]
    ) -> str:
        """Build the query analysis prompt."""
        prompt_parts = [
            "Analyze the following user query and extract the requested information:",
            ""
        ]
        
        if context:
            prompt_parts.extend([
                "Context:",
                context,
                ""
            ])
        
        if user_history:
            prompt_parts.extend([
                "Previous queries from this user:",
                "\n".join(f"- {prev_query}" for prev_query in user_history[-5:]),  # Last 5 queries
                ""
            ])
        
        prompt_parts.extend([
            "Query to analyze:",
            query
        ])
        
        return "\n".join(prompt_parts)
