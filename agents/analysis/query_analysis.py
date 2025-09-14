"""Query analysis agent."""

from typing import Type, List, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig

from .models import QueryAnalysisResult

logger = structlog.get_logger(__name__)


class QueryAnalysisAgent(BaseAgent[QueryAnalysisResult]):
    """Agent specialized for analyzing user queries and detecting intent."""
    
    def _get_default_config(self) -> AgentConfig:
        """Get default configuration for query analysis."""
        return AgentConfig(
            name="query_analysis",
            description="Analyzes user queries to extract intent, entities, and requirements",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                include_context=True,
                output_format="json",
                strict_json=True,
                include_confidence=True,
                min_confidence=0.7,
            ),
            agent_config={
                "extract_entities": True,
                "extract_keywords": True,
                "analyze_complexity": True,
                "detect_temporal_context": True,
            }
        )
    

    
    def get_result_type(self) -> Type[QueryAnalysisResult]:
        """Get the result type for query analysis."""
        return QueryAnalysisResult
    
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
                intent="search",
                entities=[],
                keywords=[],
                query_type="factual",
                complexity="simple",
                confidence="low",
                metadata={"error": "Empty query"}
            )
        
        self.logger.info(
            "Starting query analysis",
            query_length=len(query),
            has_context=context is not None,
            has_history=user_history is not None and len(user_history) > 0
        )
        
        try:
            result = await self.execute(
                query,
                context=context,
                user_history=user_history
            )
            
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
            self.logger.error("Query analysis failed", error=str(e))
            raise
    
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
