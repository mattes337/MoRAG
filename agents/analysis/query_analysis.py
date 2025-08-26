"""Query analysis agent."""

from typing import Type, List, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate, PromptExample
from .models import QueryAnalysisResult, QueryIntent, QueryType, ComplexityLevel, ConfidenceLevel

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
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        """Create the prompt template for query analysis."""
        
        system_prompt = """You are an expert query analysis agent. Your task is to analyze user queries and extract meaningful information about their intent, entities, and requirements.

## Your Role
Analyze queries to determine:
- **Intent**: Primary purpose (search, question, comparison, analysis, etc.)
- **Entities**: Important entities mentioned (people, places, concepts, products, etc.)
- **Keywords**: Key terms and phrases for understanding
- **Query Type**: Nature of the query (factual, analytical, procedural, etc.)
- **Complexity**: Complexity level (simple, medium, complex)

## Intent Categories
- **SEARCH**: Looking for specific information or documents
- **QUESTION**: Asking for explanations, definitions, or answers
- **COMPARISON**: Comparing different items, concepts, or options
- **ANALYSIS**: Requesting analysis, insights, or interpretation
- **PROCEDURE**: Asking for step-by-step instructions or how-to information
- **RECOMMENDATION**: Seeking suggestions or recommendations
- **CLARIFICATION**: Asking for clarification or more details
- **SUMMARY**: Requesting a summary or overview
- **CREATION**: Asking for content creation or generation
- **TROUBLESHOOTING**: Seeking help with problems or issues

## Query Types
- **FACTUAL**: Seeking specific facts or information
- **ANALYTICAL**: Requiring analysis or interpretation
- **PROCEDURAL**: Asking for processes or procedures
- **COMPARATIVE**: Comparing multiple items
- **TEMPORAL**: Related to time, dates, or sequences
- **SPATIAL**: Related to location or geography
- **CAUSAL**: About causes and effects
- **HYPOTHETICAL**: About possibilities or scenarios

## Complexity Assessment
- **SIMPLE**: Single concept, direct question, clear intent
- **MEDIUM**: Multiple concepts, some ambiguity, moderate complexity
- **COMPLEX**: Multiple interrelated concepts, high ambiguity, requires deep analysis

## Analysis Guidelines
- Extract all relevant entities (proper nouns, technical terms, key concepts)
- Identify the most important keywords that capture the query's essence
- Consider context and implied meaning, not just literal text
- Be precise about intent classification
- Assess complexity based on number of concepts and required reasoning

{% if config.include_examples %}
{{ examples }}
{% endif %}

{{ output_requirements }}"""

        user_prompt = """Analyze the following user query comprehensively:

Query: "{{ input }}"

{% if context %}
Context: {{ context }}
{% endif %}

{% if user_history %}
Previous queries from this user:
{% for prev_query in user_history[-3:] %}
- {{ prev_query }}
{% endfor %}
{% endif %}

Return a JSON object with the following structure:
{
  "intent": "search|question|comparison|analysis|procedure|recommendation|clarification|summary|creation|troubleshooting",
  "entities": ["entity1", "entity2"],
  "keywords": ["keyword1", "keyword2"],
  "query_type": "factual|analytical|procedural|comparative|temporal|spatial|causal|hypothetical",
  "complexity": "simple|medium|complex",
  "confidence": "low|medium|high|very_high",
  "metadata": {
    "original_query": "{{ input }}",
    "query_length": {{ input|length }},
    "word_count": {{ input.split()|length }},
    "has_context": {% if context %}true{% else %}false{% endif %},
    "analysis_method": "llm"
  }
}"""

        template = ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
        
        # Add examples
        examples = [
            PromptExample(
                input="How does machine learning compare to traditional programming approaches in terms of performance and maintainability?",
                output="""{
  "intent": "comparison",
  "entities": ["machine learning", "traditional programming"],
  "keywords": ["machine learning", "traditional programming", "performance", "maintainability", "approaches", "compare"],
  "query_type": "comparative",
  "complexity": "medium",
  "confidence": "high",
  "metadata": {
    "original_query": "How does machine learning compare to traditional programming approaches in terms of performance and maintainability?",
    "query_length": 118,
    "word_count": 16,
    "has_context": false,
    "analysis_method": "llm"
  }
}""",
                explanation="This is a comparative query asking about two different approaches with specific evaluation criteria."
            ),
            PromptExample(
                input="What is Python?",
                output="""{
  "intent": "question",
  "entities": ["Python"],
  "keywords": ["Python"],
  "query_type": "factual",
  "complexity": "simple",
  "confidence": "very_high",
  "metadata": {
    "original_query": "What is Python?",
    "query_length": 15,
    "word_count": 3,
    "has_context": false,
    "analysis_method": "llm"
  }
}""",
                explanation="Simple factual question asking for a definition with clear intent and single concept."
            )
        ]
        
        template.get_examples = lambda: examples
        return template
    
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
                intent=QueryIntent.SEARCH,
                entities=[],
                keywords=[],
                query_type=QueryType.FACTUAL,
                complexity=ComplexityLevel.SIMPLE,
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
