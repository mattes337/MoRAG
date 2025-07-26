"""Enhanced fact relevance scoring with multi-dimensional analysis."""

import asyncio
import time
from typing import List, Dict, Any, Optional, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from .llm import LLMClient
from .graph_fact_extractor import ExtractedFact, FactType

logger = structlog.get_logger(__name__)

# Optional imports for enhanced functionality
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class ScoringDimensions:
    """Multi-dimensional scoring components."""
    query_relevance: float  # Semantic similarity to user query
    source_quality: float  # Document authority and reliability
    confidence: float  # Extraction and validation confidence
    recency: float  # Temporal relevance of information
    completeness: float  # Availability of supporting evidence
    specificity: float  # How specific the fact is to the query


@dataclass
class ScoredFact:
    """Fact with comprehensive scoring information."""
    fact: ExtractedFact
    overall_score: float
    scoring_dimensions: ScoringDimensions
    reasoning: str
    metadata: Dict[str, Any]


class ScoringStrategy(Enum):
    """Strategies for fact scoring."""
    BALANCED = "balanced"  # Equal weight to all dimensions
    RELEVANCE_FOCUSED = "relevance_focused"  # Prioritize query relevance
    QUALITY_FOCUSED = "quality_focused"  # Prioritize source quality
    CONFIDENCE_FOCUSED = "confidence_focused"  # Prioritize extraction confidence
    ADAPTIVE = "adaptive"  # Adapt based on query characteristics


class FactRelevanceScorer:
    """Enhanced fact relevance scoring with multi-dimensional analysis."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the fact relevance scorer.
        
        Args:
            llm_client: LLM client for relevance assessment
            config: Optional configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.settings = get_settings()
        
        # Scoring parameters
        self.scoring_strategy = ScoringStrategy(
            self.config.get('scoring_strategy', 'adaptive')
        )
        self.min_score_threshold = self.config.get('min_score_threshold', 0.2)
        self.max_facts_to_score = self.config.get('max_facts_to_score', 100)
        
        # LLM configuration
        self.llm_enabled = self.config.get('llm_enabled', True) and GEMINI_AVAILABLE
        self.model_name = self.config.get('model_name', 'gemini-1.5-flash')
        
        # Semantic similarity configuration
        self.semantic_enabled = (
            self.config.get('semantic_enabled', True) and 
            SENTENCE_TRANSFORMERS_AVAILABLE
        )
        self.embedding_model_name = self.config.get(
            'embedding_model_name', 
            'all-MiniLM-L6-v2'
        )
        
        # Scoring weights (can be adjusted based on strategy)
        self.default_weights = {
            'query_relevance': 0.3,
            'source_quality': 0.2,
            'confidence': 0.2,
            'recency': 0.1,
            'completeness': 0.1,
            'specificity': 0.1
        }
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 10)
        self.enable_caching = self.config.get('enable_caching', True)
        
        # Initialize components
        self._llm_client = None
        self._embedding_model = None
        self._cache = {} if self.enable_caching else None
        
        logger.info(
            "Fact relevance scorer initialized",
            scoring_strategy=self.scoring_strategy.value,
            llm_enabled=self.llm_enabled,
            semantic_enabled=self.semantic_enabled,
            min_score_threshold=self.min_score_threshold
        )
    
    async def initialize(self) -> None:
        """Initialize LLM and embedding models."""
        # Initialize LLM
        if self.llm_enabled and not self._llm_client:
            try:
                genai.configure(api_key=self.settings.gemini_api_key)
                self._llm_client = genai.GenerativeModel(self.model_name)
                logger.info("LLM client initialized for fact scoring")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
                self.llm_enabled = False
        
        # Initialize embedding model
        if self.semantic_enabled and not self._embedding_model:
            try:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model initialized for fact scoring")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding model: {e}")
                self.semantic_enabled = False
    
    async def score_facts(
        self,
        facts: List[ExtractedFact],
        query: str,
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[ScoredFact]:
        """Score facts based on relevance to the query.
        
        Args:
            facts: List of extracted facts to score
            query: User query
            query_context: Optional query context information
            
        Returns:
            List of facts with relevance scores and explanations
        """
        if not facts:
            return []
        
        try:
            logger.info(
                "Starting fact relevance scoring",
                num_facts=len(facts),
                query=query,
                strategy=self.scoring_strategy.value
            )
            
            # Initialize models if needed
            await self.initialize()
            
            # Limit facts to score
            facts_to_score = facts[:self.max_facts_to_score]
            
            # Determine scoring weights based on strategy
            weights = self._get_scoring_weights(query, query_context)
            
            # Score facts in batches
            scored_facts = []
            for i in range(0, len(facts_to_score), self.batch_size):
                batch = facts_to_score[i:i + self.batch_size]
                batch_scores = await self._score_fact_batch(batch, query, weights)
                scored_facts.extend(batch_scores)
            
            # Filter by minimum score threshold
            filtered_facts = [
                f for f in scored_facts 
                if f.overall_score >= self.min_score_threshold
            ]
            
            # Sort by overall score
            filtered_facts.sort(key=lambda x: x.overall_score, reverse=True)
            
            logger.info(
                "Fact relevance scoring completed",
                total_facts=len(facts),
                scored_facts=len(scored_facts),
                filtered_facts=len(filtered_facts),
                strategy=self.scoring_strategy.value
            )
            
            return filtered_facts
            
        except Exception as e:
            logger.error(f"Fact relevance scoring failed: {e}")
            # Return facts with default scores as fallback
            return [
                ScoredFact(
                    fact=fact,
                    overall_score=0.5,
                    scoring_dimensions=ScoringDimensions(
                        query_relevance=0.5, source_quality=0.5, confidence=0.5,
                        recency=0.5, completeness=0.5, specificity=0.5
                    ),
                    reasoning="Fallback scoring due to error",
                    metadata={'scoring_error': str(e)}
                )
                for fact in facts[:10]  # Limit fallback results
            ]
    
    def _get_scoring_weights(
        self,
        query: str,
        query_context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Get scoring weights based on strategy and query characteristics."""
        if self.scoring_strategy == ScoringStrategy.RELEVANCE_FOCUSED:
            return {
                'query_relevance': 0.5,
                'source_quality': 0.15,
                'confidence': 0.15,
                'recency': 0.1,
                'completeness': 0.05,
                'specificity': 0.05
            }
        elif self.scoring_strategy == ScoringStrategy.QUALITY_FOCUSED:
            return {
                'query_relevance': 0.25,
                'source_quality': 0.4,
                'confidence': 0.2,
                'recency': 0.05,
                'completeness': 0.05,
                'specificity': 0.05
            }
        elif self.scoring_strategy == ScoringStrategy.CONFIDENCE_FOCUSED:
            return {
                'query_relevance': 0.25,
                'source_quality': 0.15,
                'confidence': 0.4,
                'recency': 0.05,
                'completeness': 0.1,
                'specificity': 0.05
            }
        elif self.scoring_strategy == ScoringStrategy.ADAPTIVE:
            # Adapt based on query characteristics
            return self._adapt_weights_to_query(query, query_context)
        else:  # BALANCED
            return self.default_weights.copy()
    
    def _adapt_weights_to_query(
        self,
        query: str,
        query_context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Adapt scoring weights based on query characteristics."""
        weights = self.default_weights.copy()
        
        query_lower = query.lower()
        
        # For factual questions, prioritize relevance and confidence
        if any(word in query_lower for word in ['what', 'who', 'when', 'where']):
            weights['query_relevance'] = 0.4
            weights['confidence'] = 0.3
            weights['source_quality'] = 0.2
            weights['recency'] = 0.05
            weights['completeness'] = 0.03
            weights['specificity'] = 0.02
        
        # For recent information queries, prioritize recency
        elif any(word in query_lower for word in ['recent', 'latest', 'current', 'now']):
            weights['recency'] = 0.3
            weights['query_relevance'] = 0.3
            weights['source_quality'] = 0.2
            weights['confidence'] = 0.15
            weights['completeness'] = 0.03
            weights['specificity'] = 0.02
        
        # For analytical questions, prioritize completeness and quality
        elif any(word in query_lower for word in ['analyze', 'explain', 'why', 'how']):
            weights['completeness'] = 0.25
            weights['source_quality'] = 0.25
            weights['query_relevance'] = 0.25
            weights['confidence'] = 0.15
            weights['recency'] = 0.05
            weights['specificity'] = 0.05
        
        return weights
    
    async def _score_fact_batch(
        self,
        facts: List[ExtractedFact],
        query: str,
        weights: Dict[str, float]
    ) -> List[ScoredFact]:
        """Score a batch of facts."""
        scored_facts = []
        
        for fact in facts:
            try:
                scored_fact = await self._score_single_fact(fact, query, weights)
                scored_facts.append(scored_fact)
            except Exception as e:
                logger.warning(f"Failed to score fact {fact.fact_id}: {e}")
                # Add fact with low score as fallback
                fallback_score = ScoredFact(
                    fact=fact,
                    overall_score=0.1,
                    scoring_dimensions=ScoringDimensions(
                        query_relevance=0.1, source_quality=0.1, confidence=0.1,
                        recency=0.1, completeness=0.1, specificity=0.1
                    ),
                    reasoning="Error during scoring",
                    metadata={'scoring_error': str(e)}
                )
                scored_facts.append(fallback_score)
        
        return scored_facts
    
    async def _score_single_fact(
        self,
        fact: ExtractedFact,
        query: str,
        weights: Dict[str, float]
    ) -> ScoredFact:
        """Score a single fact across all dimensions."""
        # Calculate individual dimension scores
        query_relevance = await self._calculate_query_relevance(fact, query)
        source_quality = self._calculate_source_quality(fact)
        confidence = self._calculate_confidence_score(fact)
        recency = self._calculate_recency_score(fact)
        completeness = self._calculate_completeness_score(fact)
        specificity = self._calculate_specificity_score(fact, query)
        
        # Create scoring dimensions
        dimensions = ScoringDimensions(
            query_relevance=query_relevance,
            source_quality=source_quality,
            confidence=confidence,
            recency=recency,
            completeness=completeness,
            specificity=specificity
        )
        
        # Calculate weighted overall score
        overall_score = (
            query_relevance * weights['query_relevance'] +
            source_quality * weights['source_quality'] +
            confidence * weights['confidence'] +
            recency * weights['recency'] +
            completeness * weights['completeness'] +
            specificity * weights['specificity']
        )
        
        # Generate reasoning
        reasoning = self._generate_scoring_reasoning(dimensions, weights, fact)
        
        return ScoredFact(
            fact=fact,
            overall_score=min(overall_score, 1.0),
            scoring_dimensions=dimensions,
            reasoning=reasoning,
            metadata={
                'scoring_method': 'multi_dimensional',
                'weights_used': weights,
                'fact_type': fact.fact_type.value
            }
        )
    
    async def _calculate_query_relevance(
        self,
        fact: ExtractedFact,
        query: str
    ) -> float:
        """Calculate semantic relevance to the query."""
        if self.semantic_enabled and self._embedding_model:
            try:
                # Use semantic similarity
                fact_embedding = self._embedding_model.encode([fact.content])
                query_embedding = self._embedding_model.encode([query])
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(fact_embedding, query_embedding)[0][0]
                return max(0.0, min(1.0, similarity))
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
        
        # Fallback to keyword-based relevance
        return self._calculate_keyword_relevance(fact.content, query)
    
    def _calculate_keyword_relevance(self, fact_content: str, query: str) -> float:
        """Calculate keyword-based relevance."""
        fact_words = set(fact_content.lower().split())
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(fact_words & query_words)
        return min(1.0, overlap / len(query_words))
    
    def _calculate_source_quality(self, fact: ExtractedFact) -> float:
        """Calculate source quality score."""
        quality_score = 0.5  # Base score
        
        # Boost for multiple sources
        if len(fact.source_documents) > 1:
            quality_score += 0.2
        
        # Boost for direct facts (more reliable)
        if fact.fact_type == FactType.DIRECT:
            quality_score += 0.2
        elif fact.fact_type == FactType.CHAIN:
            quality_score += 0.1
        
        # Boost for facts with entity sources
        if fact.source_entities:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _calculate_confidence_score(self, fact: ExtractedFact) -> float:
        """Calculate confidence score."""
        return fact.confidence
    
    def _calculate_recency_score(self, fact: ExtractedFact) -> float:
        """Calculate recency score."""
        # For now, return neutral score
        # This could be enhanced with temporal analysis
        return 0.5
    
    def _calculate_completeness_score(self, fact: ExtractedFact) -> float:
        """Calculate completeness score."""
        completeness = 0.5  # Base score
        
        # Boost for facts with context
        if fact.context:
            completeness += 0.2
        
        # Boost for facts with metadata
        if fact.metadata:
            completeness += 0.1
        
        # Boost for facts with extraction path
        if fact.extraction_path:
            completeness += 0.2
        
        return min(1.0, completeness)
    
    def _calculate_specificity_score(self, fact: ExtractedFact, query: str) -> float:
        """Calculate how specific the fact is to the query."""
        # Simple heuristic: longer facts are more specific
        fact_length = len(fact.content.split())
        query_length = len(query.split())
        
        if query_length == 0:
            return 0.5
        
        # Normalize by query length
        specificity = min(1.0, fact_length / (query_length * 2))
        return specificity
    
    def _generate_scoring_reasoning(
        self,
        dimensions: ScoringDimensions,
        weights: Dict[str, float],
        fact: ExtractedFact
    ) -> str:
        """Generate human-readable reasoning for the score."""
        top_dimensions = sorted(
            [
                ('query_relevance', dimensions.query_relevance),
                ('source_quality', dimensions.source_quality),
                ('confidence', dimensions.confidence)
            ],
            key=lambda x: x[1],
            reverse=True
        )
        
        reasoning_parts = []
        for dim_name, score in top_dimensions[:2]:  # Top 2 dimensions
            if score > 0.7:
                reasoning_parts.append(f"High {dim_name.replace('_', ' ')} ({score:.2f})")
            elif score > 0.4:
                reasoning_parts.append(f"Moderate {dim_name.replace('_', ' ')} ({score:.2f})")
        
        if not reasoning_parts:
            reasoning_parts.append("Low overall relevance")
        
        return f"Scored based on: {', '.join(reasoning_parts)}. Fact type: {fact.fact_type.value}."
