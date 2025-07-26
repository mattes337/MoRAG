"""Query-relevant entity discovery for intelligent graph traversal."""

import asyncio
import time
from typing import List, Dict, Any, Optional, NamedTuple, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from ..models import Entity
from ..query.entity_extractor import QueryEntityExtractor

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


class EntityRelevanceScore(NamedTuple):
    """Represents an entity with its relevance score."""
    entity: Entity
    relevance_score: float
    confidence: float
    reasoning: str
    match_type: str
    metadata: Dict[str, Any] = {}


class QueryIntent(Enum):
    """Types of query intent for entity discovery."""
    FACTUAL = "factual"
    EXPLORATORY = "exploratory"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    NAVIGATIONAL = "navigational"


@dataclass
class QueryAnalysis:
    """Analysis of user query for entity discovery."""
    query: str
    intent: QueryIntent
    keywords: List[str]
    entities_mentioned: List[str]
    complexity_score: float
    domain: Optional[str] = None
    temporal_context: Optional[str] = None
    metadata: Dict[str, Any] = None


class QueryEntityDiscovery:
    """Intelligent entity discovery based on query analysis."""
    
    def __init__(
        self,
        entity_extractor: QueryEntityExtractor,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the entity discovery system.
        
        Args:
            entity_extractor: Query entity extractor
            config: Optional configuration dictionary
        """
        self.entity_extractor = entity_extractor
        self.config = config or {}
        self.settings = get_settings()
        
        # Discovery parameters
        self.max_entities_to_discover = self.config.get('max_entities_to_discover', 50)
        self.min_relevance_threshold = self.config.get('min_relevance_threshold', 0.3)
        self.semantic_similarity_threshold = self.config.get('semantic_similarity_threshold', 0.7)
        
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
        
        # Performance settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_size = self.config.get('cache_size', 1000)
        self.batch_size = self.config.get('batch_size', 20)
        
        # Initialize components
        self._llm_client = None
        self._embedding_model = None
        self._cache = {} if self.enable_caching else None
        
        logger.info(
            "Entity discovery initialized",
            llm_enabled=self.llm_enabled,
            semantic_enabled=self.semantic_enabled,
            max_entities_to_discover=self.max_entities_to_discover,
            min_relevance_threshold=self.min_relevance_threshold
        )
    
    async def initialize(self) -> None:
        """Initialize LLM and embedding models."""
        # Initialize LLM
        if self.llm_enabled and not self._llm_client:
            try:
                genai.configure(api_key=self.settings.gemini_api_key)
                self._llm_client = genai.GenerativeModel(self.model_name)
                logger.info("LLM client initialized for entity discovery")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
                self.llm_enabled = False
        
        # Initialize embedding model
        if self.semantic_enabled and not self._embedding_model:
            try:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model initialized for entity discovery")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding model: {e}")
                self.semantic_enabled = False
    
    async def discover_entities(
        self,
        query: str,
        available_entities: Optional[List[Entity]] = None,
        max_entities: Optional[int] = None
    ) -> List[EntityRelevanceScore]:
        """Discover entities relevant to the query.
        
        Args:
            query: User query
            available_entities: Optional list of available entities to search
            max_entities: Optional override for maximum entities to return
            
        Returns:
            List of entities ranked by relevance
        """
        if not query or not query.strip():
            return []
        
        max_entities = max_entities or self.max_entities_to_discover
        
        try:
            logger.debug(
                "Starting entity discovery",
                query=query,
                available_entities_count=len(available_entities) if available_entities else 0,
                max_entities=max_entities
            )
            
            # Initialize models if needed
            await self.initialize()
            
            # Analyze query
            query_analysis = await self._analyze_query(query)
            
            # Extract entities from query
            extracted_entities = await self.entity_extractor.extract_and_link_entities(query)
            
            # Discover additional relevant entities
            if available_entities:
                discovered_entities = await self._discover_from_available(
                    query_analysis, available_entities
                )
            else:
                discovered_entities = await self._discover_from_graph(
                    query_analysis, extracted_entities
                )
            
            # Combine and rank entities
            all_entities = self._combine_entity_sources(
                extracted_entities, discovered_entities
            )
            
            # Filter by relevance threshold
            relevant_entities = [
                e for e in all_entities 
                if e.relevance_score >= self.min_relevance_threshold
            ]
            
            # Sort by relevance and limit results
            relevant_entities.sort(key=lambda x: x.relevance_score, reverse=True)
            final_entities = relevant_entities[:max_entities]
            
            logger.info(
                "Entity discovery completed",
                query=query,
                extracted_entities=len(extracted_entities.entities) if extracted_entities else 0,
                discovered_entities=len(discovered_entities),
                relevant_entities=len(relevant_entities),
                final_entities=len(final_entities)
            )
            
            return final_entities
            
        except Exception as e:
            logger.error(f"Entity discovery failed: {e}")
            return []
    
    async def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to understand intent and extract keywords."""
        try:
            # Basic keyword extraction
            keywords = self._extract_keywords(query)
            
            # Detect query intent
            intent = self._detect_intent(query, keywords)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity(query, keywords)
            
            # Extract mentioned entities (simple approach)
            entities_mentioned = self._extract_mentioned_entities(query)
            
            return QueryAnalysis(
                query=query,
                intent=intent,
                keywords=keywords,
                entities_mentioned=entities_mentioned,
                complexity_score=complexity_score,
                domain=None,  # Could be enhanced with domain detection
                temporal_context=None,  # Could be enhanced with temporal extraction
                metadata={}
            )
            
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            return QueryAnalysis(
                query=query,
                intent=QueryIntent.FACTUAL,
                keywords=query.split(),
                entities_mentioned=[],
                complexity_score=0.5,
                metadata={'analysis_error': str(e)}
            )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Simple keyword extraction (could be enhanced with NLP)
        import re
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _detect_intent(self, query: str, keywords: List[str]) -> QueryIntent:
        """Detect query intent based on patterns."""
        query_lower = query.lower()
        
        # Factual questions
        if any(word in query_lower for word in ['what', 'who', 'when', 'where', 'define']):
            return QueryIntent.FACTUAL
        
        # Comparative questions
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better']):
            return QueryIntent.COMPARATIVE
        
        # Analytical questions
        if any(word in query_lower for word in ['analyze', 'explain', 'why', 'how', 'relationship']):
            return QueryIntent.ANALYTICAL
        
        # Exploratory questions
        if any(word in query_lower for word in ['explore', 'discover', 'find', 'search', 'related']):
            return QueryIntent.EXPLORATORY
        
        # Default to factual
        return QueryIntent.FACTUAL
    
    def _calculate_complexity(self, query: str, keywords: List[str]) -> float:
        """Calculate query complexity score."""
        complexity = 0.0
        
        # Length factor
        complexity += min(len(query) / 100.0, 0.3)
        
        # Keyword count factor
        complexity += min(len(keywords) / 10.0, 0.3)
        
        # Question words factor
        question_words = ['what', 'who', 'when', 'where', 'why', 'how']
        question_count = sum(1 for word in question_words if word in query.lower())
        complexity += min(question_count / 3.0, 0.2)
        
        # Conjunction factor (multiple clauses)
        conjunctions = ['and', 'or', 'but', 'however', 'moreover', 'furthermore']
        conjunction_count = sum(1 for word in conjunctions if word in query.lower())
        complexity += min(conjunction_count / 2.0, 0.2)
        
        return min(complexity, 1.0)
    
    def _extract_mentioned_entities(self, query: str) -> List[str]:
        """Extract entities mentioned in the query (simple approach)."""
        import re
        
        # Look for capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        
        # Look for quoted strings
        quoted_strings = re.findall(r'"([^"]*)"', query)
        
        # Combine and deduplicate
        entities = list(set(capitalized_words + quoted_strings))
        
        return entities
    
    async def _discover_from_available(
        self,
        query_analysis: QueryAnalysis,
        available_entities: List[Entity]
    ) -> List[EntityRelevanceScore]:
        """Discover relevant entities from available entity list."""
        discovered = []
        
        # Process entities in batches
        for i in range(0, len(available_entities), self.batch_size):
            batch = available_entities[i:i + self.batch_size]
            batch_scores = await self._score_entity_batch(query_analysis, batch)
            discovered.extend(batch_scores)
        
        return discovered
    
    async def _discover_from_graph(
        self,
        query_analysis: QueryAnalysis,
        extracted_entities: Any
    ) -> List[EntityRelevanceScore]:
        """Discover entities from graph based on extracted entities."""
        # This would typically involve graph traversal to find related entities
        # For now, return empty list as this requires graph storage integration
        logger.debug("Graph-based entity discovery not yet implemented")
        return []
    
    async def _score_entity_batch(
        self,
        query_analysis: QueryAnalysis,
        entities: List[Entity]
    ) -> List[EntityRelevanceScore]:
        """Score a batch of entities for relevance."""
        scored_entities = []
        
        for entity in entities:
            # Calculate relevance score
            relevance_score = self._calculate_entity_relevance(query_analysis, entity)
            
            if relevance_score >= self.min_relevance_threshold:
                scored_entity = EntityRelevanceScore(
                    entity=entity,
                    relevance_score=relevance_score,
                    confidence=0.8,  # Could be enhanced with more sophisticated confidence calculation
                    reasoning=f"Entity relevance based on name and type matching",
                    match_type="heuristic",
                    metadata={'scoring_method': 'heuristic'}
                )
                scored_entities.append(scored_entity)
        
        return scored_entities
    
    def _calculate_entity_relevance(
        self,
        query_analysis: QueryAnalysis,
        entity: Entity
    ) -> float:
        """Calculate relevance score for an entity."""
        relevance = 0.0
        
        # Exact name match in query
        if entity.name.lower() in query_analysis.query.lower():
            relevance += 0.8
        
        # Keyword overlap
        entity_words = set(entity.name.lower().split())
        query_keywords = set(query_analysis.keywords)
        
        if entity_words & query_keywords:
            overlap_ratio = len(entity_words & query_keywords) / len(entity_words)
            relevance += overlap_ratio * 0.6
        
        # Type relevance (could be enhanced with type-specific scoring)
        if entity.type and any(keyword in entity.type.lower() for keyword in query_analysis.keywords):
            relevance += 0.3
        
        # Description relevance
        if entity.description:
            description_words = set(entity.description.lower().split())
            if description_words & query_keywords:
                overlap_ratio = len(description_words & query_keywords) / len(description_words)
                relevance += overlap_ratio * 0.4
        
        return min(relevance, 1.0)
    
    def _combine_entity_sources(
        self,
        extracted_entities: Any,
        discovered_entities: List[EntityRelevanceScore]
    ) -> List[EntityRelevanceScore]:
        """Combine entities from different sources."""
        combined = []
        seen_entity_ids = set()
        
        # Add extracted entities (if available)
        if extracted_entities and hasattr(extracted_entities, 'entities'):
            for entity_link in extracted_entities.entities:
                if entity_link.linked_entity_id and entity_link.linked_entity_id not in seen_entity_ids:
                    # Create EntityRelevanceScore from extracted entity
                    # This is a simplified approach - in practice, you'd need to fetch the full Entity object
                    seen_entity_ids.add(entity_link.linked_entity_id)
        
        # Add discovered entities
        for scored_entity in discovered_entities:
            if scored_entity.entity.id not in seen_entity_ids:
                combined.append(scored_entity)
                seen_entity_ids.add(scored_entity.entity.id)
        
        return combined
