"""Response quality assessment system for evaluating generated responses."""

import asyncio
import time
from typing import List, Dict, Any, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from .citation_integrator import CitedResponse
from .citation_manager import CitedFact

logger = structlog.get_logger(__name__)

# Optional imports for enhanced functionality
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class QualityDimension(Enum):
    """Quality assessment dimensions."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    CITATION_QUALITY = "citation_quality"
    READABILITY = "readability"
    CONSISTENCY = "consistency"


@dataclass
class QualityMetrics:
    """Quality metrics for response assessment."""
    completeness: float  # How well the response addresses the query
    accuracy: float  # Factual correctness of statements
    relevance: float  # Alignment with user query intent
    coherence: float  # Logical flow and structure
    citation_quality: float  # Quality and accuracy of citations
    readability: float  # Clarity and ease of understanding
    consistency: float  # Internal logical consistency
    overall_score: float  # Weighted overall quality score


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment of a response."""
    metrics: QualityMetrics
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    confidence: float
    assessment_time: float
    metadata: Dict[str, Any]


@dataclass
class AssessmentOptions:
    """Options for response quality assessment."""
    enable_llm_assessment: bool = True
    enable_semantic_analysis: bool = True
    enable_citation_verification: bool = True
    enable_readability_analysis: bool = True
    include_improvement_suggestions: bool = True
    assessment_depth: str = "standard"  # "quick", "standard", "thorough"
    metadata: Dict[str, Any] = None


class ResponseQualityAssessor:
    """Comprehensive response quality assessment system."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the response quality assessor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = get_settings()
        
        # Assessment parameters
        self.quality_weights = self.config.get('quality_weights', {
            'completeness': 0.25,
            'accuracy': 0.20,
            'relevance': 0.20,
            'coherence': 0.15,
            'citation_quality': 0.10,
            'readability': 0.05,
            'consistency': 0.05
        })
        
        # LLM configuration
        self.llm_enabled = self.config.get('llm_enabled', True) and GEMINI_AVAILABLE
        self.model_name = self.config.get('model_name', 'gemini-1.5-flash')
        
        # Semantic analysis configuration
        self.semantic_enabled = (
            self.config.get('semantic_enabled', True) and 
            SENTENCE_TRANSFORMERS_AVAILABLE
        )
        self.embedding_model_name = self.config.get(
            'embedding_model_name', 
            'all-MiniLM-L6-v2'
        )
        
        # Assessment thresholds
        self.min_completeness_threshold = self.config.get('min_completeness_threshold', 0.7)
        self.min_accuracy_threshold = self.config.get('min_accuracy_threshold', 0.8)
        self.min_citation_quality_threshold = self.config.get('min_citation_quality_threshold', 0.6)
        
        # Performance settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_size = self.config.get('cache_size', 100)
        
        # Initialize components
        self._llm_client = None
        self._embedding_model = None
        self._cache = {} if self.enable_caching else None
        
        logger.info(
            "Response quality assessor initialized",
            llm_enabled=self.llm_enabled,
            semantic_enabled=self.semantic_enabled,
            quality_weights=self.quality_weights
        )
    
    async def initialize(self) -> None:
        """Initialize LLM and embedding models."""
        # Initialize LLM
        if self.llm_enabled and not self._llm_client:
            try:
                genai.configure(api_key=self.settings.gemini_api_key)
                self._llm_client = genai.GenerativeModel(self.model_name)
                logger.info("LLM client initialized for quality assessment")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
                self.llm_enabled = False
        
        # Initialize embedding model
        if self.semantic_enabled and not self._embedding_model:
            try:
                # Verify numpy availability before model loading
                try:
                    import numpy as np
                    logger.debug("NumPy is available for embedding model loading", version=np.__version__)
                except ImportError as numpy_error:
                    logger.error("NumPy not available for SentenceTransformer model loading", error=str(numpy_error))
                    self.semantic_enabled = False
                    return

                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model initialized for quality assessment")
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Failed to initialize embedding model: {error_msg}")
                
                # Check if it's a numpy deserialization error and try cache clear
                if "numpy" in error_msg.lower() and "deserialization" in error_msg.lower():
                    logger.info("Detected numpy deserialization error, clearing cache and retrying")
                    try:
                        import os
                        import shutil
                        
                        # Clear sentence transformers cache
                        cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
                        if os.path.exists(cache_dir):
                            shutil.rmtree(cache_dir)
                            logger.info("Cleared SentenceTransformer cache directory", path=cache_dir)
                        
                        # Retry model loading
                        self._embedding_model = SentenceTransformer(self.embedding_model_name)
                        logger.info("Embedding model initialized for quality assessment after cache clear")
                        return
                    except Exception as cache_error:
                        logger.warning(f"Failed to clear cache and retry: {cache_error}")
                
                self.semantic_enabled = False
    
    async def assess_response(
        self,
        response: CitedResponse,
        original_query: str,
        facts: Optional[List[CitedFact]] = None,
        options: Optional[AssessmentOptions] = None
    ) -> QualityAssessment:
        """Assess response quality across multiple dimensions.
        
        Args:
            response: Cited response to assess
            original_query: Original user query
            facts: Optional list of facts used in response
            options: Assessment options
            
        Returns:
            Comprehensive quality assessment
        """
        start_time = time.time()
        options = options or AssessmentOptions()
        facts = facts or []
        
        try:
            logger.info(
                "Starting response quality assessment",
                response_length=len(response.content),
                citation_count=response.citation_count,
                assessment_depth=options.assessment_depth
            )
            
            # Initialize models if needed
            await self.initialize()
            
            # Assess individual quality dimensions
            completeness = await self._assess_completeness(response, original_query, options)
            accuracy = await self._assess_accuracy(response, facts, options)
            relevance = await self._assess_relevance(response, original_query, options)
            coherence = await self._assess_coherence(response, options)
            citation_quality = await self._assess_citation_quality(response, options)
            readability = await self._assess_readability(response, options)
            consistency = await self._assess_consistency(response, options)
            
            # Calculate overall score
            overall_score = (
                completeness * self.quality_weights['completeness'] +
                accuracy * self.quality_weights['accuracy'] +
                relevance * self.quality_weights['relevance'] +
                coherence * self.quality_weights['coherence'] +
                citation_quality * self.quality_weights['citation_quality'] +
                readability * self.quality_weights['readability'] +
                consistency * self.quality_weights['consistency']
            )
            
            # Create quality metrics
            metrics = QualityMetrics(
                completeness=completeness,
                accuracy=accuracy,
                relevance=relevance,
                coherence=coherence,
                citation_quality=citation_quality,
                readability=readability,
                consistency=consistency,
                overall_score=overall_score
            )
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(metrics)
            
            # Generate improvement suggestions
            suggestions = []
            if options.include_improvement_suggestions:
                suggestions = await self._generate_improvement_suggestions(
                    metrics, response, original_query
                )
            
            # Calculate assessment confidence
            confidence = self._calculate_assessment_confidence(metrics, options)
            
            assessment_time = time.time() - start_time
            
            logger.info(
                "Response quality assessment completed",
                overall_score=overall_score,
                completeness=completeness,
                accuracy=accuracy,
                assessment_time=assessment_time
            )
            
            return QualityAssessment(
                metrics=metrics,
                strengths=strengths,
                weaknesses=weaknesses,
                improvement_suggestions=suggestions,
                confidence=confidence,
                assessment_time=assessment_time,
                metadata={
                    'assessment_method': 'comprehensive',
                    'options_used': options.__dict__,
                    'models_used': {
                        'llm_enabled': self.llm_enabled,
                        'semantic_enabled': self.semantic_enabled
                    }
                }
            )
            
        except Exception as e:
            assessment_time = time.time() - start_time
            logger.error(f"Response quality assessment failed: {e}")
            
            # Return fallback assessment
            return QualityAssessment(
                metrics=QualityMetrics(
                    completeness=0.5, accuracy=0.5, relevance=0.5, coherence=0.5,
                    citation_quality=0.5, readability=0.5, consistency=0.5,
                    overall_score=0.5
                ),
                strengths=[],
                weaknesses=["Assessment failed"],
                improvement_suggestions=[],
                confidence=0.0,
                assessment_time=assessment_time,
                metadata={'error': str(e)}
            )
    
    async def _assess_completeness(
        self,
        response: CitedResponse,
        query: str,
        options: AssessmentOptions
    ) -> float:
        """Assess how completely the response addresses the query."""
        # Simple heuristic: longer responses are generally more complete
        response_length = len(response.content.split())
        query_length = len(query.split())
        
        # Base score on response length relative to query complexity
        base_score = min(1.0, response_length / (query_length * 10))
        
        # Boost for presence of citations
        citation_boost = min(0.2, response.citation_count * 0.05)
        
        # Check for key question words being addressed
        question_words = ['what', 'who', 'when', 'where', 'why', 'how']
        query_lower = query.lower()
        response_lower = response.content.lower()
        
        addressed_questions = 0
        for word in question_words:
            if word in query_lower:
                # Simple check if response contains related content
                if any(related in response_lower for related in [word, word + 's']):
                    addressed_questions += 1
        
        question_coverage = addressed_questions / max(1, len([w for w in question_words if w in query_lower]))
        
        return min(1.0, base_score + citation_boost + (question_coverage * 0.3))
    
    async def _assess_accuracy(
        self,
        response: CitedResponse,
        facts: List[CitedFact],
        options: AssessmentOptions
    ) -> float:
        """Assess factual accuracy of the response."""
        if not facts:
            return 0.7  # Neutral score when no facts to verify against
        
        # Calculate average confidence of facts used
        avg_fact_confidence = sum(fact.score for fact in facts) / len(facts)
        
        # Boost for verified citations
        citation_verification_boost = 0.0
        if response.verification_status == "verified":
            citation_verification_boost = 0.2
        elif response.verification_status == "partially_verified":
            citation_verification_boost = 0.1
        
        return min(1.0, avg_fact_confidence + citation_verification_boost)
    
    async def _assess_relevance(
        self,
        response: CitedResponse,
        query: str,
        options: AssessmentOptions
    ) -> float:
        """Assess relevance of response to the original query."""
        if self.semantic_enabled and self._embedding_model:
            try:
                # Use semantic similarity
                query_embedding = self._embedding_model.encode([query])
                response_embedding = self._embedding_model.encode([response.content])
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
                return max(0.0, min(1.0, similarity))
            except Exception as e:
                logger.warning(f"Semantic relevance assessment failed: {e}")
        
        # Fallback to keyword-based relevance
        query_words = set(query.lower().split())
        response_words = set(response.content.lower().split())
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words & response_words)
        return min(1.0, overlap / len(query_words))
    
    async def _assess_coherence(
        self,
        response: CitedResponse,
        options: AssessmentOptions
    ) -> float:
        """Assess logical flow and structure of the response."""
        content = response.content
        
        # Check for basic structure indicators
        structure_score = 0.0
        
        # Presence of paragraphs
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            structure_score += 0.3
        
        # Presence of transitions
        transitions = ['however', 'therefore', 'furthermore', 'moreover', 'additionally', 'consequently']
        transition_count = sum(1 for word in transitions if word in content.lower())
        structure_score += min(0.3, transition_count * 0.1)
        
        # Sentence length variation (good coherence has varied sentence lengths)
        sentences = content.split('.')
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            if sentence_lengths:
                length_variance = len(set(sentence_lengths)) / len(sentence_lengths)
                structure_score += min(0.4, length_variance)
        
        return min(1.0, structure_score)
    
    async def _assess_citation_quality(
        self,
        response: CitedResponse,
        options: AssessmentOptions
    ) -> float:
        """Assess quality and accuracy of citations."""
        if response.citation_count == 0:
            return 0.0
        
        quality_score = 0.0
        
        # Verification status
        if response.verification_status == "verified":
            quality_score += 0.5
        elif response.verification_status == "partially_verified":
            quality_score += 0.3
        
        # Citation density (citations per 100 words)
        word_count = len(response.content.split())
        citation_density = (response.citation_count / word_count) * 100
        optimal_density = 2.0  # 2 citations per 100 words
        density_score = 1.0 - abs(citation_density - optimal_density) / optimal_density
        quality_score += max(0.0, min(0.3, density_score))
        
        # Citation completeness
        if response.citations_list:
            quality_score += 0.2
        
        return min(1.0, quality_score)
    
    async def _assess_readability(
        self,
        response: CitedResponse,
        options: AssessmentOptions
    ) -> float:
        """Assess readability and clarity of the response."""
        content = response.content
        
        # Simple readability metrics
        words = content.split()
        sentences = content.split('.')
        
        if not words or not sentences:
            return 0.5
        
        # Average words per sentence
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Optimal range: 15-20 words per sentence
        if 15 <= avg_words_per_sentence <= 20:
            sentence_score = 1.0
        else:
            sentence_score = max(0.0, 1.0 - abs(avg_words_per_sentence - 17.5) / 17.5)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Optimal range: 4-6 characters per word
        if 4 <= avg_word_length <= 6:
            word_score = 1.0
        else:
            word_score = max(0.0, 1.0 - abs(avg_word_length - 5) / 5)
        
        return (sentence_score + word_score) / 2
    
    async def _assess_consistency(
        self,
        response: CitedResponse,
        options: AssessmentOptions
    ) -> float:
        """Assess internal logical consistency of the response."""
        # Simple consistency check: look for contradictory statements
        content = response.content.lower()
        
        # Check for obvious contradictions
        contradiction_patterns = [
            ('is', 'is not'),
            ('can', 'cannot'),
            ('will', 'will not'),
            ('always', 'never'),
            ('all', 'none')
        ]
        
        contradiction_count = 0
        for positive, negative in contradiction_patterns:
            if positive in content and negative in content:
                contradiction_count += 1
        
        # Penalize contradictions
        consistency_score = 1.0 - (contradiction_count * 0.2)
        
        return max(0.0, consistency_score)
    
    def _identify_strengths_weaknesses(
        self,
        metrics: QualityMetrics
    ) -> tuple[List[str], List[str]]:
        """Identify strengths and weaknesses based on metrics."""
        strengths = []
        weaknesses = []
        
        # Check each dimension
        if metrics.completeness >= 0.8:
            strengths.append("Comprehensive coverage of the query")
        elif metrics.completeness < 0.6:
            weaknesses.append("Incomplete response to the query")
        
        if metrics.accuracy >= 0.8:
            strengths.append("High factual accuracy")
        elif metrics.accuracy < 0.6:
            weaknesses.append("Questionable factual accuracy")
        
        if metrics.relevance >= 0.8:
            strengths.append("Highly relevant to the query")
        elif metrics.relevance < 0.6:
            weaknesses.append("Limited relevance to the query")
        
        if metrics.citation_quality >= 0.7:
            strengths.append("Good citation quality and coverage")
        elif metrics.citation_quality < 0.4:
            weaknesses.append("Poor or missing citations")
        
        if metrics.coherence >= 0.8:
            strengths.append("Well-structured and coherent")
        elif metrics.coherence < 0.6:
            weaknesses.append("Poor structure and flow")
        
        return strengths, weaknesses
    
    async def _generate_improvement_suggestions(
        self,
        metrics: QualityMetrics,
        response: CitedResponse,
        query: str
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        if metrics.completeness < self.min_completeness_threshold:
            suggestions.append("Provide more comprehensive coverage of all aspects of the query")
        
        if metrics.accuracy < self.min_accuracy_threshold:
            suggestions.append("Verify facts and improve source quality")
        
        if metrics.citation_quality < self.min_citation_quality_threshold:
            suggestions.append("Add more citations and improve source references")
        
        if metrics.coherence < 0.7:
            suggestions.append("Improve logical flow and structure with better transitions")
        
        if metrics.readability < 0.7:
            suggestions.append("Simplify language and improve sentence structure")
        
        if metrics.relevance < 0.7:
            suggestions.append("Focus more directly on addressing the specific query")
        
        return suggestions
    
    def _calculate_assessment_confidence(
        self,
        metrics: QualityMetrics,
        options: AssessmentOptions
    ) -> float:
        """Calculate confidence in the assessment."""
        confidence = 0.7  # Base confidence
        
        # Boost confidence if multiple assessment methods were used
        if self.llm_enabled:
            confidence += 0.1
        if self.semantic_enabled:
            confidence += 0.1
        if options.enable_citation_verification:
            confidence += 0.1
        
        return min(1.0, confidence)
