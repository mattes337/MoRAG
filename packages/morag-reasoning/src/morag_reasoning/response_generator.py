"""Response generation system for synthesizing facts into coherent responses."""

import asyncio
import time
from typing import List, Dict, Any, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from .llm import LLMClient
from .citation_manager import CitedFact, CitationFormat

logger = structlog.get_logger(__name__)

# Optional imports for enhanced functionality
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class ResponseFormat(Enum):
    """Response format options."""
    DETAILED = "detailed"
    SUMMARY = "summary"
    BULLET_POINTS = "bullet_points"
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"


class ResponseStructure(Enum):
    """Response structure templates."""
    STANDARD = "standard"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    EXPLANATORY = "explanatory"
    FACTUAL = "factual"


@dataclass
class ResponseOptions:
    """Options for response generation."""
    format: ResponseFormat = ResponseFormat.DETAILED
    structure: ResponseStructure = ResponseStructure.STANDARD
    max_length: int = 2000
    include_reasoning: bool = True
    include_confidence: bool = True
    citation_format: CitationFormat = CitationFormat.SIMPLE
    language: str = "en"
    tone: str = "professional"
    metadata: Dict[str, Any] = None


@dataclass
class GeneratedResponse:
    """Generated response with metadata."""
    content: str
    summary: str
    key_points: List[str]
    reasoning: str
    confidence_score: float
    word_count: int
    generation_time: float
    facts_used: List[str]
    metadata: Dict[str, Any]


class ResponseGenerator:
    """LLM-based response generation system."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the response generator.
        
        Args:
            llm_client: LLM client for response generation
            config: Optional configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.settings = get_settings()
        
        # Generation parameters
        self.default_max_length = self.config.get('default_max_length', 2000)
        self.min_facts_required = self.config.get('min_facts_required', 1)
        self.max_facts_to_use = self.config.get('max_facts_to_use', 20)
        
        # LLM configuration
        self.llm_enabled = self.config.get('llm_enabled', True) and GEMINI_AVAILABLE
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 4000)
        
        # Quality settings
        self.enable_fact_verification = self.config.get('enable_fact_verification', True)
        self.enable_consistency_check = self.config.get('enable_consistency_check', True)
        self.enable_reasoning_explanation = self.config.get('enable_reasoning_explanation', True)
        
        # Performance settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_size = self.config.get('cache_size', 100)
        
        # Initialize components
        self._llm_client = None
        self._cache = {} if self.enable_caching else None
        
        logger.info(
            "Response generator initialized",
            llm_enabled=self.llm_enabled,
            model_name=self.model_name,
            default_max_length=self.default_max_length,
            enable_fact_verification=self.enable_fact_verification
        )
    
    async def initialize(self) -> None:
        """Initialize the LLM client."""
        if not self.llm_enabled:
            logger.info("LLM response generation disabled")
            return
        
        if self._llm_client:
            return
        
        try:
            # Configure Gemini
            genai.configure(api_key=self.settings.gemini_api_key)
            self._llm_client = genai.GenerativeModel(self.model_name)
            
            logger.info("LLM client initialized for response generation")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_enabled = False
    
    async def generate_response(
        self,
        facts: List[CitedFact],
        query: str,
        options: Optional[ResponseOptions] = None
    ) -> GeneratedResponse:
        """Generate comprehensive response from gathered facts.
        
        Args:
            facts: List of cited facts to synthesize
            query: Original user query
            options: Response generation options
            
        Returns:
            Generated response with metadata
        """
        if not facts:
            return self._create_empty_response(query, "No facts available for response generation")
        
        start_time = time.time()
        options = options or ResponseOptions()
        
        try:
            logger.info(
                "Starting response generation",
                query=query,
                num_facts=len(facts),
                format=options.format.value,
                structure=options.structure.value
            )
            
            # Initialize LLM if needed
            await self.initialize()
            
            # Filter and prepare facts
            prepared_facts = self._prepare_facts(facts, options)
            
            if len(prepared_facts) < self.min_facts_required:
                return self._create_empty_response(
                    query, 
                    f"Insufficient facts for response generation (need {self.min_facts_required}, got {len(prepared_facts)})"
                )
            
            # Generate response using LLM
            if self.llm_enabled:
                response = await self._generate_with_llm(query, prepared_facts, options)
            else:
                response = await self._generate_fallback(query, prepared_facts, options)
            
            generation_time = time.time() - start_time
            
            # Add metadata
            response.generation_time = generation_time
            response.facts_used = [fact.fact.fact_id for fact in prepared_facts]
            response.metadata = {
                'generation_method': 'llm' if self.llm_enabled else 'fallback',
                'options_used': options.__dict__,
                'num_facts_used': len(prepared_facts),
                'query_length': len(query),
                'response_length': len(response.content)
            }
            
            logger.info(
                "Response generation completed",
                query=query,
                response_length=len(response.content),
                confidence_score=response.confidence_score,
                generation_time=generation_time
            )
            
            return response
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Response generation failed: {e}")
            
            return GeneratedResponse(
                content=f"Error generating response: {e}",
                summary="Response generation failed",
                key_points=[],
                reasoning="Error occurred during generation",
                confidence_score=0.0,
                word_count=0,
                generation_time=generation_time,
                facts_used=[],
                metadata={'error': str(e)}
            )
    
    def _prepare_facts(
        self,
        facts: List[CitedFact],
        options: ResponseOptions
    ) -> List[CitedFact]:
        """Prepare and filter facts for response generation."""
        # Sort facts by score (highest first)
        sorted_facts = sorted(facts, key=lambda f: f.score, reverse=True)
        
        # Limit number of facts
        limited_facts = sorted_facts[:self.max_facts_to_use]
        
        # Filter by minimum score if needed
        min_score = self.config.get('min_fact_score', 0.1)
        filtered_facts = [f for f in limited_facts if f.score >= min_score]
        
        return filtered_facts
    
    async def _generate_with_llm(
        self,
        query: str,
        facts: List[CitedFact],
        options: ResponseOptions
    ) -> GeneratedResponse:
        """Generate response using LLM."""
        # Create generation prompt
        prompt = self._create_generation_prompt(query, facts, options)
        
        # Generate response
        response = await self._llm_client.generate_content_async(prompt)
        
        # Parse response
        return self._parse_llm_response(response.text, query, facts, options)
    
    def _create_generation_prompt(
        self,
        query: str,
        facts: List[CitedFact],
        options: ResponseOptions
    ) -> str:
        """Create prompt for LLM response generation."""
        # Format facts for prompt
        facts_text = "\n".join([
            f"Fact {i+1}: {fact.fact.content} (Confidence: {fact.score:.2f})"
            for i, fact in enumerate(facts)
        ])
        
        # Create prompt based on format and structure
        prompt = f"""
You are an expert research assistant tasked with generating a comprehensive response to a user query based on gathered facts.

User Query: "{query}"

Available Facts:
{facts_text}

Instructions:
1. Generate a {options.format.value} response in {options.structure.value} structure
2. Maximum length: {options.max_length} words
3. Tone: {options.tone}
4. Language: {options.language}
5. Include proper reasoning if requested: {options.include_reasoning}
6. Include confidence assessment if requested: {options.include_confidence}

Response Requirements:
- Synthesize the facts into a coherent, well-structured response
- Maintain logical flow and clear organization
- Address the user query directly and comprehensively
- Use appropriate citations (will be added separately)
- Ensure factual accuracy and consistency
- Provide reasoning for conclusions when appropriate

Format your response as JSON:
{{
  "content": "Main response content",
  "summary": "Brief summary of key findings",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "reasoning": "Explanation of reasoning process",
  "confidence_score": 0.85
}}
"""
        
        return prompt
    
    def _parse_llm_response(
        self,
        response_text: str,
        query: str,
        facts: List[CitedFact],
        options: ResponseOptions
    ) -> GeneratedResponse:
        """Parse LLM response into structured format."""
        try:
            import json
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[start_idx:end_idx]
            data = json.loads(json_text)
            
            # Extract components
            content = data.get('content', '')
            summary = data.get('summary', '')
            key_points = data.get('key_points', [])
            reasoning = data.get('reasoning', '')
            confidence_score = float(data.get('confidence_score', 0.5))
            
            # Calculate word count
            word_count = len(content.split())
            
            return GeneratedResponse(
                content=content,
                summary=summary,
                key_points=key_points,
                reasoning=reasoning,
                confidence_score=confidence_score,
                word_count=word_count,
                generation_time=0.0,  # Will be set by caller
                facts_used=[],  # Will be set by caller
                metadata={}  # Will be set by caller
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback to treating entire response as content
            return GeneratedResponse(
                content=response_text,
                summary="Generated response (parsing failed)",
                key_points=[],
                reasoning="LLM response parsing failed",
                confidence_score=0.5,
                word_count=len(response_text.split()),
                generation_time=0.0,
                facts_used=[],
                metadata={'parsing_error': str(e)}
            )
    
    async def _generate_fallback(
        self,
        query: str,
        facts: List[CitedFact],
        options: ResponseOptions
    ) -> GeneratedResponse:
        """Generate response using fallback method when LLM is not available."""
        # Simple fact concatenation with basic structure
        content_parts = [f"Response to: {query}\n"]
        
        if options.format == ResponseFormat.SUMMARY:
            content_parts.append("Summary of findings:")
            for i, fact in enumerate(facts[:5], 1):
                content_parts.append(f"{i}. {fact.fact.content}")
        
        elif options.format == ResponseFormat.BULLET_POINTS:
            content_parts.append("Key findings:")
            for fact in facts:
                content_parts.append(f"• {fact.fact.content}")
        
        else:  # DETAILED or other formats
            content_parts.append("Based on the available information:")
            for fact in facts:
                content_parts.append(f"\n{fact.fact.content}")
        
        content = "\n".join(content_parts)
        
        # Extract key points
        key_points = [fact.fact.content for fact in facts[:3]]
        
        # Calculate average confidence
        avg_confidence = sum(fact.score for fact in facts) / len(facts) if facts else 0.0
        
        return GeneratedResponse(
            content=content,
            summary=f"Fallback response for: {query}",
            key_points=key_points,
            reasoning="Generated using fallback method due to LLM unavailability",
            confidence_score=avg_confidence,
            word_count=len(content.split()),
            generation_time=0.0,
            facts_used=[],
            metadata={'generation_method': 'fallback'}
        )
    
    def _create_empty_response(self, query: str, reason: str) -> GeneratedResponse:
        """Create empty response when generation is not possible."""
        return GeneratedResponse(
            content=f"Unable to generate response for: {query}. Reason: {reason}",
            summary="No response generated",
            key_points=[],
            reasoning=reason,
            confidence_score=0.0,
            word_count=0,
            generation_time=0.0,
            facts_used=[],
            metadata={'empty_response_reason': reason}
        )
