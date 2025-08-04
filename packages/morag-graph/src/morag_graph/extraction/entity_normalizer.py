"""LLM-based entity normalization for converting entities to canonical forms."""

import json
import structlog
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
import re

from ..utils.llm_response_parser import parse_json_response, LLMResponseParseError

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None

logger = structlog.get_logger(__name__)


@dataclass
class EntityVariation:
    """Represents an entity normalization result."""
    original: str
    normalized: str
    confidence: float
    rule_applied: str


class LLMEntityNormalizer:
    """LLM-based entity normalizer that converts entities to canonical forms."""
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        language: str = "auto",
        cache_size: int = 1000
    ):
        """Initialize the LLM entity normalizer.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service
            language: Primary language for normalization
            cache_size: Maximum number of cached normalization results
        """
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()
        self.language = language
        self.logger = logger.bind(component="llm_entity_normalizer")
        
        # Cache for normalization results
        self.normalization_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_size = cache_size
        
        # Initialize Google AI if available
        if GOOGLE_AI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None
            self.logger.warning("Google AI not available or no API key provided")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        import os
        return os.getenv("GOOGLE_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
    
    async def normalize_entity(self, entity_name: str, entity_type: Optional[str] = None) -> EntityVariation:
        """Normalize a single entity to its canonical form.
        
        Args:
            entity_name: The entity name to normalize
            entity_type: Optional entity type for context
            
        Returns:
            EntityVariation with normalization result
        """
        if not entity_name or not entity_name.strip():
            return EntityVariation(
                original=entity_name,
                normalized="",
                confidence=0.0,
                rule_applied="empty_input"
            )
        
        # Check cache first
        cache_key = f"{entity_name.lower()}:{entity_type or 'unknown'}:{self.language}"
        if cache_key in self.normalization_cache:
            cached = self.normalization_cache[cache_key]
            return EntityVariation(
                original=entity_name,
                normalized=cached['normalized'],
                confidence=cached['confidence'],
                rule_applied=cached['rule_applied']
            )
        
        # Apply basic normalization first
        basic_normalized = self._apply_basic_normalization(entity_name)
        
        # Use LLM for advanced normalization if available
        if self.model:
            try:
                llm_result = await self._llm_normalize(entity_name, entity_type)
                result = EntityVariation(
                    original=entity_name,
                    normalized=llm_result[0],
                    confidence=llm_result[1],
                    rule_applied=llm_result[2]
                )
            except Exception as e:
                self.logger.warning(
                    "LLM normalization failed, using basic normalization",
                    entity=entity_name,
                    error=str(e)
                )
                result = EntityVariation(
                    original=entity_name,
                    normalized=basic_normalized,
                    confidence=0.7,
                    rule_applied="basic_normalization"
                )
        else:
            result = EntityVariation(
                original=entity_name,
                normalized=basic_normalized,
                confidence=0.7,
                rule_applied="basic_normalization"
            )
        
        # Cache the result
        self._cache_result(cache_key, result)
        
        return result
    
    def _apply_basic_normalization(self, entity_name: str) -> str:
        """Apply basic rule-based normalization."""
        normalized = entity_name.strip()
        
        # Apply minimal normalization to preserve entity integrity
        # Remove only basic articles that are clearly not part of entity names
        basic_prefixes = ["the ", "a ", "an "]

        for prefix in basic_prefixes:
            if normalized.lower().startswith(prefix.lower()):
                normalized = normalized[len(prefix):].strip()
                break
        
        # Capitalize first letter
        if normalized:
            normalized = normalized[0].upper() + normalized[1:]
        
        return normalized
    
    async def _llm_normalize(self, entity_name: str, entity_type: Optional[str] = None) -> Tuple[str, float, str]:
        """Use LLM to normalize entity name."""
        prompt = self._create_normalization_prompt(entity_name, entity_type)
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.1
                )
            )
            
            # Parse JSON response using robust utility
            response_text = response.text if response.text else ""

            # Use the robust JSON parser with fallback
            fallback_result = {
                'normalized': entity_name,
                'confidence': 0.5,
                'reasoning': 'fallback_due_to_parse_error'
            }

            result = parse_json_response(
                response_text,
                fallback_value=fallback_result,
                context=f"entity_normalization:{entity_name}"
            )
            
            return (
                result.get('normalized', entity_name),
                float(result.get('confidence', 0.8)),
                result.get('reasoning', 'llm_normalization')
            )
            
        except Exception as e:
            self.logger.warning(
                "LLM normalization failed",
                entity=entity_name,
                error=str(e)
            )
            return entity_name, 0.5, f"llm_error: {str(e)}"
    
    def _create_normalization_prompt(self, entity_name: str, entity_type: Optional[str] = None) -> str:
        """Create prompt for LLM normalization."""
        type_context = f" (Type: {entity_type})" if entity_type else ""
        
        return f"""
Normalize the following entity name to its canonical form. Focus on:

1. Remove unnecessary adjectives, qualifiers, and brand suffixes
2. Convert to singular form if plural
3. Use the base term without modifiers
4. Maintain the core meaning and identity
5. Use proper capitalization for the detected language

Entity: "{entity_name}"{type_context}
Language: {self.language}

Examples:
- "Silizium Pur" → "Silizium"
- "normalem Siliziumdioxid" → "Siliziumdioxid"
- "natural vitamins" → "vitamin"
- "heavy metals" → "heavy metal"
- "pure water" → "water"

Respond with JSON:
{{
    "normalized": "canonical form",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of changes made"
}}
"""
    
    def _cache_result(self, cache_key: str, result: EntityVariation) -> None:
        """Cache normalization result."""
        # Implement simple LRU cache
        if len(self.normalization_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.normalization_cache))
            del self.normalization_cache[oldest_key]
        
        self.normalization_cache[cache_key] = {
            'normalized': result.normalized,
            'confidence': result.confidence,
            'rule_applied': result.rule_applied
        }
    
    async def normalize_entities_batch(self, entities: List[str], entity_types: Optional[List[str]] = None) -> List[EntityVariation]:
        """Normalize multiple entities in batch.
        
        Args:
            entities: List of entity names to normalize
            entity_types: Optional list of entity types (same length as entities)
            
        Returns:
            List of EntityVariation results
        """
        if not entities:
            return []
        
        # Ensure entity_types has same length as entities
        if entity_types is None:
            entity_types = [None] * len(entities)
        elif len(entity_types) != len(entities):
            entity_types = entity_types + [None] * (len(entities) - len(entity_types))
        
        # Process entities concurrently
        tasks = [
            self.normalize_entity(entity, entity_type)
            for entity, entity_type in zip(entities, entity_types)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        normalized_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(
                    "Entity normalization failed",
                    entity=entities[i],
                    error=str(result)
                )
                normalized_results.append(EntityVariation(
                    original=entities[i],
                    normalized=entities[i],
                    confidence=0.3,
                    rule_applied="error_fallback"
                ))
            else:
                normalized_results.append(result)
        
        return normalized_results
