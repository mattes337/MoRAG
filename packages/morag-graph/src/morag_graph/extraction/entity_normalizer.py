"""LLM-based entity normalization for converting entities to canonical forms."""

import asyncio
import os

# Import agents framework - required dependency
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import structlog

# Add the project root to the path if not already there
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents import get_agent
from agents.base import LLMResponseParser

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
        cache_size: int = 1000,
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

    async def normalize_entity(
        self, entity_name: str, entity_type: Optional[str] = None
    ) -> EntityVariation:
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
                rule_applied="empty_input",
            )

        # Check cache first
        cache_key = f"{entity_name.lower()}:{entity_type or 'unknown'}:{self.language}"
        if cache_key in self.normalization_cache:
            cached = self.normalization_cache[cache_key]
            return EntityVariation(
                original=entity_name,
                normalized=cached["normalized"],
                confidence=cached["confidence"],
                rule_applied=cached["rule_applied"],
            )

        # Apply basic normalization first
        basic_normalized = self._apply_basic_normalization(entity_name)

        # Use entity extraction agent for normalization - ALWAYS
        try:
            # Pass model override to ensure consistent model usage
            from agents.base.config import ModelConfig

            model_config = ModelConfig(model=self.model_name)
            entity_agent = get_agent("entity_extraction", model=model_config)

            # Use the agent to extract and normalize the entity
            extraction_result = await entity_agent.extract_entities(
                text=f"Normalize this entity: {entity_name}", domain="normalization"
            )

            if extraction_result.entities:
                # Use the first normalized entity
                normalized_entity = extraction_result.entities[0]
                result = EntityVariation(
                    original=entity_name,
                    normalized=normalized_entity.canonical_name,
                    confidence=normalized_entity.confidence,
                    rule_applied="agent_normalization",
                )
            else:
                # Fallback to basic normalization
                result = EntityVariation(
                    original=entity_name,
                    normalized=basic_normalized,
                    confidence=0.7,
                    rule_applied="basic_normalization",
                )
        except Exception as e:
            self.logger.warning(
                "Agent normalization failed, using basic normalization",
                entity=entity_name,
                error=str(e),
            )
            result = EntityVariation(
                original=entity_name,
                normalized=basic_normalized,
                confidence=0.7,
                rule_applied="basic_normalization",
            )
        else:
            result = EntityVariation(
                original=entity_name,
                normalized=basic_normalized,
                confidence=0.7,
                rule_applied="basic_normalization",
            )

        # Cache the result
        self._cache_result(cache_key, result)

        return result

    def _apply_basic_normalization(self, entity_name: str) -> str:
        """Apply comprehensive rule-based normalization to create canonical forms."""
        normalized = entity_name.strip()

        # Remove basic articles that are clearly not part of entity names
        basic_prefixes = [
            "the ",
            "a ",
            "an ",
            "der ",
            "die ",
            "das ",
            "le ",
            "la ",
            "les ",
        ]
        for prefix in basic_prefixes:
            if normalized.lower().startswith(prefix.lower()):
                normalized = normalized[len(prefix) :].strip()
                break

        # Remove content in parentheses (e.g., "Engelwurz (Wurzel)" -> "Engelwurz")
        import re

        normalized = re.sub(r"\s*\([^)]*\)", "", normalized).strip()

        # Remove common adjectives and qualifiers
        qualifiers_to_remove = [
            "pure",
            "natural",
            "organic",
            "fresh",
            "raw",
            "whole",
            "complete",
            "pur",
            "natürlich",
            "organisch",
            "frisch",
            "roh",
            "ganz",
            "komplett",
            "normal",
            "regular",
            "standard",
            "basic",
            "simple",
            "plain",
        ]

        words = normalized.split()
        filtered_words = []
        for word in words:
            if word.lower() not in qualifiers_to_remove:
                filtered_words.append(word)

        if filtered_words:
            normalized = " ".join(filtered_words)

        # Convert to singular form (basic rules)
        normalized = self._to_singular(normalized)

        # Capitalize first letter of each word (proper case)
        normalized = " ".join(word.capitalize() for word in normalized.split())

        return normalized

    def _to_singular(self, text: str) -> str:
        """Convert plural forms to singular (basic rules)."""
        words = text.split()
        singular_words = []

        for word in words:
            word_lower = word.lower()

            # German plural rules
            if word_lower.endswith("e") and len(word) > 3:
                # Remove 'e' ending for many German plurals
                singular = word[:-1]
            elif word_lower.endswith("en") and len(word) > 4:
                # Remove 'en' ending
                singular = word[:-2]
            elif word_lower.endswith("er") and len(word) > 4:
                # Remove 'er' ending
                singular = word[:-2]
            # English plural rules
            elif word_lower.endswith("ies") and len(word) > 4:
                # "berries" -> "berry"
                singular = word[:-3] + "y"
            elif word_lower.endswith("ves") and len(word) > 4:
                # "leaves" -> "leaf"
                singular = word[:-3] + "f"
            elif word_lower.endswith("ses") and len(word) > 4:
                # "glasses" -> "glass"
                singular = word[:-2]
            elif (
                word_lower.endswith("s")
                and len(word) > 3
                and not word_lower.endswith("ss")
            ):
                # General 's' removal, but not for words ending in 'ss'
                singular = word[:-1]
            else:
                singular = word

            singular_words.append(singular)

        return " ".join(singular_words)

    async def _llm_normalize(
        self, entity_name: str, entity_type: Optional[str] = None
    ) -> Tuple[str, float, str]:
        """Use LLM to normalize entity name."""
        prompt = self._create_normalization_prompt(entity_name, entity_type)

        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200, temperature=0.1
                ),
            )

            # Parse JSON response using robust utility
            response_text = response.text if response.text else ""

            # Use the robust JSON parser with fallback
            fallback_result = {
                "normalized": entity_name,
                "confidence": 0.5,
                "reasoning": "fallback_due_to_parse_error",
            }

            # Use agent system for response parsing
            result = LLMResponseParser.parse_json_response(
                response=response_text,
                fallback_value=fallback_result,
                context=f"entity_normalization:{entity_name}",
            )

            return (
                result.get("normalized", entity_name),
                float(result.get("confidence", 0.8)),
                result.get("reasoning", "llm_normalization"),
            )

        except Exception as e:
            self.logger.warning(
                "LLM normalization failed", entity=entity_name, error=str(e)
            )
            return entity_name, 0.5, f"llm_error: {str(e)}"

    def _create_normalization_prompt(
        self, entity_name: str, entity_type: Optional[str] = None
    ) -> str:
        """Create prompt for LLM normalization."""
        type_context = f" (Type: {entity_type})" if entity_type else ""

        return f"""
Normalize the following entity name to its canonical form. Follow these STRICT rules:

1. Remove ALL adjectives, qualifiers, brand names, and descriptive modifiers
2. Convert to SINGULAR form (no plurals)
3. Remove content in parentheses like "Engelwurz (Wurzel)" → "Engelwurz"
4. Use the base, canonical term without any additions
5. Remove gender-specific forms (use base form)
6. Use proper capitalization for the detected language
7. Keep only the core entity identity

Entity: "{entity_name}"{type_context}
Language: {self.language}

Examples:
- "Silizium Pur" → "Silizium"
- "normalem Siliziumdioxid" → "Siliziumdioxid"
- "natural vitamins" → "Vitamin"
- "heavy metals" → "Metal"
- "pure water" → "Wasser"
- "Engelwurz (Wurzel)" → "Engelwurz"
- "fresh herbs" → "Herb"
- "organic compounds" → "Compound"
- "männliche Hormone" → "Hormon"
- "weibliche Geschlechtshormone" → "Geschlechtshormon"

CRITICAL: The result must be a single canonical entity that can be used globally across all documents.
Multiple facts about the same entity should all reference the same normalized name.

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
            "normalized": result.normalized,
            "confidence": result.confidence,
            "rule_applied": result.rule_applied,
        }

    async def normalize_entities_batch(
        self, entities: List[str], entity_types: Optional[List[str]] = None
    ) -> List[EntityVariation]:
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
                    "Entity normalization failed", entity=entities[i], error=str(result)
                )
                normalized_results.append(
                    EntityVariation(
                        original=entities[i],
                        normalized=entities[i],
                        confidence=0.3,
                        rule_applied="error_fallback",
                    )
                )
            else:
                normalized_results.append(result)

        return normalized_results
