"""LLM-based entity normalization utilities for consistent entity naming across all languages."""

import asyncio
import logging
from typing import Dict, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

class EntityNormalizer:
    """LLM-based normalizer for entity names to canonical singular, non-conjugated forms."""

    def __init__(self, llm_client=None):
        """Initialize the entity normalizer with LLM client."""
        self.llm_client = llm_client

        # Only keep essential abbreviations that are universally recognized
        self._common_abbreviations = {
            'who': 'WHO',
            'weltgesundheitsorganisation': 'WHO',
            'world health organization': 'WHO',
            'organización mundial de la salud': 'WHO',
            'organisation mondiale de la santé': 'WHO',
            'dna': 'DNA',
            'desoxyribonukleinsäure': 'DNA',
            'ácido desoxirribonucleico': 'DNA',
            'acide désoxyribonucléique': 'DNA',
            'rna': 'RNA',
            'ribonukleinsäure': 'RNA',
            'ácido ribonucleico': 'RNA',
            'acide ribonucléique': 'RNA',
            'adhs': 'ADHS',
            'aufmerksamkeitsdefizit-hyperaktivitätsstörung': 'ADHS',
            'adhd': 'ADHD',
            'attention deficit hyperactivity disorder': 'ADHD',
            'trastorno por déficit de atención e hiperactividad': 'ADHD',
            'trouble du déficit de l\'attention avec hyperactivité': 'ADHD',
            'covid': 'COVID',
            'sars': 'SARS',
            'hiv': 'HIV',
            'aids': 'AIDS',
            'usa': 'USA',
            'estados unidos': 'USA',
            'états-unis': 'USA',
            'eu': 'EU',
            'europäische union': 'EU',
            'european union': 'EU',
            'unión europea': 'EU',
            'union européenne': 'EU',
        }

    async def normalize_entity_name(self, name: str, language: Optional[str] = None) -> str:
        """
        Normalize entity name to canonical singular, non-conjugated form using LLM.

        Args:
            name: The entity name to normalize
            language: Optional language hint ('de', 'en', 'es', 'fr', etc.)

        Returns:
            Normalized entity name in canonical form
        """
        if not name or not name.strip():
            return name

        # Clean and prepare the name
        original_name = name.strip()
        normalized_key = original_name.lower().strip()

        # Handle common abbreviations first (these are universal)
        if normalized_key in self._common_abbreviations:
            return self._common_abbreviations[normalized_key]

        # Remove obvious contextual information before LLM processing
        cleaned_name = self._remove_obvious_context(original_name)

        # Use LLM for intelligent normalization
        if self.llm_client:
            try:
                normalized = await self._llm_normalize(cleaned_name, language)
                return normalized
            except Exception as e:
                logger.warning(f"LLM normalization failed for '{name}': {e}")
                # Fall back to basic cleanup
                return self._basic_cleanup(cleaned_name)
        else:
            # No LLM available, use basic cleanup
            return self._basic_cleanup(cleaned_name)

    async def _llm_normalize(self, name: str, language: Optional[str] = None) -> str:
        """Use LLM to normalize entity name intelligently."""
        language_hint = f" (language: {language})" if language else ""

        prompt = f"""Normalize the following entity name to its canonical form{language_hint}:

Entity: "{name}"

Requirements:
1. Convert to SINGULAR form (not plural)
2. Use UNCONJUGATED base form (not inflected)
3. For gendered languages, use MASCULINE/NEUTRAL form when applicable
4. Preserve the original language but use canonical form
5. Remove any contextual or positional information
6. Keep proper nouns capitalized, common nouns lowercase
7. Resolve common abbreviations to their standard form when appropriate

Examples:
- "Hunde" → "Hund" (German plural to singular)
- "Pilotinnen" → "Pilot" (German feminine to masculine)
- "children" → "child" (English plural to singular)
- "médicos" → "médico" (Spanish plural to singular)
- "protein in cells" → "protein" (remove context)

Return only the normalized entity name, nothing else."""

        # This would use your LLM client to get the response
        # For now, return the basic cleanup as fallback
        response = await self._call_llm(prompt)
        return response.strip() if response else self._basic_cleanup(name)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for entity normalization."""
        if not self.llm_client:
            raise ValueError("No LLM client available")

        # This would integrate with your existing LLM infrastructure
        # For now, this is a placeholder that would need to be implemented
        # based on your specific LLM client setup
        try:
            response = await self.llm_client.complete(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _remove_obvious_context(self, name: str) -> str:
        """Remove obvious contextual information before LLM processing."""
        import re

        # Remove parenthetical and bracketed information
        name = re.sub(r'\s*\([^)]+\)', '', name)
        name = re.sub(r'\s*\[[^\]]+\]', '', name)

        # Remove common contextual phrases (basic patterns only)
        contextual_patterns = [
            r'\s+in\s+\w+$',      # "protein in cells" -> "protein"
            r'\s+from\s+\w+$',    # "data from study" -> "data"
            r'\s+of\s+\w+$',      # "analysis of results" -> "analysis"
            r'\s+bei\s+\w+$',     # German: "Protein bei Patienten" -> "Protein"
            r'\s+von\s+\w+$',     # German: "Analyse von Daten" -> "Analyse"
            r'\s+de\s+\w+$',      # Spanish: "análisis de datos" -> "análisis"
            r'\s+du\s+\w+$',      # French: "analyse du système" -> "analyse"
        ]

        for pattern in contextual_patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)

        return name.strip()

    def _basic_cleanup(self, name: str) -> str:
        """Basic cleanup when LLM is not available."""
        import re

        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name.strip())

        # Basic capitalization (first letter of each word for proper nouns)
        if name and name[0].isupper():
            name = ' '.join(word.capitalize() for word in name.split())

        return name

    async def are_entities_equivalent(self, name1: str, name2: str, language: Optional[str] = None) -> bool:
        """
        Check if two entity names refer to the same canonical entity using LLM.

        Args:
            name1: First entity name
            name2: Second entity name
            language: Optional language hint

        Returns:
            True if the entities are equivalent after normalization
        """
        norm1 = await self.normalize_entity_name(name1, language)
        norm2 = await self.normalize_entity_name(name2, language)

        return norm1.lower() == norm2.lower()

    def normalize_entity_name_sync(self, name: str, language: Optional[str] = None) -> str:
        """
        Synchronous version for backward compatibility - uses basic cleanup only.

        For full LLM-based normalization, use the async normalize_entity_name method.
        """
        if not name or not name.strip():
            return name

        original_name = name.strip()
        normalized_key = original_name.lower().strip()

        # Handle common abbreviations
        if normalized_key in self._common_abbreviations:
            return self._common_abbreviations[normalized_key]

        # Basic cleanup
        cleaned_name = self._remove_obvious_context(original_name)
        return self._basic_cleanup(cleaned_name)
