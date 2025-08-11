"""Fact filtering system for domain-specific relevance."""

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import structlog

from ..models.fact import Fact


@dataclass
class DomainFilterConfig:
    """Configuration for domain-specific fact filtering."""
    
    required_keywords: List[str]
    excluded_keywords: List[str]
    confidence_threshold: float = 0.7
    relevance_threshold: float = 6.0
    similarity_threshold: float = 0.4
    enable_llm_scoring: bool = True
    domain_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.domain_multipliers is None:
            self.domain_multipliers = {}


class FactFilter:
    """Multi-stage fact filtering system for domain relevance."""
    
    def __init__(self, domain_configs: Optional[Dict[str, DomainFilterConfig]] = None):
        """Initialize the fact filter.

        Args:
            domain_configs: Optional dictionary of domain configurations.
                           If None, uses minimal default configuration.
        """
        self.logger = structlog.get_logger(__name__)

        # Use provided configurations or minimal defaults
        self.domain_configs = domain_configs or {
            "general": DomainFilterConfig(
                required_keywords=[],
                excluded_keywords=["advertisement", "promotion", "marketing"],
                confidence_threshold=0.5,
                relevance_threshold=4.0,
                enable_llm_scoring=False
            )
        }
    
    def add_domain_config(self, domain: str, config: DomainFilterConfig) -> None:
        """Add or update domain configuration."""
        self.domain_configs[domain] = config
        self.logger.info("Domain configuration added", domain=domain)

    def _get_default_config(self) -> DomainFilterConfig:
        """Get a minimal default configuration as fallback."""
        return DomainFilterConfig(
            required_keywords=[],
            excluded_keywords=["advertisement", "promotion", "marketing"],
            confidence_threshold=0.5,
            relevance_threshold=4.0,
            enable_llm_scoring=False
        )
    
    def filter_facts(
        self, 
        facts: List[Fact], 
        domain: str = "general",
        document_context: Optional[Dict[str, Any]] = None
    ) -> List[Fact]:
        """Filter facts using multi-stage approach.
        
        Args:
            facts: List of facts to filter
            domain: Domain context for filtering
            document_context: Additional document context
            
        Returns:
            Filtered list of facts
        """
        if not facts:
            return []
        
        config = self.domain_configs.get(domain, self.domain_configs.get("general", self._get_default_config()))
        
        self.logger.info(
            "Starting fact filtering",
            total_facts=len(facts),
            domain=domain,
            config_threshold=config.confidence_threshold
        )
        
        # Stage 1: Keyword filtering
        keyword_filtered = self._filter_by_keywords(facts, config)
        
        # Stage 2: Confidence adjustment and filtering
        confidence_filtered = self._filter_by_adjusted_confidence(keyword_filtered, config, domain)
        
        # Stage 3: Content relevance filtering
        final_filtered = self._filter_by_content_relevance(confidence_filtered, config, document_context)
        
        self.logger.info(
            "Fact filtering completed",
            original_count=len(facts),
            keyword_filtered=len(keyword_filtered),
            confidence_filtered=len(confidence_filtered),
            final_count=len(final_filtered),
            filter_ratio=len(final_filtered) / len(facts) if facts else 0
        )
        
        return final_filtered
    
    def _filter_by_keywords(self, facts: List[Fact], config: DomainFilterConfig) -> List[Fact]:
        """Filter facts based on required and excluded keywords."""
        filtered_facts = []
        
        for fact in facts:
            # Combine all fact text for keyword matching
            fact_text = f"{fact.fact_text} {' '.join(fact.keywords)}"
            fact_text_lower = fact_text.lower()
            
            # Check excluded keywords first (immediate rejection)
            if any(excluded.lower() in fact_text_lower for excluded in config.excluded_keywords):
                self.logger.debug("Fact excluded by keyword", fact_id=fact.id, reason="excluded_keyword")
                continue
            
            # Check required keywords (if any specified)
            if config.required_keywords:
                has_required = any(required.lower() in fact_text_lower for required in config.required_keywords)
                if not has_required:
                    self.logger.debug("Fact excluded by keyword", fact_id=fact.id, reason="missing_required_keyword")
                    continue
            
            filtered_facts.append(fact)
        
        return filtered_facts
    
    def _filter_by_adjusted_confidence(
        self, 
        facts: List[Fact], 
        config: DomainFilterConfig, 
        domain: str
    ) -> List[Fact]:
        """Filter facts by domain-adjusted confidence scores."""
        filtered_facts = []
        
        for fact in facts:
            # Calculate domain-adjusted confidence
            adjusted_confidence = self._calculate_domain_adjusted_confidence(fact, config, domain)
            
            if adjusted_confidence >= config.confidence_threshold:
                # Update fact confidence with adjusted value
                fact.extraction_confidence = adjusted_confidence
                filtered_facts.append(fact)
            else:
                self.logger.debug(
                    "Fact excluded by confidence",
                    fact_id=fact.id,
                    original_confidence=fact.extraction_confidence,
                    adjusted_confidence=adjusted_confidence,
                    threshold=config.confidence_threshold
                )
        
        return filtered_facts
    
    def _calculate_domain_adjusted_confidence(
        self,
        fact: Fact,
        config: DomainFilterConfig,
        domain: str
    ) -> float:
        """Calculate domain-adjusted confidence score."""
        base_confidence = fact.extraction_confidence
        
        # Apply domain-specific multipliers based on keywords
        multiplier = 1.0
        fact_keywords_lower = [kw.lower() for kw in fact.keywords]
        
        for domain_term, domain_multiplier in config.domain_multipliers.items():
            if domain_term.lower() in fact_keywords_lower:
                multiplier = max(multiplier, domain_multiplier)
        
        # Apply multiplier and ensure it doesn't exceed 1.0
        adjusted_confidence = min(1.0, base_confidence * multiplier)
        
        return adjusted_confidence
    
    def _filter_by_content_relevance(
        self, 
        facts: List[Fact], 
        config: DomainFilterConfig,
        document_context: Optional[Dict[str, Any]]
    ) -> List[Fact]:
        """Filter facts by content relevance (placeholder for future LLM integration)."""
        # For now, just return all facts that passed previous stages
        # This is where LLM-based relevance scoring would be implemented
        
        if not document_context:
            return facts
        
        # Simple heuristic: prefer facts that mention document topics
        document_topics = document_context.get('topics', [])
        if not document_topics:
            return facts
        
        scored_facts = []
        for fact in facts:
            relevance_score = self._calculate_topic_relevance(fact, document_topics)
            if relevance_score >= 0.2:  # Lower threshold for more inclusive filtering
                scored_facts.append(fact)
            else:
                self.logger.debug(
                    "Fact excluded by topic relevance",
                    fact_id=fact.id,
                    relevance_score=relevance_score
                )
        
        return scored_facts
    
    def _calculate_topic_relevance(self, fact: Fact, document_topics: List[str]) -> float:
        """Calculate simple topic relevance score."""
        fact_text = fact.fact_text.lower()
        fact_keywords = [kw.lower() for kw in fact.keywords]

        matches = 0
        for topic in document_topics:
            topic_lower = topic.lower()
            # Direct text match
            if topic_lower in fact_text:
                matches += 1
            # Keyword match
            elif any(topic_lower in keyword for keyword in fact_keywords):
                matches += 1
            # Partial matches for related terms
            elif topic_lower == 'adhd' and any(term in fact_text for term in ['attention', 'focus', 'concentration', 'hyperactivity']):
                matches += 0.8
            elif topic_lower == 'herbs' and any(term in fact_text for term in ['herbal', 'plant', 'extract', 'natural']):
                matches += 0.8
            elif 'treatment' in topic_lower and any(term in fact_text for term in ['therapy', 'medication', 'remedy', 'approach']):
                matches += 0.6

        return min(1.0, matches / len(document_topics)) if document_topics else 0.0
    
    def get_filtering_stats(self, original_facts: List[Fact], filtered_facts: List[Fact]) -> Dict[str, Any]:
        """Get filtering statistics."""
        return {
            "original_count": len(original_facts),
            "filtered_count": len(filtered_facts),
            "retention_rate": len(filtered_facts) / len(original_facts) if original_facts else 0,
            "filtered_out": len(original_facts) - len(filtered_facts)
        }
