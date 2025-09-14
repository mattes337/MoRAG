"""Configuration utilities for fact filtering."""

from typing import Dict, Any, Optional
from .fact_filter import DomainFilterConfig


class FactFilterConfigBuilder:
    """Builder for creating domain-specific fact filter configurations."""
    
    @staticmethod
    def create_medical_config(
        confidence_threshold: float = 0.6,
        excluded_keywords: Optional[list] = None,
        language: str = "en"
    ) -> DomainFilterConfig:
        """Create configuration for medical domain filtering.
        
        Args:
            confidence_threshold: Minimum confidence threshold
            excluded_keywords: Additional keywords to exclude
            language: Language code for localized keywords
            
        Returns:
            DomainFilterConfig for medical domain
        """
        base_excluded = ["advertisement", "promotion", "sale"]
        if language == "de":
            base_excluded.extend(["werbung", "verkauf", "anzeige"])
        
        if excluded_keywords:
            base_excluded.extend(excluded_keywords)
        
        return DomainFilterConfig(
            required_keywords=[],
            excluded_keywords=base_excluded,
            confidence_threshold=confidence_threshold,
            relevance_threshold=5.0,
            domain_multipliers={
                "medical": 1.3,
                "clinical": 1.2,
                "therapeutic": 1.2,
                "treatment": 1.2
            }
        )
    
    @staticmethod
    def create_herbal_config(
        confidence_threshold: float = 0.7,
        excluded_keywords: Optional[list] = None,
        language: str = "en"
    ) -> DomainFilterConfig:
        """Create configuration for herbal/natural medicine domain filtering.
        
        Args:
            confidence_threshold: Minimum confidence threshold
            excluded_keywords: Additional keywords to exclude
            language: Language code for localized keywords
            
        Returns:
            DomainFilterConfig for herbal domain
        """
        base_excluded = ["newsletter", "subscription", "marketing", "advertisement"]
        if language == "de":
            base_excluded.extend(["werbung", "anmeldung", "newsletter", "abonnement"])
        
        if excluded_keywords:
            base_excluded.extend(excluded_keywords)
        
        return DomainFilterConfig(
            required_keywords=[],
            excluded_keywords=base_excluded,
            confidence_threshold=confidence_threshold,
            relevance_threshold=6.0,
            domain_multipliers={
                "herbal": 1.3,
                "natural": 1.2,
                "plant": 1.2,
                "botanical": 1.2,
                "remedy": 1.1
            }
        )
    
    @staticmethod
    def create_adhd_config(
        confidence_threshold: float = 0.7,
        excluded_keywords: Optional[list] = None,
        language: str = "en"
    ) -> DomainFilterConfig:
        """Create configuration for ADHD domain filtering.
        
        Args:
            confidence_threshold: Minimum confidence threshold
            excluded_keywords: Additional keywords to exclude
            language: Language code for localized keywords
            
        Returns:
            DomainFilterConfig for ADHD domain
        """
        base_excluded = ["newsletter", "subscription", "marketing", "advertisement"]
        if language == "de":
            base_excluded.extend(["werbung", "anmeldung", "newsletter"])
        
        if excluded_keywords:
            base_excluded.extend(excluded_keywords)
        
        return DomainFilterConfig(
            required_keywords=[],
            excluded_keywords=base_excluded,
            confidence_threshold=confidence_threshold,
            relevance_threshold=6.0,
            domain_multipliers={
                "adhd": 1.4,
                "attention": 1.3,
                "focus": 1.3,
                "concentration": 1.2,
                "hyperactivity": 1.2
            }
        )
    
    @staticmethod
    def create_general_config(
        confidence_threshold: float = 0.5,
        excluded_keywords: Optional[list] = None,
        language: str = "en"
    ) -> DomainFilterConfig:
        """Create configuration for general domain filtering.
        
        Args:
            confidence_threshold: Minimum confidence threshold
            excluded_keywords: Additional keywords to exclude
            language: Language code for localized keywords
            
        Returns:
            DomainFilterConfig for general domain
        """
        base_excluded = ["advertisement", "promotion", "marketing"]
        if language == "de":
            base_excluded.extend(["werbung", "anzeige"])
        
        if excluded_keywords:
            base_excluded.extend(excluded_keywords)
        
        return DomainFilterConfig(
            required_keywords=[],
            excluded_keywords=base_excluded,
            confidence_threshold=confidence_threshold,
            relevance_threshold=4.0,
            enable_llm_scoring=False
        )
    
    @staticmethod
    def create_custom_config(
        required_keywords: list = None,
        excluded_keywords: list = None,
        confidence_threshold: float = 0.3,
        relevance_threshold: float = 5.0,
        domain_multipliers: Dict[str, float] = None,
        enable_llm_scoring: bool = True
    ) -> DomainFilterConfig:
        """Create a custom domain filter configuration.
        
        Args:
            required_keywords: Keywords that must be present
            excluded_keywords: Keywords that exclude facts
            confidence_threshold: Minimum confidence threshold
            relevance_threshold: Minimum relevance threshold
            domain_multipliers: Confidence multipliers by domain term
            enable_llm_scoring: Whether to enable LLM-based scoring
            
        Returns:
            Custom DomainFilterConfig
        """
        return DomainFilterConfig(
            required_keywords=required_keywords or [],
            excluded_keywords=excluded_keywords or [],
            confidence_threshold=confidence_threshold,
            relevance_threshold=relevance_threshold,
            domain_multipliers=domain_multipliers or {},
            enable_llm_scoring=enable_llm_scoring
        )


def create_domain_configs_for_language(language: str = "en") -> Dict[str, DomainFilterConfig]:
    """Create a standard set of domain configurations for a specific language.
    
    Args:
        language: Language code (e.g., 'en', 'de')
        
    Returns:
        Dictionary of domain configurations
    """
    builder = FactFilterConfigBuilder()
    
    return {
        "general": builder.create_general_config(language=language),
        "medical": builder.create_medical_config(language=language),
        "herbal": builder.create_herbal_config(language=language),
        "adhd": builder.create_adhd_config(language=language),
        "adhd_herbal": builder.create_herbal_config(
            confidence_threshold=0.7,
            language=language
        )
    }


def load_config_from_dict(config_dict: Dict[str, Dict[str, Any]]) -> Dict[str, DomainFilterConfig]:
    """Load domain filter configurations from a dictionary.
    
    Args:
        config_dict: Dictionary with domain names as keys and config parameters as values
        
    Returns:
        Dictionary of DomainFilterConfig objects
    """
    configs = {}
    builder = FactFilterConfigBuilder()
    
    for domain, params in config_dict.items():
        configs[domain] = builder.create_custom_config(**params)
    
    return configs
