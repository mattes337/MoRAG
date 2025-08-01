"""Configuration management for entity normalization."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class NormalizationRulesConfig:
    """Configuration for normalization rules."""
    acronym_expansion: bool = True
    person_name_standardization: bool = True
    organization_cleanup: bool = True
    general_lowercasing: bool = False
    remove_articles: bool = True
    handle_plurals: bool = True
    handle_conjugations: bool = True


@dataclass
class MergeThresholdsConfig:
    """Configuration for merge confidence thresholds by entity type."""
    acronym: float = 0.9
    person: float = 0.8
    organization: float = 0.85
    location: float = 0.9
    general: float = 0.95


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and metrics."""
    enabled: bool = True
    track_effectiveness: bool = True
    log_decisions: bool = False
    metrics: List[str] = field(default_factory=lambda: [
        "entities_processed",
        "entities_normalized", 
        "merge_candidates_found",
        "merges_applied",
        "normalization_rules_used",
        "average_confidence",
        "processing_time",
        "error_rate"
    ])


@dataclass
class CacheConfig:
    """Configuration for normalization caching."""
    enabled: bool = True
    max_size: int = 10000
    ttl: int = 3600  # 1 hour
    key_strategy: str = "entity_language_type"


@dataclass
class PerformanceConfig:
    """Configuration for performance tuning."""
    max_concurrent_tasks: int = 5
    llm_timeout: int = 30
    max_retries: int = 3
    retry_backoff: str = "exponential"


@dataclass
class EntityNormalizationConfig:
    """Complete configuration for entity normalization."""
    enabled: bool = True
    confidence_threshold: float = 0.8
    merge_confidence_threshold: float = 0.8
    batch_size: int = 20
    min_confidence: float = 0.7
    enable_llm_normalization: bool = True
    enable_rule_based_fallback: bool = True
    supported_languages: List[str] = field(default_factory=lambda: ["en", "es", "de", "fr"])
    
    # Sub-configurations
    rules: NormalizationRulesConfig = field(default_factory=NormalizationRulesConfig)
    merge_thresholds: MergeThresholdsConfig = field(default_factory=MergeThresholdsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Custom mappings
    custom_acronyms: Dict[str, str] = field(default_factory=dict)
    language_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class NormalizationConfigLoader:
    """Loader for entity normalization configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration loader.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config_cache: Optional[EntityNormalizationConfig] = None
    
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path."""
        # Look for config file in several locations
        possible_paths = [
            Path(__file__).parent / "entity_normalization.yaml",
            Path.cwd() / "config" / "entity_normalization.yaml",
            Path.cwd() / "entity_normalization.yaml",
            Path(os.getenv("MORAG_CONFIG_PATH", "")) / "entity_normalization.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Return the default path even if it doesn't exist
        return Path(__file__).parent / "entity_normalization.yaml"
    
    def load_config(self, reload: bool = False) -> EntityNormalizationConfig:
        """Load configuration from file.
        
        Args:
            reload: Whether to reload configuration from file
            
        Returns:
            EntityNormalizationConfig instance
        """
        if self._config_cache is not None and not reload:
            return self._config_cache
        
        try:
            if self.config_path.exists():
                logger.info("Loading entity normalization config", path=str(self.config_path))
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # Extract entity_normalization section
                norm_config = config_data.get('entity_normalization', {})
                
                # Create configuration object
                self._config_cache = self._create_config_from_dict(norm_config)
                
            else:
                logger.warning(
                    "Configuration file not found, using defaults",
                    path=str(self.config_path)
                )
                self._config_cache = EntityNormalizationConfig()
            
            return self._config_cache
            
        except Exception as e:
            logger.error(
                "Failed to load configuration, using defaults",
                error=str(e),
                path=str(self.config_path)
            )
            return EntityNormalizationConfig()
    
    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> EntityNormalizationConfig:
        """Create configuration object from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            EntityNormalizationConfig instance
        """
        # Extract sub-configurations
        rules_config = NormalizationRulesConfig(**config_dict.get('rules', {}))
        merge_thresholds_config = MergeThresholdsConfig(**config_dict.get('merge_thresholds', {}))
        monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))
        cache_config = CacheConfig(**config_dict.get('cache', {}))
        performance_config = PerformanceConfig(**config_dict.get('performance', {}))
        
        # Create main configuration
        main_config_dict = {k: v for k, v in config_dict.items() 
                           if k not in ['rules', 'merge_thresholds', 'monitoring', 'cache', 'performance']}
        
        return EntityNormalizationConfig(
            rules=rules_config,
            merge_thresholds=merge_thresholds_config,
            monitoring=monitoring_config,
            cache=cache_config,
            performance=performance_config,
            **main_config_dict
        )
    
    def save_config(self, config: EntityNormalizationConfig) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
        """
        try:
            # Convert to dictionary
            config_dict = {
                'entity_normalization': self._config_to_dict(config)
            }
            
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info("Configuration saved", path=str(self.config_path))
            
        except Exception as e:
            logger.error("Failed to save configuration", error=str(e))
            raise
    
    def _config_to_dict(self, config: EntityNormalizationConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary.
        
        Args:
            config: Configuration object
            
        Returns:
            Configuration dictionary
        """
        # Convert dataclass to dict, handling nested dataclasses
        result = {}
        
        for field_name, field_value in config.__dict__.items():
            if hasattr(field_value, '__dict__'):
                # Nested dataclass
                result[field_name] = field_value.__dict__
            else:
                result[field_name] = field_value
        
        return result
    
    def get_config_for_component(self, component: str) -> Dict[str, Any]:
        """Get configuration dictionary for a specific component.
        
        Args:
            component: Component name (e.g., 'normalizer', 'deduplicator')
            
        Returns:
            Configuration dictionary for the component
        """
        config = self.load_config()
        
        if component == 'normalizer':
            return {
                'enable_llm_normalization': config.enable_llm_normalization,
                'enable_rule_based_fallback': config.enable_rule_based_fallback,
                'supported_languages': config.supported_languages,
                'batch_size': config.batch_size,
                'min_confidence': config.min_confidence,
                'confidence_threshold': config.confidence_threshold,
                'custom_acronyms': config.custom_acronyms,
                'language_patterns': config.language_patterns,
                'rules': config.rules.__dict__,
                'cache': config.cache.__dict__,
                'performance': config.performance.__dict__
            }
        
        elif component == 'deduplicator':
            return {
                'merge_confidence_threshold': config.merge_confidence_threshold,
                'batch_size': config.batch_size,
                'merge_thresholds': config.merge_thresholds.__dict__,
                'monitoring': config.monitoring.__dict__,
                'performance': config.performance.__dict__
            }
        
        else:
            # Return full config as dict
            return self._config_to_dict(config)


# Global configuration loader instance
_config_loader: Optional[NormalizationConfigLoader] = None


def get_normalization_config(reload: bool = False) -> EntityNormalizationConfig:
    """Get the global entity normalization configuration.
    
    Args:
        reload: Whether to reload configuration from file
        
    Returns:
        EntityNormalizationConfig instance
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = NormalizationConfigLoader()
    
    return _config_loader.load_config(reload=reload)


def get_config_for_component(component: str) -> Dict[str, Any]:
    """Get configuration for a specific component.
    
    Args:
        component: Component name
        
    Returns:
        Configuration dictionary
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = NormalizationConfigLoader()
    
    return _config_loader.get_config_for_component(component)
