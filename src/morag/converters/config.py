"""Configuration for universal document conversion."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for document conversion system."""
    
    # Default conversion options
    default_options: Dict[str, Any] = field(default_factory=lambda: {
        'preserve_formatting': True,
        'extract_images': True,
        'include_metadata': True,
        'chunking_strategy': 'page',
        'min_quality_threshold': 0.7,
        'enable_fallback': True,
        'include_toc': False,
        'clean_whitespace': True
    })
    
    # Format-specific options
    format_specific: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'pdf': {
            'use_docling': True,
            'use_ocr': True,
            'extract_tables': True,
            'preserve_layout': False
        },
        'audio': {
            'enable_diarization': False,
            'include_timestamps': True,
            'confidence_threshold': 0.8,
            'model': 'whisper-large-v3'
        },
        'video': {
            'extract_keyframes': True,
            'keyframe_interval': 30,
            'include_audio': True,
            'extract_scenes': True
        },
        'office': {
            'preserve_comments': False,
            'extract_embedded_objects': True,
            'convert_tables': True,
            'include_formulas': True
        },
        'web': {
            'follow_redirects': True,
            'extract_main_content': True,
            'include_navigation': False,
            'timeout': 30
        }
    })
    
    # Quality assessment settings
    quality_settings: Dict[str, Any] = field(default_factory=lambda: {
        'min_content_length': 50,
        'min_word_count': 10,
        'max_noise_ratio': 0.3,
        'enable_quality_validation': True
    })
    
    # Performance settings
    performance_settings: Dict[str, Any] = field(default_factory=lambda: {
        'max_file_size': 100 * 1024 * 1024,  # 100MB - default, overridden by type-specific limits
        'timeout_seconds': 300,  # 5 minutes
        'max_concurrent_conversions': 5,
        'enable_caching': True,
        'cache_ttl_seconds': 3600  # 1 hour
    })
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'ConversionConfig':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ConversionConfig instance
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            logger.info("Loaded conversion configuration", config_path=str(config_path))
            
            return cls(
                default_options=config_data.get('default_options', cls().default_options),
                format_specific=config_data.get('format_specific', cls().format_specific),
                quality_settings=config_data.get('quality_settings', cls().quality_settings),
                performance_settings=config_data.get('performance_settings', cls().performance_settings)
            )
            
        except FileNotFoundError:
            logger.warning("Configuration file not found, using defaults", config_path=str(config_path))
            return cls()
        except Exception as e:
            logger.error("Failed to load configuration, using defaults", error=str(e))
            return cls()
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file
        """
        try:
            config_data = {
                'default_options': self.default_options,
                'format_specific': self.format_specific,
                'quality_settings': self.quality_settings,
                'performance_settings': self.performance_settings
            }
            
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info("Saved conversion configuration", config_path=str(config_path))
            
        except Exception as e:
            logger.error("Failed to save configuration", error=str(e))
    
    def get_format_options(self, format_type: str) -> Dict[str, Any]:
        """Get options for a specific format.
        
        Args:
            format_type: Format to get options for
            
        Returns:
            Dictionary of format-specific options
        """
        return self.format_specific.get(format_type, {})
    
    def update_format_options(self, format_type: str, options: Dict[str, Any]) -> None:
        """Update options for a specific format.
        
        Args:
            format_type: Format to update options for
            options: New options to merge
        """
        if format_type not in self.format_specific:
            self.format_specific[format_type] = {}
        
        self.format_specific[format_type].update(options)
        logger.info("Updated format options", format_type=format_type, options=options)


# Default configuration instance
_default_config = ConversionConfig()

def get_conversion_config() -> ConversionConfig:
    """Get the global conversion configuration.
    
    Returns:
        ConversionConfig instance
    """
    return _default_config

def set_conversion_config(config: ConversionConfig) -> None:
    """Set the global conversion configuration.
    
    Args:
        config: New configuration to use
    """
    global _default_config
    _default_config = config
    logger.info("Updated global conversion configuration")

def load_conversion_config(config_path: Optional[Path] = None) -> ConversionConfig:
    """Load conversion configuration from file.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        ConversionConfig instance
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            Path('config/conversion.yaml'),
            Path('conversion.yaml'),
            Path.home() / '.morag' / 'conversion.yaml'
        ]
        
        for path in default_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path and config_path.exists():
        config = ConversionConfig.load_from_file(config_path)
        set_conversion_config(config)
        return config
    else:
        logger.info("No configuration file found, using defaults")
        return get_conversion_config()


# Sample configuration YAML content
SAMPLE_CONFIG_YAML = """
# Universal Document Conversion Configuration

default_options:
  preserve_formatting: true
  extract_images: true
  include_metadata: true
  chunking_strategy: "page"  # page, sentence, paragraph, semantic
  min_quality_threshold: 0.7
  enable_fallback: true
  include_toc: false
  clean_whitespace: true

format_specific:
  pdf:
    use_docling: true
    use_ocr: true
    extract_tables: true
    preserve_layout: false
    
  audio:
    enable_diarization: false
    include_timestamps: true
    confidence_threshold: 0.8
    model: "whisper-large-v3"
    
  video:
    extract_keyframes: true
    keyframe_interval: 30  # seconds
    include_audio: true
    extract_scenes: true
    
  office:
    preserve_comments: false
    extract_embedded_objects: true
    convert_tables: true
    include_formulas: true
    
  web:
    follow_redirects: true
    extract_main_content: true
    include_navigation: false
    timeout: 30

quality_settings:
  min_content_length: 50
  min_word_count: 10
  max_noise_ratio: 0.3
  enable_quality_validation: true

performance_settings:
  max_file_size: 104857600  # 100MB - default, overridden by type-specific limits
  timeout_seconds: 300  # 5 minutes
  max_concurrent_conversions: 5
  enable_caching: true
  cache_ttl_seconds: 3600  # 1 hour
"""

def create_sample_config(config_path: Path) -> None:
    """Create a sample configuration file.
    
    Args:
        config_path: Path to create the sample configuration
    """
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(SAMPLE_CONFIG_YAML)
        
        logger.info("Created sample configuration file", config_path=str(config_path))
        
    except Exception as e:
        logger.error("Failed to create sample configuration", error=str(e))
