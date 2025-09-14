"""Configuration validation for maintenance jobs.

Provides comprehensive validation for all maintenance job configurations
with helpful error messages and recommendations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    def __bool__(self) -> bool:
        return self.is_valid


class MaintenanceConfigValidator:
    """Validates configuration for all maintenance jobs."""
    
    # Configuration schemas with validation rules
    SCHEMAS = {
        'keyword_deduplication': {
            'similarity_threshold': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.75,
                'description': 'Minimum similarity score for merge candidates'
            },
            'max_cluster_size': {
                'type': int,
                'min': 2,
                'max': 20,
                'default': 8,
                'description': 'Maximum entities per merge cluster'
            },
            'batch_size': {
                'type': int,
                'min': 1,
                'max': 1000,
                'default': 50,
                'description': 'Batch size for merge operations'
            },
            'limit_entities': {
                'type': int,
                'min': 1,
                'max': 10000,
                'default': 100,
                'description': 'Maximum entities to process per run'
            },
            'semantic_similarity_weight': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.6,
                'description': 'Weight for embedding-based similarity vs string similarity'
            },
            'preserve_high_confidence': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.95,
                'description': 'Don\'t merge entities with confidence above this threshold'
            }
        },
        
        'keyword_hierarchization': {
            'threshold_min_facts': {
                'type': int,
                'min': 5,
                'max': 1000,
                'default': 50,
                'description': 'Minimum facts required for hierarchization candidate'
            },
            'min_new_keywords': {
                'type': int,
                'min': 1,
                'max': 10,
                'default': 3,
                'description': 'Minimum new keywords to propose'
            },
            'max_new_keywords': {
                'type': int,
                'min': 2,
                'max': 20,
                'default': 6,
                'description': 'Maximum new keywords to propose'
            },
            'max_move_ratio': {
                'type': float,
                'min': 0.1,
                'max': 0.95,
                'default': 0.8,
                'description': 'Maximum ratio of facts to move from original keyword'
            },
            'min_per_new_keyword': {
                'type': int,
                'min': 1,
                'max': 50,
                'default': 5,
                'description': 'Minimum facts required per new keyword to keep it'
            },
            'cooccurrence_share': {
                'type': float,
                'min': 0.05,
                'max': 0.5,
                'default': 0.18,
                'description': 'Minimum co-occurrence share for keyword proposals'
            }
        },
        
        'keyword_linking': {
            'cooccurrence_min_share': {
                'type': float,
                'min': 0.05,
                'max': 0.5,
                'default': 0.18,
                'description': 'Minimum co-occurrence share for link candidates'
            },
            'limit_parents': {
                'type': int,
                'min': 1,
                'max': 100,
                'default': 10,
                'description': 'Number of parent keywords to process per run'
            },
            'max_links_per_parent': {
                'type': int,
                'min': 1,
                'max': 20,
                'default': 6,
                'description': 'Maximum links to create per parent keyword'
            },
            'batch_size': {
                'type': int,
                'min': 1,
                'max': 1000,
                'default': 200,
                'description': 'Batch size for link creation operations'
            }
        },
        
        'relationship_cleanup': {
            'batch_size': {
                'type': int,
                'min': 1,
                'max': 1000,
                'default': 100,
                'description': 'Batch size for cleanup operations'
            },
            'min_confidence': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.3,
                'description': 'Minimum confidence threshold for relationships'
            },
            'similarity_threshold': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.85,
                'description': 'Threshold for semantic similarity merging'
            }
        },
        
        'relationship_merger': {
            'similarity_threshold': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.8,
                'description': 'Threshold for relationship similarity merging'
            },
            'batch_size': {
                'type': int,
                'min': 1,
                'max': 1000,
                'default': 100,
                'description': 'Batch size for merge operations'
            },
            'limit_relations': {
                'type': int,
                'min': 1,
                'max': 10000,
                'default': 1000,
                'description': 'Maximum relationships to process per run'
            },
            'min_confidence': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.5,
                'description': 'Minimum confidence for relationship merging'
            }
        }
    }
    
    @classmethod
    def validate_job_config(cls, job_name: str, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration for a specific maintenance job."""
        if job_name not in cls.SCHEMAS:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unknown job name: {job_name}"],
                warnings=[],
                recommendations=[]
            )
        
        schema = cls.SCHEMAS[job_name]
        errors = []
        warnings = []
        recommendations = []
        
        # Validate each configuration parameter
        for param_name, param_schema in schema.items():
            value = config.get(param_name)
            
            # Check if required parameter is missing
            if value is None:
                if 'default' in param_schema:
                    recommendations.append(
                        f"{param_name}: Using default value {param_schema['default']} "
                        f"({param_schema['description']})"
                    )
                    continue
                else:
                    errors.append(f"{param_name}: Required parameter is missing")
                    continue
            
            # Validate parameter type
            expected_type = param_schema['type']
            if not isinstance(value, expected_type):
                errors.append(
                    f"{param_name}: Expected {expected_type.__name__}, got {type(value).__name__}"
                )
                continue
            
            # Validate parameter range
            if 'min' in param_schema and value < param_schema['min']:
                errors.append(
                    f"{param_name}: Value {value} is below minimum {param_schema['min']}"
                )
            
            if 'max' in param_schema and value > param_schema['max']:
                errors.append(
                    f"{param_name}: Value {value} is above maximum {param_schema['max']}"
                )
            
            # Generate warnings for potentially problematic values
            cls._check_parameter_warnings(param_name, value, param_schema, warnings)
        
        # Check for unknown parameters
        unknown_params = set(config.keys()) - set(schema.keys()) - {'dry_run', 'job_tag'}
        if unknown_params:
            warnings.extend([f"Unknown parameter: {param}" for param in unknown_params])
        
        # Generate job-specific recommendations
        job_recommendations = cls._get_job_specific_recommendations(job_name, config)
        recommendations.extend(job_recommendations)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    @classmethod
    def _check_parameter_warnings(
        cls, 
        param_name: str, 
        value: Any, 
        schema: Dict[str, Any], 
        warnings: List[str]
    ):
        """Check for parameter values that might cause issues."""
        # Check for values at extremes
        if 'min' in schema and 'max' in schema:
            range_size = schema['max'] - schema['min']
            
            # Warn if value is very close to minimum
            if value <= schema['min'] + range_size * 0.1:
                warnings.append(
                    f"{param_name}: Value {value} is very close to minimum, "
                    f"consider increasing for better results"
                )
            
            # Warn if value is very close to maximum
            if value >= schema['max'] - range_size * 0.1:
                warnings.append(
                    f"{param_name}: Value {value} is very close to maximum, "
                    f"consider decreasing to avoid issues"
                )
    
    @classmethod
    def _get_job_specific_recommendations(cls, job_name: str, config: Dict[str, Any]) -> List[str]:
        """Get job-specific configuration recommendations."""
        recommendations = []
        
        if job_name == 'keyword_deduplication':
            similarity_threshold = config.get('similarity_threshold', 0.75)
            if similarity_threshold > 0.9:
                recommendations.append(
                    "High similarity threshold may miss valid merges. "
                    "Consider starting with 0.75-0.85 for initial runs."
                )
            
            max_cluster_size = config.get('max_cluster_size', 8)
            if max_cluster_size > 15:
                recommendations.append(
                    "Large cluster sizes may create overly broad merges. "
                    "Consider keeping cluster size under 10 for quality."
                )
        
        elif job_name == 'keyword_hierarchization':
            min_facts = config.get('threshold_min_facts', 50)
            max_move_ratio = config.get('max_move_ratio', 0.8)
            
            if min_facts < 20:
                recommendations.append(
                    "Low fact threshold may create too many hierarchies. "
                    "Consider starting with 30-50 facts for stable results."
                )
            
            if max_move_ratio > 0.9:
                recommendations.append(
                    "High move ratio may leave original keywords with too few facts. "
                    "Consider keeping move ratio under 0.8."
                )
        
        elif job_name == 'keyword_linking':
            min_share = config.get('cooccurrence_min_share', 0.18)
            max_links = config.get('max_links_per_parent', 6)
            
            if min_share < 0.1:
                recommendations.append(
                    "Low co-occurrence threshold may create weak links. "
                    "Consider using 0.15-0.25 for meaningful relationships."
                )
            
            if max_links > 10:
                recommendations.append(
                    "Too many links per parent may create graph clutter. "
                    "Consider limiting to 5-8 links for focused relationships."
                )
        
        return recommendations
    
    @classmethod
    def get_recommended_config(cls, job_name: str) -> Dict[str, Any]:
        """Get recommended configuration for a job."""
        if job_name not in cls.SCHEMAS:
            return {}
        
        schema = cls.SCHEMAS[job_name]
        return {
            param_name: param_schema.get('default')
            for param_name, param_schema in schema.items()
            if 'default' in param_schema
        }
    
    @classmethod
    def validate_all_configs(cls, configs: Dict[str, Dict[str, Any]]) -> Dict[str, ValidationResult]:
        """Validate configurations for multiple jobs."""
        results = {}
        
        for job_name, config in configs.items():
            results[job_name] = cls.validate_job_config(job_name, config)
        
        return results
