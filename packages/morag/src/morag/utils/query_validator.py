"""Query validation utilities for enhanced API."""

import re
from typing import Optional, List
from dataclasses import dataclass
import structlog

from morag.models.enhanced_query import EnhancedQueryRequest, QueryType

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of query validation."""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class QueryValidator:
    """Validator for enhanced query requests."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        
        # Validation rules
        self.min_query_length = 3
        self.max_query_length = 1000
        self.forbidden_patterns = [
            r'<script.*?>.*?</script>',  # Script injection
            r'javascript:',              # JavaScript URLs
            r'data:.*base64',           # Base64 data URLs
        ]
        
        # Entity/relation type validation patterns
        self.valid_entity_type_pattern = r'^[a-zA-Z][a-zA-Z0-9_]*$'
        self.valid_relation_type_pattern = r'^[a-zA-Z][a-zA-Z0-9_]*$'
    
    async def validate_query_request(self, request: EnhancedQueryRequest) -> ValidationResult:
        """Validate an enhanced query request.
        
        Args:
            request: The query request to validate
            
        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        errors = []
        warnings = []
        
        # Basic query validation
        query_validation = self._validate_query_text(request.query)
        if not query_validation.is_valid:
            errors.append(query_validation.error_message)
        warnings.extend(query_validation.warnings)
        
        # Parameter validation
        param_validation = self._validate_parameters(request)
        if not param_validation.is_valid:
            errors.append(param_validation.error_message)
        warnings.extend(param_validation.warnings)
        
        # Type-specific validation
        type_validation = self._validate_query_type_compatibility(request)
        if not type_validation.is_valid:
            errors.append(type_validation.error_message)
        warnings.extend(type_validation.warnings)
        
        # Filter validation
        filter_validation = self._validate_filters(request)
        if not filter_validation.is_valid:
            errors.append(filter_validation.error_message)
        warnings.extend(filter_validation.warnings)
        
        # Performance validation
        perf_validation = self._validate_performance_parameters(request)
        warnings.extend(perf_validation.warnings)
        
        if errors:
            return ValidationResult(
                is_valid=False,
                error_message="; ".join(errors),
                warnings=warnings
            )
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings
        )
    
    def _validate_query_text(self, query: str) -> ValidationResult:
        """Validate the query text itself."""
        errors = []
        warnings = []
        
        # Length validation
        if len(query) < self.min_query_length:
            errors.append(f"Query too short (minimum {self.min_query_length} characters)")
        
        if len(query) > self.max_query_length:
            errors.append(f"Query too long (maximum {self.max_query_length} characters)")
        
        # Security validation
        for pattern in self.forbidden_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                errors.append("Query contains potentially unsafe content")
                break
        
        # Content quality warnings
        if query.strip() != query:
            warnings.append("Query has leading/trailing whitespace")
        
        if len(query.split()) < 2:
            warnings.append("Very short query may produce limited results")
        
        if query.isupper():
            warnings.append("All-caps query may affect search quality")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings
        )
    
    def _validate_parameters(self, request: EnhancedQueryRequest) -> ValidationResult:
        """Validate request parameters."""
        errors = []
        warnings = []
        
        # Expansion depth vs strategy compatibility
        if request.expansion_strategy == "none" and request.expansion_depth > 1:
            warnings.append("Expansion depth ignored when strategy is 'none'")
        
        if request.expansion_strategy == "direct_neighbors" and request.expansion_depth > 1:
            warnings.append("Expansion depth > 1 not used with 'direct_neighbors' strategy")
        
        # Fusion strategy compatibility
        if request.fusion_strategy == "vector_only" and request.include_graph_context:
            warnings.append("Graph context may be limited with 'vector_only' fusion")
        
        if request.fusion_strategy == "graph_only" and not request.enable_multi_hop:
            warnings.append("Graph-only fusion works best with multi-hop enabled")
        
        # Performance warnings
        if request.max_results > 50:
            warnings.append("Large result sets may impact performance")
        
        if request.expansion_depth > 3:
            warnings.append("Deep expansion may significantly impact performance")
        
        if request.timeout_seconds < 10:
            warnings.append("Short timeout may cause incomplete results")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings
        )
    
    def _validate_query_type_compatibility(self, request: EnhancedQueryRequest) -> ValidationResult:
        """Validate query type compatibility with other parameters."""
        warnings = []
        
        if request.query_type == QueryType.SIMPLE:
            if request.enable_multi_hop:
                warnings.append("Multi-hop reasoning not typically used with simple queries")
            if request.include_reasoning_path:
                warnings.append("Reasoning path not typically included for simple queries")
        
        elif request.query_type == QueryType.ENTITY_FOCUSED:
            if request.expansion_strategy == "none":
                warnings.append("Entity-focused queries benefit from context expansion")
        
        elif request.query_type == QueryType.MULTI_HOP:
            if not request.enable_multi_hop:
                warnings.append("Multi-hop query type should have multi-hop reasoning enabled")
            if request.expansion_depth < 2:
                warnings.append("Multi-hop queries typically benefit from deeper expansion")
        
        return ValidationResult(is_valid=True, warnings=warnings)
    
    def _validate_filters(self, request: EnhancedQueryRequest) -> ValidationResult:
        """Validate entity and relation type filters."""
        errors = []
        warnings = []
        
        # Entity type validation
        if request.entity_types:
            for entity_type in request.entity_types:
                if not re.match(self.valid_entity_type_pattern, entity_type):
                    errors.append(f"Invalid entity type format: {entity_type}")
            
            if len(request.entity_types) > 10:
                warnings.append("Many entity type filters may limit results")
        
        # Relation type validation
        if request.relation_types:
            for relation_type in request.relation_types:
                if not re.match(self.valid_relation_type_pattern, relation_type):
                    errors.append(f"Invalid relation type format: {relation_type}")
            
            if len(request.relation_types) > 10:
                warnings.append("Many relation type filters may limit results")
        
        # Time range validation
        if request.time_range:
            if "start" in request.time_range and "end" in request.time_range:
                if request.time_range["start"] >= request.time_range["end"]:
                    errors.append("Invalid time range: start must be before end")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings
        )
    
    def _validate_performance_parameters(self, request: EnhancedQueryRequest) -> ValidationResult:
        """Validate parameters that affect performance."""
        warnings = []
        
        # Calculate complexity score
        complexity_score = 0
        complexity_score += request.max_results * 0.1
        complexity_score += request.expansion_depth * 2
        complexity_score += len(request.entity_types or []) * 0.5
        complexity_score += len(request.relation_types or []) * 0.5
        
        if request.enable_multi_hop:
            complexity_score += 3
        if request.include_reasoning_path:
            complexity_score += 2
        if request.include_graph_context:
            complexity_score += 1
        
        if complexity_score > 20:
            warnings.append("High complexity query may take longer to process")
        elif complexity_score > 30:
            warnings.append("Very high complexity query may timeout")
        
        return ValidationResult(is_valid=True, warnings=warnings)
