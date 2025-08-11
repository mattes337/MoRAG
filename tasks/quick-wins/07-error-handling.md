# Quick Win 7: Improved Error Handling and Logging

## Overview

**Priority**: 📋 **Planned** (1 week, Low Impact, Medium ROI)  
**Source**: General system engineering best practices  
**Expected Impact**: Better system reliability and debugging capabilities

## Problem Statement

MoRAG currently has basic error handling but lacks:
- Detailed logging for entity extraction failures
- Fallback strategies when graph traversal fails
- Data quality metrics tracking (entity coverage, relation density)
- Comprehensive error categorization and recovery
- Performance monitoring and alerting
- User-friendly error messages

This makes debugging difficult and reduces system reliability when components fail.

## Solution Overview

Implement comprehensive error handling with detailed logging, graceful degradation strategies, data quality monitoring, and user-friendly error reporting to improve system reliability and debugging capabilities.

## Technical Implementation

### 1. Enhanced Error Handling Framework

Create `packages/morag-core/src/morag_core/error_handling/error_manager.py`:

```python
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback
import logging
import asyncio
from collections import defaultdict

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    EXTRACTION = "extraction"
    GRAPH_TRAVERSAL = "graph_traversal"
    EMBEDDING = "embedding"
    DATABASE = "database"
    API = "api"
    VALIDATION = "validation"
    PERFORMANCE = "performance"

@dataclass
class ErrorContext:
    component: str
    operation: str
    input_data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    collection_name: Optional[str] = None

@dataclass
class MoragError:
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    technical_details: str
    context: ErrorContext
    timestamp: datetime
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    user_message: Optional[str] = None

@dataclass
class FallbackStrategy:
    name: str
    description: str
    handler: Callable
    conditions: List[str]
    priority: int

class ErrorManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history = []
        self.error_stats = defaultdict(int)
        self.fallback_strategies = {}
        self.recovery_handlers = {}
        
        # Initialize fallback strategies
        self._register_fallback_strategies()
        
    def handle_error(self, 
                    error: Exception, 
                    category: ErrorCategory,
                    severity: ErrorSeverity,
                    context: ErrorContext,
                    user_message: Optional[str] = None) -> MoragError:
        """Handle an error with comprehensive logging and recovery."""
        
        error_id = self._generate_error_id()
        
        morag_error = MoragError(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            technical_details=self._extract_technical_details(error),
            context=context,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            user_message=user_message or self._generate_user_message(category, severity)
        )
        
        # Log the error
        self._log_error(morag_error)
        
        # Update statistics
        self._update_error_stats(morag_error)
        
        # Store in history
        self.error_history.append(morag_error)
        
        # Attempt recovery if applicable
        if severity in [ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]:
            self._attempt_recovery(morag_error)
        
        return morag_error

    async def execute_with_fallback(self, 
                                  primary_operation: Callable,
                                  context: ErrorContext,
                                  fallback_key: str = None) -> Any:
        """Execute operation with automatic fallback on failure."""
        
        try:
            result = await primary_operation()
            return result
            
        except Exception as e:
            error = self.handle_error(
                e, 
                ErrorCategory.EXTRACTION,  # Default category
                ErrorSeverity.MEDIUM,
                context
            )
            
            # Try fallback strategies
            if fallback_key and fallback_key in self.fallback_strategies:
                try:
                    fallback_result = await self.fallback_strategies[fallback_key].handler(context, e)
                    error.recovery_attempted = True
                    error.recovery_successful = True
                    self.logger.info(f"Fallback successful for error {error.error_id}")
                    return fallback_result
                    
                except Exception as fallback_error:
                    self.logger.error(f"Fallback failed for error {error.error_id}: {fallback_error}")
                    error.recovery_attempted = True
                    error.recovery_successful = False
            
            # If no fallback or fallback failed, re-raise
            raise e

    def _register_fallback_strategies(self):
        """Register fallback strategies for different failure scenarios."""
        
        # Entity extraction fallback
        self.fallback_strategies['entity_extraction'] = FallbackStrategy(
            name="basic_entity_extraction",
            description="Use simple NER when advanced extraction fails",
            handler=self._basic_entity_extraction_fallback,
            conditions=["extraction_timeout", "model_unavailable"],
            priority=1
        )
        
        # Graph traversal fallback
        self.fallback_strategies['graph_traversal'] = FallbackStrategy(
            name="vector_search_fallback",
            description="Use vector search when graph traversal fails",
            handler=self._vector_search_fallback,
            conditions=["graph_unavailable", "traversal_timeout"],
            priority=1
        )
        
        # Embedding fallback
        self.fallback_strategies['embedding'] = FallbackStrategy(
            name="cached_embedding_fallback",
            description="Use cached embeddings when service fails",
            handler=self._cached_embedding_fallback,
            conditions=["embedding_service_down", "rate_limit_exceeded"],
            priority=1
        )

    async def _basic_entity_extraction_fallback(self, context: ErrorContext, original_error: Exception) -> List[Dict[str, Any]]:
        """Fallback to basic entity extraction."""
        self.logger.info("Using basic entity extraction fallback")
        
        # Simple regex-based entity extraction
        text = context.input_data.get('text', '')
        entities = []
        
        # Extract capitalized words as potential entities
        import re
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for word in set(capitalized_words):
            if len(word) > 2:  # Filter very short words
                entities.append({
                    'name': word,
                    'type': 'UNKNOWN',
                    'confidence': 0.3,
                    'extraction_method': 'fallback_regex'
                })
        
        return entities

    async def _vector_search_fallback(self, context: ErrorContext, original_error: Exception) -> List[Dict[str, Any]]:
        """Fallback to vector search when graph traversal fails."""
        self.logger.info("Using vector search fallback")
        
        # This would call the vector search service directly
        # For now, return empty results
        return []

    async def _cached_embedding_fallback(self, context: ErrorContext, original_error: Exception) -> List[float]:
        """Fallback to cached embeddings."""
        self.logger.info("Using cached embedding fallback")
        
        # This would check cache for similar text embeddings
        # For now, return zero vector
        return [0.0] * 768  # Standard embedding dimension

    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def _extract_technical_details(self, error: Exception) -> str:
        """Extract technical details from exception."""
        details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'error_args': error.args if hasattr(error, 'args') else None
        }
        
        # Add specific details for known error types
        if hasattr(error, 'response'):
            details['http_status'] = getattr(error.response, 'status_code', None)
        
        return str(details)

    def _generate_user_message(self, category: ErrorCategory, severity: ErrorSeverity) -> str:
        """Generate user-friendly error message."""
        
        messages = {
            (ErrorCategory.EXTRACTION, ErrorSeverity.LOW): "Some entities may not have been extracted properly.",
            (ErrorCategory.EXTRACTION, ErrorSeverity.MEDIUM): "Entity extraction encountered issues. Results may be incomplete.",
            (ErrorCategory.EXTRACTION, ErrorSeverity.HIGH): "Entity extraction failed. Please try again or contact support.",
            
            (ErrorCategory.GRAPH_TRAVERSAL, ErrorSeverity.LOW): "Graph search may have missed some connections.",
            (ErrorCategory.GRAPH_TRAVERSAL, ErrorSeverity.MEDIUM): "Graph search encountered issues. Some results may be missing.",
            (ErrorCategory.GRAPH_TRAVERSAL, ErrorSeverity.HIGH): "Graph search failed. Using alternative search method.",
            
            (ErrorCategory.DATABASE, ErrorSeverity.HIGH): "Database connection issues. Please try again later.",
            (ErrorCategory.API, ErrorSeverity.HIGH): "Service temporarily unavailable. Please try again later."
        }
        
        return messages.get((category, severity), "An unexpected error occurred. Please try again.")

    def _log_error(self, error: MoragError):
        """Log error with appropriate level."""
        
        log_message = f"Error {error.error_id}: {error.message} in {error.context.component}.{error.context.operation}"
        
        if error.severity == ErrorSeverity.LOW:
            self.logger.warning(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.error(log_message)
        elif error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.critical(log_message)
            self.logger.critical(f"Stack trace: {error.stack_trace}")

    def _update_error_stats(self, error: MoragError):
        """Update error statistics."""
        self.error_stats[f"{error.category.value}_{error.severity.value}"] += 1
        self.error_stats[f"total_{error.category.value}"] += 1
        self.error_stats["total_errors"] += 1

    def _attempt_recovery(self, error: MoragError):
        """Attempt to recover from error."""
        recovery_key = f"{error.category.value}_{error.context.operation}"
        
        if recovery_key in self.recovery_handlers:
            try:
                self.recovery_handlers[recovery_key](error)
                error.recovery_attempted = True
                error.recovery_successful = True
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {error.error_id}: {recovery_error}")
                error.recovery_attempted = True
                error.recovery_successful = False

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return dict(self.error_stats)

    def get_recent_errors(self, limit: int = 50) -> List[MoragError]:
        """Get recent errors."""
        return self.error_history[-limit:]
```

### 2. Data Quality Monitor

Create `packages/morag-core/src/morag_core/monitoring/data_quality_monitor.py`:

```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

@dataclass
class QualityMetric:
    name: str
    value: float
    threshold: float
    status: str  # "good", "warning", "critical"
    timestamp: datetime

@dataclass
class QualityReport:
    collection_name: str
    metrics: List[QualityMetric]
    overall_score: float
    recommendations: List[str]
    timestamp: datetime

class DataQualityMonitor:
    def __init__(self):
        self.quality_history = defaultdict(list)
        self.thresholds = {
            'entity_coverage': 0.7,
            'relation_density': 0.3,
            'extraction_confidence': 0.6,
            'graph_connectivity': 0.5,
            'chunk_entity_ratio': 0.1
        }

    async def assess_collection_quality(self, collection_name: str, neo4j_service) -> QualityReport:
        """Assess data quality for a collection."""
        
        metrics = []
        
        # Entity coverage metric
        entity_coverage = await self._calculate_entity_coverage(collection_name, neo4j_service)
        metrics.append(QualityMetric(
            name="entity_coverage",
            value=entity_coverage,
            threshold=self.thresholds['entity_coverage'],
            status=self._get_status(entity_coverage, self.thresholds['entity_coverage']),
            timestamp=datetime.now()
        ))
        
        # Relation density metric
        relation_density = await self._calculate_relation_density(collection_name, neo4j_service)
        metrics.append(QualityMetric(
            name="relation_density",
            value=relation_density,
            threshold=self.thresholds['relation_density'],
            status=self._get_status(relation_density, self.thresholds['relation_density']),
            timestamp=datetime.now()
        ))
        
        # Extraction confidence metric
        avg_confidence = await self._calculate_average_confidence(collection_name, neo4j_service)
        metrics.append(QualityMetric(
            name="extraction_confidence",
            value=avg_confidence,
            threshold=self.thresholds['extraction_confidence'],
            status=self._get_status(avg_confidence, self.thresholds['extraction_confidence']),
            timestamp=datetime.now()
        ))
        
        # Graph connectivity metric
        connectivity = await self._calculate_graph_connectivity(collection_name, neo4j_service)
        metrics.append(QualityMetric(
            name="graph_connectivity",
            value=connectivity,
            threshold=self.thresholds['graph_connectivity'],
            status=self._get_status(connectivity, self.thresholds['graph_connectivity']),
            timestamp=datetime.now()
        ))
        
        # Calculate overall score
        overall_score = sum(m.value for m in metrics) / len(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        report = QualityReport(
            collection_name=collection_name,
            metrics=metrics,
            overall_score=overall_score,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.quality_history[collection_name].append(report)
        
        return report

    async def _calculate_entity_coverage(self, collection_name: str, neo4j_service) -> float:
        """Calculate what percentage of chunks have entities."""
        
        query = """
        MATCH (c:Chunk {collection_name: $collection_name})
        OPTIONAL MATCH (c)<-[:MENTIONED_IN]-(e:Entity)
        RETURN 
            count(DISTINCT c) as total_chunks,
            count(DISTINCT CASE WHEN e IS NOT NULL THEN c END) as chunks_with_entities
        """
        
        result = await neo4j_service.execute_query(query, {'collection_name': collection_name})
        
        if result and result[0]['total_chunks'] > 0:
            return result[0]['chunks_with_entities'] / result[0]['total_chunks']
        return 0.0

    async def _calculate_relation_density(self, collection_name: str, neo4j_service) -> float:
        """Calculate relation density in the graph."""
        
        query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        WHERE e1.collection_name = $collection_name 
          AND e2.collection_name = $collection_name
        WITH count(r) as total_relations
        MATCH (e:Entity {collection_name: $collection_name})
        WITH total_relations, count(e) as total_entities
        RETURN 
            total_relations,
            total_entities,
            CASE WHEN total_entities > 1 
                 THEN toFloat(total_relations) / (total_entities * (total_entities - 1))
                 ELSE 0.0 END as density
        """
        
        result = await neo4j_service.execute_query(query, {'collection_name': collection_name})
        
        if result:
            return result[0]['density']
        return 0.0

    async def _calculate_average_confidence(self, collection_name: str, neo4j_service) -> float:
        """Calculate average extraction confidence."""
        
        query = """
        MATCH (e:Entity {collection_name: $collection_name})
        WHERE e.confidence IS NOT NULL
        RETURN avg(e.confidence) as avg_confidence
        """
        
        result = await neo4j_service.execute_query(query, {'collection_name': collection_name})
        
        if result and result[0]['avg_confidence'] is not None:
            return result[0]['avg_confidence']
        return 0.0

    async def _calculate_graph_connectivity(self, collection_name: str, neo4j_service) -> float:
        """Calculate graph connectivity (percentage of connected components)."""
        
        query = """
        MATCH (e:Entity {collection_name: $collection_name})
        OPTIONAL MATCH (e)-[r]-(other:Entity {collection_name: $collection_name})
        RETURN 
            count(DISTINCT e) as total_entities,
            count(DISTINCT CASE WHEN r IS NOT NULL THEN e END) as connected_entities
        """
        
        result = await neo4j_service.execute_query(query, {'collection_name': collection_name})
        
        if result and result[0]['total_entities'] > 0:
            return result[0]['connected_entities'] / result[0]['total_entities']
        return 0.0

    def _get_status(self, value: float, threshold: float) -> str:
        """Get status based on value and threshold."""
        if value >= threshold:
            return "good"
        elif value >= threshold * 0.7:
            return "warning"
        else:
            return "critical"

    def _generate_recommendations(self, metrics: List[QualityMetric]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        for metric in metrics:
            if metric.status == "critical":
                if metric.name == "entity_coverage":
                    recommendations.append("Consider improving entity extraction - many chunks have no entities")
                elif metric.name == "relation_density":
                    recommendations.append("Improve relation extraction - entities are poorly connected")
                elif metric.name == "extraction_confidence":
                    recommendations.append("Review extraction quality - confidence scores are low")
                elif metric.name == "graph_connectivity":
                    recommendations.append("Many entities are isolated - improve relationship detection")
            
            elif metric.status == "warning":
                if metric.name == "entity_coverage":
                    recommendations.append("Some chunks lack entities - review extraction coverage")
                elif metric.name == "relation_density":
                    recommendations.append("Consider extracting more relationships between entities")
        
        if not recommendations:
            recommendations.append("Data quality looks good - no immediate action needed")
        
        return recommendations

    def get_quality_trends(self, collection_name: str, days: int = 7) -> Dict[str, List[float]]:
        """Get quality trends over time."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_reports = [
            report for report in self.quality_history[collection_name]
            if report.timestamp >= cutoff_date
        ]
        
        trends = defaultdict(list)
        for report in recent_reports:
            for metric in report.metrics:
                trends[metric.name].append(metric.value)
        
        return dict(trends)
```

### 3. Integration with Services

Update services to use error handling:

```python
# packages/morag-graph/src/morag_graph/entity_extractor.py

from morag_core.error_handling.error_manager import ErrorManager, ErrorCategory, ErrorSeverity, ErrorContext

class EntityExtractor:
    def __init__(self):
        # ... existing initialization
        self.error_manager = ErrorManager()
    
    async def extract_entities(self, text: str, chunk_id: str = None) -> List[Dict[str, Any]]:
        """Extract entities with comprehensive error handling."""
        
        context = ErrorContext(
            component="EntityExtractor",
            operation="extract_entities",
            input_data={'text_length': len(text), 'chunk_id': chunk_id}
        )
        
        try:
            return await self.error_manager.execute_with_fallback(
                lambda: self._extract_entities_internal(text),
                context,
                fallback_key='entity_extraction'
            )
            
        except Exception as e:
            error = self.error_manager.handle_error(
                e,
                ErrorCategory.EXTRACTION,
                ErrorSeverity.HIGH,
                context,
                user_message="Entity extraction failed. Some entities may be missing from results."
            )
            
            # Return empty list as last resort
            return []
```

## Configuration

```yaml
# error_handling.yml
error_handling:
  enabled: true
  
  logging:
    level: "INFO"
    max_history_size: 1000
    log_to_file: true
    log_file_path: "logs/morag_errors.log"
    
  fallback_strategies:
    entity_extraction: true
    graph_traversal: true
    embedding: true
    
  data_quality:
    monitoring_enabled: true
    assessment_interval_hours: 24
    thresholds:
      entity_coverage: 0.7
      relation_density: 0.3
      extraction_confidence: 0.6
      graph_connectivity: 0.5
      
  alerting:
    enabled: false  # Future implementation
    critical_error_threshold: 5
    email_notifications: false
```

## Testing Strategy

```python
# tests/unit/test_error_handling.py
import pytest
from morag_core.error_handling.error_manager import ErrorManager, ErrorCategory, ErrorSeverity, ErrorContext

class TestErrorHandling:
    def setup_method(self):
        self.error_manager = ErrorManager()

    def test_error_handling_and_logging(self):
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            input_data={"test": "data"}
        )
        
        test_error = ValueError("Test error")
        
        morag_error = self.error_manager.handle_error(
            test_error,
            ErrorCategory.EXTRACTION,
            ErrorSeverity.MEDIUM,
            context
        )
        
        assert morag_error.category == ErrorCategory.EXTRACTION
        assert morag_error.severity == ErrorSeverity.MEDIUM
        assert "Test error" in morag_error.message

    @pytest.mark.asyncio
    async def test_fallback_execution(self):
        # Test fallback strategy execution
        pass
```

## Monitoring

```python
error_monitoring = {
    'error_rates': {
        'extraction_errors_per_hour': 0,
        'graph_errors_per_hour': 0,
        'database_errors_per_hour': 0
    },
    'recovery_rates': {
        'successful_fallbacks': 0,
        'failed_fallbacks': 0,
        'recovery_success_rate': 0.0
    },
    'data_quality': {
        'avg_entity_coverage': 0.0,
        'avg_relation_density': 0.0,
        'avg_extraction_confidence': 0.0
    }
}
```

## Success Metrics

- **Error Recovery Rate**: >80% of medium/high severity errors successfully handled
- **System Uptime**: >99% availability despite component failures
- **Data Quality**: Maintain >70% entity coverage and >60% extraction confidence
- **Debug Time**: 50% reduction in time to identify and fix issues

## Future Enhancements

1. **Automated Alerting**: Email/Slack notifications for critical errors
2. **Self-Healing**: Automatic recovery and system optimization
3. **Predictive Monitoring**: ML-based error prediction and prevention
4. **User Feedback Integration**: Learn from user corrections to improve quality
