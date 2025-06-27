# Task 4.3: Monitoring & Analytics

**Phase**: 4 - Advanced Features  
**Priority**: Medium  
**Estimated Time**: 6-8 days total  
**Dependencies**: Task 4.2 (Performance Optimization)

## Overview

This task implements comprehensive monitoring and analytics capabilities for the graph-augmented RAG system. It provides real-time system metrics, performance dashboards, quality analytics, and alerting mechanisms to ensure system health and enable data-driven optimization.

## Subtasks

### 4.3.1: System Metrics Collection
**Estimated Time**: 3-4 days  
**Priority**: High

#### Implementation Steps

1. **Performance Metrics System**
   ```python
   # src/morag_monitoring/metrics.py
   from typing import Dict, Any, List, Optional, Callable
   from dataclasses import dataclass, field
   from datetime import datetime, timedelta
   import time
   import asyncio
   import logging
   from collections import defaultdict, deque
   from abc import ABC, abstractmethod
   import json
   
   @dataclass
   class MetricPoint:
       timestamp: datetime
       value: float
       tags: Dict[str, str] = field(default_factory=dict)
       metadata: Dict[str, Any] = field(default_factory=dict)
   
   @dataclass
   class MetricSummary:
       name: str
       count: int
       min_value: float
       max_value: float
       avg_value: float
       p50: float
       p95: float
       p99: float
       last_updated: datetime
   
   class MetricCollector(ABC):
       @abstractmethod
       async def collect(self) -> Dict[str, float]:
           pass
   
   class PerformanceMetrics:
       def __init__(self, retention_hours: int = 24, max_points_per_metric: int = 10000):
           self.retention_hours = retention_hours
           self.max_points_per_metric = max_points_per_metric
           self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
           self.collectors: List[MetricCollector] = []
           self.logger = logging.getLogger(__name__)
           self._collection_task: Optional[asyncio.Task] = None
           self._running = False
       
       def add_collector(self, collector: MetricCollector):
           """Add a metric collector."""
           self.collectors.append(collector)
       
       async def start_collection(self, interval_seconds: int = 30):
           """Start automatic metric collection."""
           if self._running:
               return
           
           self._running = True
           self._collection_task = asyncio.create_task(
               self._collection_loop(interval_seconds)
           )
           self.logger.info(f"Started metric collection with {interval_seconds}s interval")
       
       async def stop_collection(self):
           """Stop automatic metric collection."""
           self._running = False
           if self._collection_task:
               self._collection_task.cancel()
               try:
                   await self._collection_task
               except asyncio.CancelledError:
                   pass
           self.logger.info("Stopped metric collection")
       
       async def _collection_loop(self, interval_seconds: int):
           """Main collection loop."""
           while self._running:
               try:
                   await self._collect_all_metrics()
                   await self._cleanup_old_metrics()
                   await asyncio.sleep(interval_seconds)
               except Exception as e:
                   self.logger.error(f"Error in metric collection: {str(e)}")
                   await asyncio.sleep(interval_seconds)
       
       async def _collect_all_metrics(self):
           """Collect metrics from all registered collectors."""
           timestamp = datetime.now()
           
           for collector in self.collectors:
               try:
                   metrics = await collector.collect()
                   for metric_name, value in metrics.items():
                       self.record_metric(metric_name, value, timestamp)
               except Exception as e:
                   self.logger.error(f"Error collecting from {collector.__class__.__name__}: {str(e)}")
       
       def record_metric(
           self, 
           name: str, 
           value: float, 
           timestamp: Optional[datetime] = None,
           tags: Optional[Dict[str, str]] = None
       ):
           """Record a single metric point."""
           if timestamp is None:
               timestamp = datetime.now()
           
           point = MetricPoint(
               timestamp=timestamp,
               value=value,
               tags=tags or {}
           )
           
           self.metrics[name].append(point)
       
       def get_metric_summary(self, name: str, hours: int = 1) -> Optional[MetricSummary]:
           """Get summary statistics for a metric."""
           if name not in self.metrics:
               return None
           
           cutoff_time = datetime.now() - timedelta(hours=hours)
           recent_points = [
               point for point in self.metrics[name]
               if point.timestamp >= cutoff_time
           ]
           
           if not recent_points:
               return None
           
           values = [point.value for point in recent_points]
           values.sort()
           
           return MetricSummary(
               name=name,
               count=len(values),
               min_value=min(values),
               max_value=max(values),
               avg_value=sum(values) / len(values),
               p50=self._percentile(values, 50),
               p95=self._percentile(values, 95),
               p99=self._percentile(values, 99),
               last_updated=recent_points[-1].timestamp
           )
       
       def _percentile(self, values: List[float], percentile: int) -> float:
           """Calculate percentile value."""
           if not values:
               return 0.0
           
           index = int((percentile / 100.0) * (len(values) - 1))
           return values[index]
       
       async def _cleanup_old_metrics(self):
           """Remove old metric points beyond retention period."""
           cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
           
           for metric_name, points in self.metrics.items():
               # Remove old points
               while points and points[0].timestamp < cutoff_time:
                   points.popleft()
       
       def get_all_metrics(self, hours: int = 1) -> Dict[str, MetricSummary]:
           """Get summaries for all metrics."""
           summaries = {}
           for metric_name in self.metrics.keys():
               summary = self.get_metric_summary(metric_name, hours)
               if summary:
                   summaries[metric_name] = summary
           return summaries
   
   class GraphMetricsCollector(MetricCollector):
       def __init__(self, graph_engine, cache_system):
           self.graph_engine = graph_engine
           self.cache_system = cache_system
           self.logger = logging.getLogger(__name__)
       
       async def collect(self) -> Dict[str, float]:
           """Collect graph-specific metrics."""
           metrics = {}
           
           try:
               # Graph statistics
               graph_stats = await self.graph_engine.get_statistics()
               metrics.update({
                   "graph.entity_count": float(graph_stats.get("entity_count", 0)),
                   "graph.relation_count": float(graph_stats.get("relation_count", 0)),
                   "graph.avg_degree": float(graph_stats.get("avg_degree", 0)),
                   "graph.connected_components": float(graph_stats.get("connected_components", 0))
               })
               
               # Cache statistics
               cache_stats = await self.cache_system.get_stats()
               l1_stats = cache_stats.get("l1_cache", {})
               metrics.update({
                   "cache.hit_rate": float(l1_stats.get("hit_rate", 0)),
                   "cache.entry_count": float(l1_stats.get("entry_count", 0)),
                   "cache.size_mb": float(l1_stats.get("total_size_mb", 0)),
                   "cache.evictions": float(l1_stats.get("evictions", 0))
               })
               
           except Exception as e:
               self.logger.error(f"Error collecting graph metrics: {str(e)}")
           
           return metrics
   
   class QueryMetricsCollector(MetricCollector):
       def __init__(self):
           self.query_times = deque(maxlen=1000)
           self.query_counts = defaultdict(int)
           self.error_counts = defaultdict(int)
           self.logger = logging.getLogger(__name__)
       
       def record_query(self, query_type: str, duration: float, success: bool):
           """Record a query execution."""
           self.query_times.append(duration)
           self.query_counts[query_type] += 1
           
           if not success:
               self.error_counts[query_type] += 1
       
       async def collect(self) -> Dict[str, float]:
           """Collect query performance metrics."""
           metrics = {}
           
           if self.query_times:
               times = list(self.query_times)
               metrics.update({
                   "query.avg_duration_ms": sum(times) / len(times) * 1000,
                   "query.max_duration_ms": max(times) * 1000,
                   "query.min_duration_ms": min(times) * 1000
               })
           
           # Query counts by type
           total_queries = sum(self.query_counts.values())
           total_errors = sum(self.error_counts.values())
           
           metrics.update({
               "query.total_count": float(total_queries),
               "query.error_count": float(total_errors),
               "query.error_rate": float(total_errors / total_queries) if total_queries > 0 else 0.0
           })
           
           return metrics
   
   class SystemMetricsCollector(MetricCollector):
       def __init__(self):
           self.logger = logging.getLogger(__name__)
       
       async def collect(self) -> Dict[str, float]:
           """Collect system resource metrics."""
           metrics = {}
           
           try:
               import psutil
               
               # CPU metrics
               cpu_percent = psutil.cpu_percent(interval=1)
               metrics["system.cpu_percent"] = float(cpu_percent)
               
               # Memory metrics
               memory = psutil.virtual_memory()
               metrics.update({
                   "system.memory_percent": float(memory.percent),
                   "system.memory_used_gb": float(memory.used / (1024**3)),
                   "system.memory_available_gb": float(memory.available / (1024**3))
               })
               
               # Disk metrics
               disk = psutil.disk_usage('/')
               metrics.update({
                   "system.disk_percent": float(disk.percent),
                   "system.disk_used_gb": float(disk.used / (1024**3)),
                   "system.disk_free_gb": float(disk.free / (1024**3))
               })
               
           except ImportError:
               self.logger.warning("psutil not available, skipping system metrics")
           except Exception as e:
               self.logger.error(f"Error collecting system metrics: {str(e)}")
           
           return metrics
   ```

2. **Quality Metrics System**
   ```python
   # src/morag_monitoring/quality_metrics.py
   from typing import Dict, Any, List, Optional, Tuple
   from dataclasses import dataclass
   from datetime import datetime
   import logging
   import asyncio
   from collections import defaultdict
   
   @dataclass
   class QualityAssessment:
       query_id: str
       relevance_score: float
       completeness_score: float
       accuracy_score: float
       response_time: float
       timestamp: datetime
       feedback: Optional[str] = None
       user_rating: Optional[int] = None
   
   @dataclass
   class QualityMetrics:
       avg_relevance: float
       avg_completeness: float
       avg_accuracy: float
       avg_response_time: float
       user_satisfaction: float
       total_assessments: int
       period_start: datetime
       period_end: datetime
   
   class QualityMonitor:
       def __init__(self, llm_client, retention_days: int = 30):
           self.llm_client = llm_client
           self.retention_days = retention_days
           self.assessments: List[QualityAssessment] = []
           self.logger = logging.getLogger(__name__)
       
       async def assess_response_quality(
           self, 
           query: str, 
           response: str, 
           context: Dict[str, Any],
           query_id: str
       ) -> QualityAssessment:
           """Assess the quality of a response using LLM."""
           try:
               start_time = datetime.now()
               
               # Create assessment prompt
               prompt = self._create_assessment_prompt(query, response, context)
               
               # Get LLM assessment
               assessment_response = await self.llm_client.generate(
                   prompt=prompt,
                   max_tokens=500,
                   temperature=0.1
               )
               
               response_time = (datetime.now() - start_time).total_seconds()
               
               # Parse assessment
               scores = self._parse_assessment(assessment_response)
               
               assessment = QualityAssessment(
                   query_id=query_id,
                   relevance_score=scores.get("relevance", 0.0),
                   completeness_score=scores.get("completeness", 0.0),
                   accuracy_score=scores.get("accuracy", 0.0),
                   response_time=response_time,
                   timestamp=datetime.now()
               )
               
               self.assessments.append(assessment)
               return assessment
           
           except Exception as e:
               self.logger.error(f"Error assessing response quality: {str(e)}")
               # Return default assessment
               return QualityAssessment(
                   query_id=query_id,
                   relevance_score=0.5,
                   completeness_score=0.5,
                   accuracy_score=0.5,
                   response_time=0.0,
                   timestamp=datetime.now()
               )
       
       def _create_assessment_prompt(self, query: str, response: str, context: Dict[str, Any]) -> str:
           """Create prompt for quality assessment."""
           return f"""Assess the quality of this response to the given query.

Query: {query}

Response: {response}

Context Information:
- Entities found: {len(context.get('entities', {}))}
- Relations found: {len(context.get('relations', []))}
- Documents used: {len(context.get('documents', []))}

Please rate the response on a scale of 0-10 for:
1. Relevance: How well does the response address the query?
2. Completeness: How complete is the information provided?
3. Accuracy: How accurate is the information based on the context?

Format your response as JSON:
{{
  "relevance": 8.5,
  "completeness": 7.0,
  "accuracy": 9.0,
  "reasoning": "Brief explanation of the scores"
}}"""
       
       def _parse_assessment(self, response: str) -> Dict[str, float]:
           """Parse LLM assessment response."""
           try:
               import json
               data = json.loads(response)
               return {
                   "relevance": float(data.get("relevance", 0)) / 10.0,
                   "completeness": float(data.get("completeness", 0)) / 10.0,
                   "accuracy": float(data.get("accuracy", 0)) / 10.0
               }
           except Exception as e:
               self.logger.error(f"Error parsing assessment: {str(e)}")
               return {"relevance": 0.5, "completeness": 0.5, "accuracy": 0.5}
       
       def record_user_feedback(self, query_id: str, rating: int, feedback: Optional[str] = None):
           """Record user feedback for a query."""
           for assessment in self.assessments:
               if assessment.query_id == query_id:
                   assessment.user_rating = rating
                   assessment.feedback = feedback
                   break
       
       def get_quality_metrics(self, hours: int = 24) -> QualityMetrics:
           """Get quality metrics for the specified time period."""
           cutoff_time = datetime.now() - timedelta(hours=hours)
           recent_assessments = [
               a for a in self.assessments
               if a.timestamp >= cutoff_time
           ]
           
           if not recent_assessments:
               return QualityMetrics(
                   avg_relevance=0.0,
                   avg_completeness=0.0,
                   avg_accuracy=0.0,
                   avg_response_time=0.0,
                   user_satisfaction=0.0,
                   total_assessments=0,
                   period_start=cutoff_time,
                   period_end=datetime.now()
               )
           
           # Calculate averages
           avg_relevance = sum(a.relevance_score for a in recent_assessments) / len(recent_assessments)
           avg_completeness = sum(a.completeness_score for a in recent_assessments) / len(recent_assessments)
           avg_accuracy = sum(a.accuracy_score for a in recent_assessments) / len(recent_assessments)
           avg_response_time = sum(a.response_time for a in recent_assessments) / len(recent_assessments)
           
           # Calculate user satisfaction from ratings
           rated_assessments = [a for a in recent_assessments if a.user_rating is not None]
           user_satisfaction = 0.0
           if rated_assessments:
               user_satisfaction = sum(a.user_rating for a in rated_assessments) / len(rated_assessments) / 5.0
           
           return QualityMetrics(
               avg_relevance=avg_relevance,
               avg_completeness=avg_completeness,
               avg_accuracy=avg_accuracy,
               avg_response_time=avg_response_time,
               user_satisfaction=user_satisfaction,
               total_assessments=len(recent_assessments),
               period_start=cutoff_time,
               period_end=datetime.now()
           )
   ```

#### Deliverables
- Comprehensive metrics collection system
- Performance and quality monitoring
- Real-time metric aggregation
- Configurable retention and collection intervals

### 4.3.2: Dashboard and Visualization
**Estimated Time**: 3-4 days  
**Priority**: Medium

#### Implementation Steps

1. **Grafana Dashboard Integration**
   ```python
   # src/morag_monitoring/dashboard.py
   from typing import Dict, Any, List, Optional
   import json
   import logging
   from datetime import datetime, timedelta
   
   class GrafanaDashboardExporter:
       def __init__(self, metrics_system):
           self.metrics_system = metrics_system
           self.logger = logging.getLogger(__name__)
       
       def generate_dashboard_config(self) -> Dict[str, Any]:
           """Generate Grafana dashboard configuration."""
           return {
               "dashboard": {
                   "id": None,
                   "title": "MoRAG System Monitoring",
                   "tags": ["morag", "rag", "graph"],
                   "timezone": "browser",
                   "panels": [
                       self._create_performance_panel(),
                       self._create_cache_panel(),
                       self._create_graph_panel(),
                       self._create_quality_panel(),
                       self._create_system_panel()
                   ],
                   "time": {
                       "from": "now-1h",
                       "to": "now"
                   },
                   "refresh": "30s"
               }
           }
       
       def _create_performance_panel(self) -> Dict[str, Any]:
           """Create performance metrics panel."""
           return {
               "id": 1,
               "title": "Query Performance",
               "type": "graph",
               "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
               "targets": [
                   {
                       "expr": "query.avg_duration_ms",
                       "legendFormat": "Average Duration (ms)"
                   },
                   {
                       "expr": "query.p95_duration_ms",
                       "legendFormat": "95th Percentile (ms)"
                   }
               ],
               "yAxes": [
                   {
                       "label": "Duration (ms)",
                       "min": 0
                   }
               ]
           }
       
       def _create_cache_panel(self) -> Dict[str, Any]:
           """Create cache metrics panel."""
           return {
               "id": 2,
               "title": "Cache Performance",
               "type": "stat",
               "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
               "targets": [
                   {
                       "expr": "cache.hit_rate",
                       "legendFormat": "Hit Rate"
                   },
                   {
                       "expr": "cache.size_mb",
                       "legendFormat": "Size (MB)"
                   }
               ],
               "fieldConfig": {
                   "defaults": {
                       "unit": "percent",
                       "min": 0,
                       "max": 100
                   }
               }
           }
       
       def _create_graph_panel(self) -> Dict[str, Any]:
           """Create graph statistics panel."""
           return {
               "id": 3,
               "title": "Graph Statistics",
               "type": "table",
               "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
               "targets": [
                   {
                       "expr": "graph.entity_count",
                       "legendFormat": "Entities"
                   },
                   {
                       "expr": "graph.relation_count",
                       "legendFormat": "Relations"
                   },
                   {
                       "expr": "graph.avg_degree",
                       "legendFormat": "Avg Degree"
                   }
               ]
           }
       
       def _create_quality_panel(self) -> Dict[str, Any]:
           """Create quality metrics panel."""
           return {
               "id": 4,
               "title": "Response Quality",
               "type": "gauge",
               "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
               "targets": [
                   {
                       "expr": "quality.avg_relevance",
                       "legendFormat": "Relevance"
                   },
                   {
                       "expr": "quality.avg_accuracy",
                       "legendFormat": "Accuracy"
                   },
                   {
                       "expr": "quality.user_satisfaction",
                       "legendFormat": "User Satisfaction"
                   }
               ],
               "fieldConfig": {
                   "defaults": {
                       "unit": "percent",
                       "min": 0,
                       "max": 100,
                       "thresholds": {
                           "steps": [
                               {"color": "red", "value": 0},
                               {"color": "yellow", "value": 70},
                               {"color": "green", "value": 85}
                           ]
                       }
                   }
               }
           }
       
       def _create_system_panel(self) -> Dict[str, Any]:
           """Create system metrics panel."""
           return {
               "id": 5,
               "title": "System Resources",
               "type": "graph",
               "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
               "targets": [
                   {
                       "expr": "system.cpu_percent",
                       "legendFormat": "CPU %"
                   },
                   {
                       "expr": "system.memory_percent",
                       "legendFormat": "Memory %"
                   },
                   {
                       "expr": "system.disk_percent",
                       "legendFormat": "Disk %"
                   }
               ],
               "yAxes": [
                   {
                       "label": "Percentage",
                       "min": 0,
                       "max": 100
                   }
               ]
           }
   
   class AlertManager:
       def __init__(self, metrics_system, notification_channels: List[str]):
           self.metrics_system = metrics_system
           self.notification_channels = notification_channels
           self.alert_rules = []
           self.logger = logging.getLogger(__name__)
       
       def add_alert_rule(
           self, 
           name: str, 
           metric: str, 
           condition: str, 
           threshold: float,
           severity: str = "warning"
       ):
           """Add an alert rule."""
           rule = {
               "name": name,
               "metric": metric,
               "condition": condition,  # ">", "<", "=="
               "threshold": threshold,
               "severity": severity,
               "last_triggered": None
           }
           self.alert_rules.append(rule)
       
       async def check_alerts(self):
           """Check all alert rules and trigger notifications."""
           current_metrics = self.metrics_system.get_all_metrics(hours=1)
           
           for rule in self.alert_rules:
               metric_name = rule["metric"]
               if metric_name in current_metrics:
                   metric_summary = current_metrics[metric_name]
                   current_value = metric_summary.avg_value
                   
                   if self._evaluate_condition(current_value, rule):
                       await self._trigger_alert(rule, current_value)
       
       def _evaluate_condition(self, value: float, rule: Dict[str, Any]) -> bool:
           """Evaluate if alert condition is met."""
           condition = rule["condition"]
           threshold = rule["threshold"]
           
           if condition == ">":
               return value > threshold
           elif condition == "<":
               return value < threshold
           elif condition == "==":
               return abs(value - threshold) < 0.01
           
           return False
       
       async def _trigger_alert(self, rule: Dict[str, Any], current_value: float):
           """Trigger an alert notification."""
           # Avoid spam - only trigger if not triggered recently
           now = datetime.now()
           if (rule["last_triggered"] and 
               now - rule["last_triggered"] < timedelta(minutes=15)):
               return
           
           rule["last_triggered"] = now
           
           alert_message = (
               f"ALERT: {rule['name']}\n"
               f"Metric: {rule['metric']}\n"
               f"Current Value: {current_value:.2f}\n"
               f"Threshold: {rule['threshold']}\n"
               f"Severity: {rule['severity']}"
           )
           
           self.logger.warning(alert_message)
           
           # Send notifications (implement based on your notification system)
           for channel in self.notification_channels:
               await self._send_notification(channel, alert_message)
       
       async def _send_notification(self, channel: str, message: str):
           """Send notification to specified channel."""
           # Implement notification logic (email, Slack, etc.)
           self.logger.info(f"Sending alert to {channel}: {message}")
   ```

#### Deliverables
- Grafana dashboard configuration
- Real-time visualization panels
- Alert management system
- Notification integration

## Testing Requirements

### Unit Tests
```python
# tests/test_monitoring.py
import pytest
import asyncio
from datetime import datetime, timedelta
from morag_monitoring.metrics import PerformanceMetrics, GraphMetricsCollector
from morag_monitoring.quality_metrics import QualityMonitor

class TestPerformanceMetrics:
    @pytest.mark.asyncio
    async def test_metric_recording(self):
        metrics = PerformanceMetrics(retention_hours=1)
        
        # Record some metrics
        metrics.record_metric("test_metric", 10.5)
        metrics.record_metric("test_metric", 15.2)
        metrics.record_metric("test_metric", 8.7)
        
        # Get summary
        summary = metrics.get_metric_summary("test_metric")
        assert summary is not None
        assert summary.count == 3
        assert summary.min_value == 8.7
        assert summary.max_value == 15.2
        assert abs(summary.avg_value - 11.47) < 0.1
    
    @pytest.mark.asyncio
    async def test_metric_collection(self, mock_graph_engine, mock_cache_system):
        metrics = PerformanceMetrics()
        collector = GraphMetricsCollector(mock_graph_engine, mock_cache_system)
        
        metrics.add_collector(collector)
        
        # Start collection
        await metrics.start_collection(interval_seconds=1)
        await asyncio.sleep(2)  # Let it collect a few times
        await metrics.stop_collection()
        
        # Check that metrics were collected
        all_metrics = metrics.get_all_metrics()
        assert len(all_metrics) > 0

class TestQualityMonitor:
    @pytest.mark.asyncio
    async def test_quality_assessment(self, mock_llm_client):
        monitor = QualityMonitor(mock_llm_client)
        
        assessment = await monitor.assess_response_quality(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            context={"entities": {"France": {}, "Paris": {}}, "relations": []},
            query_id="test-query-1"
        )
        
        assert assessment.query_id == "test-query-1"
        assert 0 <= assessment.relevance_score <= 1
        assert 0 <= assessment.completeness_score <= 1
        assert 0 <= assessment.accuracy_score <= 1
    
    def test_user_feedback_recording(self):
        monitor = QualityMonitor(None)
        
        # Create a mock assessment
        from morag_monitoring.quality_metrics import QualityAssessment
        assessment = QualityAssessment(
            query_id="test-query",
            relevance_score=0.8,
            completeness_score=0.7,
            accuracy_score=0.9,
            response_time=1.5,
            timestamp=datetime.now()
        )
        monitor.assessments.append(assessment)
        
        # Record feedback
        monitor.record_user_feedback("test-query", rating=4, feedback="Good response")
        
        # Check feedback was recorded
        assert assessment.user_rating == 4
        assert assessment.feedback == "Good response"
```

### Integration Tests
```python
# tests/integration/test_monitoring_integration.py
class TestMonitoringIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring(self, test_system):
        """Test complete monitoring pipeline."""
        # Set up monitoring
        metrics = PerformanceMetrics()
        quality_monitor = QualityMonitor(test_system.llm_client)
        
        # Add collectors
        graph_collector = GraphMetricsCollector(
            test_system.graph_engine, 
            test_system.cache_system
        )
        metrics.add_collector(graph_collector)
        
        # Start monitoring
        await metrics.start_collection(interval_seconds=1)
        
        # Perform some operations
        for i in range(5):
            query = f"Test query {i}"
            response = await test_system.process_query(query)
            
            # Assess quality
            await quality_monitor.assess_response_quality(
                query, response.text, response.context, f"query-{i}"
            )
        
        # Wait for metrics collection
        await asyncio.sleep(3)
        
        # Check metrics were collected
        all_metrics = metrics.get_all_metrics()
        assert len(all_metrics) > 0
        
        # Check quality metrics
        quality_metrics = quality_monitor.get_quality_metrics()
        assert quality_metrics.total_assessments == 5
        
        await metrics.stop_collection()
```

## Success Criteria

- [ ] Comprehensive metrics collection for all system components
- [ ] Real-time dashboard with key performance indicators
- [ ] Quality assessment system provides meaningful insights
- [ ] Alert system triggers notifications for critical issues
- [ ] Metrics retention and cleanup work correctly
- [ ] Dashboard visualizations are informative and responsive
- [ ] Performance impact of monitoring is minimal (< 5% overhead)
- [ ] Unit test coverage > 85%
- [ ] Integration tests validate end-to-end monitoring

## Performance Targets

- **Metric Collection**: < 100ms per collection cycle
- **Dashboard Response**: < 2 seconds for data refresh
- **Quality Assessment**: < 3 seconds per response
- **Alert Processing**: < 1 second for rule evaluation
- **Storage Overhead**: < 100MB for 24 hours of metrics
- **System Impact**: < 5% CPU overhead for monitoring

## Next Steps

After completing this task:
1. Deploy monitoring to production environment
2. Configure alerting thresholds based on baseline metrics
3. Create operational runbooks for common alerts
4. Implement automated response to certain alert conditions

## Dependencies

**Requires**:
- Task 4.2: Performance Optimization
- Grafana for visualization
- LLM client for quality assessment

**Enables**:
- Production monitoring and alerting
- Data-driven optimization decisions
- Proactive issue detection and resolution