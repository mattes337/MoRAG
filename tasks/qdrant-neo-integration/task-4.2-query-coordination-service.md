# Task 4.2: Query Coordination Service

## Overview

Develop a comprehensive query coordination service that manages query routing, load balancing, performance optimization, and intelligent caching across the hybrid Qdrant-Neo4j retrieval system. This service acts as the central orchestrator for all retrieval operations.

## Objectives

- Implement intelligent query routing and load balancing
- Create adaptive performance optimization strategies
- Establish comprehensive monitoring and analytics
- Enable query planning and execution optimization
- Provide fault tolerance and graceful degradation

## Current State Analysis

### Existing Infrastructure

**Hybrid Retriever**:
- Basic retrieval strategies implemented
- Simple result fusion capabilities
- Limited performance optimization

**Query Coordinator**:
- Basic caching and batching
- Simple performance monitoring
- No advanced optimization

**Integration Gaps**:
- No intelligent query planning
- Limited load balancing capabilities
- No adaptive optimization
- Missing comprehensive analytics

## Implementation Plan

### Step 1: Advanced Query Planning Service

Implement `src/morag_graph/coordination/query_planner.py`:

```python
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict

from ..retrieval.hybrid_retriever import RetrievalQuery, RetrievalStrategy, QueryType

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    VECTOR_DB = "vector_db"
    GRAPH_DB = "graph_db"

@dataclass
class QueryPlan:
    """Execution plan for a retrieval query."""
    query_id: str
    original_query: RetrievalQuery
    
    # Execution strategy
    execution_strategy: RetrievalStrategy
    estimated_complexity: QueryComplexity
    estimated_cost: float
    estimated_time: float
    
    # Resource requirements
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Optimization decisions
    use_cache: bool = True
    cache_ttl: int = 3600
    enable_batching: bool = False
    parallel_execution: bool = True
    
    # Fallback strategies
    fallback_strategies: List[RetrievalStrategy] = field(default_factory=list)
    
    # Monitoring
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 5  # 1-10, higher is more important
    
@dataclass
class SystemResources:
    """Current system resource utilization."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_usage: float = 0.0
    vector_db_load: float = 0.0
    graph_db_load: float = 0.0
    
    # Connection pools
    vector_db_connections: int = 0
    graph_db_connections: int = 0
    max_vector_connections: int = 100
    max_graph_connections: int = 50
    
    # Queue sizes
    pending_queries: int = 0
    max_queue_size: int = 1000
    
    timestamp: datetime = field(default_factory=datetime.now)

class QueryPlanner:
    """Advanced query planning and optimization service."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Historical performance data
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.strategy_performance: Dict[RetrievalStrategy, Dict[str, float]] = {}
        
        # Resource monitoring
        self.current_resources = SystemResources()
        self.resource_history: List[SystemResources] = []
        
        # Query patterns
        self.query_patterns: Dict[str, Dict[str, Any]] = {}
        self.optimization_rules: List[Dict[str, Any]] = []
        
        # Performance thresholds
        self.performance_thresholds = {
            "max_response_time": 5.0,  # seconds
            "max_cpu_usage": 0.8,
            "max_memory_usage": 0.8,
            "max_db_load": 0.9
        }
        
        # Initialize default optimization rules
        self._initialize_optimization_rules()
    
    async def create_query_plan(self, query: RetrievalQuery) -> QueryPlan:
        """Create an optimized execution plan for a query."""
        try:
            # Generate unique query ID
            query_id = self._generate_query_id(query)
            
            # Analyze query characteristics
            query_analysis = await self._analyze_query_characteristics(query)
            
            # Estimate complexity and cost
            complexity = self._estimate_query_complexity(query, query_analysis)
            cost = self._estimate_query_cost(query, complexity)
            estimated_time = self._estimate_execution_time(query, complexity)
            
            # Select optimal strategy
            strategy = await self._select_optimal_strategy(query, query_analysis)
            
            # Calculate resource requirements
            resources = self._calculate_resource_requirements(query, strategy, complexity)
            
            # Determine optimization settings
            optimization_settings = await self._determine_optimization_settings(
                query, strategy, complexity
            )
            
            # Generate fallback strategies
            fallback_strategies = self._generate_fallback_strategies(strategy, query)
            
            # Create query plan
            plan = QueryPlan(
                query_id=query_id,
                original_query=query,
                execution_strategy=strategy,
                estimated_complexity=complexity,
                estimated_cost=cost,
                estimated_time=estimated_time,
                resource_requirements=resources,
                fallback_strategies=fallback_strategies,
                **optimization_settings
            )
            
            # Validate plan against current resources
            if not await self._validate_plan_feasibility(plan):
                plan = await self._adjust_plan_for_resources(plan)
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Query planning failed: {e}")
            # Return basic plan as fallback
            return QueryPlan(
                query_id=self._generate_query_id(query),
                original_query=query,
                execution_strategy=RetrievalStrategy.HYBRID_PARALLEL,
                estimated_complexity=QueryComplexity.MODERATE,
                estimated_cost=1.0,
                estimated_time=1.0
            )
    
    async def optimize_query_batch(self, queries: List[RetrievalQuery]) -> List[QueryPlan]:
        """Create optimized execution plans for a batch of queries."""
        try:
            # Create individual plans
            individual_plans = await asyncio.gather(*[
                self.create_query_plan(query) for query in queries
            ])
            
            # Analyze batch characteristics
            batch_analysis = self._analyze_batch_characteristics(individual_plans)
            
            # Optimize batch execution order
            optimized_plans = self._optimize_batch_execution_order(individual_plans, batch_analysis)
            
            # Apply batch-specific optimizations
            optimized_plans = self._apply_batch_optimizations(optimized_plans, batch_analysis)
            
            return optimized_plans
            
        except Exception as e:
            self.logger.error(f"Batch optimization failed: {e}")
            return [await self.create_query_plan(query) for query in queries]
    
    def _generate_query_id(self, query: RetrievalQuery) -> str:
        """Generate unique identifier for query."""
        import hashlib
        query_str = f"{query.query_text}_{query.query_type.value}_{datetime.now().isoformat()}"
        return hashlib.md5(query_str.encode()).hexdigest()[:12]
    
    async def _analyze_query_characteristics(self, query: RetrievalQuery) -> Dict[str, Any]:
        """Analyze query characteristics for optimization."""
        analysis = {
            "text_length": len(query.query_text),
            "word_count": len(query.query_text.split()),
            "has_vector": query.query_vector is not None,
            "has_filters": query.vector_filters is not None,
            "entity_types_count": len(query.entity_types) if query.entity_types else 0,
            "relationship_types_count": len(query.relationship_types) if query.relationship_types else 0,
            "max_depth": query.max_depth,
            "result_limits": {
                "vector": query.vector_limit,
                "graph": query.graph_limit,
                "final": query.final_limit
            }
        }
        
        # Analyze query text for patterns
        text_analysis = await self._analyze_query_text(query.query_text)
        analysis.update(text_analysis)
        
        # Check for similar historical queries
        similar_queries = self._find_similar_historical_queries(query)
        analysis["similar_queries_count"] = len(similar_queries)
        analysis["avg_historical_performance"] = self._calculate_avg_performance(similar_queries)
        
        return analysis
    
    async def _analyze_query_text(self, text: str) -> Dict[str, Any]:
        """Analyze query text for semantic patterns."""
        import re
        
        analysis = {
            "question_words": len(re.findall(r'\b(what|who|where|when|why|how)\b', text.lower())),
            "entity_mentions": len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)),
            "relationship_keywords": len(re.findall(
                r'\b(related|connected|associated|linked|between|relationship|works|knows)\b', 
                text.lower()
            )),
            "temporal_keywords": len(re.findall(
                r'\b(recent|latest|current|past|future|before|after|during)\b', 
                text.lower()
            )),
            "comparison_keywords": len(re.findall(
                r'\b(similar|different|compare|contrast|like|unlike)\b', 
                text.lower()
            )),
            "aggregation_keywords": len(re.findall(
                r'\b(all|most|least|average|total|count|sum)\b', 
                text.lower()
            ))
        }
        
        return analysis
    
    def _estimate_query_complexity(self, query: RetrievalQuery, analysis: Dict[str, Any]) -> QueryComplexity:
        """Estimate query complexity based on characteristics."""
        complexity_score = 0
        
        # Text complexity
        complexity_score += min(analysis["word_count"] / 20, 1.0) * 0.2
        
        # Query type complexity
        type_complexity = {
            QueryType.SEMANTIC_SEARCH: 0.3,
            QueryType.ENTITY_LOOKUP: 0.2,
            QueryType.RELATIONSHIP_EXPLORATION: 0.6,
            QueryType.CONTEXTUAL_SEARCH: 0.7,
            QueryType.MULTI_HOP_REASONING: 1.0
        }
        complexity_score += type_complexity.get(query.query_type, 0.5) * 0.3
        
        # Filter complexity
        if analysis["has_filters"]:
            complexity_score += 0.1
        
        # Graph traversal complexity
        complexity_score += min(query.max_depth / 5, 1.0) * 0.2
        complexity_score += min(analysis["entity_types_count"] / 10, 1.0) * 0.1
        complexity_score += min(analysis["relationship_types_count"] / 10, 1.0) * 0.1
        
        # Result size complexity
        total_results = query.vector_limit + query.graph_limit
        complexity_score += min(total_results / 100, 1.0) * 0.1
        
        # Map to complexity enum
        if complexity_score < 0.3:
            return QueryComplexity.SIMPLE
        elif complexity_score < 0.6:
            return QueryComplexity.MODERATE
        elif complexity_score < 0.8:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.VERY_COMPLEX
    
    def _estimate_query_cost(self, query: RetrievalQuery, complexity: QueryComplexity) -> float:
        """Estimate computational cost of query execution."""
        base_costs = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MODERATE: 2.5,
            QueryComplexity.COMPLEX: 5.0,
            QueryComplexity.VERY_COMPLEX: 10.0
        }
        
        base_cost = base_costs[complexity]
        
        # Strategy cost multipliers
        strategy_multipliers = {
            RetrievalStrategy.VECTOR_ONLY: 0.5,
            RetrievalStrategy.GRAPH_ONLY: 0.7,
            RetrievalStrategy.HYBRID_PARALLEL: 1.0,
            RetrievalStrategy.HYBRID_SEQUENTIAL: 1.2,
            RetrievalStrategy.ADAPTIVE: 1.1
        }
        
        strategy_cost = strategy_multipliers.get(query.strategy, 1.0)
        
        # Result size cost
        result_cost = 1.0 + (query.final_limit / 100) * 0.5
        
        return base_cost * strategy_cost * result_cost
    
    def _estimate_execution_time(self, query: RetrievalQuery, complexity: QueryComplexity) -> float:
        """Estimate execution time in seconds."""
        base_times = {
            QueryComplexity.SIMPLE: 0.1,
            QueryComplexity.MODERATE: 0.3,
            QueryComplexity.COMPLEX: 0.8,
            QueryComplexity.VERY_COMPLEX: 2.0
        }
        
        base_time = base_times[complexity]
        
        # Adjust based on current system load
        load_multiplier = 1.0 + (self.current_resources.cpu_usage * 0.5)
        
        # Adjust based on historical performance
        historical_avg = self._get_historical_avg_time(query.query_type, query.strategy)
        if historical_avg > 0:
            base_time = (base_time + historical_avg) / 2
        
        return base_time * load_multiplier
    
    async def _select_optimal_strategy(self, query: RetrievalQuery, analysis: Dict[str, Any]) -> RetrievalStrategy:
        """Select optimal execution strategy based on analysis."""
        if query.strategy != RetrievalStrategy.ADAPTIVE:
            return query.strategy
        
        # Strategy selection logic based on query characteristics
        if query.query_type == QueryType.ENTITY_LOOKUP:
            return RetrievalStrategy.GRAPH_ONLY
        
        elif query.query_type == QueryType.RELATIONSHIP_EXPLORATION:
            return RetrievalStrategy.GRAPH_ONLY
        
        elif query.query_type == QueryType.MULTI_HOP_REASONING:
            return RetrievalStrategy.HYBRID_SEQUENTIAL
        
        elif analysis["relationship_keywords"] > 2:
            return RetrievalStrategy.HYBRID_PARALLEL
        
        elif analysis["entity_mentions"] == 0 and query.query_vector:
            return RetrievalStrategy.VECTOR_ONLY
        
        else:
            # Check system load to decide between parallel and sequential
            if self.current_resources.cpu_usage > 0.7:
                return RetrievalStrategy.HYBRID_SEQUENTIAL
            else:
                return RetrievalStrategy.HYBRID_PARALLEL
    
    def _calculate_resource_requirements(self, 
                                       query: RetrievalQuery, 
                                       strategy: RetrievalStrategy, 
                                       complexity: QueryComplexity) -> Dict[ResourceType, float]:
        """Calculate estimated resource requirements."""
        base_requirements = {
            QueryComplexity.SIMPLE: {
                ResourceType.CPU: 0.1,
                ResourceType.MEMORY: 0.05,
                ResourceType.NETWORK: 0.02
            },
            QueryComplexity.MODERATE: {
                ResourceType.CPU: 0.3,
                ResourceType.MEMORY: 0.15,
                ResourceType.NETWORK: 0.05
            },
            QueryComplexity.COMPLEX: {
                ResourceType.CPU: 0.6,
                ResourceType.MEMORY: 0.3,
                ResourceType.NETWORK: 0.1
            },
            QueryComplexity.VERY_COMPLEX: {
                ResourceType.CPU: 1.0,
                ResourceType.MEMORY: 0.5,
                ResourceType.NETWORK: 0.2
            }
        }
        
        requirements = base_requirements[complexity].copy()
        
        # Adjust based on strategy
        if strategy in [RetrievalStrategy.VECTOR_ONLY]:
            requirements[ResourceType.VECTOR_DB] = 0.8
            requirements[ResourceType.GRAPH_DB] = 0.0
        elif strategy == RetrievalStrategy.GRAPH_ONLY:
            requirements[ResourceType.VECTOR_DB] = 0.0
            requirements[ResourceType.GRAPH_DB] = 0.8
        else:
            requirements[ResourceType.VECTOR_DB] = 0.6
            requirements[ResourceType.GRAPH_DB] = 0.6
        
        return requirements
    
    async def _determine_optimization_settings(self, 
                                             query: RetrievalQuery, 
                                             strategy: RetrievalStrategy, 
                                             complexity: QueryComplexity) -> Dict[str, Any]:
        """Determine optimization settings for the query."""
        settings = {
            "use_cache": True,
            "cache_ttl": 3600,
            "enable_batching": False,
            "parallel_execution": True,
            "priority": 5
        }
        
        # Cache settings based on query type
        if query.query_type in [QueryType.ENTITY_LOOKUP, QueryType.RELATIONSHIP_EXPLORATION]:
            settings["cache_ttl"] = 7200  # 2 hours for entity queries
        elif complexity == QueryComplexity.VERY_COMPLEX:
            settings["cache_ttl"] = 1800  # 30 minutes for complex queries
        
        # Batching settings
        if complexity == QueryComplexity.SIMPLE and strategy == RetrievalStrategy.VECTOR_ONLY:
            settings["enable_batching"] = True
        
        # Parallel execution settings
        if self.current_resources.cpu_usage > 0.8:
            settings["parallel_execution"] = False
        
        # Priority based on user and complexity
        if query.user_id and query.user_id.startswith("premium_"):
            settings["priority"] = 8
        elif complexity == QueryComplexity.VERY_COMPLEX:
            settings["priority"] = 3
        
        return settings
    
    def _generate_fallback_strategies(self, primary_strategy: RetrievalStrategy, 
                                    query: RetrievalQuery) -> List[RetrievalStrategy]:
        """Generate fallback strategies in case primary strategy fails."""
        fallbacks = []
        
        if primary_strategy == RetrievalStrategy.HYBRID_PARALLEL:
            fallbacks = [RetrievalStrategy.HYBRID_SEQUENTIAL, RetrievalStrategy.VECTOR_ONLY]
        elif primary_strategy == RetrievalStrategy.HYBRID_SEQUENTIAL:
            fallbacks = [RetrievalStrategy.VECTOR_ONLY, RetrievalStrategy.GRAPH_ONLY]
        elif primary_strategy == RetrievalStrategy.VECTOR_ONLY:
            if query.query_type != QueryType.SEMANTIC_SEARCH:
                fallbacks = [RetrievalStrategy.GRAPH_ONLY]
        elif primary_strategy == RetrievalStrategy.GRAPH_ONLY:
            if query.query_vector:
                fallbacks = [RetrievalStrategy.VECTOR_ONLY]
        
        return fallbacks
    
    async def _validate_plan_feasibility(self, plan: QueryPlan) -> bool:
        """Validate if the plan can be executed with current resources."""
        # Check resource availability
        for resource_type, required in plan.resource_requirements.items():
            current_usage = getattr(self.current_resources, resource_type.value + "_usage", 0.0)
            if current_usage + required > self.performance_thresholds.get(f"max_{resource_type.value}_usage", 1.0):
                return False
        
        # Check queue capacity
        if self.current_resources.pending_queries >= self.current_resources.max_queue_size:
            return False
        
        # Check connection pools
        if (plan.execution_strategy in [RetrievalStrategy.VECTOR_ONLY, RetrievalStrategy.HYBRID_PARALLEL, RetrievalStrategy.HYBRID_SEQUENTIAL] and
            self.current_resources.vector_db_connections >= self.current_resources.max_vector_connections):
            return False
        
        return True
    
    async def _adjust_plan_for_resources(self, plan: QueryPlan) -> QueryPlan:
        """Adjust plan to fit current resource constraints."""
        # Try fallback strategies
        for fallback_strategy in plan.fallback_strategies:
            adjusted_plan = QueryPlan(
                query_id=plan.query_id,
                original_query=plan.original_query,
                execution_strategy=fallback_strategy,
                estimated_complexity=plan.estimated_complexity,
                estimated_cost=plan.estimated_cost * 0.8,  # Assume fallback is cheaper
                estimated_time=plan.estimated_time * 1.2,  # But might take longer
                priority=max(1, plan.priority - 1)  # Lower priority
            )
            
            if await self._validate_plan_feasibility(adjusted_plan):
                self.logger.info(f"Adjusted plan to use fallback strategy: {fallback_strategy}")
                return adjusted_plan
        
        # If no fallback works, reduce limits
        reduced_query = plan.original_query
        reduced_query.vector_limit = min(reduced_query.vector_limit, 10)
        reduced_query.graph_limit = min(reduced_query.graph_limit, 10)
        reduced_query.final_limit = min(reduced_query.final_limit, 5)
        
        plan.original_query = reduced_query
        plan.estimated_cost *= 0.5
        plan.priority = max(1, plan.priority - 2)
        
        self.logger.warning(f"Reduced query limits due to resource constraints")
        return plan
    
    def _analyze_batch_characteristics(self, plans: List[QueryPlan]) -> Dict[str, Any]:
        """Analyze characteristics of a query batch."""
        if not plans:
            return {}
        
        strategies = [plan.execution_strategy for plan in plans]
        complexities = [plan.estimated_complexity for plan in plans]
        
        return {
            "batch_size": len(plans),
            "strategy_distribution": {strategy: strategies.count(strategy) for strategy in set(strategies)},
            "complexity_distribution": {complexity: complexities.count(complexity) for complexity in set(complexities)},
            "total_estimated_cost": sum(plan.estimated_cost for plan in plans),
            "total_estimated_time": sum(plan.estimated_time for plan in plans),
            "avg_priority": sum(plan.priority for plan in plans) / len(plans),
            "has_vector_queries": any(plan.execution_strategy in [RetrievalStrategy.VECTOR_ONLY, RetrievalStrategy.HYBRID_PARALLEL, RetrievalStrategy.HYBRID_SEQUENTIAL] for plan in plans),
            "has_graph_queries": any(plan.execution_strategy in [RetrievalStrategy.GRAPH_ONLY, RetrievalStrategy.HYBRID_PARALLEL, RetrievalStrategy.HYBRID_SEQUENTIAL] for plan in plans)
        }
    
    def _optimize_batch_execution_order(self, plans: List[QueryPlan], batch_analysis: Dict[str, Any]) -> List[QueryPlan]:
        """Optimize the execution order of queries in a batch."""
        # Sort by priority first, then by estimated time
        return sorted(plans, key=lambda p: (-p.priority, p.estimated_time))
    
    def _apply_batch_optimizations(self, plans: List[QueryPlan], batch_analysis: Dict[str, Any]) -> List[QueryPlan]:
        """Apply batch-specific optimizations."""
        # Enable batching for similar simple queries
        simple_vector_plans = [
            plan for plan in plans 
            if (plan.estimated_complexity == QueryComplexity.SIMPLE and 
                plan.execution_strategy == RetrievalStrategy.VECTOR_ONLY)
        ]
        
        for plan in simple_vector_plans:
            plan.enable_batching = True
        
        # Adjust cache TTL for batch queries
        if batch_analysis["batch_size"] > 10:
            for plan in plans:
                plan.cache_ttl = min(plan.cache_ttl, 1800)  # Shorter TTL for large batches
        
        return plans
    
    def _find_similar_historical_queries(self, query: RetrievalQuery) -> List[Dict[str, Any]]:
        """Find similar queries in performance history."""
        query_key = f"{query.query_type.value}_{query.strategy.value}"
        return self.performance_history.get(query_key, [])
    
    def _calculate_avg_performance(self, historical_queries: List[Dict[str, Any]]) -> float:
        """Calculate average performance from historical data."""
        if not historical_queries:
            return 0.0
        
        total_time = sum(q.get("execution_time", 0.0) for q in historical_queries)
        return total_time / len(historical_queries)
    
    def _get_historical_avg_time(self, query_type: QueryType, strategy: RetrievalStrategy) -> float:
        """Get historical average execution time for query type and strategy."""
        key = f"{query_type.value}_{strategy.value}"
        if key in self.strategy_performance:
            return self.strategy_performance[key].get("avg_time", 0.0)
        return 0.0
    
    def _initialize_optimization_rules(self):
        """Initialize default optimization rules."""
        self.optimization_rules = [
            {
                "name": "high_load_sequential",
                "condition": lambda: self.current_resources.cpu_usage > 0.8,
                "action": "prefer_sequential_execution"
            },
            {
                "name": "low_memory_reduce_limits",
                "condition": lambda: self.current_resources.memory_usage > 0.9,
                "action": "reduce_result_limits"
            },
            {
                "name": "high_queue_increase_priority",
                "condition": lambda: self.current_resources.pending_queries > 500,
                "action": "increase_cache_usage"
            }
        ]
    
    def update_performance_history(self, plan: QueryPlan, actual_metrics: Dict[str, Any]):
        """Update performance history with actual execution metrics."""
        try:
            key = f"{plan.original_query.query_type.value}_{plan.execution_strategy.value}"
            
            performance_record = {
                "query_id": plan.query_id,
                "estimated_time": plan.estimated_time,
                "actual_time": actual_metrics.get("total_time", 0.0),
                "estimated_cost": plan.estimated_cost,
                "actual_cost": actual_metrics.get("total_cost", 0.0),
                "complexity": plan.estimated_complexity.value,
                "timestamp": datetime.now().isoformat(),
                "success": actual_metrics.get("success", True)
            }
            
            self.performance_history[key].append(performance_record)
            
            # Keep only recent history (last 1000 records per key)
            if len(self.performance_history[key]) > 1000:
                self.performance_history[key] = self.performance_history[key][-1000:]
            
            # Update strategy performance averages
            self._update_strategy_performance(plan.execution_strategy, performance_record)
            
        except Exception as e:
            self.logger.error(f"Failed to update performance history: {e}")
    
    def _update_strategy_performance(self, strategy: RetrievalStrategy, record: Dict[str, Any]):
        """Update strategy performance averages."""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                "avg_time": 0.0,
                "avg_cost": 0.0,
                "success_rate": 0.0,
                "total_queries": 0
            }
        
        stats = self.strategy_performance[strategy]
        stats["total_queries"] += 1
        
        # Update running averages
        n = stats["total_queries"]
        stats["avg_time"] = ((stats["avg_time"] * (n - 1)) + record["actual_time"]) / n
        stats["avg_cost"] = ((stats["avg_cost"] * (n - 1)) + record["actual_cost"]) / n
        stats["success_rate"] = ((stats["success_rate"] * (n - 1)) + (1 if record["success"] else 0)) / n
    
    def update_system_resources(self, resources: SystemResources):
        """Update current system resource information."""
        self.current_resources = resources
        self.resource_history.append(resources)
        
        # Keep only recent history (last 1000 records)
        if len(self.resource_history) > 1000:
            self.resource_history = self.resource_history[-1000:]
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and recommendations."""
        insights = {
            "strategy_performance": self.strategy_performance.copy(),
            "resource_trends": self._analyze_resource_trends(),
            "optimization_opportunities": self._identify_optimization_opportunities(),
            "performance_recommendations": self._generate_performance_recommendations()
        }
        
        return insights
    
    def _analyze_resource_trends(self) -> Dict[str, Any]:
        """Analyze resource usage trends."""
        if len(self.resource_history) < 10:
            return {"insufficient_data": True}
        
        recent_resources = self.resource_history[-10:]
        
        return {
            "cpu_trend": "increasing" if recent_resources[-1].cpu_usage > recent_resources[0].cpu_usage else "decreasing",
            "memory_trend": "increasing" if recent_resources[-1].memory_usage > recent_resources[0].memory_usage else "decreasing",
            "avg_cpu_usage": sum(r.cpu_usage for r in recent_resources) / len(recent_resources),
            "avg_memory_usage": sum(r.memory_usage for r in recent_resources) / len(recent_resources),
            "peak_queue_size": max(r.pending_queries for r in recent_resources)
        }
    
    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify optimization opportunities based on performance data."""
        opportunities = []
        
        # Check for consistently slow strategies
        for strategy, stats in self.strategy_performance.items():
            if stats["avg_time"] > 2.0 and stats["total_queries"] > 10:
                opportunities.append({
                    "type": "slow_strategy",
                    "strategy": strategy.value,
                    "avg_time": stats["avg_time"],
                    "recommendation": "Consider optimizing or replacing this strategy"
                })
        
        # Check for low success rates
        for strategy, stats in self.strategy_performance.items():
            if stats["success_rate"] < 0.9 and stats["total_queries"] > 10:
                opportunities.append({
                    "type": "low_success_rate",
                    "strategy": strategy.value,
                    "success_rate": stats["success_rate"],
                    "recommendation": "Investigate failures and improve error handling"
                })
        
        # Check for resource bottlenecks
        if self.current_resources.cpu_usage > 0.8:
            opportunities.append({
                "type": "cpu_bottleneck",
                "current_usage": self.current_resources.cpu_usage,
                "recommendation": "Consider scaling CPU resources or optimizing algorithms"
            })
        
        return opportunities
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations based on analysis."""
        recommendations = []
        
        # Resource-based recommendations
        if self.current_resources.cpu_usage > 0.8:
            recommendations.append("Consider enabling more sequential execution to reduce CPU load")
        
        if self.current_resources.memory_usage > 0.8:
            recommendations.append("Reduce result limits or implement more aggressive caching")
        
        if self.current_resources.pending_queries > 100:
            recommendations.append("Increase cache TTL or add more processing capacity")
        
        # Strategy-based recommendations
        best_strategy = min(self.strategy_performance.items(), 
                          key=lambda x: x[1]["avg_time"] if x[1]["total_queries"] > 5 else float('inf'),
                          default=(None, None))
        
        if best_strategy[0]:
            recommendations.append(f"Consider using {best_strategy[0].value} more frequently - it has the best average performance")
        
        return recommendations
```

### Step 2: Load Balancing and Resource Management

Implement `src/morag_graph/coordination/load_balancer.py`:

```python
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import heapq
from collections import defaultdict, deque

from .query_planner import QueryPlan, SystemResources

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_AWARE = "resource_aware"
    ADAPTIVE = "adaptive"

@dataclass
class ServiceInstance:
    """Represents a service instance (Neo4j or Qdrant)."""
    instance_id: str
    service_type: str  # 'neo4j' or 'qdrant'
    endpoint: str
    weight: float = 1.0
    
    # Health status
    is_healthy: bool = True
    last_health_check: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    
    # Performance metrics
    current_connections: int = 0
    max_connections: int = 100
    avg_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # Load metrics
    requests_per_second: float = 0.0
    active_queries: int = 0
    queue_size: int = 0
    
    def get_load_score(self) -> float:
        """Calculate current load score (0-1, higher means more loaded)."""
        connection_load = self.current_connections / self.max_connections
        resource_load = (self.cpu_usage + self.memory_usage) / 2
        queue_load = min(self.queue_size / 100, 1.0)  # Assume max queue of 100
        
        return (connection_load + resource_load + queue_load) / 3
    
    def can_handle_request(self) -> bool:
        """Check if instance can handle another request."""
        return (self.is_healthy and 
                self.current_connections < self.max_connections and
                self.queue_size < 100)

@dataclass
class QueryExecution:
    """Represents an executing query."""
    query_id: str
    plan: QueryPlan
    assigned_instances: Dict[str, ServiceInstance]
    start_time: datetime
    estimated_completion: datetime
    priority: int
    
    def is_overdue(self) -> bool:
        """Check if query execution is overdue."""
        return datetime.now() > self.estimated_completion

class LoadBalancer:
    """Advanced load balancer for hybrid retrieval system."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Service instances
        self.neo4j_instances: List[ServiceInstance] = []
        self.qdrant_instances: List[ServiceInstance] = []
        
        # Load balancing state
        self.round_robin_counters = {"neo4j": 0, "qdrant": 0}
        
        # Query execution tracking
        self.active_executions: Dict[str, QueryExecution] = {}
        self.execution_queue: List[Tuple[int, datetime, QueryPlan]] = []  # Priority queue
        
        # Performance monitoring
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Health checking
        self.health_check_interval = 30  # seconds
        self.health_check_task = None
        
        # Circuit breaker settings
        self.circuit_breaker_threshold = 5  # consecutive failures
        self.circuit_breaker_timeout = 300  # 5 minutes
    
    async def start(self):
        """Start the load balancer and health checking."""
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.logger.info("Load balancer started")
    
    async def stop(self):
        """Stop the load balancer."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Load balancer stopped")
    
    def register_neo4j_instance(self, instance: ServiceInstance):
        """Register a Neo4j service instance."""
        instance.service_type = "neo4j"
        self.neo4j_instances.append(instance)
        self.logger.info(f"Registered Neo4j instance: {instance.instance_id}")
    
    def register_qdrant_instance(self, instance: ServiceInstance):
        """Register a Qdrant service instance."""
        instance.service_type = "qdrant"
        self.qdrant_instances.append(instance)
        self.logger.info(f"Registered Qdrant instance: {instance.instance_id}")
    
    async def assign_query_execution(self, plan: QueryPlan) -> Dict[str, ServiceInstance]:
        """Assign service instances for query execution."""
        try:
            assignments = {}
            
            # Determine required services based on strategy
            needs_neo4j = plan.execution_strategy in [
                "graph_only", "hybrid_parallel", "hybrid_sequential"
            ]
            needs_qdrant = plan.execution_strategy in [
                "vector_only", "hybrid_parallel", "hybrid_sequential"
            ]
            
            # Assign Neo4j instance if needed
            if needs_neo4j:
                neo4j_instance = await self._select_best_instance(self.neo4j_instances, plan)
                if neo4j_instance:
                    assignments["neo4j"] = neo4j_instance
                    neo4j_instance.active_queries += 1
                else:
                    raise Exception("No available Neo4j instances")
            
            # Assign Qdrant instance if needed
            if needs_qdrant:
                qdrant_instance = await self._select_best_instance(self.qdrant_instances, plan)
                if qdrant_instance:
                    assignments["qdrant"] = qdrant_instance
                    qdrant_instance.active_queries += 1
                else:
                    raise Exception("No available Qdrant instances")
            
            # Track execution
            execution = QueryExecution(
                query_id=plan.query_id,
                plan=plan,
                assigned_instances=assignments,
                start_time=datetime.now(),
                estimated_completion=datetime.now() + timedelta(seconds=plan.estimated_time),
                priority=plan.priority
            )
            
            self.active_executions[plan.query_id] = execution
            
            return assignments
            
        except Exception as e:
            self.logger.error(f"Failed to assign query execution: {e}")
            return {}
    
    async def _select_best_instance(self, instances: List[ServiceInstance], plan: QueryPlan) -> Optional[ServiceInstance]:
        """Select the best instance based on current strategy."""
        # Filter healthy instances that can handle the request
        available_instances = [
            instance for instance in instances 
            if instance.is_healthy and instance.can_handle_request()
        ]
        
        if not available_instances:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_instances)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_instances)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_instances)
        
        elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._resource_aware_selection(available_instances, plan)
        
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return await self._adaptive_selection(available_instances, plan)
        
        else:
            # Default to least connections
            return self._least_connections_selection(available_instances)
    
    def _round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance using round-robin."""
        service_type = instances[0].service_type
        counter = self.round_robin_counters[service_type]
        selected = instances[counter % len(instances)]
        self.round_robin_counters[service_type] = (counter + 1) % len(instances)
        return selected
    
    def _least_connections_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least connections."""
        return min(instances, key=lambda x: x.current_connections)
    
    def _weighted_round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance using weighted round-robin."""
        # Calculate weighted selection
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return instances[0]
        
        # Simple weighted selection (can be optimized)
        import random
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.weight
            if rand_val <= current_weight:
                return instance
        
        return instances[-1]  # Fallback
    
    def _resource_aware_selection(self, instances: List[ServiceInstance], plan: QueryPlan) -> ServiceInstance:
        """Select instance based on resource requirements and availability."""
        # Score instances based on resource fit
        scored_instances = []
        
        for instance in instances:
            load_score = instance.get_load_score()
            response_time_score = min(instance.avg_response_time / 5.0, 1.0)  # Normalize to 5s max
            
            # Lower score is better
            total_score = (load_score * 0.6) + (response_time_score * 0.4)
            scored_instances.append((total_score, instance))
        
        # Return instance with lowest score
        return min(scored_instances, key=lambda x: x[0])[1]
    
    async def _adaptive_selection(self, instances: List[ServiceInstance], plan: QueryPlan) -> ServiceInstance:
        """Adaptive selection based on historical performance and current conditions."""
        # Use resource-aware selection as base
        base_selection = self._resource_aware_selection(instances, plan)
        
        # Check if we should override based on historical performance
        best_performer = self._get_best_historical_performer(instances, plan)
        
        if best_performer and best_performer != base_selection:
            # Check if best performer is significantly better and available
            base_load = base_selection.get_load_score()
            best_load = best_performer.get_load_score()
            
            # Use best performer if load difference is not too high
            if best_load - base_load < 0.3:
                return best_performer
        
        return base_selection
    
    def _get_best_historical_performer(self, instances: List[ServiceInstance], plan: QueryPlan) -> Optional[ServiceInstance]:
        """Get the instance with best historical performance for similar queries."""
        # Simple implementation - can be enhanced with ML
        performance_scores = {}
        
        for instance in instances:
            history_key = f"{instance.instance_id}_{plan.estimated_complexity.value}"
            if history_key in self.performance_history:
                recent_times = list(self.performance_history[history_key])[-10:]  # Last 10 executions
                if recent_times:
                    avg_time = sum(recent_times) / len(recent_times)
                    performance_scores[instance.instance_id] = avg_time
        
        if performance_scores:
            best_instance_id = min(performance_scores.keys(), key=lambda x: performance_scores[x])
            return next((inst for inst in instances if inst.instance_id == best_instance_id), None)
        
        return None
    
    async def complete_query_execution(self, query_id: str, execution_time: float, success: bool = True):
        """Mark query execution as complete and update metrics."""
        try:
            if query_id not in self.active_executions:
                return
            
            execution = self.active_executions[query_id]
            
            # Update instance metrics
            for service_type, instance in execution.assigned_instances.items():
                instance.active_queries = max(0, instance.active_queries - 1)
                
                # Update performance history
                history_key = f"{instance.instance_id}_{execution.plan.estimated_complexity.value}"
                self.performance_history[history_key].append(execution_time)
                
                # Update average response time
                if instance.avg_response_time == 0:
                    instance.avg_response_time = execution_time
                else:
                    instance.avg_response_time = (instance.avg_response_time * 0.9) + (execution_time * 0.1)
                
                # Update failure count
                if success:
                    instance.consecutive_failures = 0
                else:
                    instance.consecutive_failures += 1
                    if instance.consecutive_failures >= self.circuit_breaker_threshold:
                        instance.is_healthy = False
                        self.logger.warning(f"Instance {instance.instance_id} marked unhealthy due to consecutive failures")
            
            # Remove from active executions
            del self.active_executions[query_id]
            
        except Exception as e:
            self.logger.error(f"Failed to complete query execution: {e}")
    
    async def _health_check_loop(self):
        """Periodic health check for all instances."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all instances."""
        all_instances = self.neo4j_instances + self.qdrant_instances
        
        for instance in all_instances:
            try:
                # Simple health check - can be enhanced with actual service calls
                is_healthy = await self._check_instance_health(instance)
                
                if is_healthy:
                    if not instance.is_healthy:
                        self.logger.info(f"Instance {instance.instance_id} recovered")
                    instance.is_healthy = True
                    instance.consecutive_failures = 0
                else:
                    instance.consecutive_failures += 1
                    if instance.consecutive_failures >= self.circuit_breaker_threshold:
                        if instance.is_healthy:
                            self.logger.warning(f"Instance {instance.instance_id} marked unhealthy")
                        instance.is_healthy = False
                
                instance.last_health_check = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Health check failed for {instance.instance_id}: {e}")
                instance.consecutive_failures += 1
    
    async def _check_instance_health(self, instance: ServiceInstance) -> bool:
        """Check health of a specific instance."""
        # Placeholder implementation - should be replaced with actual health checks
        # For Neo4j: simple Cypher query
        # For Qdrant: collection info request
        
        # Simulate health check
        import random
        return random.random() > 0.1  # 90% success rate for simulation
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get current load balancer statistics."""
        neo4j_stats = self._get_service_stats(self.neo4j_instances)
        qdrant_stats = self._get_service_stats(self.qdrant_instances)
        
        return {
            "strategy": self.strategy.value,
            "neo4j": neo4j_stats,
            "qdrant": qdrant_stats,
            "active_executions": len(self.active_executions),
            "queue_size": len(self.execution_queue),
            "overdue_executions": sum(1 for exec in self.active_executions.values() if exec.is_overdue())
        }
    
    def _get_service_stats(self, instances: List[ServiceInstance]) -> Dict[str, Any]:
        """Get statistics for a service type."""
        if not instances:
            return {"total_instances": 0}
        
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        
        return {
            "total_instances": len(instances),
            "healthy_instances": len(healthy_instances),
            "total_connections": sum(inst.current_connections for inst in instances),
            "total_active_queries": sum(inst.active_queries for inst in instances),
            "avg_response_time": sum(inst.avg_response_time for inst in instances) / len(instances),
            "avg_load_score": sum(inst.get_load_score() for inst in instances) / len(instances)
        }
    
    def update_instance_metrics(self, instance_id: str, metrics: Dict[str, Any]):
        """Update metrics for a specific instance."""
        # Find instance
        instance = None
        for inst in self.neo4j_instances + self.qdrant_instances:
            if inst.instance_id == instance_id:
                instance = inst
                break
        
        if not instance:
            return
        
        # Update metrics
        instance.current_connections = metrics.get("current_connections", instance.current_connections)
        instance.cpu_usage = metrics.get("cpu_usage", instance.cpu_usage)
        instance.memory_usage = metrics.get("memory_usage", instance.memory_usage)
        instance.requests_per_second = metrics.get("requests_per_second", instance.requests_per_second)
        instance.queue_size = metrics.get("queue_size", instance.queue_size)
        
        # Store in load history
        load_key = f"{instance_id}_load"
        self.load_history[load_key].append({
            "timestamp": datetime.now(),
            "load_score": instance.get_load_score(),
            "connections": instance.current_connections,
            "cpu_usage": instance.cpu_usage,
            "memory_usage": instance.memory_usage
        })
```

## Testing Strategy

### Unit Tests

Create `tests/test_query_coordination.py`:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from morag_graph.coordination.query_planner import (
    QueryPlanner, QueryPlan, QueryComplexity, SystemResources
)
from morag_graph.coordination.load_balancer import (
    LoadBalancer, ServiceInstance, LoadBalancingStrategy
)
from morag_graph.retrieval.hybrid_retriever import (
    RetrievalQuery, RetrievalStrategy, QueryType
)

@pytest.fixture
def query_planner():
    return QueryPlanner()

@pytest.fixture
def load_balancer():
    return LoadBalancer(LoadBalancingStrategy.ADAPTIVE)

@pytest.fixture
def sample_query():
    return RetrievalQuery(
        query_text="Find documents about machine learning",
        query_type=QueryType.SEMANTIC_SEARCH,
        strategy=RetrievalStrategy.ADAPTIVE,
        query_vector=[0.1] * 384
    )

@pytest.mark.asyncio
async def test_query_plan_creation(query_planner, sample_query):
    plan = await query_planner.create_query_plan(sample_query)
    
    assert isinstance(plan, QueryPlan)
    assert plan.query_id is not None
    assert plan.original_query == sample_query
    assert isinstance(plan.estimated_complexity, QueryComplexity)
    assert plan.estimated_cost > 0
    assert plan.estimated_time > 0

@pytest.mark.asyncio
async def test_complexity_estimation(query_planner):
    # Simple query
    simple_query = RetrievalQuery(
        query_text="test",
        query_type=QueryType.SEMANTIC_SEARCH
    )
    
    analysis = await query_planner._analyze_query_characteristics(simple_query)
    complexity = query_planner._estimate_query_complexity(simple_query, analysis)
    
    assert complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
    
    # Complex query
    complex_query = RetrievalQuery(
        query_text="Find all relationships between entities in the knowledge graph related to artificial intelligence research conducted by universities in the past five years",
        query_type=QueryType.MULTI_HOP_REASONING,
        max_depth=5,
        entity_types=["University", "Researcher", "Publication"],
        relationship_types=["AUTHORED", "AFFILIATED_WITH", "CITES"]
    )
    
    analysis = await query_planner._analyze_query_characteristics(complex_query)
    complexity = query_planner._estimate_query_complexity(complex_query, analysis)
    
    assert complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]

@pytest.mark.asyncio
async def test_strategy_selection(query_planner):
    # Entity lookup should prefer graph-only
    entity_query = RetrievalQuery(
        query_text="Find John Doe",
        query_type=QueryType.ENTITY_LOOKUP,
        strategy=RetrievalStrategy.ADAPTIVE
    )
    
    analysis = await query_planner._analyze_query_characteristics(entity_query)
    strategy = await query_planner._select_optimal_strategy(entity_query, analysis)
    
    assert strategy == RetrievalStrategy.GRAPH_ONLY

@pytest.mark.asyncio
async def test_batch_optimization(query_planner):
    queries = [
        RetrievalQuery(f"Query {i}", QueryType.SEMANTIC_SEARCH) 
        for i in range(5)
    ]
    
    plans = await query_planner.optimize_query_batch(queries)
    
    assert len(plans) == len(queries)
    assert all(isinstance(plan, QueryPlan) for plan in plans)
    
    # Check that plans are ordered by priority
    priorities = [plan.priority for plan in plans]
    assert priorities == sorted(priorities, reverse=True)

@pytest.mark.asyncio
async def test_load_balancer_instance_registration(load_balancer):
    neo4j_instance = ServiceInstance(
        instance_id="neo4j-1",
        service_type="neo4j",
        endpoint="bolt://localhost:7687"
    )
    
    qdrant_instance = ServiceInstance(
        instance_id="qdrant-1",
        service_type="qdrant",
        endpoint="http://localhost:6333"
    )
    
    load_balancer.register_neo4j_instance(neo4j_instance)
    load_balancer.register_qdrant_instance(qdrant_instance)
    
    assert len(load_balancer.neo4j_instances) == 1
    assert len(load_balancer.qdrant_instances) == 1
    assert load_balancer.neo4j_instances[0].instance_id == "neo4j-1"
    assert load_balancer.qdrant_instances[0].instance_id == "qdrant-1"

@pytest.mark.asyncio
async def test_query_assignment(load_balancer, sample_query):
    # Register instances
    neo4j_instance = ServiceInstance("neo4j-1", "neo4j", "bolt://localhost:7687")
    qdrant_instance = ServiceInstance("qdrant-1", "qdrant", "http://localhost:6333")
    
    load_balancer.register_neo4j_instance(neo4j_instance)
    load_balancer.register_qdrant_instance(qdrant_instance)
    
    # Create plan
    plan = QueryPlan(
        query_id="test-123",
        original_query=sample_query,
        execution_strategy=RetrievalStrategy.HYBRID_PARALLEL,
        estimated_complexity=QueryComplexity.MODERATE,
        estimated_cost=1.0,
        estimated_time=0.5
    )
    
    assignments = await load_balancer.assign_query_execution(plan)
    
    assert "neo4j" in assignments
    assert "qdrant" in assignments
    assert assignments["neo4j"].instance_id == "neo4j-1"
    assert assignments["qdrant"].instance_id == "qdrant-1"

@pytest.mark.asyncio
async def test_load_balancing_strategies(load_balancer):
    # Create multiple instances
    instances = [
        ServiceInstance(f"test-{i}", "neo4j", f"bolt://host{i}:7687", weight=i+1)
        for i in range(3)
    ]
    
    # Test round robin
    load_balancer.strategy = LoadBalancingStrategy.ROUND_ROBIN
    selected1 = load_balancer._round_robin_selection(instances)
    selected2 = load_balancer._round_robin_selection(instances)
    selected3 = load_balancer._round_robin_selection(instances)
    
    assert selected1 != selected2 or selected2 != selected3
    
    # Test least connections
    instances[0].current_connections = 5
    instances[1].current_connections = 2
    instances[2].current_connections = 8
    
    selected = load_balancer._least_connections_selection(instances)
    assert selected.instance_id == "test-1"  # Has least connections

def test_service_instance_load_calculation():
    instance = ServiceInstance("test", "neo4j", "bolt://localhost:7687")
    instance.current_connections = 50
    instance.max_connections = 100
    instance.cpu_usage = 0.6
    instance.memory_usage = 0.4
    instance.queue_size = 20
    
    load_score = instance.get_load_score()
    
    # Should be around (0.5 + 0.5 + 0.2) / 3 = 0.4
    assert 0.3 <= load_score <= 0.5

def test_service_instance_can_handle_request():
    instance = ServiceInstance("test", "neo4j", "bolt://localhost:7687")
    
    # Healthy instance with capacity
    assert instance.can_handle_request()
    
    # Unhealthy instance
    instance.is_healthy = False
    assert not instance.can_handle_request()
    
    # At capacity
    instance.is_healthy = True
    instance.current_connections = instance.max_connections
    assert not instance.can_handle_request()

@pytest.mark.asyncio
async def test_performance_history_update(query_planner, sample_query):
    plan = await query_planner.create_query_plan(sample_query)
    
    # Update performance history
    metrics = {
        "total_time": 0.8,
        "total_cost": 1.2,
        "success": True
    }
    
    query_planner.update_performance_history(plan, metrics)
    
    # Check that history was updated
    key = f"{plan.original_query.query_type.value}_{plan.execution_strategy.value}"
    assert key in query_planner.performance_history
    assert len(query_planner.performance_history[key]) == 1
    
    record = query_planner.performance_history[key][0]
    assert record["actual_time"] == 0.8
    assert record["success"] is True

@pytest.mark.asyncio
async def test_resource_constraint_handling(query_planner, sample_query):
    # Set high resource usage
    high_usage_resources = SystemResources(
        cpu_usage=0.9,
        memory_usage=0.9,
        pending_queries=900
    )
    
    query_planner.update_system_resources(high_usage_resources)
    
    plan = await query_planner.create_query_plan(sample_query)
    
    # Should adjust for high resource usage
    assert plan.estimated_time > 0  # Should still create a plan
    # May have reduced limits or different strategy

@pytest.mark.asyncio
async def test_circuit_breaker_functionality(load_balancer):
    instance = ServiceInstance("test", "neo4j", "bolt://localhost:7687")
    load_balancer.register_neo4j_instance(instance)
    
    # Simulate consecutive failures
    for _ in range(load_balancer.circuit_breaker_threshold):
        await load_balancer.complete_query_execution("test-query", 1.0, success=False)
        instance.consecutive_failures += 1
    
    # Instance should be marked unhealthy after threshold
    if instance.consecutive_failures >= load_balancer.circuit_breaker_threshold:
        instance.is_healthy = False
    
    assert not instance.is_healthy
```

### Integration Tests

Create `tests/test_coordination_integration.py`:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from morag_graph.coordination.query_planner import QueryPlanner
from morag_graph.coordination.load_balancer import LoadBalancer, ServiceInstance
from morag_graph.retrieval.hybrid_retriever import HybridRetriever

@pytest.mark.asyncio
async def test_end_to_end_coordination():
    """Test complete coordination workflow."""
    # Setup components
    planner = QueryPlanner()
    load_balancer = LoadBalancer()
    
    # Register service instances
    neo4j_instance = ServiceInstance("neo4j-1", "neo4j", "bolt://localhost:7687")
    qdrant_instance = ServiceInstance("qdrant-1", "qdrant", "http://localhost:6333")
    
    load_balancer.register_neo4j_instance(neo4j_instance)
    load_balancer.register_qdrant_instance(qdrant_instance)
    
    # Create and plan query
    from morag_graph.retrieval.hybrid_retriever import RetrievalQuery, QueryType
    
    query = RetrievalQuery(
        query_text="Find research papers about neural networks",
        query_type=QueryType.SEMANTIC_SEARCH
    )
    
    plan = await planner.create_query_plan(query)
    assignments = await load_balancer.assign_query_execution(plan)
    
    # Verify coordination
    assert plan.query_id is not None
    assert len(assignments) > 0
    
    # Complete execution
    await load_balancer.complete_query_execution(plan.query_id, 0.5, success=True)
    
    # Verify cleanup
    assert plan.query_id not in load_balancer.active_executions

@pytest.mark.asyncio
async def test_coordination_under_load():
    """Test coordination behavior under high load."""
    planner = QueryPlanner()
    load_balancer = LoadBalancer()
    
    # Register multiple instances
    for i in range(3):
        neo4j_instance = ServiceInstance(f"neo4j-{i}", "neo4j", f"bolt://host{i}:7687")
        qdrant_instance = ServiceInstance(f"qdrant-{i}", "qdrant", f"http://host{i}:6333")
        
        load_balancer.register_neo4j_instance(neo4j_instance)
        load_balancer.register_qdrant_instance(qdrant_instance)
    
    # Create multiple queries
    from morag_graph.retrieval.hybrid_retriever import RetrievalQuery, QueryType
    
    queries = [
        RetrievalQuery(f"Query {i}", QueryType.SEMANTIC_SEARCH)
        for i in range(10)
    ]
    
    # Plan and assign all queries
    plans = await planner.optimize_query_batch(queries)
    assignments = []
    
    for plan in plans:
        assignment = await load_balancer.assign_query_execution(plan)
        assignments.append(assignment)
    
    # Verify all queries were assigned
    assert len(assignments) == len(queries)
    assert all(len(assignment) > 0 for assignment in assignments)
    
    # Verify load distribution
    neo4j_usage = {}
    qdrant_usage = {}
    
    for assignment in assignments:
        if "neo4j" in assignment:
            instance_id = assignment["neo4j"].instance_id
            neo4j_usage[instance_id] = neo4j_usage.get(instance_id, 0) + 1
        
        if "qdrant" in assignment:
            instance_id = assignment["qdrant"].instance_id
            qdrant_usage[instance_id] = qdrant_usage.get(instance_id, 0) + 1
    
    # Should distribute load across instances
    if len(neo4j_usage) > 1:
        assert max(neo4j_usage.values()) - min(neo4j_usage.values()) <= 2
```

## Performance Considerations

### Optimization Strategies

1. **Query Planning Cache**:
   - Cache query plans for similar queries
   - Use query fingerprinting for cache keys
   - Implement TTL-based cache invalidation

2. **Adaptive Load Balancing**:
   - Machine learning-based instance selection
   - Real-time performance feedback
   - Predictive load balancing

3. **Resource Monitoring**:
   - Real-time resource utilization tracking
   - Predictive scaling triggers
   - Automated failover mechanisms

4. **Query Optimization**:
   - Query rewriting for better performance
   - Automatic index recommendations
   - Query execution plan caching

### Performance Targets

- **Query Planning Time**: < 10ms for simple queries, < 50ms for complex queries
- **Load Balancing Decision**: < 5ms per assignment
- **Health Check Frequency**: Every 30 seconds
- **Circuit Breaker Recovery**: 5 minutes timeout
- **Resource Monitoring**: Real-time updates

## Success Criteria

- [ ] Query planner creates optimized execution plans
- [ ] Load balancer distributes queries efficiently across instances
- [ ] Health checking detects and handles instance failures
- [ ] Circuit breaker prevents cascade failures
- [ ] Performance monitoring provides actionable insights
- [ ] Resource-aware optimization reduces system load
- [ ] Adaptive strategies improve over time
- [ ] All tests pass with >95% coverage

## Risk Assessment

**Risk Level**: High

**Key Risks**:
1. **Coordination Overhead**: Complex coordination logic may introduce latency
2. **Single Point of Failure**: Centralized coordination service
3. **Resource Contention**: Multiple queries competing for limited resources
4. **State Synchronization**: Keeping coordination state consistent

**Mitigation Strategies**:
1. Implement coordination service clustering
2. Use asynchronous processing where possible
3. Implement graceful degradation modes
4. Add comprehensive monitoring and alerting

## Rollback Plan

1. **Immediate Rollback**: Disable coordination service, use simple round-robin
2. **Partial Rollback**: Disable specific optimization features
3. **Configuration Rollback**: Revert to previous coordination settings
4. **Data Preservation**: Maintain performance history during rollback

## Next Steps

- **Task 4.3**: Performance Monitoring and Analytics Dashboard
- **Integration**: Connect with existing hybrid retrieval pipeline
- **Optimization**: Implement ML-based query optimization
- **Scaling**: Add support for distributed coordination

## Dependencies

- **Task 4.1**: Hybrid Retrieval Pipeline (provides query execution interface)
- **Task 2.1**: Bidirectional Reference Storage (for cross-system coordination)
- **Task 3.1**: Vector Integration Service (for Qdrant coordination)
- **Task 3.2**: Graph Query Enhancement (for Neo4j coordination)

## Estimated Time

**6-8 days**

## Status

- [ ] Query planner implementation
- [ ] Load balancer implementation  
- [ ] Health checking system
- [ ] Performance monitoring
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance optimization
- [ ] Documentation