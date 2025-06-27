# Task 3.2: Selective Vector Strategy

## Overview

This task focuses on implementing an intelligent strategy for determining when and where to store vector embeddings - whether in Neo4j for graph-aware operations, Qdrant for high-performance similarity search, or both systems for hybrid approaches.

## Objectives

- Define criteria for selective vector storage placement
- Implement decision engine for vector storage strategy
- Create performance-based routing for vector operations
- Establish vector lifecycle management policies
- Enable dynamic strategy adjustment based on usage patterns

## Implementation Plan

### 1. Vector Storage Strategy Framework

```python
from typing import Dict, List, Optional, Tuple, Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
from enum import Enum

class VectorStorageLocation(Enum):
    """Enum for vector storage locations"""
    NEO4J_ONLY = "neo4j_only"
    QDRANT_ONLY = "qdrant_only"
    BOTH_SYSTEMS = "both_systems"
    NONE = "none"

class VectorType(Enum):
    """Enum for vector types"""
    ENTITY_EMBEDDING = "entity_embedding"
    RELATION_EMBEDDING = "relation_embedding"
    DOCUMENT_SUMMARY = "document_summary"
    CHUNK_EMBEDDING = "chunk_embedding"
    TOPIC_EMBEDDING = "topic_embedding"

@dataclass
class VectorStorageDecision:
    """Decision result for vector storage"""
    location: VectorStorageLocation
    reasoning: str
    confidence: float
    metadata: Dict
    estimated_cost: float
    performance_impact: str

@dataclass
class VectorUsagePattern:
    """Usage pattern for vector operations"""
    vector_id: str
    vector_type: VectorType
    similarity_queries_count: int
    graph_traversal_count: int
    last_accessed: datetime
    access_frequency: float  # accesses per day
    avg_query_time: float
    storage_size: int

class VectorStorageStrategy:
    """Main strategy engine for vector storage decisions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Strategy thresholds
        self.neo4j_graph_threshold = config.get("neo4j_graph_threshold", 0.3)
        self.qdrant_similarity_threshold = config.get("qdrant_similarity_threshold", 0.7)
        self.both_systems_threshold = config.get("both_systems_threshold", 0.8)
        self.access_frequency_threshold = config.get("access_frequency_threshold", 10.0)  # per day
        
        # Performance thresholds
        self.max_neo4j_vector_size = config.get("max_neo4j_vector_size", 1000000)  # 1M vectors
        self.max_query_time_ms = config.get("max_query_time_ms", 500)
        
        # Cost factors
        self.neo4j_storage_cost_factor = config.get("neo4j_storage_cost", 1.5)
        self.qdrant_storage_cost_factor = config.get("qdrant_storage_cost", 1.0)
        self.dual_storage_cost_factor = config.get("dual_storage_cost", 2.2)
    
    async def decide_vector_storage(
        self, 
        vector_type: VectorType,
        vector_size: int,
        expected_usage: Dict,
        entity_context: Optional[Dict] = None
    ) -> VectorStorageDecision:
        """Decide where to store a vector based on multiple factors"""
        try:
            # Calculate decision scores
            neo4j_score = await self._calculate_neo4j_score(
                vector_type, vector_size, expected_usage, entity_context
            )
            
            qdrant_score = await self._calculate_qdrant_score(
                vector_type, vector_size, expected_usage, entity_context
            )
            
            # Determine storage location
            location, reasoning, confidence = self._determine_location(
                neo4j_score, qdrant_score, vector_type
            )
            
            # Calculate estimated costs
            estimated_cost = self._calculate_storage_cost(location, vector_size)
            
            # Assess performance impact
            performance_impact = self._assess_performance_impact(
                location, vector_type, expected_usage
            )
            
            decision = VectorStorageDecision(
                location=location,
                reasoning=reasoning,
                confidence=confidence,
                metadata={
                    "neo4j_score": neo4j_score,
                    "qdrant_score": qdrant_score,
                    "vector_type": vector_type.value,
                    "vector_size": vector_size,
                    "decision_timestamp": datetime.utcnow().isoformat()
                },
                estimated_cost=estimated_cost,
                performance_impact=performance_impact
            )
            
            self.logger.info(
                f"Vector storage decision for {vector_type.value}: {location.value} "
                f"(confidence: {confidence:.2f})"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in vector storage decision: {e}")
            # Default to Qdrant for safety
            return VectorStorageDecision(
                location=VectorStorageLocation.QDRANT_ONLY,
                reasoning=f"Error in decision process: {e}",
                confidence=0.5,
                metadata={"error": str(e)},
                estimated_cost=0.0,
                performance_impact="unknown"
            )
    
    async def _calculate_neo4j_score(
        self, 
        vector_type: VectorType,
        vector_size: int,
        expected_usage: Dict,
        entity_context: Optional[Dict]
    ) -> float:
        """Calculate score for storing in Neo4j"""
        score = 0.0
        
        # Base scores by vector type
        type_scores = {
            VectorType.ENTITY_EMBEDDING: 0.8,  # High - entities are core to graph
            VectorType.RELATION_EMBEDDING: 0.7,  # High - relations are graph-native
            VectorType.DOCUMENT_SUMMARY: 0.4,  # Medium - useful for document clustering
            VectorType.CHUNK_EMBEDDING: 0.2,  # Low - primarily for similarity search
            VectorType.TOPIC_EMBEDDING: 0.5   # Medium - useful for topic clustering
        }
        
        score += type_scores.get(vector_type, 0.3)
        
        # Graph traversal usage factor
        graph_usage = expected_usage.get("graph_traversal_ratio", 0.0)
        score += graph_usage * 0.4
        
        # Entity connectivity factor
        if entity_context:
            connection_count = entity_context.get("connection_count", 0)
            if connection_count > 10:
                score += 0.2
            elif connection_count > 5:
                score += 0.1
        
        # Size penalty for Neo4j
        if vector_size > 1000:
            score -= 0.1
        if vector_size > 10000:
            score -= 0.2
        
        # Frequency bonus for frequently accessed vectors
        access_frequency = expected_usage.get("access_frequency", 0.0)
        if access_frequency > self.access_frequency_threshold:
            score += 0.15
        
        return min(max(score, 0.0), 1.0)
    
    async def _calculate_qdrant_score(
        self, 
        vector_type: VectorType,
        vector_size: int,
        expected_usage: Dict,
        entity_context: Optional[Dict]
    ) -> float:
        """Calculate score for storing in Qdrant"""
        score = 0.0
        
        # Base scores by vector type
        type_scores = {
            VectorType.ENTITY_EMBEDDING: 0.6,  # Medium - useful for entity similarity
            VectorType.RELATION_EMBEDDING: 0.5,  # Medium - less critical in Qdrant
            VectorType.DOCUMENT_SUMMARY: 0.7,  # High - good for document similarity
            VectorType.CHUNK_EMBEDDING: 0.9,  # Very high - primary use case
            VectorType.TOPIC_EMBEDDING: 0.8   # High - excellent for topic similarity
        }
        
        score += type_scores.get(vector_type, 0.5)
        
        # Similarity search usage factor
        similarity_usage = expected_usage.get("similarity_search_ratio", 0.0)
        score += similarity_usage * 0.4
        
        # Large dataset bonus
        if vector_size > 1000:
            score += 0.1
        if vector_size > 10000:
            score += 0.2
        
        # Performance requirement factor
        performance_requirement = expected_usage.get("performance_requirement", "medium")
        if performance_requirement == "high":
            score += 0.2
        elif performance_requirement == "critical":
            score += 0.3
        
        # Batch operation factor
        batch_operations = expected_usage.get("batch_operations", False)
        if batch_operations:
            score += 0.15
        
        return min(max(score, 0.0), 1.0)
    
    def _determine_location(
        self, 
        neo4j_score: float, 
        qdrant_score: float, 
        vector_type: VectorType
    ) -> Tuple[VectorStorageLocation, str, float]:
        """Determine storage location based on scores"""
        
        score_diff = abs(neo4j_score - qdrant_score)
        max_score = max(neo4j_score, qdrant_score)
        
        # If both scores are high and close, use both systems
        if max_score > self.both_systems_threshold and score_diff < 0.2:
            return (
                VectorStorageLocation.BOTH_SYSTEMS,
                f"Both systems beneficial (Neo4j: {neo4j_score:.2f}, Qdrant: {qdrant_score:.2f})",
                max_score
            )
        
        # If Neo4j score is significantly higher
        if neo4j_score > qdrant_score and neo4j_score > self.neo4j_graph_threshold:
            confidence = neo4j_score if score_diff > 0.2 else neo4j_score * 0.8
            return (
                VectorStorageLocation.NEO4J_ONLY,
                f"Graph operations favored (score: {neo4j_score:.2f})",
                confidence
            )
        
        # If Qdrant score is significantly higher
        if qdrant_score > neo4j_score and qdrant_score > self.qdrant_similarity_threshold:
            confidence = qdrant_score if score_diff > 0.2 else qdrant_score * 0.8
            return (
                VectorStorageLocation.QDRANT_ONLY,
                f"Similarity search favored (score: {qdrant_score:.2f})",
                confidence
            )
        
        # Default to Qdrant for chunk embeddings, Neo4j for entity/relation embeddings
        if vector_type in [VectorType.CHUNK_EMBEDDING, VectorType.DOCUMENT_SUMMARY]:
            return (
                VectorStorageLocation.QDRANT_ONLY,
                "Default strategy for content vectors",
                0.6
            )
        else:
            return (
                VectorStorageLocation.NEO4J_ONLY,
                "Default strategy for graph vectors",
                0.6
            )
    
    def _calculate_storage_cost(
        self, 
        location: VectorStorageLocation, 
        vector_size: int
    ) -> float:
        """Calculate estimated storage cost"""
        base_cost = vector_size * 0.001  # Base cost per vector dimension
        
        if location == VectorStorageLocation.NEO4J_ONLY:
            return base_cost * self.neo4j_storage_cost_factor
        elif location == VectorStorageLocation.QDRANT_ONLY:
            return base_cost * self.qdrant_storage_cost_factor
        elif location == VectorStorageLocation.BOTH_SYSTEMS:
            return base_cost * self.dual_storage_cost_factor
        else:
            return 0.0
    
    def _assess_performance_impact(
        self, 
        location: VectorStorageLocation,
        vector_type: VectorType,
        expected_usage: Dict
    ) -> str:
        """Assess performance impact of storage decision"""
        if location == VectorStorageLocation.BOTH_SYSTEMS:
            return "High performance for both graph and similarity operations"
        elif location == VectorStorageLocation.NEO4J_ONLY:
            return "Optimized for graph traversal, limited similarity search performance"
        elif location == VectorStorageLocation.QDRANT_ONLY:
            return "Optimized for similarity search, no graph-aware operations"
        else:
            return "No vector operations available"

class VectorUsageAnalyzer:
    """Analyzer for vector usage patterns to inform strategy decisions"""
    
    def __init__(self, neo4j_storage, qdrant_storage):
        self.neo4j_storage = neo4j_storage
        self.qdrant_storage = qdrant_storage
        self.logger = logging.getLogger(__name__)
        self.usage_cache = {}
    
    async def analyze_vector_usage(
        self, 
        vector_id: str, 
        vector_type: VectorType,
        time_window_days: int = 30
    ) -> VectorUsagePattern:
        """Analyze usage patterns for a specific vector"""
        try:
            # Check cache first
            cache_key = f"{vector_id}_{time_window_days}"
            if cache_key in self.usage_cache:
                cached_result = self.usage_cache[cache_key]
                if (datetime.utcnow() - cached_result['timestamp']).seconds < 3600:  # 1 hour cache
                    return cached_result['pattern']
            
            # Analyze Neo4j usage
            neo4j_usage = await self._analyze_neo4j_usage(vector_id, time_window_days)
            
            # Analyze Qdrant usage
            qdrant_usage = await self._analyze_qdrant_usage(vector_id, time_window_days)
            
            # Combine usage patterns
            pattern = VectorUsagePattern(
                vector_id=vector_id,
                vector_type=vector_type,
                similarity_queries_count=qdrant_usage.get('similarity_queries', 0),
                graph_traversal_count=neo4j_usage.get('graph_traversals', 0),
                last_accessed=max(
                    neo4j_usage.get('last_accessed', datetime.min),
                    qdrant_usage.get('last_accessed', datetime.min)
                ),
                access_frequency=(
                    neo4j_usage.get('access_frequency', 0) + 
                    qdrant_usage.get('access_frequency', 0)
                ),
                avg_query_time=(
                    neo4j_usage.get('avg_query_time', 0) + 
                    qdrant_usage.get('avg_query_time', 0)
                ) / 2,
                storage_size=max(
                    neo4j_usage.get('storage_size', 0),
                    qdrant_usage.get('storage_size', 0)
                )
            )
            
            # Cache result
            self.usage_cache[cache_key] = {
                'pattern': pattern,
                'timestamp': datetime.utcnow()
            }
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Error analyzing vector usage for {vector_id}: {e}")
            # Return default pattern
            return VectorUsagePattern(
                vector_id=vector_id,
                vector_type=vector_type,
                similarity_queries_count=0,
                graph_traversal_count=0,
                last_accessed=datetime.utcnow(),
                access_frequency=0.0,
                avg_query_time=0.0,
                storage_size=0
            )
    
    async def _analyze_neo4j_usage(self, vector_id: str, time_window_days: int) -> Dict:
        """Analyze Neo4j usage patterns"""
        try:
            # Query Neo4j for usage statistics
            # Note: This would require implementing query logging in Neo4j
            query = """
            MATCH (e:Entity {id: $vector_id})
            OPTIONAL MATCH (e)-[r]-()
            WITH e, count(r) as connection_count
            RETURN e.id as id, 
                   e.embedding_vector IS NOT NULL as has_embedding,
                   connection_count,
                   e.last_accessed as last_accessed,
                   e.access_count as access_count
            """
            
            result = await self.neo4j_storage.execute_query(query, {"vector_id": vector_id})
            
            if result and len(result) > 0:
                data = result[0]
                return {
                    'graph_traversals': data.get('access_count', 0),
                    'last_accessed': data.get('last_accessed', datetime.min),
                    'access_frequency': data.get('access_count', 0) / time_window_days,
                    'avg_query_time': 100,  # Placeholder - would need query performance tracking
                    'storage_size': 384 if data.get('has_embedding') else 0
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error analyzing Neo4j usage: {e}")
            return {}
    
    async def _analyze_qdrant_usage(self, vector_id: str, time_window_days: int) -> Dict:
        """Analyze Qdrant usage patterns"""
        try:
            # Query Qdrant for usage statistics
            # Note: This would require implementing usage tracking in Qdrant
            
            # For now, return placeholder data
            # In a real implementation, you would query Qdrant metrics or logs
            return {
                'similarity_queries': 0,  # Would come from Qdrant metrics
                'last_accessed': datetime.min,
                'access_frequency': 0.0,
                'avg_query_time': 50,  # Placeholder
                'storage_size': 384  # Standard embedding size
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing Qdrant usage: {e}")
            return {}
    
    async def get_system_usage_summary(self) -> Dict:
        """Get overall system usage summary"""
        try:
            # Neo4j statistics
            neo4j_stats = await self.neo4j_storage.execute_query("""
                MATCH (e:Entity)
                WITH count(e) as total_entities,
                     count(e.embedding_vector) as entities_with_embeddings
                MATCH ()-[r:RELATION]-()
                WITH total_entities, entities_with_embeddings,
                     count(r) as total_relations,
                     count(r.embedding_vector) as relations_with_embeddings
                RETURN total_entities, entities_with_embeddings,
                       total_relations, relations_with_embeddings
            """)
            
            # Qdrant statistics (placeholder)
            qdrant_stats = {
                'total_vectors': 0,  # Would query Qdrant collection info
                'collection_size': 0,
                'avg_query_time': 0
            }
            
            return {
                'neo4j': neo4j_stats[0] if neo4j_stats else {},
                'qdrant': qdrant_stats,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system usage summary: {e}")
            return {}

class VectorMigrationService:
    """Service for migrating vectors between storage systems"""
    
    def __init__(self, neo4j_vector_storage, qdrant_storage, strategy_engine):
        self.neo4j_vectors = neo4j_vector_storage
        self.qdrant_storage = qdrant_storage
        self.strategy = strategy_engine
        self.logger = logging.getLogger(__name__)
    
    async def migrate_vector(
        self, 
        vector_id: str,
        from_location: VectorStorageLocation,
        to_location: VectorStorageLocation,
        vector_type: VectorType
    ) -> bool:
        """Migrate a vector from one storage system to another"""
        try:
            # Get vector data from source
            vector_data = await self._get_vector_data(vector_id, from_location, vector_type)
            
            if not vector_data:
                self.logger.error(f"Could not retrieve vector data for {vector_id}")
                return False
            
            # Store in destination
            success = await self._store_vector_data(vector_id, to_location, vector_type, vector_data)
            
            if success:
                # Remove from source if migration to single system
                if to_location != VectorStorageLocation.BOTH_SYSTEMS:
                    await self._remove_vector_data(vector_id, from_location, vector_type)
                
                self.logger.info(f"Successfully migrated vector {vector_id} from {from_location.value} to {to_location.value}")
                return True
            else:
                self.logger.error(f"Failed to store vector {vector_id} in {to_location.value}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error migrating vector {vector_id}: {e}")
            return False
    
    async def _get_vector_data(self, vector_id: str, location: VectorStorageLocation, vector_type: VectorType) -> Optional[Dict]:
        """Get vector data from specified location"""
        if location == VectorStorageLocation.NEO4J_ONLY:
            if vector_type == VectorType.ENTITY_EMBEDDING:
                result = await self.neo4j_vectors.neo4j_storage.execute_query(
                    "MATCH (e:Entity {id: $id}) RETURN e.embedding_vector as vector, e.name as name",
                    {"id": vector_id}
                )
                if result:
                    return {
                        "vector": result[0]["vector"],
                        "metadata": {"name": result[0]["name"]}
                    }
        
        elif location == VectorStorageLocation.QDRANT_ONLY:
            # Query Qdrant for vector data
            # Implementation would depend on Qdrant client
            pass
        
        return None
    
    async def _store_vector_data(self, vector_id: str, location: VectorStorageLocation, vector_type: VectorType, vector_data: Dict) -> bool:
        """Store vector data in specified location"""
        try:
            if location == VectorStorageLocation.NEO4J_ONLY:
                if vector_type == VectorType.ENTITY_EMBEDDING:
                    return await self.neo4j_vectors.store_entity_embedding(
                        entity_id=vector_id,
                        text=vector_data["metadata"].get("name", "")
                    )
            
            elif location == VectorStorageLocation.QDRANT_ONLY:
                # Store in Qdrant
                # Implementation would depend on Qdrant client
                pass
            
            elif location == VectorStorageLocation.BOTH_SYSTEMS:
                # Store in both systems
                neo4j_success = await self._store_vector_data(vector_id, VectorStorageLocation.NEO4J_ONLY, vector_type, vector_data)
                qdrant_success = await self._store_vector_data(vector_id, VectorStorageLocation.QDRANT_ONLY, vector_type, vector_data)
                return neo4j_success and qdrant_success
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error storing vector data: {e}")
            return False
    
    async def _remove_vector_data(self, vector_id: str, location: VectorStorageLocation, vector_type: VectorType) -> bool:
        """Remove vector data from specified location"""
        try:
            if location == VectorStorageLocation.NEO4J_ONLY:
                if vector_type == VectorType.ENTITY_EMBEDDING:
                    await self.neo4j_vectors.neo4j_storage.execute_query(
                        "MATCH (e:Entity {id: $id}) SET e.embedding_vector = null",
                        {"id": vector_id}
                    )
                    return True
            
            elif location == VectorStorageLocation.QDRANT_ONLY:
                # Remove from Qdrant
                # Implementation would depend on Qdrant client
                pass
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error removing vector data: {e}")
            return False
    
    async def bulk_migrate_by_strategy(self, vector_ids: List[str], vector_type: VectorType) -> Dict:
        """Migrate multiple vectors based on current strategy"""
        results = {
            "successful": 0,
            "failed": 0,
            "migrations": []
        }
        
        for vector_id in vector_ids:
            try:
                # Analyze current usage
                usage_analyzer = VectorUsageAnalyzer(self.neo4j_vectors.neo4j_storage, self.qdrant_storage)
                usage_pattern = await usage_analyzer.analyze_vector_usage(vector_id, vector_type)
                
                # Get new strategy decision
                expected_usage = {
                    "similarity_search_ratio": usage_pattern.similarity_queries_count / max(usage_pattern.similarity_queries_count + usage_pattern.graph_traversal_count, 1),
                    "graph_traversal_ratio": usage_pattern.graph_traversal_count / max(usage_pattern.similarity_queries_count + usage_pattern.graph_traversal_count, 1),
                    "access_frequency": usage_pattern.access_frequency,
                    "performance_requirement": "high" if usage_pattern.avg_query_time > 200 else "medium"
                }
                
                decision = await self.strategy.decide_vector_storage(
                    vector_type=vector_type,
                    vector_size=usage_pattern.storage_size,
                    expected_usage=expected_usage
                )
                
                # Determine current location (simplified)
                current_location = VectorStorageLocation.NEO4J_ONLY  # Would need to detect actual location
                
                # Migrate if needed
                if decision.location != current_location:
                    success = await self.migrate_vector(
                        vector_id=vector_id,
                        from_location=current_location,
                        to_location=decision.location,
                        vector_type=vector_type
                    )
                    
                    if success:
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
                    
                    results["migrations"].append({
                        "vector_id": vector_id,
                        "from": current_location.value,
                        "to": decision.location.value,
                        "success": success,
                        "reasoning": decision.reasoning
                    })
                
            except Exception as e:
                self.logger.error(f"Error migrating vector {vector_id}: {e}")
                results["failed"] += 1
        
        return results
```

### 2. Strategy Configuration

```python
class VectorStrategyConfig:
    """Configuration for vector storage strategy"""
    
    def __init__(self):
        self.config = {
            # Threshold configurations
            "neo4j_graph_threshold": 0.3,
            "qdrant_similarity_threshold": 0.7,
            "both_systems_threshold": 0.8,
            "access_frequency_threshold": 10.0,
            
            # Performance configurations
            "max_neo4j_vector_size": 1000000,
            "max_query_time_ms": 500,
            
            # Cost configurations
            "neo4j_storage_cost": 1.5,
            "qdrant_storage_cost": 1.0,
            "dual_storage_cost": 2.2,
            
            # Vector type specific configurations
            "vector_type_preferences": {
                "entity_embedding": {
                    "default_location": "neo4j_only",
                    "similarity_threshold": 0.8,
                    "graph_threshold": 0.6
                },
                "relation_embedding": {
                    "default_location": "neo4j_only",
                    "similarity_threshold": 0.7,
                    "graph_threshold": 0.7
                },
                "chunk_embedding": {
                    "default_location": "qdrant_only",
                    "similarity_threshold": 0.9,
                    "graph_threshold": 0.2
                },
                "document_summary": {
                    "default_location": "qdrant_only",
                    "similarity_threshold": 0.8,
                    "graph_threshold": 0.4
                },
                "topic_embedding": {
                    "default_location": "both_systems",
                    "similarity_threshold": 0.8,
                    "graph_threshold": 0.5
                }
            },
            
            # Performance monitoring
            "monitoring": {
                "enable_usage_tracking": True,
                "cache_duration_hours": 1,
                "analysis_window_days": 30,
                "migration_batch_size": 100
            }
        }
    
    def get_config(self) -> Dict:
        return self.config
    
    def update_config(self, updates: Dict) -> None:
        """Update configuration with new values"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
```

### 3. Strategy Monitoring and Optimization

```python
class VectorStrategyMonitor:
    """Monitor and optimize vector storage strategy performance"""
    
    def __init__(self, strategy_engine, usage_analyzer):
        self.strategy = strategy_engine
        self.usage_analyzer = usage_analyzer
        self.logger = logging.getLogger(__name__)
        self.performance_history = []
    
    async def monitor_strategy_performance(self) -> Dict:
        """Monitor overall strategy performance"""
        try:
            # Get system usage summary
            usage_summary = await self.usage_analyzer.get_system_usage_summary()
            
            # Calculate performance metrics
            metrics = await self._calculate_performance_metrics()
            
            # Analyze strategy effectiveness
            effectiveness = await self._analyze_strategy_effectiveness()
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(metrics, effectiveness)
            
            performance_report = {
                "timestamp": datetime.utcnow().isoformat(),
                "usage_summary": usage_summary,
                "performance_metrics": metrics,
                "strategy_effectiveness": effectiveness,
                "recommendations": recommendations
            }
            
            # Store in history
            self.performance_history.append(performance_report)
            
            # Keep only last 100 reports
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            return performance_report
            
        except Exception as e:
            self.logger.error(f"Error monitoring strategy performance: {e}")
            return {"error": str(e)}
    
    async def _calculate_performance_metrics(self) -> Dict:
        """Calculate key performance metrics"""
        # Placeholder implementation
        # In a real system, these would be calculated from actual performance data
        return {
            "avg_query_time_neo4j": 150,  # ms
            "avg_query_time_qdrant": 75,   # ms
            "storage_efficiency_neo4j": 0.85,
            "storage_efficiency_qdrant": 0.92,
            "cache_hit_rate": 0.78,
            "migration_success_rate": 0.95
        }
    
    async def _analyze_strategy_effectiveness(self) -> Dict:
        """Analyze how effective the current strategy is"""
        # Placeholder implementation
        return {
            "decision_accuracy": 0.87,
            "cost_optimization": 0.82,
            "performance_improvement": 0.15,
            "user_satisfaction": 0.89
        }
    
    async def _generate_recommendations(self, metrics: Dict, effectiveness: Dict) -> List[Dict]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check query performance
        if metrics.get("avg_query_time_neo4j", 0) > 200:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "description": "Neo4j query times are high, consider migrating some vectors to Qdrant",
                "action": "migrate_slow_vectors"
            })
        
        # Check storage efficiency
        if metrics.get("storage_efficiency_neo4j", 1.0) < 0.8:
            recommendations.append({
                "type": "storage",
                "priority": "medium",
                "description": "Neo4j storage efficiency is low, review vector storage strategy",
                "action": "optimize_neo4j_storage"
            })
        
        # Check decision accuracy
        if effectiveness.get("decision_accuracy", 1.0) < 0.85:
            recommendations.append({
                "type": "strategy",
                "priority": "high",
                "description": "Strategy decision accuracy is low, review thresholds and criteria",
                "action": "tune_strategy_parameters"
            })
        
        return recommendations
    
    async def optimize_strategy_parameters(self) -> Dict:
        """Automatically optimize strategy parameters based on performance history"""
        try:
            if len(self.performance_history) < 10:
                return {"message": "Insufficient data for optimization"}
            
            # Analyze performance trends
            recent_performance = self.performance_history[-10:]
            
            # Calculate average metrics
            avg_metrics = {}
            for metric in ["avg_query_time_neo4j", "avg_query_time_qdrant", "decision_accuracy"]:
                values = [report["performance_metrics"].get(metric, 0) for report in recent_performance]
                avg_metrics[metric] = sum(values) / len(values)
            
            # Generate parameter adjustments
            adjustments = {}
            
            # Adjust thresholds based on performance
            if avg_metrics["avg_query_time_neo4j"] > 200:
                adjustments["neo4j_graph_threshold"] = self.strategy.neo4j_graph_threshold * 0.9
            
            if avg_metrics["decision_accuracy"] < 0.85:
                adjustments["both_systems_threshold"] = self.strategy.both_systems_threshold * 0.95
            
            # Apply adjustments
            if adjustments:
                for param, value in adjustments.items():
                    setattr(self.strategy, param, value)
                
                self.logger.info(f"Applied strategy optimizations: {adjustments}")
                
                return {
                    "optimizations_applied": adjustments,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {"message": "No optimizations needed"}
                
        except Exception as e:
            self.logger.error(f"Error optimizing strategy parameters: {e}")
            return {"error": str(e)}
```

## Testing Strategy

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestVectorStorageStrategy:
    @pytest.fixture
    def strategy_engine(self):
        config = {
            "neo4j_graph_threshold": 0.3,
            "qdrant_similarity_threshold": 0.7,
            "both_systems_threshold": 0.8,
            "access_frequency_threshold": 10.0
        }
        return VectorStorageStrategy(config)
    
    async def test_entity_embedding_decision(self, strategy_engine):
        """Test decision for entity embeddings"""
        expected_usage = {
            "graph_traversal_ratio": 0.8,
            "similarity_search_ratio": 0.2,
            "access_frequency": 15.0
        }
        
        entity_context = {
            "connection_count": 12
        }
        
        decision = await strategy_engine.decide_vector_storage(
            vector_type=VectorType.ENTITY_EMBEDDING,
            vector_size=384,
            expected_usage=expected_usage,
            entity_context=entity_context
        )
        
        assert decision.location == VectorStorageLocation.NEO4J_ONLY
        assert decision.confidence > 0.7
        assert "graph operations" in decision.reasoning.lower()
    
    async def test_chunk_embedding_decision(self, strategy_engine):
        """Test decision for chunk embeddings"""
        expected_usage = {
            "graph_traversal_ratio": 0.1,
            "similarity_search_ratio": 0.9,
            "access_frequency": 5.0,
            "performance_requirement": "high"
        }
        
        decision = await strategy_engine.decide_vector_storage(
            vector_type=VectorType.CHUNK_EMBEDDING,
            vector_size=384,
            expected_usage=expected_usage
        )
        
        assert decision.location == VectorStorageLocation.QDRANT_ONLY
        assert decision.confidence > 0.7
        assert "similarity search" in decision.reasoning.lower()
    
    async def test_both_systems_decision(self, strategy_engine):
        """Test decision for both systems storage"""
        expected_usage = {
            "graph_traversal_ratio": 0.6,
            "similarity_search_ratio": 0.7,
            "access_frequency": 20.0,
            "performance_requirement": "critical"
        }
        
        decision = await strategy_engine.decide_vector_storage(
            vector_type=VectorType.TOPIC_EMBEDDING,
            vector_size=384,
            expected_usage=expected_usage
        )
        
        assert decision.location == VectorStorageLocation.BOTH_SYSTEMS
        assert decision.confidence > 0.8
        assert "both systems" in decision.reasoning.lower()

class TestVectorUsageAnalyzer:
    @pytest.fixture
    def usage_analyzer(self):
        neo4j_storage = AsyncMock()
        qdrant_storage = AsyncMock()
        return VectorUsageAnalyzer(neo4j_storage, qdrant_storage)
    
    async def test_analyze_vector_usage(self, usage_analyzer):
        """Test vector usage analysis"""
        # Mock Neo4j response
        usage_analyzer.neo4j_storage.execute_query.return_value = [{
            "id": "entity_123",
            "has_embedding": True,
            "access_count": 150,
            "last_accessed": datetime.utcnow()
        }]
        
        pattern = await usage_analyzer.analyze_vector_usage(
            vector_id="entity_123",
            vector_type=VectorType.ENTITY_EMBEDDING,
            time_window_days=30
        )
        
        assert pattern.vector_id == "entity_123"
        assert pattern.vector_type == VectorType.ENTITY_EMBEDDING
        assert pattern.graph_traversal_count == 150
        assert pattern.access_frequency == 5.0  # 150/30

class TestVectorMigrationService:
    @pytest.fixture
    def migration_service(self):
        neo4j_vector_storage = AsyncMock()
        qdrant_storage = AsyncMock()
        strategy_engine = AsyncMock()
        
        return VectorMigrationService(
            neo4j_vector_storage, qdrant_storage, strategy_engine
        )
    
    async def test_migrate_vector(self, migration_service):
        """Test vector migration"""
        # Mock vector data retrieval
        migration_service._get_vector_data = AsyncMock(return_value={
            "vector": [0.1] * 384,
            "metadata": {"name": "Test Entity"}
        })
        
        # Mock vector storage
        migration_service._store_vector_data = AsyncMock(return_value=True)
        migration_service._remove_vector_data = AsyncMock(return_value=True)
        
        success = await migration_service.migrate_vector(
            vector_id="entity_123",
            from_location=VectorStorageLocation.NEO4J_ONLY,
            to_location=VectorStorageLocation.QDRANT_ONLY,
            vector_type=VectorType.ENTITY_EMBEDDING
        )
        
        assert success is True
        migration_service._get_vector_data.assert_called_once()
        migration_service._store_vector_data.assert_called_once()
        migration_service._remove_vector_data.assert_called_once()
```

### Integration Tests

```python
class TestVectorStrategyIntegration:
    """Integration tests for vector strategy system"""
    
    @pytest.fixture
    async def integration_setup(self):
        # Set up real connections
        neo4j_storage = Neo4jStorage(test_config)
        qdrant_storage = QdrantStorage(test_config)
        embedding_service = EmbeddingService(test_config)
        
        # Create services
        neo4j_vector_storage = Neo4jVectorStorage(neo4j_storage, embedding_service)
        strategy_config = VectorStrategyConfig()
        strategy_engine = VectorStorageStrategy(strategy_config.get_config())
        usage_analyzer = VectorUsageAnalyzer(neo4j_storage, qdrant_storage)
        migration_service = VectorMigrationService(
            neo4j_vector_storage, qdrant_storage, strategy_engine
        )
        
        # Clean up test data
        await self._cleanup_test_data(neo4j_storage, qdrant_storage)
        
        yield {
            "strategy": strategy_engine,
            "analyzer": usage_analyzer,
            "migration": migration_service,
            "neo4j_vectors": neo4j_vector_storage
        }
        
        # Clean up after tests
        await self._cleanup_test_data(neo4j_storage, qdrant_storage)
    
    async def test_end_to_end_strategy_workflow(self, integration_setup):
        """Test complete strategy workflow"""
        services = integration_setup
        
        # Create test entity
        entity_id = "test_strategy_entity"
        await services["neo4j_vectors"].neo4j_storage.execute_query(
            "CREATE (e:Entity {id: $id, name: $name, type: $type})",
            {"id": entity_id, "name": "Test Strategy Entity", "type": "PERSON"}
        )
        
        # Make strategy decision
        expected_usage = {
            "graph_traversal_ratio": 0.7,
            "similarity_search_ratio": 0.3,
            "access_frequency": 12.0
        }
        
        decision = await services["strategy"].decide_vector_storage(
            vector_type=VectorType.ENTITY_EMBEDDING,
            vector_size=384,
            expected_usage=expected_usage
        )
        
        assert decision.location in [VectorStorageLocation.NEO4J_ONLY, VectorStorageLocation.BOTH_SYSTEMS]
        assert decision.confidence > 0.5
        
        # Store vector based on decision
        if decision.location in [VectorStorageLocation.NEO4J_ONLY, VectorStorageLocation.BOTH_SYSTEMS]:
            success = await services["neo4j_vectors"].store_entity_embedding(
                entity_id=entity_id,
                text="Test Strategy Entity (PERSON)"
            )
            assert success is True
        
        # Analyze usage pattern
        usage_pattern = await services["analyzer"].analyze_vector_usage(
            vector_id=entity_id,
            vector_type=VectorType.ENTITY_EMBEDDING
        )
        
        assert usage_pattern.vector_id == entity_id
        assert usage_pattern.vector_type == VectorType.ENTITY_EMBEDDING
    
    async def _cleanup_test_data(self, neo4j_storage, qdrant_storage):
        """Clean up test data"""
        await neo4j_storage.execute_query(
            "MATCH (n) WHERE n.id STARTS WITH 'test_strategy_' DELETE n"
        )
```

## Performance Considerations

### Optimization Strategies

1. **Decision Caching**:
   - Cache strategy decisions for frequently accessed vectors
   - Implement cache invalidation based on usage pattern changes
   - Use Redis for distributed caching

2. **Batch Processing**:
   - Process strategy decisions in batches
   - Implement parallel analysis for independent vectors
   - Use async processing for non-blocking operations

3. **Adaptive Thresholds**:
   - Automatically adjust thresholds based on performance metrics
   - Implement machine learning for threshold optimization
   - Monitor and react to system load changes

### Performance Targets

- **Strategy Decision**: < 50ms per vector
- **Usage Analysis**: < 200ms per vector
- **Migration Operation**: < 500ms per vector
- **Batch Processing**: 100+ decisions per second

## Success Criteria

- [ ] Strategy engine making intelligent storage decisions
- [ ] Usage pattern analysis providing accurate insights
- [ ] Migration service enabling seamless vector movement
- [ ] Performance monitoring and optimization working
- [ ] Configurable strategy parameters
- [ ] Comprehensive test coverage (>90%)
- [ ] Performance targets met consistently
- [ ] Integration with existing vector storage systems

## Risk Assessment

**Medium Risk**: Strategy decision accuracy and performance impact

**Mitigation Strategies**:
- Implement comprehensive testing with real usage patterns
- Start with conservative thresholds and adjust gradually
- Monitor performance impact continuously
- Provide manual override capabilities

## Rollback Plan

1. **Disable strategy engine** and use default storage locations
2. **Revert to previous storage configuration** for all vectors
3. **Stop migration operations** and maintain current state
4. **Monitor system stability** after rollback

## Next Steps

- **Task 3.3**: Embedding Synchronization Pipeline
- **Task 4.1**: Hybrid Query Engine
- **Integration**: Incorporate into production system

## Dependencies

- **Task 3.1**: Neo4j Vector Storage (completed)
- **Task 2.3**: ID Mapping Utilities (completed)
- Performance monitoring infrastructure
- Configuration management system

## Estimated Time

**6-7 days**

## Status

- [ ] Not Started
- [ ] In Progress
- [ ] Testing
- [ ] Completed
- [ ] Verified