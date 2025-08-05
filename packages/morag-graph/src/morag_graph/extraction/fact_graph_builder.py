"""Build knowledge graphs from extracted facts."""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import structlog

from morag_reasoning.llm import LLMClient, LLMConfig
from ..models.fact import Fact, FactRelation, FactRelationType
from ..models.graph import Graph
from .fact_prompts import FactExtractionPrompts


class FactGraphBuilder:
    """Build knowledge graph from extracted facts."""
    
    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        min_relation_confidence: float = 0.6,
        max_relations_per_fact: int = 5,
        language: str = "en"
    ):
        """Initialize fact graph builder.

        Args:
            model_id: LLM model for relationship extraction
            api_key: API key for LLM service
            min_relation_confidence: Minimum confidence for relationships
            max_relations_per_fact: Maximum relationships per fact
            language: Language for relationship extraction
        """
        self.model_id = model_id
        self.api_key = api_key
        self.min_relation_confidence = min_relation_confidence
        self.max_relations_per_fact = max_relations_per_fact
        self.language = language
        
        self.logger = structlog.get_logger(__name__)
        
        # Initialize LLM client
        llm_config = LLMConfig(
            provider="gemini",
            model=model_id,
            api_key=api_key,
            temperature=0.1,
            max_tokens=2000
        )
        self.llm_client = LLMClient(llm_config)
    
    async def build_fact_graph(self, facts: List[Fact]) -> Graph:
        """Build knowledge graph from extracted facts.
        
        Args:
            facts: List of facts to build graph from
            
        Returns:
            Graph object containing facts and their relationships
        """
        if not facts:
            return Graph(nodes=[], edges=[])
        
        self.logger.info(
            "Starting fact graph building",
            num_facts=len(facts)
        )
        
        try:
            # Create fact relationships
            relationships = await self._create_fact_relationships(facts)
            
            # Build graph structure
            graph = self._build_graph_structure(facts, relationships)
            
            # Index facts for efficient retrieval
            await self._index_facts(facts)
            
            self.logger.info(
                "Fact graph building completed",
                num_facts=len(facts),
                num_relationships=len(relationships)
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(
                "Fact graph building failed",
                error=str(e),
                error_type=type(e).__name__
            )
            # Return empty graph on failure
            return Graph(nodes=[], edges=[])
    
    async def _create_fact_relationships(self, facts: List[Fact]) -> List[FactRelation]:
        """Create semantic relationships between facts.
        
        Args:
            facts: List of facts to analyze for relationships
            
        Returns:
            List of FactRelation objects
        """
        if len(facts) < 2:
            return []
        
        relationships = []
        
        # Process facts in batches to avoid overwhelming the LLM
        batch_size = 10
        for i in range(0, len(facts), batch_size):
            batch = facts[i:i + batch_size]
            batch_relationships = await self._extract_relationships_for_batch(batch)
            relationships.extend(batch_relationships)
        
        # Filter relationships by confidence
        filtered_relationships = [
            rel for rel in relationships 
            if rel.confidence >= self.min_relation_confidence
        ]
        
        self.logger.debug(
            "Fact relationships created",
            total_relationships=len(relationships),
            filtered_relationships=len(filtered_relationships)
        )
        
        return filtered_relationships
    
    async def _extract_relationships_for_batch(self, facts: List[Fact]) -> List[FactRelation]:
        """Extract relationships for a batch of facts.
        
        Args:
            facts: Batch of facts to analyze
            
        Returns:
            List of relationships found in the batch
        """
        if len(facts) < 2:
            return []
        
        try:
            # Convert facts to dictionaries for LLM prompt
            fact_dicts = [self._fact_to_prompt_dict(fact) for fact in facts]
            
            # Create relationship extraction prompt
            prompt = FactExtractionPrompts.create_relationship_prompt(fact_dicts, self.language)
            
            # Get relationships from LLM
            response = await self.llm_client.generate(prompt)
            
            # Parse response
            relationship_data = self._parse_relationship_response(response)
            
            # Convert to FactRelation objects
            relationships = self._create_fact_relation_objects(relationship_data, facts)
            
            return relationships
            
        except Exception as e:
            self.logger.warning(
                "Relationship extraction failed for batch",
                batch_size=len(facts),
                error=str(e)
            )
            return []
    
    def _fact_to_prompt_dict(self, fact: Fact) -> Dict[str, Any]:
        """Convert fact to dictionary for LLM prompt.
        
        Args:
            fact: Fact to convert
            
        Returns:
            Dictionary representation for prompt
        """
        return {
            'id': fact.id,
            'subject': fact.subject,
            'object': fact.object,
            'approach': fact.approach,
            'solution': fact.solution,
            'remarks': fact.remarks,
            'fact_type': fact.fact_type
        }
    
    def _parse_relationship_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for relationships.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of relationship dictionaries
        """
        try:
            # Try to find JSON array in response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                relationships = json.loads(json_str)
                
                if isinstance(relationships, list):
                    return relationships
            
            # Fallback: try to parse entire response as JSON
            relationships = json.loads(response)
            if isinstance(relationships, list):
                return relationships
            elif isinstance(relationships, dict):
                return [relationships]
            
        except json.JSONDecodeError as e:
            self.logger.warning(
                "Failed to parse relationship response",
                error=str(e),
                response_preview=response[:200]
            )
        
        return []
    
    def _create_fact_relation_objects(
        self, 
        relationship_data: List[Dict[str, Any]], 
        facts: List[Fact]
    ) -> List[FactRelation]:
        """Create FactRelation objects from parsed data.
        
        Args:
            relationship_data: List of relationship dictionaries
            facts: List of facts for ID validation
            
        Returns:
            List of FactRelation objects
        """
        relationships = []
        fact_ids = {fact.id for fact in facts}
        
        for rel_data in relationship_data:
            try:
                source_id = rel_data.get('source_fact_id', '')
                target_id = rel_data.get('target_fact_id', '')
                relation_type = rel_data.get('relation_type', '')
                
                # Validate fact IDs exist
                if source_id not in fact_ids or target_id not in fact_ids:
                    continue
                
                # Validate relation type
                if relation_type not in FactRelationType.all_types():
                    continue
                
                # Avoid self-relationships
                if source_id == target_id:
                    continue
                
                relationship = FactRelation(
                    source_fact_id=source_id,
                    target_fact_id=target_id,
                    relation_type=relation_type,
                    confidence=float(rel_data.get('confidence', 0.7)),
                    context=rel_data.get('context', '')
                )
                
                relationships.append(relationship)
                
            except Exception as e:
                self.logger.warning(
                    "Failed to create relationship object",
                    rel_data=rel_data,
                    error=str(e)
                )
                continue
        
        return relationships
    
    def _build_graph_structure(self, facts: List[Fact], relationships: List[FactRelation]) -> Graph:
        """Build graph structure from facts and relationships.
        
        Args:
            facts: List of facts (nodes)
            relationships: List of relationships (edges)
            
        Returns:
            Graph object
        """
        from ..models.graph import GraphNode, GraphEdge
        
        # Convert facts to graph nodes
        nodes = []
        for fact in facts:
            node = GraphNode(
                id=fact.id,
                label="Fact",
                properties=fact.get_neo4j_properties()
            )
            nodes.append(node)
        
        # Convert relationships to graph edges
        edges = []
        for relationship in relationships:
            edge = GraphEdge(
                source=relationship.source_fact_id,
                target=relationship.target_fact_id,
                type=relationship.relation_type,
                properties=relationship.get_neo4j_properties()
            )
            edges.append(edge)
        
        return Graph(nodes=nodes, edges=edges)
    
    async def _index_facts(self, facts: List[Fact]) -> None:
        """Create keyword and domain indexes for facts.
        
        Args:
            facts: List of facts to index
        """
        # Group facts by domain
        domain_index = {}
        keyword_index = {}
        
        for fact in facts:
            # Domain indexing
            if fact.domain:
                if fact.domain not in domain_index:
                    domain_index[fact.domain] = []
                domain_index[fact.domain].append(fact.id)
            
            # Keyword indexing
            for keyword in fact.keywords:
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                keyword_index[keyword].append(fact.id)
        
        self.logger.debug(
            "Fact indexing completed",
            domains=len(domain_index),
            keywords=len(keyword_index)
        )
    
    def get_related_facts(
        self, 
        fact_id: str, 
        relationships: List[FactRelation],
        max_depth: int = 2
    ) -> List[str]:
        """Get facts related to a given fact through relationships.
        
        Args:
            fact_id: ID of the source fact
            relationships: List of all relationships
            max_depth: Maximum relationship depth to traverse
            
        Returns:
            List of related fact IDs
        """
        related_facts = set()
        current_level = {fact_id}
        
        for depth in range(max_depth):
            next_level = set()
            
            for current_fact in current_level:
                # Find outgoing relationships
                for rel in relationships:
                    if rel.source_fact_id == current_fact:
                        next_level.add(rel.target_fact_id)
                        related_facts.add(rel.target_fact_id)
                    elif rel.target_fact_id == current_fact:
                        next_level.add(rel.source_fact_id)
                        related_facts.add(rel.source_fact_id)
            
            current_level = next_level
            if not current_level:
                break
        
        # Remove the original fact ID
        related_facts.discard(fact_id)
        return list(related_facts)
    
    def analyze_fact_clusters(self, facts: List[Fact], relationships: List[FactRelation]) -> Dict[str, Any]:
        """Analyze clusters of related facts.
        
        Args:
            facts: List of all facts
            relationships: List of all relationships
            
        Returns:
            Dictionary with cluster analysis
        """
        # Build adjacency list
        adjacency = {}
        for fact in facts:
            adjacency[fact.id] = set()
        
        for rel in relationships:
            adjacency[rel.source_fact_id].add(rel.target_fact_id)
            adjacency[rel.target_fact_id].add(rel.source_fact_id)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for fact_id in adjacency:
            if fact_id not in visited:
                cluster = self._dfs_cluster(fact_id, adjacency, visited)
                if len(cluster) > 1:  # Only include clusters with multiple facts
                    clusters.append(cluster)
        
        return {
            'num_clusters': len(clusters),
            'cluster_sizes': [len(cluster) for cluster in clusters],
            'largest_cluster_size': max([len(cluster) for cluster in clusters]) if clusters else 0,
            'isolated_facts': len([cluster for cluster in clusters if len(cluster) == 1])
        }
    
    def _dfs_cluster(self, start_id: str, adjacency: Dict[str, set], visited: set) -> List[str]:
        """Depth-first search to find connected component.
        
        Args:
            start_id: Starting fact ID
            adjacency: Adjacency list representation
            visited: Set of visited fact IDs
            
        Returns:
            List of fact IDs in the cluster
        """
        cluster = []
        stack = [start_id]
        
        while stack:
            fact_id = stack.pop()
            if fact_id not in visited:
                visited.add(fact_id)
                cluster.append(fact_id)
                
                for neighbor in adjacency.get(fact_id, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return cluster
