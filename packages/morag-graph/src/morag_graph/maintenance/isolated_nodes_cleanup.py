"""Isolated Nodes Cleanup Maintenance Job

Identifies and removes completely isolated nodes (entities with no relationships).
These nodes are typically orphaned entities that were created but never connected
to the knowledge graph through relationships.

Applies changes by default (dry-run available as opt-in).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig

logger = structlog.get_logger(__name__)


@dataclass
class IsolatedNodesCleanupConfig:
    """Configuration for isolated nodes cleanup."""
    dry_run: bool = True
    batch_size: int = 100
    job_tag: str = ""
    
    def ensure_defaults(self) -> None:
        """Ensure all configuration values have defaults."""
        if self.batch_size <= 0:
            self.batch_size = 100


@dataclass
class IsolatedNodesCleanupResult:
    """Result of isolated nodes cleanup operation."""
    total_nodes_checked: int = 0
    isolated_nodes_found: int = 0
    isolated_nodes_removed: int = 0
    execution_time_seconds: float = 0.0
    dry_run: bool = True
    job_tag: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class IsolatedNodesCleanupService:
    """Service for cleaning up isolated nodes in the knowledge graph."""
    
    def __init__(self, neo4j_storage: Neo4jStorage, config: IsolatedNodesCleanupConfig):
        self.neo4j_storage = neo4j_storage
        self.cleanup_config = config
        self.logger = logger.bind(service="isolated_nodes_cleanup")
    
    async def run_cleanup(self) -> IsolatedNodesCleanupResult:
        """Run the isolated nodes cleanup process."""
        start_time = time.time()
        self.cleanup_config.ensure_defaults()
        
        self.logger.info("Starting isolated nodes cleanup", config=self.cleanup_config)
        
        result = IsolatedNodesCleanupResult(
            dry_run=self.cleanup_config.dry_run,
            job_tag=self.cleanup_config.job_tag,
            details={}
        )
        
        try:
            # Find isolated nodes
            isolated_nodes = await self._find_isolated_nodes()
            result.isolated_nodes_found = len(isolated_nodes)
            result.total_nodes_checked = await self._count_total_nodes()
            
            if not isolated_nodes:
                self.logger.info("No isolated nodes found")
                result.execution_time_seconds = time.time() - start_time
                return result
            
            self.logger.info(f"Found {len(isolated_nodes)} isolated nodes")
            
            # Remove isolated nodes
            if self.cleanup_config.dry_run:
                self.logger.info(f"[DRY RUN] Would remove {len(isolated_nodes)} isolated nodes")
                result.isolated_nodes_removed = len(isolated_nodes)
            else:
                removed_count = await self._remove_isolated_nodes(isolated_nodes)
                result.isolated_nodes_removed = removed_count
                self.logger.info(f"Removed {removed_count} isolated nodes")
            
            # Store details
            result.details = {
                "isolated_node_samples": isolated_nodes[:10] if isolated_nodes else [],
                "batch_size": self.cleanup_config.batch_size
            }
            
        except Exception as e:
            self.logger.error(f"Error during isolated nodes cleanup: {e}")
            raise
        finally:
            result.execution_time_seconds = time.time() - start_time
        
        return result
    
    async def _find_isolated_nodes(self) -> List[Dict[str, Any]]:
        """Find all nodes that have no incoming or outgoing relationships."""
        query = """
        MATCH (n)
        WHERE NOT (n)--()
        RETURN n.id as node_id, n.name as node_name, labels(n) as node_labels
        LIMIT 1000
        """
        
        result = await self.neo4j_storage._connection_ops._execute_query(query, {})
        
        isolated_nodes = []
        for record in result:
            isolated_nodes.append({
                "id": record["node_id"],
                "name": record["node_name"],
                "labels": record["node_labels"]
            })
        
        return isolated_nodes
    
    async def _count_total_nodes(self) -> int:
        """Count total number of nodes in the database."""
        query = "MATCH (n) RETURN count(n) as total"
        result = await self.neo4j_storage._connection_ops._execute_query(query, {})
        return result[0]["total"] if result else 0
    
    async def _remove_isolated_nodes(self, isolated_nodes: List[Dict[str, Any]]) -> int:
        """Remove isolated nodes in batches."""
        removed_count = 0
        
        # Process in batches
        for i in range(0, len(isolated_nodes), self.cleanup_config.batch_size):
            batch = isolated_nodes[i:i + self.cleanup_config.batch_size]
            node_ids = [node["id"] for node in batch]
            
            # Delete nodes by ID
            query = """
            MATCH (n)
            WHERE n.id IN $node_ids AND NOT (n)--()
            DELETE n
            RETURN count(n) as deleted
            """
            
            result = await self.neo4j_storage._connection_ops._execute_query(
                query, {"node_ids": node_ids}
            )
            
            batch_deleted = result[0]["deleted"] if result else 0
            removed_count += batch_deleted
            
            self.logger.info(f"Removed {batch_deleted} isolated nodes in batch {i // self.cleanup_config.batch_size + 1}")
        
        return removed_count


def parse_isolated_nodes_cleanup_overrides() -> IsolatedNodesCleanupConfig:
    """Parse environment variables for isolated nodes cleanup configuration."""
    return IsolatedNodesCleanupConfig(
        dry_run=os.getenv("MORAG_ISOLATED_CLEANUP_DRY_RUN", "true").lower() == "true",
        batch_size=int(os.getenv("MORAG_ISOLATED_CLEANUP_BATCH_SIZE", "100")),
        job_tag=os.getenv("MORAG_ISOLATED_CLEANUP_JOB_TAG", "")
    )


async def run_isolated_nodes_cleanup(config_overrides: Optional[IsolatedNodesCleanupConfig] = None) -> Dict[str, Any]:
    """Run isolated nodes cleanup with the given configuration.
    
    Args:
        config_overrides: Optional configuration overrides
        
    Returns:
        Dictionary containing cleanup results
    """
    # Use provided config or parse from environment
    config = config_overrides or parse_isolated_nodes_cleanup_overrides()
    
    # Initialize Neo4j connection
    neo4j_config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true",
        trust_all_certificates=os.getenv("NEO4J_TRUST_ALL_CERTIFICATES", "false").lower() == "true",
    )
    storage = Neo4jStorage(neo4j_config)
    
    try:
        # Initialize connection
        await storage.connect()
        
        # Run cleanup
        service = IsolatedNodesCleanupService(storage, config)
        result = await service.run_cleanup()
        
        # Convert result to dictionary
        return {
            "total_nodes_checked": result.total_nodes_checked,
            "isolated_nodes_found": result.isolated_nodes_found,
            "isolated_nodes_removed": result.isolated_nodes_removed,
            "execution_time_seconds": result.execution_time_seconds,
            "dry_run": result.dry_run,
            "job_tag": result.job_tag,
            "details": result.details
        }
    
    finally:
        await storage.disconnect()
