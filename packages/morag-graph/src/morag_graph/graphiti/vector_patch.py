#!/usr/bin/env python3
"""
Vector similarity function patch for Graphiti.

This module patches Graphiti to use GDS similarity functions instead of 
the missing vector.similarity.cosine function in Neo4j Community Edition.
"""

import re
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class GraphitiVectorPatch:
    """Patches Graphiti queries to use GDS similarity functions."""
    
    @staticmethod
    def patch_query(query: str) -> str:
        """
        Replace vector.similarity.cosine with gds.similarity.cosine in Cypher queries.
        Also fix dynamic label issues for Neo4j Community Edition compatibility.

        Args:
            query: Original Cypher query string

        Returns:
            Patched query string with GDS similarity functions and fixed dynamic labels
        """
        if not query:
            return query

        original_query = query

        # Replace vector.similarity.cosine with gds.similarity.cosine
        patched_query = re.sub(
            r'\bvector\.similarity\.cosine\b',
            'gds.similarity.cosine',
            query,
            flags=re.IGNORECASE
        )

        # Fix dynamic label setting: SET n:$(node.labels) -> SET n:Entity
        # This is a workaround for Neo4j Community Edition which doesn't support dynamic labels
        patched_query = re.sub(
            r'SET\s+(\w+):\$\((\w+)\.labels\)',
            r'SET \1:Entity',
            patched_query,
            flags=re.IGNORECASE
        )

        # Fix other dynamic label patterns
        patched_query = re.sub(
            r'SET\s+(\w+):\$\{(\w+)\.labels\}',
            r'SET \1:Entity',
            patched_query,
            flags=re.IGNORECASE
        )

        # Fix APOC dynamic label calls
        patched_query = re.sub(
            r'CALL\s+apoc\.create\.addLabels\([^)]+\)',
            'SET n:Entity',
            patched_query,
            flags=re.IGNORECASE
        )

        # Fix Neo4j Enterprise vector property procedures
        # Only patch db.create.setRelationshipVectorProperty (which doesn't exist in Community Edition)
        # Keep db.create.setNodeVectorProperty (which does exist in Community Edition)

        # Replace db.create.setRelationshipVectorProperty with a no-op comment
        patched_query = re.sub(
            r'WITH\s+([^,\n]+),\s*([^,\n]+)\s+CALL\s+db\.create\.setRelationshipVectorProperty\([^)]+\)',
            r'WITH \1, \2 // Relationship vector property skipped for Community Edition',
            patched_query,
            flags=re.IGNORECASE | re.MULTILINE
        )

        # More aggressive pattern for relationship vector property procedures
        patched_query = re.sub(
            r'CALL\s+db\.create\.setRelationshipVectorProperty\([^)]+\)',
            '// Relationship vector property skipped for Community Edition',
            patched_query,
            flags=re.IGNORECASE
        )

        # Log if we made any changes
        if patched_query != original_query:
            logger.debug(f"Patched query for Neo4j Community Edition compatibility")
            logger.debug(f"Original: {original_query[:100]}...")
            logger.debug(f"Patched:  {patched_query[:100]}...")

        return patched_query
    
    @staticmethod
    def patch_graphiti_driver():
        """
        Monkey patch the Neo4j driver to automatically patch queries.

        This patches the driver's execute_query method to automatically replace
        vector similarity functions before executing queries.
        """
        try:
            # Patch the Graphiti Neo4j driver specifically
            from graphiti_core.driver.neo4j_driver import Neo4jDriver

            # Store the original execute_query method
            original_execute_query = Neo4jDriver.execute_query

            async def patched_execute_query(self, cypher_query_, **kwargs):
                """Patched execute_query method that replaces vector similarity functions."""
                if isinstance(cypher_query_, str):
                    cypher_query_ = GraphitiVectorPatch.patch_query(cypher_query_)
                return await original_execute_query(self, cypher_query_, **kwargs)

            # Apply the patch
            Neo4jDriver.execute_query = patched_execute_query
            logger.info("✅ Successfully patched Graphiti Neo4j driver for vector similarity functions")

        except Exception as e:
            logger.warning(f"⚠️ Failed to patch Graphiti Neo4j driver: {e}")

            # Fallback to general Neo4j driver patching
            try:
                import neo4j
                from neo4j import AsyncSession, AsyncTransaction

                # Patch AsyncSession.run method
                original_session_run = AsyncSession.run

                def patched_session_run(self, query, parameters=None, **kwargs):
                    """Patched session run method that replaces vector similarity functions."""
                    if isinstance(query, str):
                        query = GraphitiVectorPatch.patch_query(query)
                    return original_session_run(self, query, parameters, **kwargs)

                # Patch AsyncTransaction.run method
                original_transaction_run = AsyncTransaction.run

                async def patched_transaction_run(self, query, parameters=None, **kwargs):
                    """Patched transaction run method that replaces vector similarity functions."""
                    if isinstance(query, str):
                        query = GraphitiVectorPatch.patch_query(query)
                    return await original_transaction_run(self, query, parameters, **kwargs)

                # Apply the patches
                AsyncSession.run = patched_session_run
                AsyncTransaction.run = patched_transaction_run
                logger.info("✅ Successfully patched Neo4j AsyncSession and AsyncTransaction for vector similarity functions")

            except Exception as e2:
                logger.warning(f"⚠️ Failed to patch Neo4j AsyncSession/AsyncTransaction: {e2}")
    
    @staticmethod
    def patch_graphiti_queries():
        """
        Patch Graphiti's query generation methods.

        This patches the bulk query generation functions that create
        queries with dynamic labels and vector similarity functions.
        """
        try:
            # Patch the bulk query generation functions
            from graphiti_core.utils import bulk_utils

            # Store the original function
            original_get_entity_node_save_bulk_query = bulk_utils.get_entity_node_save_bulk_query

            def patched_get_entity_node_save_bulk_query(nodes, provider=None):
                """Patched entity node save bulk query generator."""
                query = original_get_entity_node_save_bulk_query(nodes, provider)
                patched_query = GraphitiVectorPatch.patch_query(query)
                logger.debug(f"Original query: {query}")
                logger.debug(f"Patched query: {patched_query}")
                return patched_query

            # Apply the patch
            bulk_utils.get_entity_node_save_bulk_query = patched_get_entity_node_save_bulk_query
            logger.info("✅ Successfully patched get_entity_node_save_bulk_query function")

            # Also patch other bulk query functions if they exist
            if hasattr(bulk_utils, 'get_entity_edge_save_bulk_query'):
                original_get_entity_edge_save_bulk_query = bulk_utils.get_entity_edge_save_bulk_query

                def patched_get_entity_edge_save_bulk_query(db_type='neo4j'):
                    """Patched entity edge save bulk query generator."""
                    query = original_get_entity_edge_save_bulk_query(db_type)
                    return GraphitiVectorPatch.patch_query(query)

                bulk_utils.get_entity_edge_save_bulk_query = patched_get_entity_edge_save_bulk_query

            logger.info("✅ Successfully patched Graphiti bulk query generation functions")

        except ImportError:
            logger.debug("Graphiti bulk_utils module not available for direct patching")
        except Exception as e:
            logger.warning(f"⚠️ Failed to patch Graphiti bulk query functions: {e}")

def apply_vector_similarity_patch():
    """
    Apply all available patches for vector similarity functions.

    This function should be called before creating Graphiti instances
    to ensure compatibility with Neo4j Community Edition.
    """
    logger.info("🔧 Applying vector similarity patches for Graphiti...")

    # Apply query-level patches first (before driver patches)
    GraphitiVectorPatch.patch_graphiti_queries()

    # Apply driver-level patch
    GraphitiVectorPatch.patch_graphiti_driver()

    logger.info("✅ Vector similarity patches applied successfully")

def test_vector_similarity_patch():
    """Test the vector similarity patch functionality."""

    test_queries = [
        # Vector similarity tests
        "WITH n, vector.similarity.cosine(n.name_embedding, $search_vector) AS score",
        "RETURN vector.similarity.cosine([1,2,3], [4,5,6]) as similarity",
        "MATCH (n) WHERE vector.similarity.cosine(n.embedding, $vector) > 0.8",

        # Dynamic label tests
        "SET n:$(node.labels)",
        "SET entity:${node.labels}",
        "CALL apoc.create.addLabels(n, node.labels)",

        # Vector property tests
        "WITH e, edge CALL db.create.setRelationshipVectorProperty(e, \"fact_embedding\", edge.fact_embedding)",
    ]

    print("🧪 Testing vector similarity and dynamic label patches...")

    for i, query in enumerate(test_queries, 1):
        patched = GraphitiVectorPatch.patch_query(query)
        print(f"\nTest {i}:")
        print(f"Original: {query}")
        print(f"Patched:  {patched}")

        # Verify vector similarity patches worked
        if "vector.similarity.cosine" in query:
            assert "vector.similarity.cosine" not in patched
            assert "gds.similarity.cosine" in patched

        # Verify dynamic label patches worked
        if ":$(" in query or ":${" in query or "apoc.create.addLabels" in query:
            assert ":$(" not in patched
            assert ":${" not in patched
            assert "apoc.create.addLabels" not in patched
            assert ":Entity" in patched

        # Verify vector property patches worked
        if "db.create.setRelationshipVectorProperty" in query:
            assert "db.create.setRelationshipVectorProperty" not in patched
            assert "Relationship vector property skipped" in patched

    print("\n✅ All vector similarity and dynamic label patch tests passed!")

if __name__ == "__main__":
    test_vector_similarity_patch()
