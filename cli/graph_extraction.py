#!/usr/bin/env python3
"""
Graph Extraction Module for MoRAG CLI Scripts

This module provides common graph entity and relation extraction functionality
for all CLI scripts using the morag-graph package.
"""

import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    # Import Graphiti components (primary approach)
    from morag_graph.graphiti import (
        GraphitiConfig, GraphitiConnectionService,
        DocumentEpisodeMapper, GraphitiEntityStorage,
        GraphitiSearchService
    )
    from morag_graph.models import Document, DocumentChunk
except ImportError as e:
    print(f"âŒ Import error: {e}")
    print("Make sure you have installed the morag-graph package:")
    print("  pip install -e packages/morag-graph")
    raise

from common_schema import Entity, Relation


class GraphitiExtractionService:
    """Service for extracting and ingesting content using Graphiti."""

    def __init__(self, use_graphiti: bool = True):
        """Initialize the service.

        Args:
            use_graphiti: Whether to use Graphiti for ingestion (default: True)
        """
        self.use_graphiti = use_graphiti
        self._graphiti_config = None
        self._connection_service = None

    def _get_graphiti_config(self) -> GraphitiConfig:
        """Get or create Graphiti configuration."""
        if self._graphiti_config is None:
            # Get Gemini API key from environment
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("No Gemini API key found. Set GEMINI_API_KEY environment variable.")

            # Force local Neo4j configuration with Gemini
            self._graphiti_config = GraphitiConfig(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",
                neo4j_database="neo4j",
                openai_api_key=gemini_api_key,
                openai_model="gemini-1.5-flash",
                openai_embedding_model="text-embedding-004"
            )
        return self._graphiti_config

    async def get_connection_service(self) -> GraphitiConnectionService:
        """Get or create Graphiti connection service."""
        if self._connection_service is None:
            config = self._get_graphiti_config()
            self._connection_service = GraphitiConnectionService(config)
            await self._connection_service.connect()
        return self._connection_service

    async def ingest_document_content(
        self,
        content: str,
        doc_id: str,
        title: str = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_description: str = None
    ) -> Dict[str, Any]:
        """Ingest document content using Graphiti episodes.

        Args:
            content: Document content
            doc_id: Document identifier
            title: Document title
            metadata: Additional metadata
            source_description: Description of the source

        Returns:
            Dictionary with ingestion results
        """
        if not self.use_graphiti:
            return {"success": False, "error": "Graphiti ingestion disabled"}

        try:
            connection_service = await self.get_connection_service()

            # Create episode from document content
            episode_name = title or f"Document {doc_id}"
            success = await connection_service.create_episode(
                name=episode_name,
                content=content,
                source_description=source_description or f"Document ingestion: {doc_id}",
                metadata=metadata or {}
            )

            if success:
                return {
                    "success": True,
                    "episode_name": episode_name,
                    "doc_id": doc_id,
                    "content_length": len(content)
                }
            else:
                return {"success": False, "error": "Failed to create episode"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def search_content(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "hybrid"
    ) -> Dict[str, Any]:
        """Search content using Graphiti.

        Args:
            query: Search query
            limit: Maximum number of results
            search_type: Type of search (hybrid, semantic, keyword)

        Returns:
            Dictionary with search results
        """
        if not self.use_graphiti:
            return {"success": False, "error": "Graphiti search disabled"}

        try:
            config = self._get_graphiti_config()
            search_service = GraphitiSearchService(config)

            results, metrics = await search_service.search(
                query=query,
                limit=limit,
                search_type=search_type
            )

            return {
                "success": True,
                "results": results,
                "metrics": metrics,
                "query": query,
                "count": len(results)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def close(self):
        """Close connections."""
        if self._connection_service:
            await self._connection_service.disconnect()


# Traditional extraction functions removed - use Graphiti instead
# See extract_and_ingest_with_graphiti() for the recommended approach


# Traditional extraction services removed - use Graphiti instead


# Traditional database ingestion service removed - use Graphiti instead

# Legacy extract_and_ingest function removed - use extract_and_ingest_with_graphiti instead


async def extract_and_ingest_with_graphiti(
    text_content: str,
    doc_id: str,
    title: str = None,
    context: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Extract and ingest content using Graphiti episodes.

    This is the recommended approach that uses Graphiti's built-in
    entity extraction and knowledge graph capabilities.

    Args:
        text_content: Text content to process
        doc_id: Document identifier
        title: Document title
        context: Additional context for extraction
        metadata: Additional metadata

    Returns:
        Dictionary with extraction and ingestion results
    """
    results = {
        'graphiti': {'success': False, 'episode_name': None, 'error': None}
    }

    # Initialize Graphiti service
    graphiti_service = GraphitiExtractionService(use_graphiti=True)

    try:
        # Ingest using Graphiti (this will handle entity extraction automatically)
        print(f"ðŸš€ Ingesting content using Graphiti...")
        graphiti_result = await graphiti_service.ingest_document_content(
            content=text_content,
            doc_id=doc_id,
            title=title,
            metadata=metadata,
            source_description=context or f"CLI ingestion for document {doc_id}"
        )

        results['graphiti'] = graphiti_result

        if graphiti_result['success']:
            print(f"âœ… Graphiti ingestion successful: {graphiti_result['episode_name']}")
            print(f"   Content length: {graphiti_result['content_length']} characters")
        else:
            print(f"âŒ Graphiti ingestion failed: {graphiti_result.get('error', 'Unknown error')}")

        return results

    except Exception as e:
        print(f"âŒ Error in Graphiti extraction: {e}")
        results['graphiti']['error'] = str(e)
        return results

    finally:
        await graphiti_service.close()


async def search_with_graphiti(
    query: str,
    limit: int = 10,
    search_type: str = "hybrid"
) -> Dict[str, Any]:
    """Search content using Graphiti.

    Args:
        query: Search query
        limit: Maximum number of results
        search_type: Type of search (hybrid, semantic, keyword)

    Returns:
        Dictionary with search results
    """
    graphiti_service = GraphitiExtractionService(use_graphiti=True)

    try:
        print(f"ðŸ” Searching with Graphiti: '{query}'")
        search_result = await graphiti_service.search_content(
            query=query,
            limit=limit,
            search_type=search_type
        )

        if search_result['success']:
            print(f"âœ… Found {search_result['count']} results")
            for i, result in enumerate(search_result['results'][:3], 1):
                print(f"   {i}. {result.content[:100]}...")
        else:
            print(f"âŒ Search failed: {search_result.get('error', 'Unknown error')}")

        return search_result

    except Exception as e:
        print(f"âŒ Error in Graphiti search: {e}")
        return {"success": False, "error": str(e)}

    finally:
        await graphiti_service.close()
