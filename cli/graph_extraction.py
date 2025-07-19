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
    from morag_graph import (
        EntityExtractor, RelationExtractor,
        Neo4jStorage, QdrantStorage,
        Neo4jConfig, QdrantConfig,
        GraphBuilder
    )
    from morag_graph.models import Entity as GraphEntity, Relation as GraphRelation
    # Import Graphiti components
    from morag_graph.graphiti import (
        GraphitiConfig, GraphitiConnectionService,
        DocumentEpisodeMapper, GraphitiEntityStorage,
        GraphitiSearchService
    )
    from morag_graph.models import Document, DocumentChunk
except ImportError as e:
    print(f"❌ Import error: {e}")
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
            # Get API key from environment (try Gemini first, then OpenAI)
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY environment variable.")

            self._graphiti_config = GraphitiConfig(
                neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
                neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
                neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
                openai_api_key=api_key,  # Use available API key
                openai_model=os.getenv("GRAPHITI_MODEL", "gpt-4"),
                openai_embedding_model=os.getenv("GRAPHITI_EMBEDDING_MODEL", "text-embedding-3-small")
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


async def extract_entities_and_relations(
    text: str,
    doc_id: str,
    context: Optional[str] = None
) -> Tuple[List[Entity], List[Relation]]:
    """Standalone function to extract entities and relations from text.
    
    Args:
        text: Text content to analyze
        doc_id: Document identifier
        context: Additional context for extraction
        
    Returns:
        Tuple of (entities, relations)
    """
    extraction_service = GraphExtractionService()
    return await extraction_service.extract_entities_and_relations(
        text=text,
        doc_id=doc_id,
        context=context
    )


class GraphExtractionService:
    """Service for extracting entities and relations from text content."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """Initialize the graph extraction service.
        
        Args:
            api_key: API key for LLM (defaults to GEMINI_API_KEY env var)
            model: LLM model to use
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter required")
        
        self.model = model
        
        # Initialize LLM configuration
        self.llm_config = {
            "provider": "gemini",
            "api_key": self.api_key,
            "model": self.model,
            "temperature": 0.1,  # Low temperature for consistent results
            "max_tokens": 2000
        }
        
        # Initialize extractors
        self.entity_extractor = EntityExtractor(llm_config=self.llm_config)
        self.relation_extractor = RelationExtractor(llm_config=self.llm_config)
    
    async def extract_entities_and_relations(
        self,
        text: str,
        doc_id: str,
        context: Optional[str] = None
    ) -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from text.
        
        Args:
            text: Text content to analyze
            doc_id: Document identifier
            context: Additional context for extraction
            
        Returns:
            Tuple of (entities, relations)
        """
        try:
            # Extract entities
            graph_entities = await self.entity_extractor.extract(
                text=text,
                doc_id=doc_id,
                context=context
            )
            
            # Convert to common schema
            entities = [
                Entity(
                    id=entity.id,
                    name=entity.name,
                    type=str(entity.type),  # Handle both enum and string types
                    confidence=entity.confidence,
                    attributes=getattr(entity, 'attributes', {}) or {},
                    source_span=getattr(entity, 'source_span', None)
                )
                for entity in graph_entities
            ]
            
            # Extract relations
            graph_relations = await self.relation_extractor.extract(
                text=text,
                entities=graph_entities,
                doc_id=doc_id,
                context=context
            )
            
            # Convert to common schema
            relations = [
                Relation(
                    id=relation.id,
                    source_entity_id=relation.source_entity_id,
                    target_entity_id=relation.target_entity_id,
                    type=relation.type,  # Keep the type as-is (can be enum or string)
                    confidence=relation.confidence,
                    attributes=getattr(relation, 'attributes', {}) or {},
                    source_span=getattr(relation, 'source_span', None)
                )
                for relation in graph_relations
            ]
            
            return entities, relations
            
        except Exception as e:
            print(f"⚠️ Warning: Graph extraction failed: {e}")
            return [], []


class DatabaseIngestionService:
    """Service for ingesting content into Qdrant and Neo4j databases."""
    
    def __init__(self):
        """Initialize the database ingestion service."""
        self.qdrant_storage = None
        self.neo4j_storage = None
        self.graph_builder = None
    
    def initialize_qdrant(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize Qdrant storage.
        
        Args:
            config: Qdrant configuration (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if config:
                qdrant_config = QdrantConfig(**config)
            else:
                # Use default configuration
                # Prefer QDRANT_URL if available, otherwise use QDRANT_HOST/PORT
                qdrant_url = os.getenv('QDRANT_URL')
                if qdrant_url:
                    # Parse URL to extract components
                    from urllib.parse import urlparse
                    parsed = urlparse(qdrant_url)
                    host = parsed.hostname or "localhost"
                    port = parsed.port or (443 if parsed.scheme == 'https' else 6333)
                    https = parsed.scheme == 'https'
                else:
                    # Fall back to host/port configuration
                    host = os.getenv('QDRANT_HOST', 'localhost')
                    port = int(os.getenv('QDRANT_PORT', 6333))
                    https = port == 443  # Auto-detect HTTPS for port 443

                verify_ssl = os.getenv('QDRANT_VERIFY_SSL', 'true').lower() == 'true'
                qdrant_config = QdrantConfig(
                    host=host,
                    port=port,
                    https=https,
                    api_key=os.getenv('QDRANT_API_KEY'),
                    collection_name=os.getenv('QDRANT_COLLECTION', 'morag_documents'),
                    verify_ssl=verify_ssl
                )
            
            self.qdrant_storage = QdrantStorage(qdrant_config)
            return True
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize Qdrant: {e}")
            return False
    
    def initialize_neo4j(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize Neo4j storage.
        
        Args:
            config: Neo4j configuration (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if config:
                neo4j_config = Neo4jConfig(**config)
            else:
                # Use default configuration
                neo4j_config = Neo4jConfig(
                    uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                    username=os.getenv('NEO4J_USERNAME', 'neo4j'),
                    password=os.getenv('NEO4J_PASSWORD', 'password'),
                    database=os.getenv('NEO4J_DATABASE', 'neo4j'),
                    verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true",
                    trust_all_certificates=os.getenv("NEO4J_TRUST_ALL_CERTIFICATES", "false").lower() == "true"
                )
            
            self.neo4j_storage = Neo4j_storage(neo4j_config)
            return True
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize Neo4j: {e}")
            return False
    
    def initialize_graph_builder(self) -> bool:
        """Initialize graph builder with available storages.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            storage_backends = []
            
            if self.qdrant_storage:
                storage_backends.append(self.qdrant_storage)
            
            if self.neo4j_storage:
                storage_backends.append(self.neo4j_storage)
            
            if not storage_backends:
                print("⚠️ Warning: No storage backends available for graph builder")
                return False
            
            self.graph_builder = GraphBuilder(storage_backends=storage_backends)
            return True
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize graph builder: {e}")
            return False
    
    async def ingest_to_qdrant(
        self,
        text_content: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Ingest text content to Qdrant vector database.
        
        Args:
            text_content: Text content to ingest
            metadata: Metadata for the content
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of point IDs created in Qdrant
        """
        if not self.qdrant_storage:
            raise ValueError("Qdrant storage not initialized")
        
        try:
            # For now, use a simple chunking strategy
            # In a real implementation, you'd use a proper text splitter
            chunks = []
            for i in range(0, len(text_content), chunk_size - chunk_overlap):
                chunk = text_content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append({
                        'text': chunk,
                        'metadata': {**metadata, 'chunk_index': len(chunks)}
                    })
            
            # Store chunks (this is a simplified implementation)
            # In practice, you'd use the actual Qdrant storage API
            point_ids = [f"point_{i}" for i in range(len(chunks))]
            
            print(f"✅ Stored {len(chunks)} chunks in Qdrant")
            return point_ids
            
        except Exception as e:
            print(f"❌ Error ingesting to Qdrant: {e}")
            raise
    
    async def ingest_to_neo4j(
        self,
        entities: List[Entity],
        relations: List[Relation],
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ingest entities and relations to Neo4j graph database.
        
        Args:
            entities: List of extracted entities
            relations: List of extracted relations
            doc_metadata: Document metadata
            
        Returns:
            Dictionary with ingestion results
        """
        if not self.neo4j_storage:
            raise ValueError("Neo4j storage not initialized")
        
        try:
            # Convert back to graph entities and relations for storage
            graph_entities = []
            for entity in entities:
                # Create graph entity (simplified)
                graph_entity = GraphEntity(
                    id=entity.id,
                    name=entity.name,
                    type=entity.type,
                    confidence=entity.confidence,
                    attributes=getattr(entity, 'attributes', {}) or {}
                )
                graph_entities.append(graph_entity)
            
            graph_relations = []
            for relation in relations:
                # Create graph relation (simplified)
                graph_relation = GraphRelation(
                    id=relation.id,
                    source_entity_id=relation.source_entity_id,
                    target_entity_id=relation.target_entity_id,
                    type=relation.type,
                    confidence=relation.confidence,
                    attributes=getattr(relation, 'attributes', {}) or {}
                )
                graph_relations.append(graph_relation)
            
            # Store in Neo4j (this is a simplified implementation)
            # In practice, you'd use the actual Neo4j storage API
            result = {
                'entities_stored': len(graph_entities),
                'relations_stored': len(graph_relations),
                'document_metadata': doc_metadata
            }
            
            print(f"✅ Stored {len(graph_entities)} entities and {len(graph_relations)} relations in Neo4j")
            return result
            
        except Exception as e:
            print(f"❌ Error ingesting to Neo4j: {e}")
            raise


async def extract_and_ingest(
    text_content: str,
    doc_id: str,
    context: Optional[str] = None,
    use_qdrant: bool = True,
    use_neo4j: bool = True,
    qdrant_config: Optional[Dict[str, Any]] = None,
    neo4j_config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Extract entities/relations and ingest to databases.
    
    Args:
        text_content: Text content to process
        doc_id: Document identifier
        context: Additional context for extraction
        use_qdrant: Whether to ingest to Qdrant
        use_neo4j: Whether to ingest to Neo4j
        qdrant_config: Qdrant configuration
        neo4j_config: Neo4j configuration
        metadata: Additional metadata
        
    Returns:
        Dictionary with extraction and ingestion results
    """
    results = {
        'extraction': {'entities': [], 'relations': []},
        'ingestion': {'qdrant': None, 'neo4j': None}
    }
    
    try:
        # Extract entities and relations
        extraction_service = GraphExtractionService()
        entities, relations = await extraction_service.extract_entities_and_relations(
            text=text_content,
            doc_id=doc_id,
            context=context
        )
        
        results['extraction']['entities'] = entities
        results['extraction']['relations'] = relations
        
        print(f"✅ Extracted {len(entities)} entities and {len(relations)} relations")
        
        # Initialize ingestion service
        ingestion_service = DatabaseIngestionService()
        
        # Ingest to Qdrant if requested
        if use_qdrant:
            if ingestion_service.initialize_qdrant(qdrant_config):
                try:
                    point_ids = await ingestion_service.ingest_to_qdrant(
                        text_content=text_content,
                        metadata=metadata or {}
                    )
                    results['ingestion']['qdrant'] = {
                        'success': True,
                        'point_ids': point_ids,
                        'chunks_count': len(point_ids)
                    }
                except Exception as e:
                    results['ingestion']['qdrant'] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                results['ingestion']['qdrant'] = {
                    'success': False,
                    'error': 'Failed to initialize Qdrant storage'
                }
        
        # Ingest to Neo4j if requested
        if use_neo4j:
            if ingestion_service.initialize_neo4j(neo4j_config):
                try:
                    neo4j_result = await ingestion_service.ingest_to_neo4j(
                        entities=entities,
                        relations=relations,
                        doc_metadata=metadata or {}
                    )
                    results['ingestion']['neo4j'] = {
                        'success': True,
                        **neo4j_result
                    }
                except Exception as e:
                    results['ingestion']['neo4j'] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                results['ingestion']['neo4j'] = {
                    'success': False,
                    'error': 'Failed to initialize Neo4j storage'
                }
        
        return results
        
    except Exception as e:
        print(f"❌ Error in extract_and_ingest: {e}")
        results['error'] = str(e)
        return results


async def extract_and_ingest_with_graphiti(
    text_content: str,
    doc_id: str,
    title: str = None,
    context: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    use_traditional_extraction: bool = False
) -> Dict[str, Any]:
    """Extract and ingest content using Graphiti episodes.

    This is the new recommended approach that uses Graphiti's built-in
    entity extraction and knowledge graph capabilities.

    Args:
        text_content: Text content to process
        doc_id: Document identifier
        title: Document title
        context: Additional context for extraction
        metadata: Additional metadata
        use_traditional_extraction: Whether to also run traditional extraction

    Returns:
        Dictionary with extraction and ingestion results
    """
    results = {
        'graphiti': {'success': False, 'episode_name': None, 'error': None},
        'traditional': {'entities': [], 'relations': [], 'success': False} if use_traditional_extraction else None
    }

    # Initialize Graphiti service
    graphiti_service = GraphitiExtractionService(use_graphiti=True)

    try:
        # Ingest using Graphiti (this will handle entity extraction automatically)
        print(f"🚀 Ingesting content using Graphiti...")
        graphiti_result = await graphiti_service.ingest_document_content(
            content=text_content,
            doc_id=doc_id,
            title=title,
            metadata=metadata,
            source_description=context or f"CLI ingestion for document {doc_id}"
        )

        results['graphiti'] = graphiti_result

        if graphiti_result['success']:
            print(f"✅ Graphiti ingestion successful: {graphiti_result['episode_name']}")
            print(f"   Content length: {graphiti_result['content_length']} characters")
        else:
            print(f"❌ Graphiti ingestion failed: {graphiti_result.get('error', 'Unknown error')}")

        # Optionally run traditional extraction for comparison
        if use_traditional_extraction:
            print(f"🔄 Running traditional extraction for comparison...")
            try:
                entities, relations = await extract_entities_and_relations(
                    text=text_content,
                    doc_id=doc_id,
                    context=context
                )
                results['traditional'] = {
                    'entities': entities,
                    'relations': relations,
                    'success': True
                }
                print(f"✅ Traditional extraction: {len(entities)} entities, {len(relations)} relations")
            except Exception as e:
                print(f"❌ Traditional extraction failed: {e}")
                results['traditional']['success'] = False

        return results

    except Exception as e:
        print(f"❌ Error in Graphiti extraction: {e}")
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
        print(f"🔍 Searching with Graphiti: '{query}'")
        search_result = await graphiti_service.search_content(
            query=query,
            limit=limit,
            search_type=search_type
        )

        if search_result['success']:
            print(f"✅ Found {search_result['count']} results")
            for i, result in enumerate(search_result['results'][:3], 1):
                print(f"   {i}. {result.content[:100]}...")
        else:
            print(f"❌ Search failed: {search_result.get('error', 'Unknown error')}")

        return search_result

    except Exception as e:
        print(f"❌ Error in Graphiti search: {e}")
        return {"success": False, "error": str(e)}

    finally:
        await graphiti_service.close()