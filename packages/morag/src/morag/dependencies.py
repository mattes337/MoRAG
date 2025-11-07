"""FastAPI dependencies for MoRAG API."""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import structlog
from fastapi import HTTPException
from morag.api import MoRAGAPI
from morag_services import ServiceConfig

logger = structlog.get_logger(__name__)

# Try to import graph components, but handle gracefully if not available
try:
    # Import components directly to avoid circular import issues
    from morag_graph.operations import GraphAnalytics, GraphCRUD, GraphTraversal
    from morag_graph.query import QueryEntityExtractor
    from morag_graph.retrieval import ContextExpansionEngine, HybridRetrievalCoordinator
    from morag_graph.storage import (
        Neo4jConfig,
        Neo4jStorage,
        QdrantConfig,
        QdrantStorage,
    )

    GRAPH_AVAILABLE = True
except ImportError as e:
    logger.warning("Graph components not available", error=str(e))
    GRAPH_AVAILABLE = False
    # Create dummy classes for type hints
    HybridRetrievalCoordinator = None
    ContextExpansionEngine = None
    QueryEntityExtractor = None
    Neo4jStorage = None
    QdrantStorage = None
    Neo4jConfig = None
    QdrantConfig = None
    GraphCRUD = None
    GraphTraversal = None
    GraphAnalytics = None

# Try to import reasoning components, but handle gracefully if not available
try:
    from morag_reasoning import (
        IterativeRetriever,
        LLMClient,
        LLMConfig,
        PathSelectionAgent,
        ReasoningPathFinder,
        RecursiveFactRetrievalService,
        RetrievalContext,
    )

    REASONING_AVAILABLE = True
except ImportError as e:
    logger.warning("Reasoning components not available", error=str(e))
    REASONING_AVAILABLE = False
    # Create dummy classes for type hints
    LLMClient = None
    LLMConfig = None
    PathSelectionAgent = None
    ReasoningPathFinder = None
    IterativeRetriever = None
    RetrievalContext = None
    RecursiveFactRetrievalService = None

logger = structlog.get_logger(__name__)


class FallbackHybridRetrievalCoordinator:
    """Fallback coordinator that only does vector search when graph components are unavailable."""

    def __init__(self, vector_retriever):
        self.vector_retriever = vector_retriever
        self.logger = structlog.get_logger(__name__)

    async def retrieve(self, query: str, max_results: int = 10) -> list:
        """Retrieve using only vector search."""
        try:
            self.logger.info("Using fallback vector-only retrieval", query=query[:100])
            results = await self.vector_retriever.retrieve(query, max_results)
            return results
        except Exception as e:
            self.logger.error("Fallback retrieval failed", error=str(e))
            return []


@lru_cache()
def get_service_config() -> ServiceConfig:
    """Get service configuration."""
    return ServiceConfig()


@lru_cache()
def get_morag_api() -> MoRAGAPI:
    """Get MoRAG API instance."""
    config = get_service_config()
    return MoRAGAPI(config)


@lru_cache()
def get_neo4j_storage() -> Optional[Neo4jStorage]:
    """Get Neo4j storage instance."""
    if not GRAPH_AVAILABLE:
        return None
    try:
        config = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true",
            trust_all_certificates=os.getenv(
                "NEO4J_TRUST_ALL_CERTIFICATES", "false"
            ).lower()
            == "true",
        )
        return Neo4jStorage(config)
    except Exception as e:
        logger.warning("Neo4j storage not available", error=str(e))
        return None


@lru_cache()
def get_qdrant_storage() -> Optional[QdrantStorage]:
    """Get Qdrant storage instance."""
    if not GRAPH_AVAILABLE:
        return None
    try:
        # Prefer QDRANT_URL if available, otherwise use QDRANT_HOST/PORT
        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            # Parse URL to extract components
            from urllib.parse import urlparse

            parsed = urlparse(qdrant_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 6333)
            https = parsed.scheme == "https"
        else:
            # Fall back to host/port configuration
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            https = port == 443  # Auto-detect HTTPS for port 443

        config = QdrantConfig(
            host=host,
            port=port,
            https=https,
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION", "morag_vectors"),
        )
        return QdrantStorage(config)
    except Exception as e:
        logger.warning("Qdrant storage not available", error=str(e))
        return None


@lru_cache()
def get_graph_crud() -> Optional[GraphCRUD]:
    """Get graph CRUD operations instance."""
    if not GRAPH_AVAILABLE:
        return None
    try:
        storage = get_neo4j_storage()
        if storage is None:
            return None
        return GraphCRUD(storage)
    except Exception as e:
        logger.warning("Graph CRUD not available", error=str(e))
        return None


@lru_cache()
def get_graph_traversal() -> Optional[GraphTraversal]:
    """Get graph traversal instance."""
    if not GRAPH_AVAILABLE:
        return None
    try:
        storage = get_neo4j_storage()
        if storage is None:
            return None
        return GraphTraversal(storage)
    except Exception as e:
        logger.warning("Graph traversal not available", error=str(e))
        return None


@lru_cache()
def get_graph_analytics() -> Optional[GraphAnalytics]:
    """Get graph analytics instance."""
    if not GRAPH_AVAILABLE:
        return None
    try:
        storage = get_neo4j_storage()
        if storage is None:
            return None
        return GraphAnalytics(storage)
    except Exception as e:
        logger.warning("Graph analytics not available", error=str(e))
        return None


@lru_cache()
def get_query_entity_extractor() -> Optional[QueryEntityExtractor]:
    """Get query entity extractor instance."""
    try:
        storage = get_neo4j_storage()
        return QueryEntityExtractor(graph_storage=storage)
    except Exception as e:
        logger.warning("Query entity extractor not available", error=str(e))
        return None


@lru_cache()
def get_context_expansion_engine() -> Optional[ContextExpansionEngine]:
    """Get context expansion engine instance."""
    try:
        storage = get_neo4j_storage()
        return ContextExpansionEngine(storage)
    except Exception as e:
        logger.warning("Context expansion engine not available", error=str(e))
        return None


@lru_cache()
def get_hybrid_retrieval_coordinator() -> Optional[HybridRetrievalCoordinator]:
    """Get hybrid retrieval coordinator instance."""
    try:
        # Create a simple vector retriever wrapper for the existing search functionality
        class VectorRetrieverWrapper:
            def __init__(self, morag_api: MoRAGAPI):
                self.morag_api = morag_api

            async def retrieve(self, query: str, max_results: int = 10) -> list:
                """Retrieve using vector search."""
                try:
                    results = await self.morag_api.search(query, max_results)
                    return results
                except Exception as e:
                    logger.error("Vector retrieval failed", error=str(e))
                    return []

        morag_api = get_morag_api()
        vector_retriever = VectorRetrieverWrapper(morag_api)
        context_expansion_engine = get_context_expansion_engine()
        query_entity_extractor = get_query_entity_extractor()

        # If graph components are not available, create a fallback coordinator
        if context_expansion_engine is None or query_entity_extractor is None:
            logger.warning(
                "Graph components not available, creating fallback coordinator"
            )
            return FallbackHybridRetrievalCoordinator(vector_retriever)

        return HybridRetrievalCoordinator(
            vector_retriever=vector_retriever,
            context_expansion_engine=context_expansion_engine,
            query_entity_extractor=query_entity_extractor,
        )
    except Exception as e:
        logger.error("Failed to create hybrid retrieval coordinator", error=str(e))
        # Return fallback that only does vector search
        morag_api = get_morag_api()

        class VectorRetrieverWrapper:
            def __init__(self, morag_api: MoRAGAPI):
                self.morag_api = morag_api

            async def retrieve(self, query: str, max_results: int = 10) -> list:
                """Retrieve using vector search."""
                try:
                    results = await self.morag_api.search(query, max_results)
                    return results
                except Exception as e:
                    logger.error("Vector retrieval failed", error=str(e))
                    return []

        vector_retriever = VectorRetrieverWrapper(morag_api)
        return FallbackHybridRetrievalCoordinator(vector_retriever)


class GraphEngine:
    """Wrapper for graph operations to provide a unified interface."""

    def __init__(self):
        self.crud = get_graph_crud()
        self.traversal = get_graph_traversal()
        self.analytics = get_graph_analytics()
        self.available = self.crud is not None and self.traversal is not None

    async def get_entity(self, entity_id: str):
        """Get entity by ID."""
        if not self.available:
            raise HTTPException(status_code=503, detail="Graph engine not available")
        return await self.crud.get_entity(entity_id)

    async def find_entities_by_name(self, name: str, entity_type: Optional[str] = None):
        """Find entities by name."""
        if not self.available:
            raise HTTPException(status_code=503, detail="Graph engine not available")

        # Check if the method exists, if not, provide a fallback
        if hasattr(self.crud, "find_entities_by_name"):
            return await self.crud.find_entities_by_name(name, entity_type)
        else:
            # Fallback: search for entities with similar names
            logger.warning("find_entities_by_name method not available, using fallback")
            # Return empty list as fallback
            return []

    async def get_entity_relations(
        self, entity_id: str, depth: int = 1, max_relations: int = 50
    ):
        """Get entity relations."""
        if not self.available:
            raise HTTPException(status_code=503, detail="Graph engine not available")
        return await self.traversal.find_neighbors(entity_id, max_distance=depth)

    async def find_shortest_paths(
        self,
        start_id: str,
        end_id: str,
        max_paths: int = 10,
        relation_filters: Optional[list] = None,
    ):
        """Find shortest paths between entities."""
        if not self.available:
            raise HTTPException(status_code=503, detail="Graph engine not available")
        path = await self.traversal.find_shortest_path(
            start_id, end_id, relation_filters
        )
        return [path] if path else []

    async def explore_from_entity(
        self,
        entity_id: str,
        max_depth: int = 3,
        max_paths: int = 10,
        entity_filters: Optional[list] = None,
        relation_filters: Optional[list] = None,
    ):
        """Explore from entity."""
        if not self.available:
            raise HTTPException(status_code=503, detail="Graph engine not available")
        neighbors = await self.traversal.find_neighbors(
            entity_id, max_distance=max_depth, relation_types=relation_filters
        )
        # Convert to paths format (simplified)
        paths = []
        for neighbor in neighbors[:max_paths]:
            path = type(
                "Path",
                (),
                {
                    "entities": [entity_id, neighbor.id],
                    "relations": [],
                    "total_weight": 1.0,
                    "confidence": 0.8,
                },
            )()
            paths.append(path)
        return paths

    async def get_graph_statistics(self):
        """Get graph statistics."""
        if not self.available or self.analytics is None:
            raise HTTPException(status_code=503, detail="Graph analytics not available")
        return await self.analytics.get_graph_statistics()

    async def calculate_centrality_measures(self):
        """Calculate centrality measures."""
        if not self.available or self.analytics is None:
            raise HTTPException(status_code=503, detail="Graph analytics not available")
        return await self.analytics.calculate_centrality_measures()

    async def detect_communities(self):
        """Detect communities."""
        if not self.available or self.analytics is None:
            raise HTTPException(status_code=503, detail="Graph analytics not available")
        return await self.analytics.detect_communities()


@lru_cache()
def get_graph_engine() -> GraphEngine:
    """Get graph engine instance."""
    return GraphEngine()


@lru_cache()
def get_llm_client() -> Optional[LLMClient]:
    """Get LLM client instance for reasoning."""
    if not REASONING_AVAILABLE:
        return None
    try:
        from morag_core.config import LLMConfig as UnifiedLLMConfig

        config = UnifiedLLMConfig.from_env_and_overrides()

        # Convert to reasoning LLMConfig format
        reasoning_config = LLMConfig(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
        )
        return LLMClient(reasoning_config)
    except Exception as e:
        logger.warning("LLM client not available", error=str(e))
        return None


async def get_recursive_fact_retrieval_service() -> Optional[
    RecursiveFactRetrievalService
]:
    """Get recursive fact retrieval service instance."""
    if not REASONING_AVAILABLE:
        return None

    try:
        # Get LLM client
        llm_client = get_llm_client()
        if not llm_client:
            return None

        # Get storage instances
        neo4j_storage = await get_connected_default_neo4j_storage()
        qdrant_storage = await get_connected_default_qdrant_storage()

        if not neo4j_storage or not qdrant_storage:
            logger.warning(
                "Storage instances not available for recursive fact retrieval"
            )
            return None

        # Create stronger LLM client for final synthesis (could use a different model)
        stronger_llm_config = LLMConfig(
            provider=os.getenv("MORAG_LLM_PROVIDER", "gemini"),
            model=os.getenv(
                "MORAG_GEMINI_MODEL_STRONG",
                os.getenv("MORAG_GEMINI_MODEL", "gemini-1.5-flash"),
            ),
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=float(os.getenv("MORAG_LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("MORAG_LLM_MAX_TOKENS", "4000")),
            max_retries=int(os.getenv("MORAG_LLM_MAX_RETRIES", "5")),
        )
        stronger_llm_client = LLMClient(stronger_llm_config)

        # Initialize embedding service for enhanced retrieval
        embedding_service = None
        try:
            from morag_services.embedding import GeminiEmbeddingService

            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                embedding_service = GeminiEmbeddingService(api_key=api_key)
                logger.info("Embedding service initialized for enhanced retrieval")
            else:
                logger.warning("GEMINI_API_KEY not found - enhanced retrieval disabled")
        except Exception as e:
            logger.warning("Failed to initialize embedding service", error=str(e))

        return RecursiveFactRetrievalService(
            llm_client=llm_client,
            neo4j_storage=neo4j_storage,
            qdrant_storage=qdrant_storage,
            embedding_service=embedding_service,
            stronger_llm_client=stronger_llm_client,
        )
    except Exception as e:
        logger.warning("Recursive fact retrieval service not available", error=str(e))
        return None


@lru_cache()
def get_path_selection_agent() -> Optional[PathSelectionAgent]:
    """Get path selection agent instance."""
    if not REASONING_AVAILABLE:
        return None
    try:
        llm_client = get_llm_client()
        if llm_client is None:
            return None
        max_paths = int(os.getenv("MORAG_REASONING_MAX_PATHS", "10"))
        return PathSelectionAgent(llm_client, max_paths=max_paths)
    except Exception as e:
        logger.warning("Path selection agent not available", error=str(e))
        return None


@lru_cache()
def get_reasoning_path_finder() -> Optional[ReasoningPathFinder]:
    """Get reasoning path finder instance."""
    if not REASONING_AVAILABLE:
        return None
    try:
        graph_engine = get_graph_engine()
        path_selector = get_path_selection_agent()
        if path_selector is None:
            return None
        return ReasoningPathFinder(graph_engine, path_selector)
    except Exception as e:
        logger.warning("Reasoning path finder not available", error=str(e))
        return None


@lru_cache()
def get_iterative_retriever() -> Optional[IterativeRetriever]:
    """Get iterative retriever instance."""
    if not REASONING_AVAILABLE:
        return None
    try:
        llm_client = get_llm_client()
        graph_engine = get_graph_engine()

        # Create vector retriever wrapper
        class VectorRetrieverWrapper:
            def __init__(self, morag_api: MoRAGAPI):
                self.morag_api = morag_api

            async def search(self, query: str, limit: int = 10) -> list:
                """Search using vector similarity."""
                try:
                    results = await self.morag_api.search(query, limit)
                    return results
                except Exception as e:
                    logger.error("Vector search failed", error=str(e))
                    return []

            async def retrieve(self, query: str, max_results: int = 10) -> list:
                """Retrieve using vector search."""
                return await self.search(query, max_results)

        morag_api = get_morag_api()
        vector_retriever = VectorRetrieverWrapper(morag_api)

        if llm_client is None:
            return None

        max_iterations = int(os.getenv("MORAG_REASONING_MAX_ITERATIONS", "5"))
        sufficiency_threshold = float(
            os.getenv("MORAG_REASONING_SUFFICIENCY_THRESHOLD", "0.8")
        )

        return IterativeRetriever(
            llm_client=llm_client,
            graph_engine=graph_engine,
            vector_retriever=vector_retriever,
            max_iterations=max_iterations,
            sufficiency_threshold=sufficiency_threshold,
        )
    except Exception as e:
        logger.warning("Iterative retriever not available", error=str(e))
        return None


def create_dynamic_graph_engine(
    database_servers: Optional[List[Dict[str, Any]]] = None
) -> GraphEngine:
    """Create a graph engine with dynamic database connections."""
    if database_servers:
        from morag.database_factory import get_neo4j_storages

        neo4j_storages = get_neo4j_storages(database_servers)

        if neo4j_storages:
            # Create a custom graph engine with the first available storage
            storage = neo4j_storages[0]

            class DynamicGraphEngine(GraphEngine):
                def __init__(self, storage):
                    self.storage = storage
                    self.available = True
                    # Initialize components with custom storage
                    try:
                        from morag_graph.operations import (
                            GraphAnalytics,
                            GraphCRUD,
                            GraphTraversal,
                        )

                        self.crud = GraphCRUD(storage)
                        self.traversal = GraphTraversal(storage)
                        self.analytics = GraphAnalytics(storage)
                    except Exception as e:
                        logger.error(
                            "Failed to initialize dynamic graph engine components",
                            error=str(e),
                        )
                        self.available = False

            return DynamicGraphEngine(storage)

    # Fall back to default graph engine
    return get_graph_engine()


def create_dynamic_hybrid_retrieval_coordinator(
    database_servers: Optional[List[Dict[str, Any]]] = None
):
    """Create a hybrid retrieval coordinator with dynamic database connections."""
    if database_servers:
        from morag.database_factory import get_neo4j_storages, get_qdrant_storages

        qdrant_storages = get_qdrant_storages(database_servers)
        neo4j_storages = get_neo4j_storages(database_servers)

        # Create custom vector retriever if Qdrant storages are available
        if qdrant_storages:

            class DynamicVectorRetriever:
                def __init__(self, qdrant_storage):
                    self.storage = qdrant_storage

                async def retrieve(self, query: str, max_results: int = 10) -> list:
                    """Retrieve using custom Qdrant storage."""
                    try:
                        # Return empty results as placeholder for dynamic storage
                        return []
                    except Exception as e:
                        logger.error("Dynamic vector retrieval failed", error=str(e))
                        return []

            vector_retriever = DynamicVectorRetriever(qdrant_storages[0])

            # Create custom context expansion engine if Neo4j storages are available
            if neo4j_storages:
                try:
                    from morag_graph.retrieval import ContextExpansionEngine

                    context_expansion_engine = ContextExpansionEngine(neo4j_storages[0])

                    # Create query entity extractor
                    from morag_graph.query import QueryEntityExtractor

                    query_entity_extractor = QueryEntityExtractor()

                    from morag_graph.retrieval import HybridRetrievalCoordinator

                    return HybridRetrievalCoordinator(
                        vector_retriever=vector_retriever,
                        context_expansion_engine=context_expansion_engine,
                        query_entity_extractor=query_entity_extractor,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to create dynamic hybrid coordinator", error=str(e)
                    )

    # Fall back to default coordinator
    return get_hybrid_retrieval_coordinator()
