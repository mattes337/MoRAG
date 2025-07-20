"""Graphiti configuration management for MoRAG integration."""

import os
from typing import Optional
from pydantic import BaseModel, Field
import structlog

from .vector_patch import apply_vector_similarity_patch

logger = structlog.get_logger(__name__)


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti integration."""

    # Neo4j connection settings
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    neo4j_database: str = Field(default="morag_graphiti", description="Neo4j database name")

    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="OpenAI model for LLM inference")
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")

    # Graphiti-specific settings
    enable_telemetry: bool = Field(default=False, description="Enable Graphiti telemetry")
    parallel_runtime: bool = Field(default=False, description="Enable parallel runtime")

    class Config:
        env_prefix = "GRAPHITI_"


def create_graphiti_instance(config: Optional[GraphitiConfig] = None):
    """Create and configure a Graphiti instance.

    Args:
        config: Optional configuration. If None, loads from environment.

    Returns:
        Configured Graphiti instance

    Raises:
        ImportError: If graphiti-core is not installed
        ValueError: If required configuration is missing
    """
    try:
        from graphiti_core import Graphiti
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
        from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
        from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
    except ImportError as e:
        logger.error("Graphiti core not available", error=str(e))
        raise ImportError(
            "graphiti-core is not installed. Install with: pip install 'graphiti-core[google-genai]'"
        ) from e

    if config is None:
        # Determine which API key is available
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Choose appropriate models based on available API keys
        if gemini_key:
            # Use Gemini models if Gemini API key is available
            model = os.getenv("GRAPHITI_MODEL", "gemini-1.5-flash")
            embedding_model = os.getenv("GRAPHITI_EMBEDDING_MODEL", "text-embedding-004")
            api_key = gemini_key
        elif openai_key:
            # Use OpenAI models if only OpenAI API key is available
            model = os.getenv("GRAPHITI_MODEL", "gpt-4")
            embedding_model = os.getenv("GRAPHITI_EMBEDDING_MODEL", "text-embedding-3-small")
            api_key = openai_key
        else:
            # Default to Gemini models but will fail without API key
            model = "gemini-1.5-flash"
            embedding_model = "text-embedding-004"
            api_key = None

        config = GraphitiConfig(
            neo4j_uri=os.getenv("GRAPHITI_NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("GRAPHITI_NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("GRAPHITI_NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("GRAPHITI_NEO4J_DATABASE", "morag_graphiti"),
            openai_api_key=api_key,
            openai_model=model,
            openai_embedding_model=embedding_model,
            enable_telemetry=os.getenv("GRAPHITI_TELEMETRY_ENABLED", "false").lower() == "true",
            parallel_runtime=os.getenv("USE_PARALLEL_RUNTIME", "false").lower() == "true"
        )

    # Validate required settings
    if not config.openai_api_key:
        raise ValueError("GEMINI_API_KEY or OPENAI_API_KEY is required for Graphiti integration")

    # Apply vector similarity patches for Neo4j Community Edition compatibility
    apply_vector_similarity_patch()

    # Additional aggressive patching for bulk queries
    try:
        import graphiti_core.utils.bulk_utils as bulk_utils
        from morag_graph.graphiti.vector_patch import GraphitiVectorPatch

        # Store original functions
        original_get_entity_node_save_bulk_query = bulk_utils.get_entity_node_save_bulk_query
        original_get_entity_edge_save_bulk_query = bulk_utils.get_entity_edge_save_bulk_query

        # Create patched versions
        def patched_entity_node_query(nodes, provider=None):
            query = original_get_entity_node_save_bulk_query(nodes, provider)
            return GraphitiVectorPatch.patch_query(query)

        def patched_entity_edge_query(db_type='neo4j'):
            query = original_get_entity_edge_save_bulk_query(db_type)
            return GraphitiVectorPatch.patch_query(query)

        # Apply patches
        bulk_utils.get_entity_node_save_bulk_query = patched_entity_node_query
        bulk_utils.get_entity_edge_save_bulk_query = patched_entity_edge_query

        logger.info("✅ Applied aggressive bulk query patches")

    except Exception as e:
        logger.warning(f"⚠️ Failed to apply aggressive bulk query patches: {e}")

    # Create Neo4j driver
    driver = Neo4jDriver(
        uri=config.neo4j_uri,
        user=config.neo4j_username,  # Use 'user' instead of 'username'
        password=config.neo4j_password,
        database=config.neo4j_database
    )

    # Check if using Gemini models
    if config.openai_model and "gemini" in config.openai_model.lower():
        # Configure Gemini clients
        api_key = config.openai_api_key

        # Create Gemini LLM client
        llm_client = GeminiClient(
            config=LLMConfig(
                api_key=api_key,
                model=config.openai_model
            )
        )

        # Create Gemini embedder
        embedder = GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=api_key,
                embedding_model=config.openai_embedding_model or "text-embedding-004"
            )
        )

        # Create Gemini reranker
        cross_encoder = GeminiRerankerClient(
            config=LLMConfig(
                api_key=api_key,
                model="gemini-2.0-flash"  # Use recommended model for reranking
            )
        )

        # Create Graphiti instance with Gemini clients
        graphiti = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder
        )

        logger.info("Configured Graphiti with Gemini clients",
                   model=config.openai_model,
                   embedding_model=config.openai_embedding_model or "text-embedding-004")
    else:
        # Use standard OpenAI configuration
        os.environ["OPENAI_API_KEY"] = config.openai_api_key

        # Create Graphiti instance with default OpenAI clients
        graphiti = Graphiti(
            graph_driver=driver
        )

        logger.info("Configured Graphiti with OpenAI clients",
                   model=config.openai_model)

    logger.info(
        "Graphiti instance created",
        neo4j_uri=config.neo4j_uri,
        neo4j_database=config.neo4j_database,
        llm_model=config.openai_model,
        embedding_model=config.openai_embedding_model
    )

    return graphiti


def load_config_from_env() -> GraphitiConfig:
    """Load Graphiti configuration from environment variables.

    Returns:
        GraphitiConfig instance with values from environment
    """
    return GraphitiConfig(
        neo4j_uri=os.getenv("GRAPHITI_NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("GRAPHITI_NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("GRAPHITI_NEO4J_PASSWORD", "password"),
        neo4j_database=os.getenv("GRAPHITI_NEO4J_DATABASE", "morag_graphiti"),
        openai_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("GRAPHITI_MODEL", "gemini-1.5-flash"),
        openai_embedding_model=os.getenv("GRAPHITI_EMBEDDING_MODEL", "text-embedding-004"),
        enable_telemetry=os.getenv("GRAPHITI_TELEMETRY_ENABLED", "false").lower() == "true",
        parallel_runtime=os.getenv("USE_PARALLEL_RUNTIME", "false").lower() == "true"
    )
