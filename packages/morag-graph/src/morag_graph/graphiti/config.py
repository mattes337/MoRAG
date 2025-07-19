"""Graphiti configuration management for MoRAG integration."""

import os
from typing import Optional
from pydantic import BaseModel, Field
import structlog

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
    except ImportError as e:
        logger.error("Graphiti core not available", error=str(e))
        raise ImportError(
            "graphiti-core is not installed. Install with: pip install graphiti-core"
        ) from e

    if config is None:
        config = GraphitiConfig(
            neo4j_uri=os.getenv("GRAPHITI_NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("GRAPHITI_NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("GRAPHITI_NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("GRAPHITI_NEO4J_DATABASE", "morag_graphiti"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            enable_telemetry=os.getenv("GRAPHITI_TELEMETRY_ENABLED", "false").lower() == "true",
            parallel_runtime=os.getenv("USE_PARALLEL_RUNTIME", "false").lower() == "true"
        )

    # Validate required settings
    if not config.openai_api_key:
        raise ValueError("OpenAI API key is required for Graphiti integration")

    # Set OpenAI API key in environment for Graphiti
    os.environ["OPENAI_API_KEY"] = config.openai_api_key

    # Create Neo4j driver
    driver = Neo4jDriver(
        uri=config.neo4j_uri,
        user=config.neo4j_username,  # Use 'user' instead of 'username'
        password=config.neo4j_password,
        database=config.neo4j_database
    )

    # Create Graphiti instance with minimal parameters
    # Let Graphiti use default LLM configuration from environment
    graphiti = Graphiti(
        graph_driver=driver
    )

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
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("GRAPHITI_OPENAI_MODEL", "gpt-4"),
        openai_embedding_model=os.getenv("GRAPHITI_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        enable_telemetry=os.getenv("GRAPHITI_TELEMETRY_ENABLED", "false").lower() == "true",
        parallel_runtime=os.getenv("USE_PARALLEL_RUNTIME", "false").lower() == "true"
    )
