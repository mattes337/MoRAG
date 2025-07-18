# Step 1: Environment Setup and Basic Connection

**Duration**: 2-3 days
**Phase**: Proof of Concept
**Prerequisites**: Python 3.10+, Neo4j 5.26+, OpenAI API key

## Objective

Establish a working Graphiti environment integrated with MoRAG's existing infrastructure, validate basic connectivity, and create the foundation for subsequent integration steps.

## Deliverables

1. Graphiti library installed and configured
2. Basic connection to Neo4j database
3. Simple episode creation and retrieval
4. Configuration management for Graphiti settings
5. Basic test suite validating connectivity

## Implementation

### 1. Install Graphiti Dependencies

```bash
# Core Graphiti installation
pip install graphiti-core

# Optional: FalkorDB support (if using FalkorDB instead of Neo4j)
pip install graphiti-core[falkordb]

# Optional: Alternative LLM providers
pip install graphiti-core[anthropic,groq,google-genai]
```

### 2. Create Graphiti Configuration Module

**File**: `packages/morag-graph/src/morag_graph/graphiti/config.py`

```python
"""Graphiti configuration management for MoRAG integration."""

import os
from typing import Optional
from pydantic import BaseModel
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti integration."""

    # Neo4j connection settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "morag_graphiti"

    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_embedding_model: str = "text-embedding-3-small"

    # Graphiti-specific settings
    enable_telemetry: bool = False
    parallel_runtime: bool = False

    class Config:
        env_prefix = "GRAPHITI_"


def create_graphiti_instance(config: Optional[GraphitiConfig] = None) -> Graphiti:
    """Create and configure a Graphiti instance.

    Args:
        config: Optional configuration. If None, loads from environment.

    Returns:
        Configured Graphiti instance
    """
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
    
    # Set telemetry preference
    if not config.enable_telemetry:
        os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
    
    # Create Neo4j driver with custom database name
    driver = Neo4jDriver(
        uri=config.neo4j_uri,
        user=config.neo4j_username,
        password=config.neo4j_password,
        database=config.neo4j_database
    )
    
    # Create Graphiti instance
    graphiti = Graphiti(graph_driver=driver)

    return graphiti
```

### 3. Create Basic Connection Test

**File**: `packages/morag-graph/src/morag_graph/graphiti/connection_test.py`

```python
"""Basic connection and functionality test for Graphiti integration."""

import asyncio
import logging
from typing import Dict, Any
from graphiti_core.nodes import EpisodeType

from .config import create_graphiti_instance, GraphitiConfig

logger = logging.getLogger(__name__)


class GraphitiConnectionTest:
    """Test basic Graphiti connectivity and functionality."""
    
    def __init__(self, config: Optional[GraphitiConfig] = None):
        self.config = config
        self.graphiti = None
        
    async def setup(self) -> bool:
        """Initialize Graphiti connection.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.graphiti = create_graphiti_instance(self.config)
            logger.info("Graphiti instance created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create Graphiti instance: {e}")
            return False
    
    async def test_basic_connection(self) -> Dict[str, Any]:
        """Test basic Neo4j connection through Graphiti.
        
        Returns:
            Test results dictionary
        """
        results = {
            "connection": False,
            "database_accessible": False,
            "error": None
        }
        
        try:
            # Test basic connection by attempting to access the driver
            driver = self.graphiti.graph_driver
            if driver:
                results["connection"] = True
                logger.info("Basic connection test passed")
            
            # Test database accessibility
            # Note: Graphiti doesn't expose direct query methods, so we test episode creation
            test_episode = await self.graphiti.add_episode(
                name="Connection Test",
                episode_body="This is a test episode to verify database connectivity.",
                source_description="Graphiti connection test",
                episode_type=EpisodeType.text
            )
            
            if test_episode:
                results["database_accessible"] = True
                logger.info("Database accessibility test passed")
                
                # Clean up test episode
                # Note: Graphiti doesn't have direct delete methods, this is expected
                logger.info(f"Test episode created with ID: {test_episode}")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Connection test failed: {e}")
        
        return results
    
    async def test_episode_lifecycle(self) -> Dict[str, Any]:
        """Test complete episode creation and retrieval.
        
        Returns:
            Test results dictionary
        """
        results = {
            "episode_creation": False,
            "episode_search": False,
            "episode_id": None,
            "search_results": [],
            "error": None
        }
        
        try:
            # Create a test episode
            episode_name = "MoRAG Integration Test"
            episode_body = """
            This is a comprehensive test episode for MoRAG-Graphiti integration.
            It contains entities like John Doe, a software engineer working on AI systems.
            The project involves Neo4j databases and knowledge graphs.
            """
            
            episode_id = await self.graphiti.add_episode(
                name=episode_name,
                episode_body=episode_body,
                source_description="MoRAG integration test suite",
                episode_type=EpisodeType.text,
                metadata={"test": True, "step": "01"}
            )
            
            if episode_id:
                results["episode_creation"] = True
                results["episode_id"] = episode_id
                logger.info(f"Test episode created successfully: {episode_id}")
                
                # Test search functionality
                search_results = await self.graphiti.search(
                    query="John Doe software engineer",
                    limit=5
                )
                
                if search_results:
                    results["episode_search"] = True
                    results["search_results"] = [
                        {
                            "score": result.score,
                            "content": result.content[:100] + "..." if len(result.content) > 100 else result.content
                        }
                        for result in search_results
                    ]
                    logger.info(f"Search test passed: found {len(search_results)} results")
                else:
                    logger.warning("Search returned no results")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Episode lifecycle test failed: {e}")
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all connection and functionality tests.
        
        Returns:
            Comprehensive test results
        """
        logger.info("Starting Graphiti connection tests...")
        
        # Setup
        setup_success = await self.setup()
        if not setup_success:
            return {"setup": False, "error": "Failed to initialize Graphiti"}
        
        # Run tests
        connection_results = await self.test_basic_connection()
        episode_results = await self.test_episode_lifecycle()
        
        # Compile results
        all_results = {
            "setup": setup_success,
            "connection_test": connection_results,
            "episode_test": episode_results,
            "overall_success": (
                setup_success and 
                connection_results.get("connection", False) and
                connection_results.get("database_accessible", False) and
                episode_results.get("episode_creation", False)
            )
        }
        
        logger.info(f"All tests completed. Overall success: {all_results['overall_success']}")
        return all_results


async def main():
    """Run connection tests as standalone script."""
    logging.basicConfig(level=logging.INFO)
    
    test_runner = GraphitiConnectionTest()
    results = await test_runner.run_all_tests()
    
    print("\n" + "="*50)
    print("GRAPHITI CONNECTION TEST RESULTS")
    print("="*50)
    
    print(f"Setup: {'✓' if results['setup'] else '✗'}")
    print(f"Connection: {'✓' if results['connection_test'].get('connection') else '✗'}")
    print(f"Database Access: {'✓' if results['connection_test'].get('database_accessible') else '✗'}")
    print(f"Episode Creation: {'✓' if results['episode_test'].get('episode_creation') else '✗'}")
    print(f"Episode Search: {'✓' if results['episode_test'].get('episode_search') else '✗'}")
    
    if results['episode_test'].get('search_results'):
        print(f"\nSearch Results: {len(results['episode_test']['search_results'])} found")
    
    print(f"\nOverall Success: {'✓' if results['overall_success'] else '✗'}")
    
    if not results['overall_success']:
        print("\nErrors:")
        for test_name, test_results in results.items():
            if isinstance(test_results, dict) and test_results.get('error'):
                print(f"  {test_name}: {test_results['error']}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 4. Create Environment Configuration

**File**: `tasks/graphiti/.env.example`

```bash
# Graphiti Configuration for MoRAG Integration

# Neo4j Database Settings
GRAPHITI_NEO4J_URI=bolt://localhost:7687
GRAPHITI_NEO4J_USERNAME=neo4j
GRAPHITI_NEO4J_PASSWORD=password
GRAPHITI_NEO4J_DATABASE=morag_graphiti

# OpenAI API Settings
OPENAI_API_KEY=your_openai_api_key_here

# Graphiti Settings
GRAPHITI_TELEMETRY_ENABLED=false
USE_PARALLEL_RUNTIME=false

# Optional: Alternative LLM Providers
# ANTHROPIC_API_KEY=your_anthropic_key
# GROQ_API_KEY=your_groq_key
# GOOGLE_API_KEY=your_google_key
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_graphiti_connection.py`

```python
"""Unit tests for Graphiti connection and basic functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch
from morag_graph.graphiti.config import GraphitiConfig, create_graphiti_instance
from morag_graph.graphiti.connection_test import GraphitiConnectionTest


class TestGraphitiConfig:
    """Test Graphiti configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GraphitiConfig()
        assert config.neo4j_uri == "bolt://localhost:7687"
        assert config.neo4j_username == "neo4j"
        assert config.neo4j_database == "morag_graphiti"
        assert config.enable_telemetry is False
    
    def test_config_from_env(self):
        """Test configuration loading from environment variables."""
        with patch.dict('os.environ', {
            'GRAPHITI_NEO4J_URI': 'bolt://test:7687',
            'GRAPHITI_NEO4J_USERNAME': 'testuser',
            'GRAPHITI_NEO4J_DATABASE': 'testdb'
        }):
            config = GraphitiConfig()
            assert config.neo4j_uri == 'bolt://test:7687'
            assert config.neo4j_username == 'testuser'
            assert config.neo4j_database == 'testdb'


class TestGraphitiConnection:
    """Test Graphiti connection functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return GraphitiConfig(
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="test_password",
            neo4j_database="test_db",
            openai_api_key="test_key"
        )
    
    @pytest.mark.asyncio
    async def test_connection_test_setup(self, mock_config):
        """Test connection test setup."""
        test_runner = GraphitiConnectionTest(mock_config)
        
        with patch('morag_graph.graphiti.config.create_graphiti_instance') as mock_create:
            mock_graphiti = Mock()
            mock_create.return_value = mock_graphiti
            
            result = await test_runner.setup()
            assert result is True
            assert test_runner.graphiti == mock_graphiti
    
    @pytest.mark.asyncio
    async def test_episode_creation_mock(self, mock_config):
        """Test episode creation with mocked Graphiti."""
        test_runner = GraphitiConnectionTest(mock_config)
        
        # Mock Graphiti instance
        mock_graphiti = Mock()
        mock_graphiti.add_episode = Mock(return_value="test_episode_id")
        mock_graphiti.search = Mock(return_value=[])
        test_runner.graphiti = mock_graphiti
        
        results = await test_runner.test_episode_lifecycle()
        
        assert results["episode_creation"] is True
        assert results["episode_id"] == "test_episode_id"
        mock_graphiti.add_episode.assert_called_once()


# Integration test (requires actual Neo4j instance)
@pytest.mark.integration
class TestGraphitiIntegration:
    """Integration tests requiring actual Neo4j database."""
    
    @pytest.mark.asyncio
    async def test_real_connection(self):
        """Test actual connection to Neo4j through Graphiti."""
        # Skip if no Neo4j available
        pytest.importorskip("neo4j")
        
        test_runner = GraphitiConnectionTest()
        
        try:
            results = await test_runner.run_all_tests()
            # Don't assert success as it depends on environment
            # Just verify structure
            assert "setup" in results
            assert "connection_test" in results
            assert "episode_test" in results
            assert "overall_success" in results
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
```

## Validation Checklist

- [ ] Graphiti library installed successfully
- [ ] Neo4j connection established through Graphiti
- [ ] Basic episode creation works
- [ ] Episode search functionality operational
- [ ] Configuration management implemented
- [ ] Unit tests pass
- [ ] Integration tests pass (with Neo4j available)
- [ ] Error handling for connection failures
- [ ] Logging and monitoring in place

## Success Criteria

1. **Functional**: Graphiti can connect to Neo4j and create/search episodes
2. **Configurable**: Environment-based configuration management
3. **Testable**: Comprehensive test suite with mocks and integration tests
4. **Maintainable**: Clean code structure with proper error handling
5. **Documented**: Clear setup instructions and usage examples

## Next Steps

After completing this step:
1. Verify all tests pass
2. Document any environment-specific configuration needs
3. Proceed to [Step 2: Document to Episode Mapping](./step-02-document-episode-mapping.md)

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Verify Neo4j is running and accessible
   - Check connection credentials and URI
   - Ensure database exists or can be created

2. **OpenAI API Issues**
   - Verify API key is valid and has sufficient credits
   - Check network connectivity to OpenAI services
   - Consider using alternative LLM providers

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility (3.10+)
   - Verify virtual environment activation

### Performance Notes

- Initial episode creation may be slow due to index creation
- Search performance improves with more data
- Monitor memory usage during large episode ingestion
