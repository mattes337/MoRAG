"""Basic functionality tests for the agents framework."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import os

# Set up test environment
os.environ["GEMINI_API_KEY"] = "test-key"

from agents.base.config import AgentConfig, PromptConfig, ModelConfig
from agents.base.template import ConfigurablePromptTemplate
from agents.extraction.fact_extraction import FactExtractionAgent
from agents.extraction.models import FactExtractionResult, ExtractedFact, FactType, ConfidenceLevel
from agents.factory.utils import create_agent, get_agent


class TestAgentConfig:
    """Test agent configuration."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = AgentConfig(name="test_agent")
        assert config.name == "test_agent"
        assert config.model.provider == "gemini"
        assert config.timeout == 30
    
    def test_config_with_overrides(self):
        """Test configuration with overrides."""
        config = AgentConfig(
            name="test_agent",
            timeout=60,
            model=ModelConfig(temperature=0.5)
        )
        assert config.timeout == 60
        assert config.model.temperature == 0.5
    
    def test_agent_specific_config(self):
        """Test agent-specific configuration."""
        config = AgentConfig(name="test_agent")
        config.set_agent_config("max_facts", 50)
        assert config.get_agent_config("max_facts") == 50
        assert config.get_agent_config("nonexistent", "default") == "default"


class TestPromptTemplate:
    """Test prompt template functionality."""
    
    def test_template_creation(self):
        """Test creating a prompt template."""
        config = PromptConfig()
        template = ConfigurablePromptTemplate(
            config,
            "System: {{ config.domain }}",
            "User: {{ input }}"
        )
        assert template.config == config
    
    def test_template_rendering(self):
        """Test template rendering."""
        config = PromptConfig(domain="test")
        template = ConfigurablePromptTemplate(
            config,
            "System: {{ config.domain }}",
            "User: {{ input }}"
        )
        
        result = template.render_template("Hello {{ name }}", name="World")
        assert result == "Hello World"
    
    def test_prompt_generation(self):
        """Test full prompt generation."""
        config = PromptConfig(domain="test", include_examples=False)
        template = ConfigurablePromptTemplate(
            config,
            "System: You are a {{ config.domain }} expert.",
            "User: Process {{ input }}"
        )
        
        prompts = template.generate_full_prompt("test input")
        assert "test expert" in prompts["system"]
        assert "test input" in prompts["user"]


class TestFactExtractionAgent:
    """Test fact extraction agent."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = FactExtractionAgent()
        assert agent.config.name == "fact_extraction"
        assert isinstance(agent._template, ConfigurablePromptTemplate)
    
    def test_result_type(self):
        """Test result type."""
        agent = FactExtractionAgent()
        assert agent.get_result_type() == FactExtractionResult
    
    @patch('agents.base.agent.BaseAgent._call_model')
    async def test_fact_extraction_execution(self, mock_call_model):
        """Test fact extraction execution."""
        # Mock the model response
        mock_response = """{
            "facts": [
                {
                    "subject": "Test subject",
                    "object": "Test object",
                    "approach": null,
                    "solution": null,
                    "condition": null,
                    "remarks": null,
                    "fact_type": "declarative",
                    "confidence": 0.9,
                    "keywords": ["test"],
                    "source_text": null
                }
            ],
            "total_facts": 1,
            "confidence": "high",
            "domain": "general",
            "language": "en",
            "metadata": {}
        }"""
        
        mock_call_model.return_value = mock_response
        
        agent = FactExtractionAgent()
        result = await agent.extract_facts("Test text")
        
        assert isinstance(result, FactExtractionResult)
        assert result.total_facts == 1
        assert len(result.facts) == 1
        assert result.facts[0].subject == "Test subject"
        assert result.confidence == ConfidenceLevel.HIGH
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        agent = FactExtractionAgent()
        
        # Test with empty string
        result = asyncio.run(agent.extract_facts(""))
        assert result.total_facts == 0
        assert len(result.facts) == 0
        assert "error" in result.metadata


class TestAgentFactory:
    """Test agent factory functionality."""
    
    def test_create_agent_with_name(self):
        """Test creating agent by name."""
        # This will fail without proper registration, but tests the interface
        with pytest.raises(Exception):  # ConfigurationError expected
            create_agent("nonexistent_agent")
    
    def test_create_agent_with_config(self):
        """Test creating agent with custom config."""
        config = AgentConfig(
            name="test_fact_extraction",
            agent_config={"max_facts": 50}
        )
        
        agent = FactExtractionAgent(config)
        assert agent.config.name == "test_fact_extraction"
        assert agent.config.get_agent_config("max_facts") == 50


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_valid_configuration(self):
        """Test valid configuration passes validation."""
        config = AgentConfig(
            name="test_agent",
            model=ModelConfig(api_key="test-key")
        )
        
        agent = FactExtractionAgent(config)
        # Should not raise exception
        assert agent.config.name == "test_agent"
    
    def test_invalid_configuration(self):
        """Test invalid configuration raises error."""
        config = AgentConfig(
            name="test_agent",
            timeout=-1  # Invalid timeout
        )
        
        with pytest.raises(Exception):  # ConfigurationError expected
            FactExtractionAgent(config)


class TestErrorHandling:
    """Test error handling."""
    
    @patch('agents.base.agent.BaseAgent._call_model')
    async def test_model_error_handling(self, mock_call_model):
        """Test handling of model errors."""
        mock_call_model.side_effect = Exception("Model error")
        
        agent = FactExtractionAgent()
        
        with pytest.raises(Exception):
            await agent.extract_facts("Test text")
    
    @patch('agents.base.agent.BaseAgent._call_model')
    async def test_invalid_json_handling(self, mock_call_model):
        """Test handling of invalid JSON responses."""
        mock_call_model.return_value = "Invalid JSON response"
        
        agent = FactExtractionAgent()
        
        with pytest.raises(Exception):  # ValidationError expected
            await agent.extract_facts("Test text")


@pytest.mark.asyncio
async def test_async_functionality():
    """Test that async functionality works correctly."""
    agent = FactExtractionAgent()
    
    # Test that the agent can be created and has async methods
    assert hasattr(agent, 'extract_facts')
    assert asyncio.iscoroutinefunction(agent.extract_facts)


def test_imports():
    """Test that all necessary imports work."""
    from agents.base import BaseAgent, AgentConfig, PromptTemplate
    from agents.extraction import FactExtractionAgent
    from agents.factory import create_agent, get_agent
    
    # Should not raise import errors
    assert BaseAgent is not None
    assert AgentConfig is not None
    assert FactExtractionAgent is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
