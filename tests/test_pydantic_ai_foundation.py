"""Tests for PydanticAI foundation components."""

import asyncio
import os
import sys
from typing import Type

import pytest
from pydantic import BaseModel, Field

# Add the packages directory to the Python path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "packages", "morag-core", "src")
)

# Import the AI components we just created
from morag_core.ai import (
    AgentConfig,
    AgentError,
    AgentFactory,
    ConfidenceLevel,
    Entity,
    EntityExtractionResult,
    GeminiProvider,
    MoRAGBaseAgent,
    ProviderConfig,
    ValidationError,
    create_agent,
    create_agent_with_config,
)


class SimpleTestResult(BaseModel):
    """Simple test result for testing."""

    message: str = Field(description="A simple message")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class SimpleTestAgent(MoRAGBaseAgent[SimpleTestResult]):
    """Simple test agent for testing the foundation."""

    def get_result_type(self) -> Type[SimpleTestResult]:
        return SimpleTestResult

    def get_system_prompt(self) -> str:
        return "You are a test agent. Always respond with a simple message and confidence score."


class TestProviderConfig:
    """Test provider configuration."""

    def test_provider_config_creation(self):
        """Test creating provider configuration."""
        config = ProviderConfig(api_key="test-key", timeout=60, max_retries=5)

        assert config.api_key == "test-key"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.extra_headers == {}
        assert config.extra_params == {}

    def test_provider_config_defaults(self):
        """Test provider configuration defaults."""
        config = ProviderConfig()

        assert config.api_key is None
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.extra_headers == {}
        assert config.extra_params == {}


class TestAgentConfig:
    """Test agent configuration."""

    def test_agent_config_creation(self):
        """Test creating agent configuration."""
        config = AgentConfig(
            model="google-gla:gemini-1.5-pro",
            timeout=60,
            max_retries=5,
            temperature=0.5,
            max_tokens=1000,
        )

        assert config.model == "google-gla:gemini-1.5-pro"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.temperature == 0.5
        assert config.max_tokens == 1000

    def test_agent_config_defaults(self):
        """Test agent configuration defaults."""
        config = AgentConfig()

        assert config.model == "google-gla:gemini-1.5-flash"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.temperature == 0.1
        assert config.max_tokens is None


class TestGeminiProvider:
    """Test Gemini provider."""

    def test_provider_creation(self):
        """Test creating Gemini provider."""
        config = ProviderConfig(api_key="test-key")
        provider = GeminiProvider(config)

        assert provider.config.api_key == "test-key"
        assert provider.api_key == "test-key"

    def test_provider_without_config(self):
        """Test creating provider without config."""
        provider = GeminiProvider()

        assert provider.config is not None
        assert isinstance(provider.config, ProviderConfig)

    def test_model_name_handling(self):
        """Test model name handling."""
        provider = GeminiProvider()

        # Test different model name formats
        assert (
            provider.get_model_name("google-gla:gemini-1.5-flash")
            == "google-gla:gemini-1.5-flash"
        )
        assert provider.get_model_name("gemini-1.5-pro") == "google-gla:gemini-1.5-pro"
        assert provider.get_model_name("unknown-model") == "google-gla:gemini-1.5-flash"

    def test_provider_info(self):
        """Test getting provider information."""
        provider = GeminiProvider()
        info = provider.get_provider_info()

        assert info["name"] == "gemini"
        assert "available" in info
        assert "api_key_configured" in info
        assert "config" in info


class TestMoRAGBaseAgent:
    """Test base agent functionality."""

    def test_agent_creation(self):
        """Test creating a base agent."""
        config = AgentConfig(model="google-gla:gemini-1.5-flash")
        provider = GeminiProvider()
        agent = SimpleTestAgent(config=config, provider=provider)

        assert agent.config.model == "google-gla:gemini-1.5-flash"
        assert agent.provider is not None
        assert agent.get_result_type() == SimpleTestResult
        assert "test agent" in agent.get_system_prompt().lower()

    def test_agent_with_defaults(self):
        """Test creating agent with default configuration."""
        agent = SimpleTestAgent()

        assert agent.config is not None
        assert agent.provider is not None
        assert isinstance(agent.config, AgentConfig)
        assert isinstance(agent.provider, GeminiProvider)

    def test_result_validation(self):
        """Test result validation."""
        agent = SimpleTestAgent()

        # Test valid result
        valid_result = {"message": "test", "confidence": 0.8}
        validated = agent._validate_result(valid_result)
        assert isinstance(validated, SimpleTestResult)
        assert validated.message == "test"
        assert validated.confidence == 0.8

        # Test invalid result
        invalid_result = {"message": "test", "confidence": 1.5}  # confidence > 1.0
        with pytest.raises(ValidationError):
            agent._validate_result(invalid_result)


class TestAgentFactory:
    """Test agent factory."""

    def test_factory_creation(self):
        """Test creating agent factory."""
        factory = AgentFactory()
        assert factory.default_config is not None
        assert isinstance(factory.default_config, AgentConfig)

    def test_factory_with_config(self):
        """Test factory with custom default config."""
        config = AgentConfig(model="google-gla:gemini-1.5-pro")
        factory = AgentFactory(default_config=config)

        assert factory.default_config.model == "google-gla:gemini-1.5-pro"

    def test_create_agent(self):
        """Test creating agent with factory."""
        factory = AgentFactory()
        agent = factory.create_agent(SimpleTestAgent)

        assert isinstance(agent, SimpleTestAgent)
        assert agent.config is not None
        assert agent.provider is not None

    def test_create_agent_with_config(self):
        """Test creating agent with inline configuration."""
        factory = AgentFactory()
        agent = factory.create_agent_with_config(
            SimpleTestAgent,
            model="google-gla:gemini-1.5-pro",
            timeout=60,
            temperature=0.5,
        )

        assert isinstance(agent, SimpleTestAgent)
        assert agent.config.model == "google-gla:gemini-1.5-pro"
        assert agent.config.timeout == 60
        assert agent.config.temperature == 0.5

    def test_get_agent_info(self):
        """Test getting agent information."""
        factory = AgentFactory()
        agent = factory.create_agent(SimpleTestAgent)
        info = factory.get_agent_info(agent)

        assert info["class"] == "SimpleTestAgent"
        assert "config" in info
        assert "provider" in info
        assert "result_type" in info
        assert "system_prompt_preview" in info


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_agent_function(self):
        """Test create_agent convenience function."""
        agent = create_agent(SimpleTestAgent)

        assert isinstance(agent, SimpleTestAgent)
        assert agent.config is not None
        assert agent.provider is not None

    def test_create_agent_with_config_function(self):
        """Test create_agent_with_config convenience function."""
        agent = create_agent_with_config(
            SimpleTestAgent, model="google-gla:gemini-1.5-pro", timeout=45
        )

        assert isinstance(agent, SimpleTestAgent)
        assert agent.config.model == "google-gla:gemini-1.5-pro"
        assert agent.config.timeout == 45


class TestStructuredModels:
    """Test structured response models."""

    def test_entity_model(self):
        """Test Entity model."""
        entity = Entity(
            name="Apple Inc.",
            type="ORGANIZATION",
            confidence=0.95,
            start_pos=0,
            end_pos=10,
            context="Apple Inc. is a technology company",
        )

        assert entity.name == "Apple Inc."
        assert entity.type == "ORGANIZATION"
        assert entity.confidence == 0.95
        assert entity.start_pos == 0
        assert entity.end_pos == 10
        assert "technology company" in entity.context

    def test_entity_extraction_result(self):
        """Test EntityExtractionResult model."""
        entities = [
            Entity(name="Apple", type="ORGANIZATION", confidence=0.9),
            Entity(name="iPhone", type="PRODUCT", confidence=0.8),
        ]

        result = EntityExtractionResult(
            entities=entities, confidence=ConfidenceLevel.HIGH, processing_time=1.5
        )

        assert len(result.entities) == 2
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.processing_time == 1.5
        assert result.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__])
