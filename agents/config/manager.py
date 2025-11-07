"""Configuration manager for agents."""

import os
import json
import yaml
from typing import Dict, Any, Optional, Type
from pathlib import Path
import structlog

from ..base.config import AgentConfig, PromptConfig, ModelConfig, RetryConfig
from ..base.exceptions import ConfigurationError

logger = structlog.get_logger(__name__)


class AgentConfigManager:
    """Manages agent configurations from various sources."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the config manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("agents/config/defaults")
        self.configs: Dict[str, AgentConfig] = {}
        self.logger = logger.bind(component="config_manager")

        # Load default configurations
        self._load_default_configs()

    def _load_default_configs(self) -> None:
        """Load default configurations from files."""
        if not self.config_dir.exists():
            self.logger.warning(f"Config directory not found: {self.config_dir}")
            return

        for config_file in self.config_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)

                agent_name = config_file.stem
                self.configs[agent_name] = AgentConfig(**config_data)
                self.logger.info(f"Loaded config for {agent_name}")

            except Exception as e:
                self.logger.error(f"Failed to load config {config_file}: {e}")

    def get_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent configuration

        Raises:
            ConfigurationError: If config not found
        """
        if agent_name not in self.configs:
            # Try to create a default config
            self.configs[agent_name] = self._create_default_config(agent_name)

        # Apply environment variable overrides
        config = self.configs[agent_name].copy(deep=True)
        config.update_from_env(f"MORAG_{agent_name.upper()}_")

        return config

    def _create_default_config(self, agent_name: str) -> AgentConfig:
        """Create a default configuration for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Default configuration
        """
        return AgentConfig(
            name=agent_name,
            description=f"Default configuration for {agent_name}",
            model=ModelConfig(),
            retry=RetryConfig(),
            prompt=PromptConfig()
        )

    def register_config(self, agent_name: str, config: AgentConfig) -> None:
        """Register a configuration for an agent.

        Args:
            agent_name: Name of the agent
            config: Agent configuration
        """
        self.configs[agent_name] = config
        self.logger.info(f"Registered config for {agent_name}")

    def save_config(self, agent_name: str, config_path: Optional[str] = None) -> None:
        """Save agent configuration to file.

        Args:
            agent_name: Name of the agent
            config_path: Optional path to save to
        """
        if agent_name not in self.configs:
            raise ConfigurationError(f"No config found for {agent_name}")

        config = self.configs[agent_name]
        save_path = Path(config_path) if config_path else self.config_dir / f"{agent_name}.yaml"

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save
        config_dict = config.dict(exclude={'model': {'api_key'}})  # Don't save API keys

        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        self.logger.info(f"Saved config for {agent_name} to {save_path}")

    def list_configs(self) -> Dict[str, str]:
        """List all available configurations.

        Returns:
            Dictionary mapping agent names to descriptions
        """
        return {
            name: config.description or "No description"
            for name, config in self.configs.items()
        }

    def validate_config(self, config: AgentConfig) -> bool:
        """Validate an agent configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid

        Raises:
            ConfigurationError: If validation fails
        """
        try:
            # Basic validation
            if not config.name:
                raise ConfigurationError("Agent name is required")

            if not config.model.api_key:
                raise ConfigurationError("API key is required")

            if config.timeout <= 0:
                raise ConfigurationError("Timeout must be positive")

            if config.retry.max_retries < 0:
                raise ConfigurationError("Max retries cannot be negative")

            if not (0.0 <= config.model.temperature <= 2.0):
                raise ConfigurationError("Temperature must be between 0.0 and 2.0")

            if config.model.max_tokens <= 0:
                raise ConfigurationError("Max tokens must be positive")

            return True

        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e

    def create_config_from_env(self, agent_name: str, env_prefix: str = None) -> AgentConfig:
        """Create configuration from environment variables.

        Args:
            agent_name: Name of the agent
            env_prefix: Environment variable prefix

        Returns:
            Configuration created from environment
        """
        prefix = env_prefix or f"MORAG_{agent_name.upper()}_"

        config = self._create_default_config(agent_name)
        config.update_from_env(prefix)

        return config
