"""Prompt template system for MoRAG agents."""

import re
import os
import yaml
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from jinja2 import Template, Environment, BaseLoader
import structlog

from .config import AgentConfig, PromptConfig
from .exceptions import PromptGenerationError

logger = structlog.get_logger(__name__)


class GlobalPromptLoader:
    """Loads prompts from the global prompts.yaml file."""

    _instance = None
    _prompts = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._prompts is None:
            self._load_prompts()

    def _load_prompts(self):
        """Load prompts from the global YAML file."""
        try:
            # Find the prompts.yaml file
            current_dir = Path(__file__).parent.parent
            prompts_file = current_dir / "prompts.yaml"

            if not prompts_file.exists():
                logger.warning(f"Global prompts file not found: {prompts_file}")
                self._prompts = {}
                return

            with open(prompts_file, 'r', encoding='utf-8') as f:
                self._prompts = yaml.safe_load(f) or {}

            logger.info(f"Loaded {len(self._prompts)} agent prompts from {prompts_file}")

        except Exception as e:
            logger.error(f"Failed to load global prompts: {e}")
            self._prompts = {}

    def get_prompts(self, agent_name: str) -> Dict[str, str]:
        """Get prompts for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with system_prompt and user_prompt
        """
        if self._prompts is None:
            self._load_prompts()

        agent_prompts = self._prompts.get(agent_name, {})

        if not agent_prompts:
            logger.warning(f"No prompts found for agent: {agent_name}")
            return {
                "system_prompt": f"You are a {agent_name} agent.",
                "user_prompt": "Process: {{ input }}"
            }

        return {
            "system_prompt": agent_prompts.get("system_prompt", f"You are a {agent_name} agent."),
            "user_prompt": agent_prompts.get("user_prompt", "Process: {{ input }}")
        }

    def reload_prompts(self):
        """Reload prompts from file."""
        self._prompts = None
        self._load_prompts()


@dataclass
class PromptExample:
    """Example for few-shot prompting."""
    input: str
    output: str
    explanation: Optional[str] = None


class PromptTemplate(ABC):
    """Base class for prompt templates."""

    def __init__(self, config: PromptConfig):
        """Initialize the prompt template.

        Args:
            config: Prompt configuration
        """
        self.config = config
        self.env = Environment(loader=BaseLoader())
        self.logger = logger.bind(template=self.__class__.__name__)

    @abstractmethod
    def get_system_prompt(self, **kwargs) -> str:
        """Generate the system prompt.

        Returns:
            The system prompt string
        """
        pass

    @abstractmethod
    def get_user_prompt(self, **kwargs) -> str:
        """Generate the user prompt.

        Returns:
            The user prompt string
        """
        pass

    def get_examples(self) -> List[PromptExample]:
        """Get few-shot examples for this template.

        Returns:
            List of examples
        """
        return []

    def format_examples(self, examples: Optional[List[PromptExample]] = None) -> str:
        """Format examples for inclusion in prompts.

        Args:
            examples: Optional custom examples, uses default if None

        Returns:
            Formatted examples string
        """
        if not self.config.include_examples:
            return ""

        examples = examples or self.get_examples()
        if not examples:
            return ""

        formatted = ["## Examples\n"]
        for i, example in enumerate(examples, 1):
            formatted.append(f"### Example {i}")
            formatted.append(f"**Input:** {example.input}")
            formatted.append(f"**Output:** {example.output}")
            if example.explanation:
                formatted.append(f"**Explanation:** {example.explanation}")
            formatted.append("")

        return "\n".join(formatted)

    def format_instructions(self, instructions: str) -> str:
        """Format instructions section.

        Args:
            instructions: Raw instructions

        Returns:
            Formatted instructions
        """
        if not self.config.include_instructions:
            return ""

        formatted = ["## Instructions\n", instructions]

        if self.config.custom_instructions:
            formatted.extend(["\n### Additional Instructions", self.config.custom_instructions])

        return "\n".join(formatted)

    def format_output_requirements(self) -> str:
        """Format output requirements section.

        Returns:
            Formatted output requirements
        """
        requirements = ["## Output Requirements\n"]

        if self.config.output_format == "json":
            requirements.append("- Respond with valid JSON only")
            if self.config.strict_json:
                requirements.append("- Do not include any text outside the JSON structure")
                requirements.append("- Ensure all JSON keys are properly quoted")

        if self.config.include_confidence:
            requirements.append("- Include confidence scores (0.0-1.0) for all outputs")
            requirements.append(f"- Minimum confidence threshold: {self.config.min_confidence}")

        if self.config.max_output_length:
            requirements.append(f"- Maximum output length: {self.config.max_output_length} characters")

        requirements.append(f"- Target language: {self.config.language}")
        requirements.append(f"- Domain context: {self.config.domain}")

        return "\n".join(requirements)

    def render_template(self, template_str: str, **kwargs) -> str:
        """Render a Jinja2 template with the given variables.

        Args:
            template_str: Template string
            **kwargs: Template variables

        Returns:
            Rendered template

        Raises:
            PromptGenerationError: If template rendering fails
        """
        try:
            template = self.env.from_string(template_str)
            return template.render(**kwargs)
        except Exception as e:
            raise PromptGenerationError(f"Template rendering failed: {e}") from e

    def validate_prompt(self, prompt: str) -> bool:
        """Validate a generated prompt.

        Args:
            prompt: The prompt to validate

        Returns:
            True if valid, False otherwise
        """
        if not prompt or not prompt.strip():
            return False

        # Check for template variables that weren't replaced
        if re.search(r'\{\{.*?\}\}', prompt):
            self.logger.warning("Prompt contains unreplaced template variables")
            return False

        return True

    def generate_full_prompt(self, user_input: str, **kwargs) -> Dict[str, str]:
        """Generate the complete prompt with system and user parts.

        Args:
            user_input: The user input/query
            **kwargs: Additional template variables

        Returns:
            Dictionary with 'system' and 'user' prompt parts

        Raises:
            PromptGenerationError: If prompt generation fails
        """
        try:
            # Generate system prompt
            system_prompt = self.get_system_prompt(**kwargs)

            # Generate user prompt with input
            user_prompt = self.get_user_prompt(input=user_input, **kwargs)

            # Validate prompts
            if not self.validate_prompt(system_prompt):
                raise PromptGenerationError("Invalid system prompt generated")

            if not self.validate_prompt(user_prompt):
                raise PromptGenerationError("Invalid user prompt generated")

            return {
                "system": system_prompt,
                "user": user_prompt
            }

        except Exception as e:
            if isinstance(e, PromptGenerationError):
                raise
            raise PromptGenerationError(f"Prompt generation failed: {e}") from e


class ConfigurablePromptTemplate(PromptTemplate):
    """A prompt template that can be configured via the config object."""

    def __init__(self, config: PromptConfig, system_template: str, user_template: str, agent_config: Optional[Dict[str, Any]] = None):
        """Initialize with template strings.

        Args:
            config: Prompt configuration
            system_template: System prompt template
            user_template: User prompt template
            agent_config: Agent-specific configuration
        """
        super().__init__(config)
        self.system_template = system_template
        self.user_template = user_template
        self.agent_config = agent_config or {}

    def get_system_prompt(self, **kwargs) -> str:
        """Generate system prompt from template."""
        # Create a config object that includes agent_config for template access
        config_with_agent = type('ConfigWithAgent', (), {
            **self.config.model_dump(),
            'agent_config': self.agent_config
        })()

        context = {
            'config': config_with_agent,
            'examples': self.format_examples(),
            'output_requirements': self.format_output_requirements(),
            **kwargs
        }
        return self.render_template(self.system_template, **context)

    def get_user_prompt(self, **kwargs) -> str:
        """Generate user prompt from template."""
        # Create a config object that includes agent_config for template access
        config_with_agent = type('ConfigWithAgent', (), {
            **self.config.model_dump(),
            'agent_config': self.agent_config
        })()

        context = {
            'config': config_with_agent,
            **kwargs
        }
        return self.render_template(self.user_template, **context)
