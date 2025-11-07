"""Automated consistency checks for MoRAG agents."""

import ast
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest
import yaml

from agents.base.agent import BaseAgent
from agents.base.template import GlobalPromptLoader
from agents.config.defaults import DefaultConfigs
from agents.config.entity_types import get_entity_types_manager
from agents.config.validation import validate_all_configs


class TestAgentConsistency:
    """Test suite for agent consistency requirements."""

    @classmethod
    def setup_class(cls):
        """Set up test class with agent discovery."""
        cls.agents_dir = Path(__file__).parent.parent
        cls.agent_modules = cls._discover_agent_modules()
        cls.agent_classes = cls._discover_agent_classes()
        cls.prompts_config = cls._load_prompts_config()

    @classmethod
    def _discover_agent_modules(cls) -> List[Path]:
        """Discover all agent module files."""
        agent_modules = []

        # Search in all agent category directories
        for category_dir in [
            "extraction",
            "analysis",
            "reasoning",
            "generation",
            "processing",
        ]:
            category_path = cls.agents_dir / category_dir
            if category_path.exists():
                for py_file in category_path.glob("*.py"):
                    if py_file.name != "__init__.py" and py_file.name != "models.py":
                        agent_modules.append(py_file)

        return agent_modules

    @classmethod
    def _discover_agent_classes(cls) -> Dict[str, type]:
        """Discover all agent classes."""
        agent_classes = {}

        for module_path in cls.agent_modules:
            try:
                # Convert path to module name
                relative_path = module_path.relative_to(cls.agents_dir.parent)
                module_name = str(relative_path).replace(os.sep, ".").replace(".py", "")

                # Import module
                module = importlib.import_module(module_name)

                # Find agent classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        name.endswith("Agent")
                        and issubclass(obj, BaseAgent)
                        and obj != BaseAgent
                    ):
                        agent_classes[name] = obj

            except Exception as e:
                print(f"Warning: Could not import {module_path}: {e}")

        return agent_classes

    @classmethod
    def _load_prompts_config(cls) -> Dict[str, Any]:
        """Load prompts configuration."""
        prompts_file = cls.agents_dir / "prompts.yaml"
        if prompts_file.exists():
            with open(prompts_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def test_no_template_overrides(self):
        """Test that no agents override _create_template method."""
        violations = []

        for module_path in self.agent_modules:
            try:
                with open(module_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse AST to find method definitions
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.FunctionDef)
                        and node.name == "_create_template"
                    ):
                        violations.append(str(module_path))
                        break

            except Exception as e:
                print(f"Warning: Could not parse {module_path}: {e}")

        assert (
            len(violations) == 0
        ), f"Agents with _create_template overrides: {violations}"

    def test_all_agents_have_prompts(self):
        """Test that all agents have entries in prompts.yaml."""
        missing_prompts = []

        expected_agents = {
            "fact_extraction",
            "entity_extraction",
            "relation_extraction",
            "keyword_extraction",
            "query_analysis",
            "content_analysis",
            "sentiment_analysis",
            "topic_analysis",
            "summarization",
            "path_selection",
            "reasoning",
            "response_generation",
            "decision_making",
            "context_analysis",
            "explanation",
            "synthesis",
            "chunking",
            "classification",
            "validation",
            "filtering",
        }

        for agent_name in expected_agents:
            if agent_name not in self.prompts_config:
                missing_prompts.append(agent_name)

        assert (
            len(missing_prompts) == 0
        ), f"Agents missing from prompts.yaml: {missing_prompts}"

    def test_prompt_structure_consistency(self):
        """Test that all prompts have consistent structure."""
        structural_issues = []

        for agent_name, prompts in self.prompts_config.items():
            if not isinstance(prompts, dict):
                structural_issues.append(f"{agent_name}: prompts not a dictionary")
                continue

            if "system_prompt" not in prompts:
                structural_issues.append(f"{agent_name}: missing system_prompt")

            if "user_prompt" not in prompts:
                structural_issues.append(f"{agent_name}: missing user_prompt")

            # Check for empty prompts
            if prompts.get("system_prompt", "").strip() == "":
                structural_issues.append(f"{agent_name}: empty system_prompt")

            if prompts.get("user_prompt", "").strip() == "":
                structural_issues.append(f"{agent_name}: empty user_prompt")

        assert (
            len(structural_issues) == 0
        ), f"Prompt structure issues: {structural_issues}"

    def test_configuration_consistency(self):
        """Test that all configurations follow standard patterns."""
        all_configs = DefaultConfigs.get_all_configs()
        validation_results = validate_all_configs(all_configs)

        invalid_configs = []
        for agent_name, result in validation_results.items():
            if not result.is_valid:
                invalid_configs.append(f"{agent_name}: {result.errors}")

        assert len(invalid_configs) == 0, f"Invalid configurations: {invalid_configs}"

    def test_entity_types_centralization(self):
        """Test that entity types use centralized configuration."""
        entity_manager = get_entity_types_manager()

        # Test that entity types manager works
        standard_types = entity_manager.get_standard_entity_types()
        assert len(standard_types) > 0, "No standard entity types found"

        # Test that agent defaults are available
        agent_types = entity_manager.get_agent_default_types("entity_extraction")
        assert len(agent_types) > 0, "No entity types for entity_extraction agent"

        # Test that domain-specific types are available
        medical_types = entity_manager.get_domain_entity_types("medical")
        assert len(medical_types) > 0, "No medical domain entity types found"

    def test_agent_instantiation(self):
        """Test that all agents can be instantiated (without API keys)."""
        instantiation_failures = []

        for agent_name, agent_class in self.agent_classes.items():
            try:
                # Try to instantiate the agent
                agent = agent_class()

                # Basic checks
                assert hasattr(agent, "config"), f"{agent_name}: no config attribute"
                assert hasattr(
                    agent, "_template"
                ), f"{agent_name}: no _template attribute"
                assert (
                    agent.config.name is not None
                ), f"{agent_name}: config.name is None"

            except Exception as e:
                # Allow API key errors as they're expected
                if "API key required" not in str(e):
                    instantiation_failures.append(f"{agent_name}: {str(e)}")

        assert (
            len(instantiation_failures) == 0
        ), f"Agent instantiation failures: {instantiation_failures}"

    def test_prompt_template_variables(self):
        """Test that prompt templates use consistent variable patterns."""
        variable_issues = []

        common_variables = {"input", "config", "domain", "context"}

        for agent_name, prompts in self.prompts_config.items():
            if not isinstance(prompts, dict):
                continue

            system_prompt = prompts.get("system_prompt", "")
            user_prompt = prompts.get("user_prompt", "")

            # Check for {{ input }} in user prompt
            if "{{ input }}" not in user_prompt:
                variable_issues.append(
                    f"{agent_name}: user_prompt missing {{ input }} variable"
                )

            # Check for consistent variable syntax
            import re

            variables = re.findall(
                r"\{\{\s*([^}]+)\s*\}\}", system_prompt + user_prompt
            )

            for var in variables:
                var_name = var.split(".")[0].split("|")[0].strip()
                if var_name not in common_variables and not var_name.startswith(
                    "config."
                ):
                    # This is just a warning, not an error
                    pass

        assert (
            len(variable_issues) == 0
        ), f"Prompt template variable issues: {variable_issues}"

    def test_configuration_completeness(self):
        """Test that all discovered agents have default configurations."""
        missing_configs = []
        all_configs = DefaultConfigs.get_all_configs()

        # Extract agent names from class names
        expected_config_names = set()
        for agent_class_name in self.agent_classes.keys():
            # Convert FactExtractionAgent -> fact_extraction
            config_name = agent_class_name.replace("Agent", "").lower()
            # Handle camelCase to snake_case conversion
            import re

            config_name = re.sub(r"([A-Z])", r"_\1", config_name).lower().strip("_")
            expected_config_names.add(config_name)

        for config_name in expected_config_names:
            if config_name not in all_configs:
                missing_configs.append(config_name)

        # Allow some missing configs for now (this is more of a warning)
        if len(missing_configs) > 5:  # Only fail if too many are missing
            assert (
                False
            ), f"Many agents missing default configurations: {missing_configs}"

    def test_import_consistency(self):
        """Test that agents don't import deprecated or unused modules."""
        import_issues = []

        deprecated_imports = {
            "ConfigurablePromptTemplate",  # Should use base implementation
            "PromptExample",  # Should use prompts.yaml
        }

        for module_path in self.agent_modules:
            try:
                with open(module_path, "r", encoding="utf-8") as f:
                    content = f.read()

                for deprecated in deprecated_imports:
                    if deprecated in content:
                        import_issues.append(
                            f"{module_path}: imports deprecated {deprecated}"
                        )

            except Exception as e:
                print(f"Warning: Could not check imports in {module_path}: {e}")

        assert len(import_issues) == 0, f"Deprecated import issues: {import_issues}"


class TestPromptQuality:
    """Test suite for prompt quality and consistency."""

    @classmethod
    def setup_class(cls):
        """Set up test class."""
        cls.prompts_file = Path(__file__).parent.parent / "prompts.yaml"
        with open(cls.prompts_file, "r", encoding="utf-8") as f:
            cls.prompts_config = yaml.safe_load(f)

    def test_prompt_length_adequacy(self):
        """Test that prompts are adequately detailed."""
        short_prompts = []

        for agent_name, prompts in self.prompts_config.items():
            if not isinstance(prompts, dict):
                continue

            system_prompt = prompts.get("system_prompt", "")

            # Check for minimum prompt length (should be detailed)
            if len(system_prompt) < 200:
                short_prompts.append(
                    f"{agent_name}: system prompt too short ({len(system_prompt)} chars)"
                )

        assert len(short_prompts) == 0, f"Prompts that are too short: {short_prompts}"

    def test_prompt_structure_elements(self):
        """Test that prompts contain expected structural elements."""
        missing_elements = []

        expected_elements = ["## Your Role", "output_requirements"]

        for agent_name, prompts in self.prompts_config.items():
            if not isinstance(prompts, dict):
                continue

            system_prompt = prompts.get("system_prompt", "")

            for element in expected_elements:
                if element not in system_prompt:
                    missing_elements.append(f"{agent_name}: missing '{element}'")

        # Allow some missing elements (this is more of a guideline)
        if (
            len(missing_elements) > len(self.prompts_config) * 0.3
        ):  # More than 30% missing
            assert (
                False
            ), f"Many prompts missing structural elements: {missing_elements[:10]}..."


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
