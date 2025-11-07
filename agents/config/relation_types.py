"""Relation types configuration manager for MoRAG agents."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import structlog
import yaml

logger = structlog.get_logger(__name__)


class RelationTypesManager:
    """Manages relation types configuration for MoRAG agents."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the relation types manager.

        Args:
            config_path: Path to relation types configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "relation_types.yaml"

        self.config_path = Path(config_path)
        self._config = None
        self._load_config()

    def _load_config(self) -> None:
        """Load relation types configuration from YAML file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            logger.info(
                "Relation types configuration loaded", path=str(self.config_path)
            )
        except FileNotFoundError:
            logger.error(
                "Relation types configuration file not found",
                path=str(self.config_path),
            )
            self._config = self._get_fallback_config()
        except yaml.YAMLError as e:
            logger.error("Failed to parse relation types configuration", error=str(e))
            self._config = self._get_fallback_config()

    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration if file loading fails."""
        return {
            "standard_relation_types": [
                "WORKS_FOR",
                "LOCATED_IN",
                "PART_OF",
                "CREATED_BY",
                "USES",
                "CAUSES",
                "SUPPORTS",
                "ELABORATES",
                "CONTRADICTS",
                "RELATED_TO",
            ],
            "agent_defaults": {
                "relation_extraction": [
                    "WORKS_FOR",
                    "LOCATED_IN",
                    "PART_OF",
                    "CREATED_BY",
                    "USES",
                    "CAUSES",
                    "MANAGES",
                    "COLLABORATES_WITH",
                ]
            },
            "dynamic_relation_types": {
                "enabled": True,
                "max_custom_types": 30,
                "confidence_threshold": 0.7,
            },
        }

    def get_standard_relation_types(self) -> List[str]:
        """Get the list of standard relation types.

        Returns:
            List of standard relation type names
        """
        return self._config.get("standard_relation_types", [])

    def get_extended_relation_types(self) -> List[str]:
        """Get the list of extended relation types.

        Returns:
            List of extended relation type names
        """
        return self._config.get("extended_relation_types", [])

    def get_all_relation_types(self) -> List[str]:
        """Get all relation types (standard + extended + domain-specific).

        Returns:
            List of all relation type names
        """
        all_types = set()
        all_types.update(self.get_standard_relation_types())
        all_types.update(self.get_extended_relation_types())

        # Add domain-specific types
        domain_specific = self._config.get("domain_specific", {})
        for domain_types in domain_specific.values():
            all_types.update(domain_types)

        return list(all_types)

    def get_domain_relation_types(self, domain: str) -> List[str]:
        """Get relation types for a specific domain.

        Args:
            domain: Domain name (e.g., 'medical', 'legal', 'technical')

        Returns:
            List of relation types for the domain
        """
        domain_specific = self._config.get("domain_specific", {})
        return domain_specific.get(domain, [])

    def get_agent_default_types(self, agent_name: str) -> List[str]:
        """Get default relation types for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of default relation types for the agent
        """
        agent_defaults = self._config.get("agent_defaults", {})
        return agent_defaults.get(agent_name, self.get_standard_relation_types())

    def get_category_types(self, category: str) -> List[str]:
        """Get relation types for a specific category.

        Args:
            category: Category name (e.g., 'organizational', 'spatial', 'causal')

        Returns:
            List of relation types in the category
        """
        categories = self._config.get("categories", {})
        return categories.get(category, [])

    def get_all_categories(self) -> List[str]:
        """Get all available categories.

        Returns:
            List of category names
        """
        categories = self._config.get("categories", {})
        return list(categories.keys())

    def get_dynamic_config(self) -> Dict[str, Any]:
        """Get dynamic relation types configuration.

        Returns:
            Dynamic configuration dictionary
        """
        return self._config.get("dynamic_relation_types", {})

    def is_dynamic_enabled(self) -> bool:
        """Check if dynamic relation types are enabled.

        Returns:
            True if dynamic types are enabled, False otherwise
        """
        dynamic_config = self.get_dynamic_config()
        return dynamic_config.get("enabled", True)

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for relation types.

        Returns:
            Validation rules dictionary
        """
        return self._config.get("validation_rules", {})

    def is_valid_relation_type(self, relation_type: str) -> bool:
        """Check if a relation type is valid.

        Args:
            relation_type: Relation type to validate

        Returns:
            True if valid, False otherwise
        """
        all_types = set(self.get_all_relation_types())

        # Check category types
        categories = self._config.get("categories", {})
        for category_types in categories.values():
            all_types.update(category_types)

        return relation_type.upper() in {t.upper() for t in all_types}

    def get_relation_description(self, relation_type: str) -> Optional[str]:
        """Get description for a relation type.

        Args:
            relation_type: Relation type name

        Returns:
            Description string or None if not found
        """
        descriptions = self._config.get("descriptions", {})
        return descriptions.get(relation_type.upper())

    def get_relation_strength(self, relation_type: str) -> Optional[str]:
        """Get strength indicator for a relation type.

        Args:
            relation_type: Relation type name

        Returns:
            Strength indicator ('strong', 'medium', 'weak') or None
        """
        strength_indicators = self._config.get("strength_indicators", {})
        relation_upper = relation_type.upper()

        for strength, types in strength_indicators.items():
            if relation_upper in [t.upper() for t in types]:
                return strength

        return None

    def validate_relation_type_name(self, relation_type: str) -> bool:
        """Validate relation type name against rules.

        Args:
            relation_type: Relation type name to validate

        Returns:
            True if valid, False otherwise
        """
        rules = self.get_validation_rules()

        # Check length
        min_length = rules.get("min_relation_length", 3)
        max_length = rules.get("max_relation_length", 50)

        if not (min_length <= len(relation_type) <= max_length):
            return False

        # Check characters (simplified validation)
        allowed = rules.get(
            "allowed_characters", "alphanumeric_spaces_hyphens_underscores"
        )
        if allowed == "alphanumeric_spaces_hyphens_underscores":
            import re

            if not re.match(r"^[A-Za-z0-9\s\-_]+$", relation_type):
                return False

        # Check against generic types to avoid
        avoid_generic = rules.get("avoid_generic_types", [])
        if relation_type.upper() in [t.upper() for t in avoid_generic]:
            return False

        return True

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()


# Global instance
_relation_types_manager = None


def get_relation_types_manager() -> RelationTypesManager:
    """Get the global relation types manager instance.

    Returns:
        RelationTypesManager instance
    """
    global _relation_types_manager
    if _relation_types_manager is None:
        _relation_types_manager = RelationTypesManager()
    return _relation_types_manager


def get_standard_relation_types() -> List[str]:
    """Get standard relation types.

    Returns:
        List of standard relation types
    """
    return get_relation_types_manager().get_standard_relation_types()


def get_agent_relation_types(agent_name: str) -> List[str]:
    """Get relation types for a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        List of relation types for the agent
    """
    return get_relation_types_manager().get_agent_default_types(agent_name)


def get_domain_relation_types(domain: str) -> List[str]:
    """Get relation types for a specific domain.

    Args:
        domain: Domain name

    Returns:
        List of relation types for the domain
    """
    return get_relation_types_manager().get_domain_relation_types(domain)
