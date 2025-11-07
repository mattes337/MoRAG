"""Entity types configuration manager for MoRAG agents."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import structlog
import yaml

logger = structlog.get_logger(__name__)


class EntityTypesManager:
    """Manages entity types configuration for MoRAG agents."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the entity types manager.

        Args:
            config_path: Path to entity types configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "entity_types.yaml"

        self.config_path = Path(config_path)
        self._config = None
        self._load_config()

    def _load_config(self) -> None:
        """Load entity types configuration from YAML file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            logger.info("Entity types configuration loaded", path=str(self.config_path))
        except FileNotFoundError:
            logger.error(
                "Entity types configuration file not found", path=str(self.config_path)
            )
            self._config = self._get_fallback_config()
        except yaml.YAMLError as e:
            logger.error("Failed to parse entity types configuration", error=str(e))
            self._config = self._get_fallback_config()

    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration if file loading fails."""
        return {
            "standard_entity_types": [
                "PERSON",
                "ORGANIZATION",
                "LOCATION",
                "CONCEPT",
                "PRODUCT",
                "EVENT",
                "DATE",
                "QUANTITY",
                "TECHNOLOGY",
                "PROCESS",
            ],
            "agent_defaults": {
                "entity_extraction": [
                    "PERSON",
                    "ORGANIZATION",
                    "LOCATION",
                    "CONCEPT",
                    "PRODUCT",
                    "EVENT",
                    "DATE",
                    "QUANTITY",
                    "TECHNOLOGY",
                    "PROCESS",
                ]
            },
            "dynamic_entity_types": {
                "enabled": True,
                "max_custom_types": 20,
                "confidence_threshold": 0.8,
            },
        }

    def get_standard_entity_types(self) -> List[str]:
        """Get the list of standard entity types.

        Returns:
            List of standard entity type names
        """
        return self._config.get("standard_entity_types", [])

    def get_extended_entity_types(self) -> List[str]:
        """Get the list of extended entity types.

        Returns:
            List of extended entity type names
        """
        return self._config.get("extended_entity_types", [])

    def get_all_entity_types(self) -> List[str]:
        """Get all available entity types (standard + extended + domain-specific).

        Returns:
            List of all entity type names
        """
        all_types = set()

        # Add standard types
        all_types.update(self.get_standard_entity_types())

        # Add extended types
        all_types.update(self.get_extended_entity_types())

        # Add domain-specific types
        domain_specific = self._config.get("domain_specific", {})
        for domain_types in domain_specific.values():
            all_types.update(domain_types)

        return sorted(list(all_types))

    def get_domain_entity_types(self, domain: str) -> List[str]:
        """Get entity types for a specific domain.

        Args:
            domain: Domain name (e.g., 'medical', 'legal', 'technical')

        Returns:
            List of entity types for the domain
        """
        domain_specific = self._config.get("domain_specific", {})
        return domain_specific.get(domain, [])

    def get_agent_default_types(self, agent_name: str) -> List[str]:
        """Get default entity types for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of default entity types for the agent
        """
        agent_defaults = self._config.get("agent_defaults", {})
        return agent_defaults.get(agent_name, self.get_standard_entity_types())

    def get_entity_type_description(self, entity_type: str) -> str:
        """Get description for an entity type.

        Args:
            entity_type: Entity type name

        Returns:
            Description of the entity type
        """
        descriptions = self._config.get("descriptions", {})
        return descriptions.get(entity_type, f"Entity type: {entity_type}")

    def get_entity_type_hierarchy(self, parent_type: str) -> List[str]:
        """Get child entity types for a parent type.

        Args:
            parent_type: Parent entity type name

        Returns:
            List of child entity types
        """
        hierarchies = self._config.get("hierarchies", {})
        return hierarchies.get(parent_type, [])

    def is_valid_entity_type(self, entity_type: str) -> bool:
        """Check if an entity type is valid.

        Args:
            entity_type: Entity type to validate

        Returns:
            True if valid, False otherwise
        """
        all_types = set(self.get_all_entity_types())

        # Check hierarchical types
        hierarchies = self._config.get("hierarchies", {})
        for child_types in hierarchies.values():
            all_types.update(child_types)

        return entity_type.upper() in {t.upper() for t in all_types}

    def normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type to standard format.

        Args:
            entity_type: Entity type to normalize

        Returns:
            Normalized entity type
        """
        validation_rules = self._config.get("validation_rules", {})

        if validation_rules.get("normalize_to_uppercase", True):
            return entity_type.upper()

        return entity_type

    def get_dynamic_config(self) -> Dict[str, Any]:
        """Get dynamic entity types configuration.

        Returns:
            Dynamic entity types configuration
        """
        return self._config.get("dynamic_entity_types", {})

    def is_dynamic_types_enabled(self) -> bool:
        """Check if dynamic entity types are enabled.

        Returns:
            True if dynamic types are enabled
        """
        dynamic_config = self.get_dynamic_config()
        return dynamic_config.get("enabled", False)

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get entity type validation rules.

        Returns:
            Validation rules configuration
        """
        return self._config.get("validation_rules", {})

    def validate_entity_type_name(self, entity_type: str) -> bool:
        """Validate entity type name against rules.

        Args:
            entity_type: Entity type name to validate

        Returns:
            True if valid, False otherwise
        """
        rules = self.get_validation_rules()

        # Check length
        min_length = rules.get("min_entity_length", 2)
        max_length = rules.get("max_entity_length", 100)

        if not (min_length <= len(entity_type) <= max_length):
            return False

        # Check characters (simplified validation)
        allowed = rules.get(
            "allowed_characters", "alphanumeric_spaces_hyphens_underscores"
        )
        if allowed == "alphanumeric_spaces_hyphens_underscores":
            import re

            if not re.match(r"^[A-Za-z0-9\s\-_]+$", entity_type):
                return False

        return True

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()


# Global instance
_entity_types_manager = None


def get_entity_types_manager() -> EntityTypesManager:
    """Get the global entity types manager instance.

    Returns:
        EntityTypesManager instance
    """
    global _entity_types_manager
    if _entity_types_manager is None:
        _entity_types_manager = EntityTypesManager()
    return _entity_types_manager


def get_standard_entity_types() -> List[str]:
    """Get standard entity types.

    Returns:
        List of standard entity types
    """
    return get_entity_types_manager().get_standard_entity_types()


def get_agent_entity_types(agent_name: str) -> List[str]:
    """Get entity types for a specific agent.

    Args:
        agent_name: Agent name

    Returns:
        List of entity types for the agent
    """
    return get_entity_types_manager().get_agent_default_types(agent_name)


def get_domain_entity_types(domain: str) -> List[str]:
    """Get entity types for a specific domain.

    Args:
        domain: Domain name

    Returns:
        List of entity types for the domain
    """
    return get_entity_types_manager().get_domain_entity_types(domain)
