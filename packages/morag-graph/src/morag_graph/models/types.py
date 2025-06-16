"""Type definitions for graph models."""

from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union


class EntityType(str, Enum):
    """Predefined entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    FACILITY = "FACILITY"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    CONCEPT = "CONCEPT"
    TECHNOLOGY = "TECHNOLOGY"
    CUSTOM = "CUSTOM"  # For dynamic/custom entity types


class RelationType(str, Enum):
    """Predefined relation types."""
    WORKS_FOR = "WORKS_FOR"  # Person -> Organization
    LOCATED_IN = "LOCATED_IN"  # Entity -> Location
    PART_OF = "PART_OF"  # Entity -> Entity
    CREATED_BY = "CREATED_BY"  # Entity -> Person/Organization
    FOUNDED = "FOUNDED"  # Person -> Organization (founding relationship)
    OWNS = "OWNS"  # Person/Organization -> Entity
    USES = "USES"  # Entity -> Entity
    LEADS = "LEADS"  # Person -> Organization (leadership relationship)
    COMPETES_WITH = "COMPETES_WITH"  # Organization -> Organization
    CAUSES = "CAUSES"  # Entity -> Entity (causal relationship)
    TREATS = "TREATS"  # Treatment -> Condition
    DIAGNOSED_WITH = "DIAGNOSED_WITH"  # Person -> Condition
    ASSOCIATED_WITH = "ASSOCIATED_WITH"  # Entity -> Entity (association)
    AFFECTS = "AFFECTS"  # Entity -> Entity (influence/impact)
    RELATED_TO = "RELATED_TO"  # Generic relation
    HAPPENED_ON = "HAPPENED_ON"  # Event -> Date/Time
    HAPPENED_AT = "HAPPENED_AT"  # Event -> Location
    PARTICIPATED_IN = "PARTICIPATED_IN"  # Person/Organization -> Event
    CUSTOM = "CUSTOM"  # For dynamic/custom relation types


# Type aliases for improved readability
EntityId = str
RelationId = str
EntityAttributes = Dict[str, Any]
RelationAttributes = Dict[str, Any]