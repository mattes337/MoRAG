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
    CONTAINS = "CONTAINS"  # Document -> DocumentChunk
    MENTIONS = "MENTIONS"  # DocumentChunk -> Entity
    PLAYED_ROLE = "PLAYED_ROLE"  # Person -> Role/Character
    PORTRAYED = "PORTRAYED"  # Person -> Character/Role
    PRACTICES = "PRACTICES"  # Person -> Activity/Belief
    ENGAGED_IN = "ENGAGED_IN"  # Person -> Activity
    STUDIED = "STUDIED"  # Person -> Subject/Field

    # Technical and specification relations
    DEFINED_BY = "DEFINED_BY"  # Entity -> Standard/Specification
    SPECIFIED_BY = "SPECIFIED_BY"  # Entity -> Standard/Specification
    PUBLISHED_BY = "PUBLISHED_BY"  # Entity -> Organization/Person
    COMPONENT_OF = "COMPONENT_OF"  # Entity -> Entity (structural)
    IMPLEMENTS = "IMPLEMENTS"  # Entity -> Standard/Protocol
    ESTABLISHES = "ESTABLISHES"  # Entity -> Entity (creation)
    PROVIDES = "PROVIDES"  # Entity -> Service/Function
    MANDATES = "MANDATES"  # Standard -> Requirement
    REQUIRES = "REQUIRES"  # Entity -> Requirement
    SPECIFIES = "SPECIFIES"  # Standard -> Detail
    FACILITATES = "FACILITATES"  # Entity -> Process/Function
    ENABLES = "ENABLES"  # Entity -> Capability
    WORKS_WITH = "WORKS_WITH"  # Entity -> Entity (collaboration)
    INTEROPERATES_WITH = "INTEROPERATES_WITH"  # System -> System
    COMPLIES_WITH = "COMPLIES_WITH"  # Entity -> Standard
    FOLLOWS = "FOLLOWS"  # Entity -> Standard/Protocol
    BASED_ON = "BASED_ON"  # Entity -> Foundation/Standard
    COMMUNICATES_WITH = "COMMUNICATES_WITH"  # System -> System
    PROCESSES = "PROCESSES"  # System -> Data/Entity
    VALIDATES = "VALIDATES"  # System -> Data/Entity

    CUSTOM = "CUSTOM"  # For dynamic/custom relation types (deprecated, use RELATED_TO)


# Type aliases for improved readability
EntityId = str
RelationId = str
EntityAttributes = Dict[str, Any]
RelationAttributes = Dict[str, Any]