#!/usr/bin/env python3
"""
Entity type normalizer for consistent entity classification.
"""

import logging
from typing import List, Dict, Set
from ..models import Entity

logger = logging.getLogger(__name__)


class EntityTypeNormalizer:
    """Normalizes entity types for consistent classification across extractions."""
    
    # Medical conditions should always be classified as CONCEPT
    MEDICAL_CONDITIONS = {
        "borreliose", "lyme disease", "diabetes", "cancer", "pneumonia", "influenza",
        "covid-19", "coronavirus", "malaria", "tuberculosis", "hepatitis", "aids",
        "hiv", "alzheimer", "parkinson", "multiple sclerosis", "arthritis", "asthma",
        "hypertension", "depression", "anxiety", "schizophrenia", "bipolar", "autism",
        "adhd", "epilepsy", "migraine", "stroke", "heart attack", "myocardial infarction",
        "babesiose", "chlamydien", "borrelien", "zirbeldrüse", "pineal gland"
    }
    
    # Organizations/companies should be ORGANIZATION
    ORGANIZATIONS = {
        "who", "world health organization", "cdc", "fda", "nih", "pfizer", "moderna",
        "johnson & johnson", "astrazeneca", "novartis", "roche", "merck", "abbott",
        "armin labs"
    }
    
    # People names should be PERSON
    KNOWN_PERSONS = {
        "dr. armin schwarzbach", "armin schwarzbach", "carolin tietz", "dr. schwarzbach"
    }
    
    def __init__(self):
        """Initialize the normalizer."""
        # Convert to lowercase for case-insensitive matching
        self.medical_conditions = {term.lower() for term in self.MEDICAL_CONDITIONS}
        self.organizations = {term.lower() for term in self.ORGANIZATIONS}
        self.known_persons = {term.lower() for term in self.KNOWN_PERSONS}
    
    def normalize_entity_type(self, entity: Entity) -> Entity:
        """Normalize the entity type based on predefined rules.
        
        Args:
            entity: Entity to normalize
            
        Returns:
            Entity with normalized type
        """
        name_lower = entity.name.lower().strip()
        original_type = entity.type
        
        # Check for medical conditions
        if name_lower in self.medical_conditions:
            if entity.type != "CONCEPT":
                logger.info(f"Normalizing '{entity.name}' from {entity.type} to CONCEPT (medical condition)")
                entity.type = "CONCEPT"
        
        # Check for organizations
        elif name_lower in self.organizations:
            if entity.type != "ORGANIZATION":
                logger.info(f"Normalizing '{entity.name}' from {entity.type} to ORGANIZATION")
                entity.type = "ORGANIZATION"
        
        # Check for known persons
        elif name_lower in self.known_persons:
            if entity.type != "PERSON":
                logger.info(f"Normalizing '{entity.name}' from {entity.type} to PERSON")
                entity.type = "PERSON"
        
        # Check for partial matches in medical conditions (e.g., "Borrelien" -> "borreliose")
        elif any(medical_term in name_lower or name_lower in medical_term for medical_term in self.medical_conditions):
            if entity.type != "CONCEPT":
                logger.info(f"Normalizing '{entity.name}' from {entity.type} to CONCEPT (partial medical match)")
                entity.type = "CONCEPT"
        
        return entity
    
    def normalize_entities(self, entities: List[Entity]) -> List[Entity]:
        """Normalize a list of entities.
        
        Args:
            entities: List of entities to normalize
            
        Returns:
            List of entities with normalized types
        """
        normalized = []
        for entity in entities:
            normalized_entity = self.normalize_entity_type(entity)
            normalized.append(normalized_entity)
        
        return normalized
    
    def add_medical_condition(self, condition: str) -> None:
        """Add a new medical condition to the normalization rules.
        
        Args:
            condition: Medical condition name to add
        """
        self.medical_conditions.add(condition.lower().strip())
        logger.info(f"Added medical condition: {condition}")
    
    def add_organization(self, organization: str) -> None:
        """Add a new organization to the normalization rules.
        
        Args:
            organization: Organization name to add
        """
        self.organizations.add(organization.lower().strip())
        logger.info(f"Added organization: {organization}")
    
    def add_person(self, person: str) -> None:
        """Add a new person to the normalization rules.
        
        Args:
            person: Person name to add
        """
        self.known_persons.add(person.lower().strip())
        logger.info(f"Added person: {person}")