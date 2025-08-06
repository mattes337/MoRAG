"""Entity categorization service for adding semantic categories to entities."""

import asyncio
from typing import List, Dict, Optional, Any
import structlog

from ..models import Entity


class EntityCategorizationService:
    """Service for adding semantic categories to entities to improve retrieval quality."""
    
    def __init__(self, llm_service=None, logger=None):
        """Initialize the entity categorization service.
        
        Args:
            llm_service: LLM service for category determination
            logger: Logger instance
        """
        self.llm_service = llm_service
        self.logger = logger or structlog.get_logger(__name__)
        
        # Predefined semantic categories for fallback
        self.semantic_categories = {
            'MEDICAL': ['disease', 'condition', 'symptom', 'disorder', 'syndrome', 'illness', 'diagnosis'],
            'CHEMICAL': ['vitamin', 'mineral', 'supplement', 'compound', 'element', 'molecule', 'substance'],
            'BIOLOGICAL': ['hormone', 'enzyme', 'protein', 'neurotransmitter', 'receptor', 'gene', 'cell'],
            'TECHNICAL': ['software', 'hardware', 'system', 'technology', 'device', 'tool', 'platform'],
            'PHARMACEUTICAL': ['drug', 'medication', 'medicine', 'treatment', 'therapy', 'prescription'],
            'ANATOMICAL': ['brain', 'organ', 'tissue', 'muscle', 'nerve', 'gland', 'system'],
            'NUTRITIONAL': ['food', 'nutrient', 'diet', 'nutrition', 'meal', 'ingredient'],
            'PSYCHOLOGICAL': ['behavior', 'emotion', 'cognition', 'memory', 'attention', 'mood'],
            'ORGANIZATIONAL': ['company', 'institution', 'organization', 'agency', 'corporation'],
            'GEOGRAPHICAL': ['country', 'city', 'region', 'location', 'place', 'area'],
            'TEMPORAL': ['time', 'period', 'duration', 'age', 'year', 'month', 'day'],
            'QUANTITATIVE': ['amount', 'dose', 'level', 'concentration', 'percentage', 'ratio']
        }
    
    async def categorize_entities(self, entities: List[Entity], domain: Optional[str] = None) -> List[Entity]:
        """Add semantic categories to entities.
        
        Args:
            entities: List of entities to categorize
            domain: Optional domain context for categorization
            
        Returns:
            List of entities with added categories
        """
        if not entities:
            return entities
            
        categorized_entities = []
        
        # Process entities in batches for efficiency
        batch_size = 10
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batch_results = await self._categorize_entity_batch(batch, domain)
            categorized_entities.extend(batch_results)
            
        return categorized_entities
    
    async def _categorize_entity_batch(self, entities: List[Entity], domain: Optional[str] = None) -> List[Entity]:
        """Categorize a batch of entities.
        
        Args:
            entities: Batch of entities to categorize
            domain: Optional domain context
            
        Returns:
            List of categorized entities
        """
        if self.llm_service:
            try:
                return await self._llm_categorize_batch(entities, domain)
            except Exception as e:
                self.logger.warning(f"LLM categorization failed, using fallback: {e}")
        
        # Fallback to rule-based categorization
        return [self._rule_based_categorize(entity, domain) for entity in entities]
    
    async def _llm_categorize_batch(self, entities: List[Entity], domain: Optional[str] = None) -> List[Entity]:
        """Use LLM to categorize entities in batch.
        
        Args:
            entities: Entities to categorize
            domain: Optional domain context
            
        Returns:
            List of categorized entities
        """
        entity_names = [entity.name for entity in entities]
        entity_types = [entity.type for entity in entities]
        
        prompt = f"""Analyze these entities and assign semantic categories that would improve information retrieval:

Entities to categorize:
{chr(10).join([f"- {name} (type: {etype})" for name, etype in zip(entity_names, entity_types)])}

Domain context: {domain or 'general'}

For each entity, determine the most appropriate semantic category from these options or suggest a more specific one:
- MEDICAL (diseases, conditions, symptoms, disorders)
- CHEMICAL (vitamins, minerals, compounds, substances)
- BIOLOGICAL (hormones, enzymes, proteins, genes)
- TECHNICAL (software, hardware, systems, tools)
- PHARMACEUTICAL (drugs, medications, treatments)
- ANATOMICAL (body parts, organs, tissues)
- NUTRITIONAL (foods, nutrients, dietary components)
- PSYCHOLOGICAL (behaviors, emotions, cognitive processes)
- ORGANIZATIONAL (companies, institutions, agencies)
- GEOGRAPHICAL (locations, places, regions)
- TEMPORAL (time-related concepts)
- QUANTITATIVE (measurements, amounts, levels)

Return a JSON array with the same order as input:
[
  {{"entity": "entity_name", "category": "CATEGORY_NAME"}},
  ...
]

Only return the JSON array, no explanation."""

        try:
            response = await self.llm_service.generate(prompt, max_tokens=500)
            
            # Parse JSON response
            import json
            categories_data = json.loads(response.strip())
            
            # Apply categories to entities
            categorized_entities = []
            for i, entity in enumerate(entities):
                if i < len(categories_data):
                    category = categories_data[i].get('category', 'GENERAL')
                    categorized_entity = self._add_category_to_entity(entity, category)
                    categorized_entities.append(categorized_entity)
                else:
                    # Fallback for missing entries
                    categorized_entities.append(self._rule_based_categorize(entity, domain))
                    
            return categorized_entities
            
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM categorization response: {e}")
            # Fallback to rule-based categorization
            return [self._rule_based_categorize(entity, domain) for entity in entities]
    
    def _rule_based_categorize(self, entity: Entity, domain: Optional[str] = None) -> Entity:
        """Apply rule-based categorization to an entity.
        
        Args:
            entity: Entity to categorize
            domain: Optional domain context
            
        Returns:
            Entity with added category
        """
        entity_text = f"{entity.name} {entity.type}".lower()
        
        # Check against predefined categories
        for category, keywords in self.semantic_categories.items():
            if any(keyword in entity_text for keyword in keywords):
                return self._add_category_to_entity(entity, category)
        
        # Domain-specific fallback
        if domain:
            domain_lower = domain.lower()
            if domain_lower in ['medical', 'health']:
                return self._add_category_to_entity(entity, 'MEDICAL')
            elif domain_lower in ['technical', 'technology']:
                return self._add_category_to_entity(entity, 'TECHNICAL')
            elif domain_lower in ['business', 'corporate']:
                return self._add_category_to_entity(entity, 'ORGANIZATIONAL')
        
        # Default category
        return self._add_category_to_entity(entity, 'GENERAL')
    
    def _add_category_to_entity(self, entity: Entity, category: str) -> Entity:
        """Add semantic category to entity attributes.
        
        Args:
            entity: Entity to modify
            category: Semantic category to add
            
        Returns:
            Entity with added category
        """
        # Create a copy of the entity to avoid modifying the original
        entity_dict = entity.model_dump()
        
        # Add category to attributes
        if 'attributes' not in entity_dict:
            entity_dict['attributes'] = {}
        
        entity_dict['attributes']['semantic_category'] = category
        entity_dict['attributes']['categorization_method'] = 'llm' if self.llm_service else 'rule_based'
        
        # Create new entity with category
        return Entity(**entity_dict)
    
    def get_entities_by_category(self, entities: List[Entity], category: str) -> List[Entity]:
        """Filter entities by semantic category.
        
        Args:
            entities: List of entities to filter
            category: Category to filter by
            
        Returns:
            List of entities matching the category
        """
        return [
            entity for entity in entities
            if entity.attributes and entity.attributes.get('semantic_category') == category
        ]
    
    def get_category_distribution(self, entities: List[Entity]) -> Dict[str, int]:
        """Get distribution of semantic categories in entity list.
        
        Args:
            entities: List of entities to analyze
            
        Returns:
            Dictionary mapping categories to counts
        """
        distribution = {}
        for entity in entities:
            if entity.attributes and 'semantic_category' in entity.attributes:
                category = entity.attributes['semantic_category']
                distribution[category] = distribution.get(category, 0) + 1
            else:
                distribution['UNCATEGORIZED'] = distribution.get('UNCATEGORIZED', 0) + 1
        
        return distribution
