"""LLM-based entity extraction."""

import logging
from typing import Dict, List, Optional, Any, Union

from ..models import Entity, EntityType
from .base import BaseExtractor, LLMConfig

logger = logging.getLogger(__name__)


class EntityExtractor(BaseExtractor):
    """LLM-based entity extractor.
    
    This class uses Large Language Models to extract entities from text.
    It identifies named entities like persons, organizations, locations, concepts, etc.
    """
    
    def __init__(self, config: Union[LLMConfig, Dict[str, Any]] = None, **kwargs):
        """Initialize the entity extractor.
        
        Args:
            config: LLM configuration as LLMConfig object or dictionary
            **kwargs: Additional configuration parameters (for backward compatibility)
        """
        # Handle backward compatibility with llm_config parameter
        if config is None and 'llm_config' in kwargs:
            config = kwargs['llm_config']
        
        # Convert dictionary to LLMConfig if needed
        if isinstance(config, dict):
            config = LLMConfig(**config)
        elif config is None:
            config = LLMConfig()
            
        super().__init__(config)
    
    async def extract(self, text: str, doc_id: Optional[str] = None, source_doc_id: Optional[str] = None, **kwargs) -> List[Entity]:
        """Extract entities from text.
        
        Args:
            text: Text to extract entities from
            doc_id: Optional document ID to associate with entities (deprecated, use source_doc_id)
            source_doc_id: Optional document ID to associate with entities
            **kwargs: Additional arguments
            
        Returns:
            List of Entity objects
        """
        entities = await super().extract(text, **kwargs)
        
        # Set document ID if provided (prefer source_doc_id over doc_id)
        document_id = source_doc_id or doc_id
        if document_id:
            for entity in entities:
                entity.source_doc_id = document_id
                
        return entities
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for entity extraction.
        
        Returns:
            System prompt string
        """
        return """
You are an expert entity extraction system. Your task is to identify and extract named entities from the given text.

Extract entities of the following types:
- PERSON: Names of people
- ORGANIZATION: Companies, institutions, government bodies
- LOCATION: Cities, countries, addresses, geographical locations
- DATE: Specific dates, years, time periods
- TIME: Times of day, durations
- MONEY: Monetary amounts, currencies
- PERCENT: Percentages
- FACILITY: Buildings, airports, highways, bridges
- PRODUCT: Objects, vehicles, foods, etc.
- EVENT: Named hurricanes, battles, wars, sports events
- WORK_OF_ART: Titles of books, songs, movies
- LAW: Named documents, laws, acts
- LANGUAGE: Any named language
- TECHNOLOGY: Software, hardware, technical concepts
- CONCEPT: Abstract concepts, ideas, theories

For each entity, provide:
1. name: The exact text of the entity as it appears
2. type: One of the types listed above
3. context: A brief description of the entity's role or significance in the text
4. confidence: A score from 0.0 to 1.0 indicating extraction confidence

Return the results as a JSON array of objects with the following structure:
[
  {
    "name": "entity name",
    "type": "ENTITY_TYPE",
    "context": "brief description",
    "confidence": 0.95
  }
]

Rules:
- Only extract entities that are clearly identifiable and significant
- Avoid extracting common words unless they are proper nouns
- If an entity could be multiple types, choose the most specific one
- Ensure confidence scores reflect the certainty of the extraction
- Return an empty array if no entities are found
"""
    
    def get_user_prompt(self, text: str, **kwargs) -> str:
        """Get the user prompt for entity extraction.
        
        Args:
            text: Text to extract entities from
            **kwargs: Additional arguments (unused)
            
        Returns:
            User prompt string
        """
        return f"""
Extract named entities from the following text:

{text}

Return the entities as a JSON array as specified in the system prompt.
"""
    
    def parse_response(self, response: str, text: str = "") -> List[Entity]:
        """Parse the LLM response into Entity objects.
        
        Args:
            response: Raw LLM response
            text: Original text that was processed (for source_text)
            
        Returns:
            List of Entity objects
        """
        try:
            data = self.parse_json_response(response)
            
            if not isinstance(data, list):
                logger.warning(f"Expected list response, got {type(data)}")
                return []
            
            entities = []
            for item in data:
                try:
                    entity = self._create_entity_from_dict(item)
                    if entity:
                        # Set source text (first 500 chars as context)
                        entity.source_text = text[:500] if text else ""
                        entities.append(entity)
                except Exception as e:
                    logger.warning(f"Failed to create entity from {item}: {e}")
                    continue
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to parse entity extraction response: {e}")
            return []
    
    def _create_entity_from_dict(self, data: Dict[str, Any]) -> Optional[Entity]:
        """Create an Entity object from dictionary data.
        
        Args:
            data: Dictionary containing entity data
            
        Returns:
            Entity object or None if creation fails
        """
        try:
            # Validate required fields
            if "name" not in data or not data["name"]:
                logger.warning(f"Entity missing name: {data}")
                return None
            
            # Get entity type
            entity_type = data.get("type", "CUSTOM")
            if isinstance(entity_type, str):
                # Try to convert to EntityType enum
                try:
                    entity_type = EntityType(entity_type)
                except ValueError:
                    # Convert to CUSTOM if not in enum
                    entity_type = EntityType.CUSTOM
            
            # Create entity
            entity = Entity(
                name=data["name"].strip(),
                type=entity_type,
                confidence=data.get("confidence", 1.0),
                attributes={
                    "context": data.get("context", ""),
                    "extraction_method": "llm",
                }
            )
            
            return entity
            
        except Exception as e:
            logger.error(f"Error creating entity from data {data}: {e}")
            return None
    
    async def extract_with_context(
        self, 
        text: str, 
        source_doc_id: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> List[Entity]:
        """Extract entities with additional context information.
        
        Args:
            text: Text to extract entities from
            source_doc_id: ID of the source document
            additional_context: Additional context to help with extraction
            
        Returns:
            List of Entity objects with context information
        """
        # Modify user prompt if additional context is provided
        if additional_context:
            original_get_user_prompt = self.get_user_prompt
            
            def get_user_prompt_with_context(text: str, **kwargs) -> str:
                base_prompt = original_get_user_prompt(text, **kwargs)
                return f"""
{base_prompt}

Additional context: {additional_context}

Use this context to better understand the entities and their significance.
"""
            
            self.get_user_prompt = get_user_prompt_with_context
        
        try:
            entities = await self.extract(text)
            
            # Add source document ID and source text to entities
            for entity in entities:
                entity.source_doc_id = source_doc_id
                entity.source_text = text[:500]  # First 500 chars as context
            
            return entities
            
        finally:
            # Restore original method if it was modified
            if additional_context:
                self.get_user_prompt = original_get_user_prompt