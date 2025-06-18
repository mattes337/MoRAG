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
    
    # Default entity types with descriptions
    DEFAULT_ENTITY_TYPES = {
        "PERSON": "Names of people",
        "ORGANIZATION": "Companies, institutions, government bodies",
        "LOCATION": "Cities, countries, addresses, geographical locations",
        "DATE": "Specific dates, years, time periods",
        "TIME": "Times of day, durations",
        "MONEY": "Monetary amounts, currencies",
        "PERCENT": "Percentages",
        "FACILITY": "Buildings, airports, highways, bridges",
        "PRODUCT": "Objects, vehicles, foods, etc.",
        "EVENT": "Named hurricanes, battles, wars, sports events",
        "WORK_OF_ART": "Titles of books, songs, movies",
        "LAW": "Named documents, laws, acts",
        "LANGUAGE": "Any named language",
        "TECHNOLOGY": "Software, hardware, technical concepts",
        "CONCEPT": "Abstract concepts, ideas, theories"
    }
    
    def __init__(self, config: Union[LLMConfig, Dict[str, str]] = None, chunk_size: int = 4000, entity_types: Optional[Dict[str, str]] = None, **kwargs):
        """Initialize the entity extractor.
        
        Args:
            config: LLM configuration as LLMConfig object or dictionary
            chunk_size: Maximum characters per chunk for large texts (default: 4000)
            entity_types: Optional dictionary of entity types and their descriptions.
                         If None, uses DEFAULT_ENTITY_TYPES.
                         If provided (including empty dict {}), uses EXACTLY those types.
                         Format: {"TYPE_NAME": "description"}
            **kwargs: Additional configuration parameters (for backward compatibility)
        
        Examples:
            # Use default types (general purpose)
            extractor = EntityExtractor(config)
            
            # Use custom types (domain-specific)
            medical_types = {
                "DISEASE": "Medical condition or illness",
                "TREATMENT": "Medical intervention or therapy",
                "SYMPTOM": "Observable sign of disease"
            }
            extractor = EntityExtractor(config, entity_types=medical_types)
            
            # Use minimal types (highly focused)
            minimal = {"PERSON": "Individual person"}
            extractor = EntityExtractor(config, entity_types=minimal)
            
            # Use no types (maximum control)
            extractor = EntityExtractor(config, entity_types={})
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
        self.chunk_size = chunk_size
        
        # Set entity types (use provided or default)
        # Use 'is None' check to allow empty dict for complete control
        self.entity_types = entity_types if entity_types is not None else self.DEFAULT_ENTITY_TYPES
    
    async def extract(self, text: str, doc_id: Optional[str] = None, source_doc_id: Optional[str] = None, **kwargs) -> List[Entity]:
        """Extract entities from text with automatic chunking for large texts.
        
        Args:
            text: Text to extract entities from
            doc_id: Optional document ID to associate with entities (deprecated, use source_doc_id)
            source_doc_id: Optional document ID to associate with entities
            **kwargs: Additional arguments
            
        Returns:
            List of Entity objects
        """
        if not text or not text.strip():
            return []
        
        # Check if text needs chunking
        if len(text) <= self.chunk_size:
            # Process normally for small texts
            entities = await super().extract(text, **kwargs)
        else:
            # Process in chunks for large texts
            logger.info(f"Text length ({len(text)} chars) exceeds chunk size ({self.chunk_size}). Processing in chunks...")
            entities = await self._extract_chunked(text, **kwargs)
        
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
        # Build entity types list dynamically
        entity_types_text = "\n".join([f"- {entity_type}: {description}" for entity_type, description in self.entity_types.items()])
        
        return f"""
You are an expert entity extraction system. Your task is to identify and extract named entities from the given text.

Extract entities of the following types:
{entity_types_text}

For each entity, provide:
1. name: The exact text of the entity as it appears
2. type: One of the types listed above
3. context: A brief description of the entity's role or significance in the text
4. confidence: A score from 0.0 to 1.0 indicating extraction confidence

Return the results as a JSON array of objects with the following structure:
[
  {{
    "name": "entity name",
    "type": "ENTITY_TYPE",
    "context": "brief description",
    "confidence": 0.95
  }}
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
            **kwargs: Additional arguments including context
            
        Returns:
            User prompt string
        """
        base_prompt = f"""
Extract named entities from the following text:

{text}

Return the entities as a JSON array as specified in the system prompt.
"""
        
        # Add context if provided
        context = kwargs.get('context')
        if context:
            base_prompt += f"\n\nAdditional context: {context}"
        
        return base_prompt
    
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
                        entity.attributes["source_text"] = text[:500] if text else ""
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
                # Try to convert to EntityType enum, but preserve custom strings
                try:
                    entity_type = EntityType(entity_type)
                except ValueError:
                    # Keep the custom string type as-is
                    pass
            
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
                entity.attributes["source_text"] = text[:500]  # First 500 chars as context
            
            return entities
            
        finally:
            # Restore original method if it was modified
            if additional_context:
                self.get_user_prompt = original_get_user_prompt
    
    async def _extract_chunked(self, text: str, **kwargs) -> List[Entity]:
        """Extract entities from large text by processing in chunks.
        
        Args:
            text: Large text to extract entities from
            **kwargs: Additional arguments
            
        Returns:
            List of Entity objects with duplicates removed
        """
        chunks = self._split_text_into_chunks(text)
        all_entities = []
        
        logger.info(f"Processing {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks, 1):
            try:
                logger.info(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)...")
                chunk_entities = await super().extract(chunk, **kwargs)
                
                # Add chunk information to entities
                for entity in chunk_entities:
                    entity.attributes = entity.attributes or {}
                    entity.attributes["chunk_index"] = i
                    entity.attributes["total_chunks"] = len(chunks)
                    entity.attributes["source_text"] = chunk[:500]  # First 500 chars of chunk as context
                
                all_entities.extend(chunk_entities)
                
            except Exception as e:
                logger.warning(f"Failed to process chunk {i}: {e}")
                continue
        
        # Remove duplicates and merge similar entities
        deduplicated_entities = self._deduplicate_entities(all_entities)
        
        logger.info(f"Extracted {len(all_entities)} entities from chunks, {len(deduplicated_entities)} after deduplication")
        
        return deduplicated_entities
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks at sentence boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        import re
        
        # Split on sentence boundaries (periods, exclamation marks, question marks)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, start a new chunk
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If we still have chunks that are too large, split them more aggressively
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Split large chunks by words
                words = chunk.split()
                current_word_chunk = ""
                
                for word in words:
                    if current_word_chunk and len(current_word_chunk) + len(word) + 1 > self.chunk_size:
                        if current_word_chunk.strip():
                            final_chunks.append(current_word_chunk.strip())
                        current_word_chunk = word
                    else:
                        if current_word_chunk:
                            current_word_chunk += " " + word
                        else:
                            current_word_chunk = word
                
                if current_word_chunk.strip():
                    final_chunks.append(current_word_chunk.strip())
        
        return final_chunks
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on name and type.
        
        Args:
            entities: List of entities that may contain duplicates
            
        Returns:
            List of unique entities with highest confidence scores
        """
        if not entities:
            return []
        
        # Group entities by (name, type) key
        entity_groups = {}
        
        for entity in entities:
            key = (entity.name.lower().strip(), entity.type)
            
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # For each group, keep the entity with highest confidence
        deduplicated = []
        
        for group in entity_groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Find entity with highest confidence
                best_entity = max(group, key=lambda e: e.confidence)
                
                # Merge attributes from all entities in the group
                merged_attributes = {}
                contexts = []
                
                for entity in group:
                    if entity.attributes:
                        merged_attributes.update(entity.attributes)
                    
                    context = entity.attributes.get("context", "") if entity.attributes else ""
                    if context and context not in contexts:
                        contexts.append(context)
                
                # Update best entity with merged information
                best_entity.attributes = merged_attributes
                if contexts:
                    best_entity.attributes["context"] = "; ".join(contexts)
                
                deduplicated.append(best_entity)
        
        return deduplicated