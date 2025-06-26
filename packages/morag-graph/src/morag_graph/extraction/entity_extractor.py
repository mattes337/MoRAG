"""LLM-based entity extraction."""

import logging
from typing import Dict, List, Optional, Any, Union

from ..models import Entity, EntityType
from .base import BaseExtractor, LLMConfig
from .entity_normalizer import EntityTypeNormalizer

logger = logging.getLogger(__name__)

# Sentinel values to detect when parameters are not explicitly set
_DYNAMIC_TYPES_DEFAULT = object()
_ENTITY_TYPES_DEFAULT = object()


class EntityExtractor(BaseExtractor):
    """LLM-based entity extractor.

    This class uses Large Language Models to extract entities from text.
    It identifies named entities like persons, organizations, locations, concepts, etc.
    """

    # No predefined entity types - LLM determines types dynamically
    
    def __init__(self, config: Union[LLMConfig, Dict[str, str]] = None, chunk_size: int = 4000, entity_types=_ENTITY_TYPES_DEFAULT, normalize_types: bool = True, dynamic_types=_DYNAMIC_TYPES_DEFAULT, **kwargs):
        """Initialize the entity extractor.

        Args:
            config: LLM configuration as LLMConfig object or dictionary
            chunk_size: Maximum characters per chunk for large texts (default: 4000)
            entity_types: Optional dictionary of entity types and their descriptions.
                         If None and dynamic_types=False, uses basic examples.
                         If provided, uses EXACTLY those types.
                         Format: {"TYPE_NAME": "description"}
            normalize_types: Whether to normalize entity types for consistency (default: True)
            dynamic_types: Whether to let LLM determine entity types dynamically (default: True)
            **kwargs: Additional configuration parameters (for backward compatibility)

        Examples:
            # Use dynamic types (recommended - LLM determines types)
            extractor = EntityExtractor(config, dynamic_types=True)

            # Use custom types (domain-specific)
            medical_types = {
                "DISEASE": "Medical condition or illness",
                "TREATMENT": "Medical intervention or therapy",
                "SYMPTOM": "Observable sign of disease"
            }
            extractor = EntityExtractor(config, entity_types=medical_types, dynamic_types=False)

            # Use no predefined types but still constrain to specific ones
            extractor = EntityExtractor(config, entity_types={}, dynamic_types=False)
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

        # Determine dynamic_types behavior
        self.dynamic_types = True if dynamic_types is _DYNAMIC_TYPES_DEFAULT else dynamic_types

        # Set entity types based on parameters
        if entity_types is _ENTITY_TYPES_DEFAULT or entity_types is None:
            # No entity_types parameter or explicit None: pure dynamic mode
            self.entity_types = {}
        else:
            # Explicit types provided (could be empty dict)
            self.entity_types = entity_types

        # Initialize entity type normalizer
        self.normalize_types = normalize_types
        self.normalizer = EntityTypeNormalizer() if normalize_types else None
    
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
        
        # Set source document ID if provided
        final_source_doc_id = source_doc_id or doc_id
        if final_source_doc_id:
            for entity in entities:
                entity.source_doc_id = final_source_doc_id
        
        # Normalize entity types for consistency
        if self.normalize_types and self.normalizer:
            entities = self.normalizer.normalize_entities(entities)
                
        return entities
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for entity extraction.

        Returns:
            System prompt string
        """
        if self.dynamic_types:
            return self._get_dynamic_system_prompt()
        else:
            return self._get_static_system_prompt()

    def _get_dynamic_system_prompt(self) -> str:
        """Get system prompt for dynamic entity type extraction."""
        examples_text = ""
        if self.entity_types:
            examples_text = f"""
Example entity types (you can use these or create more appropriate ones):
{chr(10).join([f"- {entity_type}: {description}" for entity_type, description in self.entity_types.items()])}
"""

        return f"""
You are an expert entity extraction system. Your task is to identify and extract named entities from the given text.

IMPORTANT: You should determine the most appropriate entity type for each entity based on its semantic meaning and context. Use ABSTRACT, BROAD entity types rather than overly specific ones.

{examples_text}

For each entity, provide:
1. name: The exact text of the entity as it appears
2. type: A broad, abstract type that captures the entity's semantic category (e.g., PERSON, ORGANIZATION, ANATOMICAL, TECHNOLOGY, CONCEPT, etc.)
3. context: A brief description of the entity's role or significance in the text
4. confidence: A score from 0.0 to 1.0 indicating extraction confidence

Return the results as a JSON array of objects with the following structure:
[
  {{
    "name": "entity name",
    "type": "SEMANTIC_TYPE",
    "context": "brief description",
    "confidence": 0.95
  }}
]

Rules:
- Only extract entities that are clearly identifiable and significant
- Avoid extracting common words unless they are proper nouns
- Use BROAD, ABSTRACT entity types - avoid overly specific categories
- Entity types should be SINGULAR (e.g., PERSON not PERSONS, TECHNOLOGY not TECHNOLOGIES)
- Prefer general categories: ANATOMICAL over BRAIN_REGION/CELL_TYPE, TECHNOLOGY over SOFTWARE_LIBRARY/FRAMEWORK
- Be consistent with type naming within the same extraction
- If an entity could be multiple types, choose the most general appropriate one
- Ensure confidence scores reflect the certainty of the extraction
- Return an empty array if no entities are found
"""

    def _get_static_system_prompt(self) -> str:
        """Get system prompt for static entity type extraction."""
        if not self.entity_types:
            return """
You are an expert entity extraction system. Your task is to identify and extract named entities from the given text.

Since no specific entity types are provided, extract any significant named entities and assign them appropriate semantic types.

For each entity, provide:
1. name: The exact text of the entity as it appears
2. type: A descriptive type that best captures the entity's semantic category
3. context: A brief description of the entity's role or significance in the text
4. confidence: A score from 0.0 to 1.0 indicating extraction confidence

Return the results as a JSON array of objects with the following structure:
[
  {{
    "name": "entity name",
    "type": "SEMANTIC_TYPE",
    "context": "brief description",
    "confidence": 0.95
  }}
]

Rules:
- Only extract entities that are clearly identifiable and significant
- Avoid extracting common words unless they are proper nouns
- Use clear, descriptive type names
- Ensure confidence scores reflect the certainty of the extraction
- Return an empty array if no entities are found
"""

        # Build entity types list for static mode
        entity_types_text = "\n".join([f"- {entity_type}: {description}" for entity_type, description in self.entity_types.items()])

        return f"""
You are an expert entity extraction system. Your task is to identify and extract named entities from the given text.

Extract entities of the following types ONLY:
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
- ONLY use the entity types listed above
- If an entity doesn't fit any of the provided types, skip it
- Ensure confidence scores reflect the certainty of the extraction
- Return an empty array if no entities are found
"""
    
    def get_user_prompt(self, text: str, **kwargs) -> str:
        """Get the user prompt for entity extraction.

        Args:
            text: Text to extract entities from
            **kwargs: Additional arguments including context and intention

        Returns:
            User prompt string
        """
        base_prompt = f"""
Extract named entities from the following text:

{text}

Return the entities as a JSON array as specified in the system prompt.
"""

        # Add document intention if provided
        intention = kwargs.get('intention')
        if intention:
            base_prompt += f"\n\nDocument intention: {intention}\n\nBased on this intention, use appropriate abstract entity types that fit the document's purpose."

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
                        # Remove document-specific attributes to make entities generic
                        # No longer adding source_text
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

    async def extract_entities(self, text: str, source_doc_id: Optional[str] = None, **kwargs) -> List[Entity]:
        """Extract entities from text (alias for extract method).

        Args:
            text: Text to extract entities from
            source_doc_id: Optional document ID to associate with entities
            **kwargs: Additional arguments including intention

        Returns:
            List of extracted entities
        """
        return await self.extract(text, source_doc_id=source_doc_id, **kwargs)

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
            entities = await self.extract(text, source_doc_id=source_doc_id)
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
                
                # Add minimal context to entities (remove document-specific attributes)
                for entity in chunk_entities:
                    entity.attributes = entity.attributes or {}
                    # Remove chunk_index, total_chunks, and source_text to make entities generic
                
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
        """Remove duplicate entities based on name, type, and source document.
        
        Args:
            entities: List of entities that may contain duplicates
            
        Returns:
            List of unique entities with highest confidence scores
        """
        if not entities:
            return []
        
        # Group entities by (name, type, source_doc_id) key
        entity_groups = {}
        
        for entity in entities:
            key = (entity.name.lower().strip(), entity.type, entity.source_doc_id or '')
            
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