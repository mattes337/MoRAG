"""LLM-based relation extraction."""

import logging
from typing import Dict, List, Optional, Any, Union

from ..models import Entity, Relation, RelationType
from .base import BaseExtractor, LLMConfig

logger = logging.getLogger(__name__)


class RelationExtractor(BaseExtractor):
    """LLM-based relation extractor.
    
    This class uses Large Language Models to extract relations between entities.
    It identifies relationships like "works for", "located in", "part of", etc.
    """
    
    # Default relation types with descriptions
    DEFAULT_RELATION_TYPES = {
        "WORKS_FOR": "Person works for an organization",
        "LOCATED_IN": "Entity is located in a place",
        "PART_OF": "Entity is part of another entity",
        "CREATED_BY": "Entity was created by a person or organization",
        "FOUNDED": "Person founded an organization",
        "OWNS": "Person or organization owns an entity",
        "USES": "Entity uses another entity",
        "CAUSES": "Entity causes another entity (especially important for diseases, pathogens, symptoms)",
        "TREATS": "Treatment or medication treats a condition",
        "DIAGNOSED_WITH": "Person is diagnosed with a condition",
        "ASSOCIATED_WITH": "Entity is associated with another entity",
        "AFFECTS": "Entity affects or impacts another entity",
        "RELATED_TO": "Generic relationship between entities",
        "HAPPENED_ON": "Event happened on a specific date/time",
        "HAPPENED_AT": "Event happened at a specific location",
        "PARTICIPATED_IN": "Person or organization participated in an event"
    }
    
    def __init__(self, config: Union[LLMConfig, Dict[str, Any]] = None, relation_types: Optional[Dict[str, str]] = None, **kwargs):
        """Initialize the relation extractor.
        
        Args:
            config: LLM configuration as LLMConfig object or dictionary
            relation_types: Optional dictionary of relation types and their descriptions.
                          If None, uses DEFAULT_RELATION_TYPES.
                          If provided (including empty dict {}), uses EXACTLY those types.
                          Format: {"TYPE_NAME": "description"}
            **kwargs: Additional configuration parameters (for backward compatibility)
        
        Examples:
            # Use default types (general purpose)
            extractor = RelationExtractor(config)
            
            # Use custom types (domain-specific)
            medical_types = {
                "CAUSES": "Pathogen causes disease",
                "TREATS": "Treatment treats condition"
            }
            extractor = RelationExtractor(config, relation_types=medical_types)
            
            # Use minimal types (highly focused)
            minimal = {"CAUSES": "Causal relationship"}
            extractor = RelationExtractor(config, relation_types=minimal)
            
            # Use no types (maximum control)
            extractor = RelationExtractor(config, relation_types={})
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
        
        # Set relation types (use provided or default)
        # Use 'is None' check to allow empty dict for complete control
        self.relation_types = relation_types if relation_types is not None else self.DEFAULT_RELATION_TYPES
    
    async def extract(self, text: str, entities: Optional[List[Entity]] = None, doc_id: Optional[str] = None, **kwargs) -> List[Relation]:
        """Extract relations from text.
        
        Args:
            text: Text to extract relations from
            entities: Optional list of known entities
            doc_id: Optional document ID to associate with relations
            **kwargs: Additional arguments
            
        Returns:
            List of Relation objects
        """
        if not text or not text.strip():
            return []
        
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(text, entities=entities, **kwargs)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = await self.call_llm(messages)
            relations = self.parse_response(response, text=text)
            
            # Resolve entity IDs if entities are provided
            if entities:
                entity_name_to_id = {entity.name: entity.id for entity in entities}
                resolved_relations = []
                
                for relation in relations:
                    # Try to resolve entity names to IDs
                    source_id = self._resolve_entity_id(
                        relation.source_entity_id, entity_name_to_id
                    )
                    target_id = self._resolve_entity_id(
                        relation.target_entity_id, entity_name_to_id
                    )
                    
                    if source_id and target_id:
                        relation.source_entity_id = source_id
                        relation.target_entity_id = target_id
                        resolved_relations.append(relation)
                    else:
                        logger.warning(
                            f"Could not resolve entity IDs for relation: "
                            f"{relation.attributes.get('source_entity_name')} -> "
                            f"{relation.attributes.get('target_entity_name')}"
                        )
                
                relations = resolved_relations
            
            # Set document ID if provided
            if doc_id:
                for relation in relations:
                    relation.source_doc_id = doc_id
                    
            return relations
            
        except Exception as e:
            logger.error(f"Error during relation extraction: {e}")
            return []
     
    def get_system_prompt(self) -> str:
        """Get the system prompt for relation extraction.
        
        Returns:
            System prompt string
        """
        # Build relation types list dynamically
        relation_types_text = "\n".join([f"- {rel_type}: {description}" for rel_type, description in self.relation_types.items()])
        
        return f"""
You are an expert relation extraction system. Your task is to identify relationships between entities in the given text. Be thorough and comprehensive in finding all possible relationships, especially in medical, scientific, and technical content.

Extract relations of the following types:
{relation_types_text}

For each relation, provide:
1. source_entity: The name of the source entity (exactly as it appears in text)
2. target_entity: The name of the target entity (exactly as it appears in text)
3. relation_type: One of the types listed above
4. context: The specific text that indicates this relationship
5. confidence: A score from 0.0 to 1.0 indicating extraction confidence

Return the results as a JSON array of objects with the following structure:
[
  {{
    "source_entity": "source entity name",
    "target_entity": "target entity name",
    "relation_type": "RELATION_TYPE",
    "context": "text that indicates the relationship",
    "confidence": 0.95
  }}
]

Rules:
- Extract ALL relations that are explicitly stated OR reasonably implied in the text
- Pay special attention to causal relationships (CAUSES) and associations between medical terms
- For medical/scientific content, be especially thorough in identifying relationships
- Look for relationships between similar terms (e.g., a pathogen and the disease it causes)
- Ensure both entities are identifiable in the text
- If a relationship could be multiple types, choose the most specific one
- Ensure confidence scores reflect the certainty of the extraction
- Return an empty array if no relations are found
- Be careful about the direction of relationships (source -> target)
"""
    
    def get_user_prompt(self, text: str, entities: Optional[List[Entity]] = None, **kwargs) -> str:
        """Get the user prompt for relation extraction.
        
        Args:
            text: Text to extract relations from
            entities: Optional list of known entities to focus on
            **kwargs: Additional arguments
            
        Returns:
            User prompt string
        """
        base_prompt = f"""
Extract relations between entities from the following text:

{text}
"""
        
        if entities:
            entity_names = [entity.name for entity in entities]
            base_prompt += f"""

Known entities in the text:
{', '.join(entity_names)}

Focus on finding relationships between these entities, but also identify any other clear relationships.
"""
        
        # Add context if provided
        context = kwargs.get('context')
        if context:
            base_prompt += f"""

Additional context: {context}
"""
        
        base_prompt += """

Return the relations as a JSON array as specified in the system prompt.
"""
        
        return base_prompt
    
    def parse_response(self, response: str, text: str = "") -> List[Relation]:
        """Parse the LLM response into Relation objects.
        
        Args:
            response: Raw LLM response
            text: Original text that was processed (for source_text)
            
        Returns:
            List of Relation objects
        """
        try:
            data = self.parse_json_response(response)
            
            if not isinstance(data, list):
                logger.warning(f"Expected list response, got {type(data)}")
                return []
            
            relations = []
            for item in data:
                try:
                    relation = self._create_relation_from_dict(item)
                    if relation:
                        # Set source text (first 500 chars as context)
                        relation.source_text = text[:500] if text else ""
                        relations.append(relation)
                except Exception as e:
                    logger.warning(f"Failed to create relation from {item}: {e}")
                    continue
            
            return relations
            
        except Exception as e:
            logger.error(f"Failed to parse relation extraction response: {e}")
            return []
    
    def _create_relation_from_dict(self, data: Dict[str, Any]) -> Optional[Relation]:
        """Create a Relation object from dictionary data.
        
        Args:
            data: Dictionary containing relation data
            
        Returns:
            Relation object or None if creation fails
        """
        try:
            # Validate required fields
            required_fields = ["source_entity", "target_entity", "relation_type"]
            for field in required_fields:
                if field not in data or not data[field]:
                    logger.warning(f"Relation missing {field}: {data}")
                    return None
            
            # Get relation type
            relation_type = data.get("relation_type", "CUSTOM")
            if isinstance(relation_type, str):
                # Try to convert to RelationType enum
                try:
                    relation_type = RelationType(relation_type)
                except ValueError:
                    # Convert to CUSTOM if not in enum
                    relation_type = RelationType.CUSTOM
            
            # For now, use entity names as IDs (will be resolved later)
            source_entity_id = data["source_entity"].strip()
            target_entity_id = data["target_entity"].strip()
            
            # Create relation
            relation = Relation(
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                type=relation_type,
                confidence=data.get("confidence", 1.0),
                attributes={
                    "context": data.get("context", ""),
                    "extraction_method": "llm",
                    "source_entity_name": source_entity_id,  # Store original names
                    "target_entity_name": target_entity_id,
                }
            )
            
            return relation
            
        except Exception as e:
            logger.error(f"Error creating relation from data {data}: {e}")
            return None
    
    async def extract_for_entity_pairs(
        self,
        text: str,
        entities: List[Entity],
        entity_pairs: List[tuple],
        doc_id: Optional[str] = None
    ) -> List[Relation]:
        """Extract relations for specific entity pairs.
        
        Args:
            text: Text to extract relations from
            entities: List of known entities
            entity_pairs: List of (source_entity_id, target_entity_id) tuples
            doc_id: Optional document ID to associate with relations
            
        Returns:
            List of Relation objects for the specified pairs
        """
        # Extract all relations first
        all_relations = await self.extract(text, entities, doc_id=doc_id)
        
        # Filter relations to only include specified pairs
        filtered_relations = []
        entity_pair_set = set(entity_pairs)
        
        for relation in all_relations:
            # Check if this relation matches any of the specified pairs (in either direction)
            relation_pair = (relation.source_entity_id, relation.target_entity_id)
            reverse_pair = (relation.target_entity_id, relation.source_entity_id)
            
            if relation_pair in entity_pair_set or reverse_pair in entity_pair_set:
                filtered_relations.append(relation)
        
        return filtered_relations
    
    async def extract_with_entities(
        self, 
        text: str, 
        entities: List[Entity],
        source_doc_id: Optional[str] = None
    ) -> List[Relation]:
        """Extract relations with known entities.
        
        Args:
            text: Text to extract relations from
            entities: List of known entities
            source_doc_id: ID of the source document
            
        Returns:
            List of Relation objects with resolved entity IDs
        """
        relations = await self.extract(text, entities=entities)
        
        # Create entity name to ID mapping
        entity_name_to_id = {entity.name: entity.id for entity in entities}
        
        # Resolve entity IDs and add context
        resolved_relations = []
        for relation in relations:
            # Try to resolve entity names to IDs
            source_id = self._resolve_entity_id(
                relation.source_entity_id, entity_name_to_id
            )
            target_id = self._resolve_entity_id(
                relation.target_entity_id, entity_name_to_id
            )
            
            if source_id and target_id:
                relation.source_entity_id = source_id
                relation.target_entity_id = target_id
                relation.source_doc_id = source_doc_id
                relation.source_text = text[:500]  # First 500 chars as context
                resolved_relations.append(relation)
            else:
                logger.warning(
                    f"Could not resolve entity IDs for relation: "
                    f"{relation.attributes.get('source_entity_name')} -> "
                    f"{relation.attributes.get('target_entity_name')}"
                )
        
        return resolved_relations
    
    def _resolve_entity_id(
        self, 
        entity_name: str, 
        entity_name_to_id: Dict[str, str]
    ) -> Optional[str]:
        """Resolve entity name to ID using fuzzy matching.
        
        Args:
            entity_name: Name of the entity to resolve
            entity_name_to_id: Mapping of entity names to IDs
            
        Returns:
            Entity ID if found, None otherwise
        """
        # Exact match
        if entity_name in entity_name_to_id:
            return entity_name_to_id[entity_name]
        
        # Case-insensitive match
        entity_name_lower = entity_name.lower()
        for name, entity_id in entity_name_to_id.items():
            if name.lower() == entity_name_lower:
                return entity_id
        
        # Partial match (entity name contains or is contained in known entity)
        for name, entity_id in entity_name_to_id.items():
            if (entity_name_lower in name.lower() or 
                name.lower() in entity_name_lower):
                return entity_id
        
        # Enhanced fuzzy matching for similar terms (especially useful for German compound words)
        # Check for root word matches (e.g., "Borrelien" and "Borreliose")
        entity_root = entity_name_lower[:min(6, len(entity_name_lower))]  # First 6 chars as root
        if len(entity_root) >= 4:  # Only for reasonably long roots
            for name, entity_id in entity_name_to_id.items():
                name_root = name.lower()[:min(6, len(name.lower()))]
                if len(name_root) >= 4 and entity_root == name_root:
                    return entity_id
        
        # Check for word similarity (Levenshtein-like approach for close matches)
        for name, entity_id in entity_name_to_id.items():
            if self._are_similar_words(entity_name_lower, name.lower()):
                return entity_id
        
        return None
    
    def _are_similar_words(self, word1: str, word2: str) -> bool:
        """Check if two words are similar enough to be considered the same entity.
        
        Args:
            word1: First word to compare
            word2: Second word to compare
            
        Returns:
            True if words are similar enough
        """
        # Skip very short words
        if len(word1) < 4 or len(word2) < 4:
            return False
        
        # Check if one word is a substring of another with reasonable length
        if len(word1) >= 6 and len(word2) >= 6:
            if word1[:6] == word2[:6]:  # Same first 6 characters
                return True
        
        # Simple edit distance check for very similar words
        if abs(len(word1) - len(word2)) <= 2:  # Similar length
            differences = sum(c1 != c2 for c1, c2 in zip(word1, word2))
            max_len = max(len(word1), len(word2))
            if differences <= max_len * 0.2:  # Allow 20% character differences
                return True
        
        return False
    
    async def extract_from_entity_pairs(
        self, 
        text: str, 
        entity_pairs: List[tuple[Entity, Entity]],
        source_doc_id: Optional[str] = None
    ) -> List[Relation]:
        """Extract relations for specific entity pairs.
        
        Args:
            text: Text to extract relations from
            entity_pairs: List of entity pairs to check for relations
            source_doc_id: ID of the source document
            
        Returns:
            List of Relation objects
        """
        if not entity_pairs:
            return []
        
        # Create focused prompt for specific entity pairs
        pair_descriptions = []
        for source_entity, target_entity in entity_pairs:
            pair_descriptions.append(f"'{source_entity.name}' and '{target_entity.name}'")
        
        focused_prompt = f"""
Analyze the relationships between these specific entity pairs in the text:
{', '.join(pair_descriptions)}

Text:
{text}

Return relations as a JSON array as specified in the system prompt.
"""
        
        # Temporarily override the user prompt
        original_get_user_prompt = self.get_user_prompt
        self.get_user_prompt = lambda text, **kwargs: focused_prompt
        
        try:
            relations = await self.extract(text)
            
            # Add context and resolve IDs
            for relation in relations:
                relation.source_doc_id = source_doc_id
                relation.source_text = text[:500]
            
            return relations
            
        finally:
            # Restore original method
            self.get_user_prompt = original_get_user_prompt