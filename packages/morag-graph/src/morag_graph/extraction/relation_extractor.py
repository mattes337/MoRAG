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
        "PARTICIPATED_IN": "Person or organization participated in an event",

        # Technical and specification relations
        "DEFINED_BY": "Entity is defined by a standard or specification",
        "SPECIFIED_BY": "Entity is specified by a standard or document",
        "PUBLISHED_BY": "Entity was published by an organization or person",
        "COMPONENT_OF": "Entity is a component of another entity",
        "IMPLEMENTS": "Entity implements a standard or protocol",
        "ESTABLISHES": "Entity establishes or creates another entity",
        "PROVIDES": "Entity provides a service or function",
        "MANDATES": "Standard mandates a requirement",
        "REQUIRES": "Entity requires another entity",
        "SPECIFIES": "Standard specifies details or requirements",
        "FACILITATES": "Entity facilitates a process or function",
        "ENABLES": "Entity enables a capability or function",
        "WORKS_WITH": "Entity works with another entity",
        "INTEROPERATES_WITH": "System interoperates with another system",
        "COMPLIES_WITH": "Entity complies with a standard",
        "FOLLOWS": "Entity follows a standard or protocol",
        "BASED_ON": "Entity is based on a foundation or standard",
        "COMMUNICATES_WITH": "System communicates with another system",
        "PROCESSES": "System processes data or entities",
        "VALIDATES": "System validates data or entities"
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
        """Extract relations from text using chunked processing for better results.
        
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
        
        # Use chunked extraction for large texts
        chunk_size = kwargs.get('chunk_size', 3000)  # Characters per chunk
        overlap = kwargs.get('overlap', 500)  # Overlap between chunks
        
        if len(text) > chunk_size and entities:
            logger.info(f"Using chunked extraction for large text ({len(text)} chars) with {len(entities)} entities")
            return await self._extract_chunked(text, entities, doc_id, chunk_size, overlap, **kwargs)
        else:
            logger.info(f"Using single-pass extraction for text ({len(text)} chars)")
            return await self._extract_single(text, entities, doc_id, **kwargs)
    
    async def _extract_single(self, text: str, entities: Optional[List[Entity]] = None, doc_id: Optional[str] = None, **kwargs) -> List[Relation]:
        """Extract relations from text in a single pass."""
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(text, entities=entities, **kwargs)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = await self.call_llm(messages)
            relations = self.parse_response(response, text=text)
            
            logger.info(f"LLM found {len(relations)} potential relations before entity ID resolution")
            
            # Resolve entity IDs if entities are provided
            if entities:
                relations = self._resolve_relations(relations, entities)
            
            # Set document ID if provided
            # Remove document-specific attributes to make relations generic
            # No longer setting source_doc_id based on doc_id
                    
            return relations
            
        except Exception as e:
            logger.error(f"Error during relation extraction: {e}")
            return []
    
    async def _extract_chunked(self, text: str, entities: List[Entity], doc_id: Optional[str] = None, chunk_size: int = 3000, overlap: int = 500, **kwargs) -> List[Relation]:
        """Extract relations using chunked processing for better coverage."""
        chunks = self._create_text_chunks(text, chunk_size, overlap)
        all_relations = []
        
        logger.info(f"Processing {len(chunks)} text chunks for relation extraction")
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            # Find entities that appear in this chunk
            chunk_entities = self._find_entities_in_chunk(chunk, entities)
            
            if len(chunk_entities) < 2:
                logger.debug(f"Chunk {i+1} has fewer than 2 entities, skipping")
                continue
            
            logger.debug(f"Chunk {i+1} contains {len(chunk_entities)} entities")
            
            # Extract relations from this chunk
            chunk_relations = await self._extract_single(chunk, chunk_entities, doc_id, **kwargs)
            
            if chunk_relations:
                logger.debug(f"Chunk {i+1} yielded {len(chunk_relations)} relations")
                all_relations.extend(chunk_relations)
        
        # Deduplicate relations
        unique_relations = self._deduplicate_relations(all_relations)
        
        logger.info(f"Chunked extraction found {len(all_relations)} total relations, {len(unique_relations)} unique")
        
        return unique_relations
    
    def _create_text_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(end - 200, start)
                sentence_end = -1
                
                for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                    pos = text.rfind(punct, search_start, end)
                    if pos > sentence_end:
                        sentence_end = pos + len(punct)
                
                if sentence_end > start:
                    end = sentence_end
            
            chunks.append(text[start:end])
            
            if end >= len(text):
                break
            
            # Move start position with overlap
            start = end - overlap
        
        return chunks
    
    def _find_entities_in_chunk(self, chunk: str, entities: List[Entity]) -> List[Entity]:
        """Find entities that appear in the given text chunk."""
        chunk_lower = chunk.lower()
        chunk_entities = []
        
        for entity in entities:
            entity_name_lower = entity.name.lower()
            
            # Check if entity name appears in chunk
            if entity_name_lower in chunk_lower:
                chunk_entities.append(entity)
                continue
            
            # Check for partial matches for compound terms
            if len(entity_name_lower) > 10:  # Only for longer entity names
                words = entity_name_lower.split()
                if len(words) > 1:
                    # Check if significant parts of the entity name appear
                    significant_words = [w for w in words if len(w) > 3]
                    if significant_words and all(w in chunk_lower for w in significant_words[:2]):
                        chunk_entities.append(entity)
        
        return chunk_entities
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate relations and self-referencing loops based on source, target, and type."""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # Skip self-referencing relations (loops)
            if relation.source_entity_id == relation.target_entity_id:
                logger.debug(f"Skipping self-referencing relation: {relation.source_entity_id} -> {relation.target_entity_id} ({str(relation.type)})")
                continue
            
            # Create a key for deduplication
            key = (relation.source_entity_id, relation.target_entity_id, str(relation.type))
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        return unique_relations
    
    def _resolve_relations(self, relations: List[Relation], entities: List[Entity]) -> List[Relation]:
        """Resolve entity IDs for relations."""
        entity_name_to_id = {entity.name: entity.id for entity in entities}
        resolved_relations = []

        # Debug: Log all available entities
        logger.info(f"Available entities for resolution ({len(entities)} total):")
        for entity in entities[:10]:  # Log first 10 entities
            logger.info(f"  Entity: '{entity.name}' -> ID: {entity.id}")
        if len(entities) > 10:
            logger.info(f"  ... and {len(entities) - 10} more entities")

        for relation in relations:
            source_name = relation.attributes.get('source_entity_name', relation.source_entity_id)
            target_name = relation.attributes.get('target_entity_name', relation.target_entity_id)

            logger.info(f"Attempting to resolve relation: '{source_name}' -> '{target_name}'")

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
                logger.info(f"✅ Successfully resolved relation: '{source_name}' -> '{target_name}' (IDs: {source_id} -> {target_id})")
            else:
                logger.warning(
                    f"❌ Could not resolve entity IDs for relation: "
                    f"'{source_name}' -> '{target_name}' "
                    f"(source_id: {source_id}, target_id: {target_id})"
                )
                # Debug: Show exact search attempts
                logger.warning(f"   Searched for source: '{relation.source_entity_id}' in {len(entity_name_to_id)} entities")
                logger.warning(f"   Searched for target: '{relation.target_entity_id}' in {len(entity_name_to_id)} entities")

        logger.info(f"Successfully resolved {len(resolved_relations)} out of {len(relations)} relations")
        return resolved_relations
     
    def get_system_prompt(self) -> str:
        """Get the system prompt for relation extraction.
        
        Returns:
            System prompt string
        """
        # Build relation types list dynamically
        relation_types_text = "\n".join([f"- {rel_type}: {description}" for rel_type, description in self.relation_types.items()])
        
        return f"""
You are an expert relation extraction system. Your task is to identify relationships between entities in the given text. Be thorough and comprehensive in finding all possible relationships, especially in medical, scientific, and technical content.

IMPORTANT: Be SPECIFIC about relationship types. NEVER use generic types like "CUSTOM" or "RELATED_TO" unless absolutely necessary.

Available relation types:
{relation_types_text}

RELATIONSHIP TYPE INFERENCE GUIDELINES:
- If an entity is defined by another: use "DEFINED_BY" or "SPECIFIED_BY"
- If an entity is created/published by another: use "CREATED_BY" or "PUBLISHED_BY"
- If an entity is part of another: use "PART_OF" or "COMPONENT_OF"
- If an entity uses/implements another: use "USES" or "IMPLEMENTS"
- If an entity establishes/provides something: use "ESTABLISHES" or "PROVIDES"
- If an entity mandates/requires something: use "MANDATES" or "REQUIRES"
- If an entity facilitates/enables something: use "FACILITATES" or "ENABLES"
- If entities work together: use "WORKS_WITH" or "INTEROPERATES_WITH"
- For standards and protocols: use "COMPLIES_WITH", "FOLLOWS", "BASED_ON"
- For technical relationships: use "COMMUNICATES_WITH", "PROCESSES", "VALIDATES"

For each relation, provide:
1. source_entity: The name of the source entity (exactly as it appears in text)
2. target_entity: The name of the target entity (exactly as it appears in text)
3. relation_type: A specific, descriptive relationship type (infer from context if not in list)
4. context: The specific text that indicates this relationship
5. confidence: A score from 0.0 to 1.0 indicating extraction confidence

Return the results as a JSON array of objects with the following structure:
[
  {{
    "source_entity": "source entity name",
    "target_entity": "target entity name",
    "relation_type": "SPECIFIC_RELATION_TYPE",
    "context": "text that indicates the relationship",
    "confidence": 0.95
  }}
]

Rules:
- Extract ALL relations that are explicitly stated OR reasonably implied in the text
- Be SPECIFIC about relation types - infer the most appropriate type from context
- For technical/scientific content, be especially thorough in identifying relationships
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
                        # Remove document-specific attributes to make relations generic
                        # No longer adding source_text
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
                    # If not a valid enum value, try context-based detection
                    detected_type = self._detect_relation_type_from_context(data.get("context", ""), data["source_entity"], data["target_entity"])
                    if detected_type is not None:
                        relation_type = detected_type
                    else:
                        # Keep the original string if it's descriptive, otherwise use CUSTOM
                        if relation_type.upper() not in ["CUSTOM", "RELATED_TO", "UNKNOWN"] and len(relation_type) > 3:
                            # Keep the descriptive relation type as a string
                            pass
                        else:
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
    
    def _detect_relation_type_from_context(self, context: str, source_entity: str, target_entity: str) -> Optional[RelationType]:
        """Detect relation type based on context patterns."""
        context_lower = context.lower()

        # Definition and specification patterns
        if any(pattern in context_lower for pattern in ["defined by", "defined in", "as defined", "specification"]):
            return RelationType.DEFINED_BY
        elif any(pattern in context_lower for pattern in ["created by", "published by", "issued by", "established by"]):
            return RelationType.CREATED_BY
        elif any(pattern in context_lower for pattern in ["part of", "component of", "element of", "within"]):
            return RelationType.PART_OF
        elif any(pattern in context_lower for pattern in ["uses", "utilizes", "implements", "employs"]):
            return RelationType.USES
        elif any(pattern in context_lower for pattern in ["mandates", "requires", "demands"]):
            return RelationType.MANDATES
        elif any(pattern in context_lower for pattern in ["specifies"]):
            return RelationType.SPECIFIES
        elif any(pattern in context_lower for pattern in ["facilitates", "allows", "supports"]):
            return RelationType.FACILITATES
        elif any(pattern in context_lower for pattern in ["enables"]):
            return RelationType.ENABLES
        elif any(pattern in context_lower for pattern in ["based on", "according to"]):
            return RelationType.BASED_ON
        elif any(pattern in context_lower for pattern in ["follows"]):
            return RelationType.FOLLOWS
        elif any(pattern in context_lower for pattern in ["complies with"]):
            return RelationType.COMPLIES_WITH
        elif any(pattern in context_lower for pattern in ["communicates with", "exchanges", "sends", "receives"]):
            return RelationType.COMMUNICATES_WITH
        elif any(pattern in context_lower for pattern in ["processes", "handles", "manages", "controls"]):
            return RelationType.PROCESSES

        # German context patterns
        elif "rolle war" in context_lower or "spielte" in context_lower:
            return RelationType.PLAYED_ROLE
        elif "darstellte" in context_lower or "verkörperte" in context_lower:
            return RelationType.PORTRAYED
        elif "betrieben habe" in context_lower or "praktiziert" in context_lower:
            return RelationType.PRACTICES
        elif "beschäftigte sich" in context_lower or "engagierte sich" in context_lower:
            return RelationType.ENGAGED_IN
        elif "studierte" in context_lower or "lernte" in context_lower:
            return RelationType.STUDIED

        # English context patterns
        elif "played role" in context_lower or "acted as" in context_lower:
            return RelationType.PLAYED_ROLE
        elif "portrayed" in context_lower or "depicted" in context_lower:
            return RelationType.PORTRAYED
        elif "practices" in context_lower or "engaged in" in context_lower:
            return RelationType.PRACTICES
        elif "studied" in context_lower or "learned" in context_lower:
            return RelationType.STUDIED

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
        missing_entities = {}  # Track entities that need to be created

        for relation in relations:
            # Try to resolve entity names to IDs
            source_name = relation.source_entity_id
            target_name = relation.target_entity_id

            source_id = self._resolve_entity_id(source_name, entity_name_to_id)
            target_id = self._resolve_entity_id(target_name, entity_name_to_id)

            # If we can't resolve an entity, create a placeholder for it
            if not source_id and source_name:
                source_id = self._create_missing_entity_id(source_name, missing_entities)
                logger.info(f"🔧 Created missing entity ID for source: '{source_name}' -> {source_id}")

            if not target_id and target_name:
                target_id = self._create_missing_entity_id(target_name, missing_entities)
                logger.info(f"🔧 Created missing entity ID for target: '{target_name}' -> {target_id}")

            if source_id and target_id:
                relation.source_entity_id = source_id
                relation.target_entity_id = target_id
                relation.source_doc_id = source_doc_id
                relation.attributes["source_text"] = text[:500]  # First 500 chars as context
                resolved_relations.append(relation)
            else:
                logger.warning(
                    f"❌ Could not resolve entity IDs for relation: "
                    f"{relation.attributes.get('source_entity_name')} -> "
                    f"{relation.attributes.get('target_entity_name')} "
                    f"(source_id: {source_id}, target_id: {target_id})"
                )
                logger.warning(f"   Searched for source: '{source_name}' in {len(entity_name_to_id)} entities")
                logger.warning(f"   Searched for target: '{target_name}' in {len(entity_name_to_id)} entities")

        # Log summary of missing entities that were created
        if missing_entities:
            logger.info(f"📝 Created {len(missing_entities)} missing entities during relation resolution:")
            for name, entity_id in missing_entities.items():
                logger.info(f"   '{name}' -> {entity_id}")

        return resolved_relations

    def _create_missing_entity_id(self, entity_name: str, missing_entities: dict) -> str:
        """Create a placeholder entity ID for entities that couldn't be resolved.

        Args:
            entity_name: Name of the missing entity
            missing_entities: Dictionary to track created entities

        Returns:
            Generated entity ID
        """
        if entity_name in missing_entities:
            return missing_entities[entity_name]

        # Generate a consistent ID based on the entity name
        import hashlib
        import re

        # Normalize the name for ID generation
        normalized_name = re.sub(r'[^a-zA-Z0-9\s]', '', entity_name.lower())
        normalized_name = '_'.join(normalized_name.split())

        # Create a short hash for uniqueness
        name_hash = hashlib.md5(entity_name.encode()).hexdigest()[:8]

        # Generate the entity ID
        entity_id = f"ent_{normalized_name}_{name_hash}"

        # Store in missing entities tracker
        missing_entities[entity_name] = entity_id

        return entity_id

    def _resolve_entity_id(
        self,
        entity_name: str,
        entity_name_to_id: Dict[str, str]
    ) -> Optional[str]:
        """Resolve entity name to ID using enhanced fuzzy matching.

        Args:
            entity_name: Name of the entity to resolve
            entity_name_to_id: Mapping of entity names to IDs

        Returns:
            Entity ID if found, None otherwise
        """
        if not entity_name or not entity_name.strip():
            logger.debug(f"Empty entity name provided for resolution")
            return None

        # Normalize entity name (remove extra whitespace, quotes, etc.)
        entity_name = entity_name.strip()
        # Remove common quote variations that LLMs might add
        entity_name = entity_name.strip('"\'""''')
        # Normalize whitespace
        entity_name = ' '.join(entity_name.split())

        logger.debug(f"Resolving entity: '{entity_name}' against {len(entity_name_to_id)} available entities")

        # Create normalized entity mapping for better matching
        normalized_entity_map = {}
        for name, entity_id in entity_name_to_id.items():
            normalized_name = name.strip().strip('"\'""''')
            normalized_name = ' '.join(normalized_name.split())
            normalized_entity_map[normalized_name] = entity_id

        # 1. Exact match (normalized)
        if entity_name in normalized_entity_map:
            logger.info(f"✅ Exact match found for '{entity_name}' -> {normalized_entity_map[entity_name]}")
            return normalized_entity_map[entity_name]

        # 2. Case-insensitive match (normalized)
        entity_name_lower = entity_name.lower()
        for normalized_name, entity_id in normalized_entity_map.items():
            if normalized_name.lower() == entity_name_lower:
                logger.info(f"✅ Case-insensitive match found for '{entity_name}' -> '{normalized_name}' -> {entity_id}")
                return entity_id

        # 3. Partial word matching - check if entity name is contained in any known entity
        # This handles cases like "status word" vs "Status Word" or "Class byte" vs "Class Byte"
        for name, entity_id in entity_name_to_id.items():
            name_lower = name.lower()
            # Check if the search term is contained in the known entity name
            if len(entity_name_lower) >= 3 and entity_name_lower in name_lower:
                logger.info(f"✅ Partial match (contained): '{entity_name}' found in '{name}' -> {entity_id}")
                return entity_id
            # Check if the known entity name is contained in the search term
            if len(name_lower) >= 3 and name_lower in entity_name_lower:
                logger.info(f"✅ Partial match (contains): '{name}' found in '{entity_name}' -> {entity_id}")
                return entity_id

        # 4. Word-by-word matching for compound terms
        entity_words = set(word.lower() for word in entity_name_lower.split() if len(word) >= 3)
        if entity_words:
            for name, entity_id in entity_name_to_id.items():
                name_words = set(word.lower() for word in name.lower().split() if len(word) >= 3)
                if name_words:
                    # Check if significant overlap in words (at least 50% of words match)
                    overlap = entity_words.intersection(name_words)
                    if overlap and len(overlap) >= min(len(entity_words), len(name_words)) * 0.5:
                        logger.info(f"✅ Word overlap match: '{entity_name}' <-> '{name}' (overlap: {overlap}) -> {entity_id}")
                        return entity_id

        # 5. Acronym matching - check if entity name could be an acronym
        if len(entity_name) <= 10 and entity_name.isupper():
            for name, entity_id in entity_name_to_id.items():
                name_words = [word for word in name.split() if len(word) >= 2]
                if len(name_words) >= 2:
                    acronym = ''.join(word[0].upper() for word in name_words)
                    if acronym == entity_name:
                        logger.info(f"✅ Acronym match: '{entity_name}' -> '{name}' (acronym: {acronym}) -> {entity_id}")
                        return entity_id

        # 6. Enhanced fuzzy matching for technical terms
        if len(entity_name_lower) >= 5:
            entity_root = entity_name_lower[:6]  # First 6 chars as root
            for name, entity_id in entity_name_to_id.items():
                if len(name.lower()) >= 5:
                    name_root = name.lower()[:6]
                    if entity_root == name_root:
                        logger.info(f"✅ Root match: '{entity_name}' <-> '{name}' (root: {entity_root}) -> {entity_id}")
                        return entity_id

        # 7. Check for word similarity (enhanced for technical terms)
        for name, entity_id in entity_name_to_id.items():
            if self._are_similar_words(entity_name_lower, name.lower()):
                logger.info(f"✅ Similar words: '{entity_name}' <-> '{name}' -> {entity_id}")
                return entity_id
        
        # Last resort: check if entity_name is a substring of any entity (for very specific terms)
        if len(entity_name_lower) >= 6:  # Only for longer terms to avoid false positives
            for name, entity_id in entity_name_to_id.items():
                if entity_name_lower in name.lower():
                    logger.debug(f"Substring match: '{entity_name}' found in '{name}'")
                    return entity_id

        # Debug: Log failure details with normalized comparison
        logger.warning(f"❌ Failed to resolve entity '{entity_name}'. Normalized search term: '{entity_name}'")
        logger.warning(f"Available normalized entities (first 5):")
        for i, (normalized_name, entity_id) in enumerate(normalized_entity_map.items()):
            if i < 5:  # Show first 5 entities
                logger.warning(f"   '{normalized_name}' -> {entity_id}")
                # Show character-by-character comparison for debugging
                if len(normalized_name) == len(entity_name):
                    differences = [i for i, (c1, c2) in enumerate(zip(entity_name, normalized_name)) if c1 != c2]
                    if differences:
                        logger.warning(f"     Character differences at positions: {differences}")
        if len(normalized_entity_map) > 5:
            logger.warning(f"   ... and {len(normalized_entity_map) - 5} more entities")

        return None
    
    def _are_similar_words(self, word1: str, word2: str) -> bool:
        """Check if two words are similar enough to be considered the same entity.
        Enhanced for technical terminology and common variations.

        Args:
            word1: First word to compare
            word2: Second word to compare

        Returns:
            True if words are similar enough
        """
        # Skip very short words
        if len(word1) < 3 or len(word2) < 3:
            return False

        # Check for common technical word patterns
        def normalize_technical_word(word):
            # Remove common technical suffixes
            suffixes = ['-command', '-byte', '-field', '-key', '-card', '-data', '-word', '-code']
            for suffix in suffixes:
                if word.endswith(suffix):
                    word = word[:-len(suffix)]

            # Remove common prefixes
            prefixes = ['ins ', 'sw', 'p1', 'p2']
            for prefix in prefixes:
                if word.startswith(prefix):
                    word = word[len(prefix):].strip()

            # Remove parentheses and their contents
            import re
            word = re.sub(r'\([^)]*\)', '', word).strip()

            return word

        norm_word1 = normalize_technical_word(word1)
        norm_word2 = normalize_technical_word(word2)

        # Check normalized versions
        if norm_word1 == norm_word2 and len(norm_word1) >= 3:
            return True

        # Check if one normalized word is contained in the other
        if len(norm_word1) >= 4 and len(norm_word2) >= 4:
            if norm_word1 in norm_word2 or norm_word2 in norm_word1:
                return True

        # Check for common technical abbreviations and expansions
        # Handle cases like "PIV" vs "Personal Identity Verification"
        if len(word1) <= 5 and word1.isupper() and len(word2) > 10:
            # Check if word1 could be an acronym of word2
            word2_words = [w for w in word2.split() if len(w) >= 2]
            if len(word2_words) >= 2:
                acronym = ''.join(w[0].upper() for w in word2_words)
                if acronym == word1:
                    return True

        # Reverse check
        if len(word2) <= 5 and word2.isupper() and len(word1) > 10:
            word1_words = [w for w in word1.split() if len(w) >= 2]
            if len(word1_words) >= 2:
                acronym = ''.join(w[0].upper() for w in word1_words)
                if acronym == word2:
                    return True

        # Check if one word is a substring of another with reasonable length
        if len(word1) >= 5 and len(word2) >= 5:
            if word1[:5] == word2[:5]:  # Same first 5 characters
                return True

        # Enhanced edit distance check for technical terms
        if abs(len(word1) - len(word2)) <= 3:  # Allow more length difference
            differences = sum(c1 != c2 for c1, c2 in zip(word1, word2))
            max_len = max(len(word1), len(word2))
            if differences <= max_len * 0.3:  # Allow 30% character differences
                return True

        # Check for common technical compound word patterns
        # Split on common separators and check parts
        separators = ['-', ' ', '.', '_']
        for sep in separators:
            if sep in word1 and sep in word2:
                parts1 = [p.strip() for p in word1.split(sep) if p.strip()]
                parts2 = [p.strip() for p in word2.split(sep) if p.strip()]

                # Check if any significant parts match
                for p1 in parts1:
                    for p2 in parts2:
                        if len(p1) >= 3 and len(p2) >= 3 and p1.lower() == p2.lower():
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
                relation.attributes["source_text"] = text[:500]
            
            return relations
            
        finally:
            # Restore original method
            self.get_user_prompt = original_get_user_prompt