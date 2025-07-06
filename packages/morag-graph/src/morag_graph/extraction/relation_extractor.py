"""PydanticAI-based relation extraction."""

import structlog
from typing import List, Optional, Dict, Any

from ..models import Entity, Relation
from ..ai import RelationExtractionAgent

logger = structlog.get_logger(__name__)


class RelationExtractor:
    """PydanticAI-based relation extractor - completely new implementation."""

    def __init__(self, config=None, min_confidence: float = 0.6, chunk_size: int = 3000, dynamic_types: bool = True, relation_types: Optional[Dict[str, str]] = None, language: Optional[str] = None, **kwargs):
        """Initialize the relation extractor.

        Args:
            config: LLMConfig object or dict with LLM configuration (for compatibility with tests)
            min_confidence: Minimum confidence threshold for relations
            chunk_size: Maximum characters per chunk for large texts
            dynamic_types: Whether to use dynamic relation types (LLM-determined)
            relation_types: Custom relation types dict (type_name -> description). If None and dynamic_types=True, uses pure dynamic mode
            language: Language code for processing (e.g., 'en', 'de', 'fr')
            **kwargs: Additional arguments passed to the agent
        """
        self.min_confidence = min_confidence
        self.chunk_size = chunk_size
        self.dynamic_types = dynamic_types
        self.relation_types = relation_types or {}
        self.language = language

        # Handle config parameter for test compatibility
        if config is not None:
            # If config is provided, it might be an LLMConfig object or dict
            if hasattr(config, 'provider'):
                # It's an LLMConfig object, convert to dict
                llm_config = {
                    'provider': config.provider,
                    'model': config.model,
                    'api_key': getattr(config, 'api_key', None),
                    'temperature': getattr(config, 'temperature', 0.1),
                    'max_tokens': getattr(config, 'max_tokens', None)
                }
                kwargs['llm_config'] = llm_config

        # Convert llm_config dict to proper agent configuration
        agent_kwargs = {}
        if 'llm_config' in kwargs:
            llm_config = kwargs.pop('llm_config')
            if isinstance(llm_config, dict):
                # Import here to avoid circular imports
                from morag_core.ai import AgentConfig
                from morag_core.ai.providers import ProviderConfig, GeminiProvider

                # Create provider config
                provider_config = ProviderConfig(
                    api_key=llm_config.get('api_key'),
                    timeout=llm_config.get('timeout', 30),
                    max_retries=llm_config.get('max_retries', 3)
                )

                # Create agent config
                agent_config = AgentConfig(
                    model=f"google-gla:{llm_config.get('model', 'gemini-1.5-flash')}",
                    timeout=llm_config.get('timeout', 30),
                    max_retries=llm_config.get('max_retries', 3),
                    temperature=llm_config.get('temperature', 0.1),
                    max_tokens=llm_config.get('max_tokens'),
                    provider_config=provider_config
                )

                # Create provider
                provider = GeminiProvider(provider_config)

                agent_kwargs['config'] = agent_config
                agent_kwargs['provider'] = provider

        # Add any remaining kwargs
        agent_kwargs.update(kwargs)

        self.agent = RelationExtractionAgent(
            min_confidence=min_confidence,
            dynamic_types=self.dynamic_types,
            relation_types=self.relation_types,
            language=self.language,
            **agent_kwargs
        )
        self.logger = logger.bind(component="relation_extractor")

    def get_system_prompt(self) -> str:
        """Get the system prompt used by the agent."""
        return self.agent.get_system_prompt()
    
    async def extract(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
        doc_id: Optional[str] = None,
        source_doc_id: Optional[str] = None,
        **kwargs
    ) -> List[Relation]:
        """Extract relations from text using PydanticAI agent.
        
        Args:
            text: Text to extract relations from
            entities: Optional list of known entities
            doc_id: Optional document ID (deprecated, use source_doc_id)
            source_doc_id: Optional document ID to associate with relations
            **kwargs: Additional arguments
            
        Returns:
            List of Relation objects
        """
        # Handle backward compatibility
        if source_doc_id is None and doc_id is not None:
            source_doc_id = doc_id
        
        try:
            # Use PydanticAI agent for extraction
            relations = await self.agent.extract_relations(
                text=text,
                entities=entities,
                chunk_size=self.chunk_size,
                source_doc_id=source_doc_id
            )
            
            self.logger.info(
                "Relation extraction completed",
                text_length=len(text),
                num_entities=len(entities) if entities else 0,
                relations_found=len(relations),
                source_doc_id=source_doc_id
            )
            
            return relations
            
        except Exception as e:
            self.logger.error(
                "Relation extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text)
            )
            raise
    
    async def extract_relations(
        self,
        text: str,
        entities: List[Entity],
        source_doc_id: Optional[str] = None,
        **kwargs
    ) -> List[Relation]:
        """Extract relations from text (alias for extract method).
        
        Args:
            text: Text to extract relations from
            entities: List of entities to consider for relations
            source_doc_id: Optional document ID to associate with relations
            **kwargs: Additional arguments
            
        Returns:
            List of extracted relations
        """
        return await self.extract(text, entities=entities, source_doc_id=source_doc_id, **kwargs)
    
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
        return await self.extract(text, entities=entities, source_doc_id=source_doc_id)
    
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
        # Convert entity pairs to a flat list of entities
        entities = []
        for source_entity, target_entity in entity_pairs:
            if source_entity not in entities:
                entities.append(source_entity)
            if target_entity not in entities:
                entities.append(target_entity)
        
        # Extract relations and filter for the specific pairs
        all_relations = await self.extract(text, entities=entities, source_doc_id=source_doc_id)
        
        # Filter relations to only include the specified pairs
        pair_ids = set()
        for source_entity, target_entity in entity_pairs:
            pair_ids.add((source_entity.id, target_entity.id))
            pair_ids.add((target_entity.id, source_entity.id))  # Include reverse direction
        
        filtered_relations = []
        for relation in all_relations:
            if (relation.source_entity_id, relation.target_entity_id) in pair_ids:
                filtered_relations.append(relation)
        
        return filtered_relations
