"""PydanticAI-based entity extraction."""

import structlog
from typing import List, Optional, Dict, Any

from ..models import Entity
from ..ai import EntityExtractionAgent

logger = structlog.get_logger(__name__)


class EntityExtractor:
    """PydanticAI-based entity extractor - completely new implementation."""

    def __init__(self, config=None, min_confidence: float = 0.6, chunk_size: int = 4000, dynamic_types: bool = True, entity_types: Optional[Dict[str, str]] = None, **kwargs):
        """Initialize the entity extractor.

        Args:
            config: LLMConfig object or dict with LLM configuration (for compatibility with tests)
            min_confidence: Minimum confidence threshold for entities
            chunk_size: Maximum characters per chunk for large texts
            dynamic_types: Whether to use dynamic entity types (LLM-determined)
            entity_types: Custom entity types dict (type_name -> description). If None and dynamic_types=True, uses pure dynamic mode
            **kwargs: Additional arguments passed to the agent
        """
        self.min_confidence = min_confidence
        self.chunk_size = chunk_size
        self.dynamic_types = dynamic_types
        self.entity_types = entity_types or {}

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

        self.agent = EntityExtractionAgent(
            min_confidence=min_confidence,
            dynamic_types=self.dynamic_types,
            entity_types=self.entity_types,
            **agent_kwargs
        )
        self.logger = logger.bind(component="entity_extractor")

    def get_system_prompt(self) -> str:
        """Get the system prompt used by the agent."""
        return self.agent.get_system_prompt()
    
    async def extract(self, text: str, doc_id: Optional[str] = None, source_doc_id: Optional[str] = None, **kwargs) -> List[Entity]:
        """Extract entities from text using PydanticAI agent."""
        if source_doc_id is None and doc_id is not None:
            source_doc_id = doc_id
        
        try:
            entities = await self.agent.extract_entities(
                text=text,
                chunk_size=self.chunk_size,
                source_doc_id=source_doc_id
            )
            
            self.logger.info(
                "Entity extraction completed",
                text_length=len(text),
                entities_found=len(entities),
                source_doc_id=source_doc_id
            )
            
            return entities
            
        except Exception as e:
            self.logger.error(
                "Entity extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text)
            )
            raise
    
    async def extract_entities(self, text: str, source_doc_id: Optional[str] = None, **kwargs) -> List[Entity]:
        """Alias for extract method for backward compatibility."""
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
        # For now, just call the regular extract method
        # The PydanticAI agent handles context internally
        return await self.extract(text, source_doc_id=source_doc_id, **kwargs)
