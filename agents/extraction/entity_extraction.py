"""Entity extraction agent."""

from typing import Type, List, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate, PromptExample
from .models import EntityExtractionResult, ExtractedEntity, EntityType, ConfidenceLevel

logger = structlog.get_logger(__name__)


class EntityExtractionAgent(BaseAgent[EntityExtractionResult]):
    """Agent specialized for extracting named entities from text."""
    
    def _get_default_config(self) -> AgentConfig:
        """Get default configuration for entity extraction."""
        return AgentConfig(
            name="entity_extraction",
            description="Extracts named entities from text content",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                include_context=True,
                output_format="json",
                strict_json=True,
                include_confidence=True,
                min_confidence=0.6,
            ),
            agent_config={
                "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "PRODUCT", "EVENT", "DATE", "QUANTITY", "TECHNOLOGY", "PROCESS"],
                "include_offsets": True,
                "normalize_entities": True,
                "min_entity_length": 2,
            }
        )
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        """Create the prompt template for entity extraction."""
        
        system_prompt = """You are an expert entity extraction agent. Your task is to identify and extract named entities from text with high precision.

## Your Role
Extract entities that are:
- Named entities (proper nouns, specific concepts)
- Technically significant terms
- Domain-specific terminology
- Measurable quantities and specifications

## Entity Types
Focus on these entity types:
{% for entity_type in config.agent_config.get('entity_types', ['PERSON', 'ORGANIZATION', 'LOCATION', 'CONCEPT']) %}
- **{{ entity_type }}**: {{ entity_type.lower().replace('_', ' ') }}
{% endfor %}

## Entity Structure
For each entity, provide:
- **name**: Exact text mention as it appears
- **canonical_name**: Normalized/standardized form
- **entity_type**: One of the supported types
- **confidence**: 0.0-1.0 confidence score
- **attributes**: Additional properties (synonyms, descriptions, etc.)
{% if config.agent_config.get('include_offsets', True) %}
- **start_offset**: Character position where entity starts
- **end_offset**: Character position where entity ends
{% endif %}
- **context**: Surrounding text for disambiguation

## Quality Standards
- Minimum entity length: {{ config.agent_config.get('min_entity_length', 2) }} characters
- Focus on domain-specific and technical terms
- Avoid common words unless they're proper nouns
- Normalize similar entities to canonical forms

{% if config.include_examples %}
{{ examples }}
{% endif %}

{{ output_requirements }}"""

        user_prompt = """Extract named entities from the following text:

Text: {{ input }}

{% if domain and domain != 'general' %}
Focus on {{ domain }}-specific entities and terminology.
{% endif %}

Return a JSON object with the following structure:
{
  "entities": [
    {
      "name": "exact text mention",
      "canonical_name": "normalized form",
      "entity_type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|PRODUCT|EVENT|DATE|QUANTITY|TECHNOLOGY|PROCESS",
      "confidence": 0.0-1.0,
      "attributes": {"key": "value"},
      {% if config.agent_config.get('include_offsets', True) %}
      "start_offset": 0,
      "end_offset": 10,
      {% endif %}
      "context": "surrounding text"
    }
  ],
  "total_entities": 0,
  "confidence": "low|medium|high|very_high",
  "metadata": {}
}"""

        template = ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
        
        # Add examples
        examples = [
            PromptExample(
                input="Apple Inc. announced a partnership with Stanford University to research machine learning applications in healthcare.",
                output="""{
  "entities": [
    {
      "name": "Apple Inc.",
      "canonical_name": "Apple Inc.",
      "entity_type": "ORGANIZATION",
      "confidence": 0.95,
      "attributes": {"type": "technology_company", "industry": "technology"},
      "start_offset": 0,
      "end_offset": 10,
      "context": "Apple Inc. announced a partnership"
    },
    {
      "name": "Stanford University",
      "canonical_name": "Stanford University",
      "entity_type": "ORGANIZATION",
      "confidence": 0.95,
      "attributes": {"type": "university", "sector": "education"},
      "start_offset": 45,
      "end_offset": 63,
      "context": "partnership with Stanford University to research"
    },
    {
      "name": "machine learning",
      "canonical_name": "Machine Learning",
      "entity_type": "TECHNOLOGY",
      "confidence": 0.9,
      "attributes": {"field": "artificial_intelligence", "domain": "computer_science"},
      "start_offset": 76,
      "end_offset": 92,
      "context": "research machine learning applications"
    },
    {
      "name": "healthcare",
      "canonical_name": "Healthcare",
      "entity_type": "CONCEPT",
      "confidence": 0.85,
      "attributes": {"domain": "medical", "sector": "health"},
      "start_offset": 109,
      "end_offset": 119,
      "context": "applications in healthcare"
    }
  ],
  "total_entities": 4,
  "confidence": "high",
  "metadata": {"extraction_method": "llm"}
}""",
                explanation="Extracted organizations, technology concepts, and domain-specific terms with proper normalization."
            )
        ]
        
        template.get_examples = lambda: examples
        return template
    
    def get_result_type(self) -> Type[EntityExtractionResult]:
        """Get the result type for entity extraction."""
        return EntityExtractionResult
    
    async def extract_entities(
        self,
        text: str,
        domain: str = "general",
        entity_types: Optional[List[str]] = None
    ) -> EntityExtractionResult:
        """Extract entities from text.
        
        Args:
            text: Text to extract entities from
            domain: Domain context for extraction
            entity_types: Optional list of entity types to focus on
            
        Returns:
            Entity extraction result
        """
        if not text or not text.strip():
            return EntityExtractionResult(
                entities=[],
                total_entities=0,
                confidence=ConfidenceLevel.HIGH,
                metadata={"error": "Empty text"}
            )
        
        # Update config for this extraction
        if entity_types:
            self.config.set_agent_config("entity_types", entity_types)
        
        self.logger.info(
            "Starting entity extraction",
            text_length=len(text),
            domain=domain,
            entity_types=entity_types
        )
        
        try:
            result = await self.execute(
                text,
                domain=domain
            )
            
            self.logger.info(
                "Entity extraction completed",
                entities_extracted=result.total_entities,
                confidence=result.confidence
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Entity extraction failed", error=str(e))
            raise
    
    async def extract_entities_batch(
        self,
        texts: List[str],
        domain: str = "general",
        entity_types: Optional[List[str]] = None
    ) -> List[EntityExtractionResult]:
        """Extract entities from multiple texts.
        
        Args:
            texts: List of texts to process
            domain: Domain context
            entity_types: Optional entity types to focus on
            
        Returns:
            List of entity extraction results
        """
        if not texts:
            return []
        
        self.logger.info(f"Starting batch entity extraction for {len(texts)} texts")
        
        # Update config if needed
        if entity_types:
            self.config.set_agent_config("entity_types", entity_types)
        
        results = await self.batch_execute(
            texts,
            domain=domain
        )
        
        self.logger.info(f"Batch entity extraction completed for {len(results)} texts")
        return results
