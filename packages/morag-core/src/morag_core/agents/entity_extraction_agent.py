"""Entity extraction agent using Outlines for guaranteed structured output."""

from typing import Type, List, Optional, Dict, Any
import structlog

from ..ai.base_agent import MoRAGBaseAgent, AgentConfig
from ..ai.models import EntityExtractionResult, Entity, ConfidenceLevel

logger = structlog.get_logger(__name__)


class EntityExtractionAgent(MoRAGBaseAgent[EntityExtractionResult]):
    """Entity extraction agent with guaranteed structured output using Outlines."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the entity extraction agent.
        
        Args:
            config: Agent configuration
        """
        if config is None:
            config = AgentConfig(
                model="google-gla:gemini-1.5-flash",
                temperature=0.1,
                outlines_provider="gemini"
            )
        
        super().__init__(config)
        
        # Entity extraction specific configuration
        self.entity_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "PRODUCT", 
            "EVENT", "DATE", "QUANTITY", "TECHNOLOGY", "PROCESS"
        ]
        self.min_confidence = 0.4
        self.include_offsets = True
        self.normalize_entities = True
        self.min_entity_length = 2
    
    def get_result_type(self) -> Type[EntityExtractionResult]:
        """Return the Pydantic model for entity extraction results.
        
        Returns:
            EntityExtractionResult class
        """
        return EntityExtractionResult
    
    def get_system_prompt(self) -> str:
        """Return the system prompt for entity extraction.
        
        Returns:
            The system prompt string
        """
        return f"""You are an expert entity extraction system. Your task is to identify and extract named entities from text with high accuracy.

ENTITY TYPES TO EXTRACT:
{', '.join(self.entity_types)}

EXTRACTION GUIDELINES:
1. Extract entities that are clearly identifiable and meaningful
2. Provide confidence scores between 0.0 and 1.0
3. Include position information when possible
4. Add relevant metadata and context
5. Minimum entity length: {self.min_entity_length} characters
6. Minimum confidence threshold: {self.min_confidence}

OUTPUT FORMAT:
Return a JSON object with the following structure:
- entities: List of extracted entities with name, type, confidence, start_pos, end_pos, context, metadata
- confidence: Overall confidence level (low, medium, high, very_high)
- processing_time: Time taken for processing (optional)
- metadata: Additional extraction metadata

ENTITY STRUCTURE:
Each entity should have:
- name: The exact text mention of the entity
- type: The entity type (use domain-appropriate types)
- confidence: Confidence score (0.0 to 1.0)
- start_pos: Character start position in text (optional)
- end_pos: Character end position in text (optional)
- context: Surrounding context (optional)
- metadata: Additional properties (optional)

IMPORTANT:
- Be precise and avoid over-extraction
- Focus on entities that add semantic value
- Ensure all confidence scores are realistic
- Use domain-appropriate entity types when the standard types don't fit"""

    async def extract_entities(
        self,
        text: str,
        domain: str = "general",
        entity_types: Optional[List[str]] = None,
        min_confidence: Optional[float] = None
    ) -> EntityExtractionResult:
        """Extract entities from text with guaranteed structured output.
        
        Args:
            text: Text to extract entities from
            domain: Domain context for extraction
            entity_types: Optional list of entity types to focus on
            min_confidence: Minimum confidence threshold
            
        Returns:
            EntityExtractionResult with guaranteed structure
        """
        if not text or not text.strip():
            return EntityExtractionResult(
                entities=[],
                total_entities=0,
                confidence=ConfidenceLevel.HIGH,
                metadata={"error": "Empty text", "domain": domain}
            )
        
        # Update configuration for this extraction
        if entity_types:
            self.entity_types = entity_types
        if min_confidence is not None:
            self.min_confidence = min_confidence
        
        self.logger.info(
            "Starting entity extraction with Outlines",
            text_length=len(text),
            domain=domain,
            entity_types=len(self.entity_types),
            structured_generation=self.is_outlines_available()
        )
        
        # Prepare the extraction prompt
        prompt = self._create_extraction_prompt(text, domain)
        
        try:
            # Use structured generation with Outlines
            result = await self.run(prompt)
            
            # Post-process the result
            result = self._post_process_result(result, text, domain)
            
            self.logger.info(
                "Entity extraction completed",
                entities_extracted=result.total_entities,
                confidence=result.confidence,
                used_outlines=self.is_outlines_available()
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Entity extraction failed", error=str(e))
            # Return a fallback result
            return EntityExtractionResult(
                entities=[],
                total_entities=0,
                confidence=ConfidenceLevel.LOW,
                metadata={
                    "error": str(e),
                    "domain": domain,
                    "fallback": True
                }
            )
    
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
        
        self.logger.info(
            "Starting batch entity extraction",
            batch_size=len(texts),
            domain=domain
        )
        
        results = []
        for i, text in enumerate(texts):
            try:
                result = await self.extract_entities(text, domain, entity_types)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                self.logger.warning(
                    "Failed to extract entities from text",
                    text_index=i,
                    error=str(e)
                )
                # Add fallback result
                results.append(EntityExtractionResult(
                    entities=[],
                    total_entities=0,
                    confidence=ConfidenceLevel.LOW,
                    metadata={"error": str(e), "text_index": i}
                ))
        
        self.logger.info(f"Batch entity extraction completed for {len(results)} texts")
        return results
    
    def _create_extraction_prompt(self, text: str, domain: str) -> str:
        """Create the extraction prompt for the given text and domain.
        
        Args:
            text: Text to extract entities from
            domain: Domain context
            
        Returns:
            Formatted prompt string
        """
        return f"""Extract named entities from the following {domain} text:

TEXT:
{text}

FOCUS ON:
- Entity types: {', '.join(self.entity_types)}
- Domain: {domain}
- Minimum confidence: {self.min_confidence}

Extract all relevant entities with their types, confidence scores, and any relevant attributes."""
    
    def _post_process_result(
        self,
        result: EntityExtractionResult,
        original_text: str,
        domain: str
    ) -> EntityExtractionResult:
        """Post-process the extraction result.
        
        Args:
            result: Raw extraction result
            original_text: Original input text
            domain: Domain context
            
        Returns:
            Post-processed result
        """
        # Filter entities by confidence threshold
        filtered_entities = [
            entity for entity in result.entities
            if entity.confidence >= self.min_confidence
        ]
        
        # Update metadata
        metadata = result.metadata or {}
        metadata.update({
            "domain": domain,
            "original_entity_count": len(result.entities),
            "filtered_entity_count": len(filtered_entities),
            "text_length": len(original_text),
            "min_confidence_threshold": self.min_confidence,
            "extraction_method": "outlines" if self.is_outlines_available() else "fallback"
        })
        
        # Create updated result
        return EntityExtractionResult(
            entities=filtered_entities,
            total_entities=len(filtered_entities),
            confidence=result.confidence,
            metadata=metadata
        )
