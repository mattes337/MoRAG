"""Advanced extraction agent with regex constraints and structured formats."""

from typing import Type, List, Optional, Dict, Any, Union
import structlog
from pydantic import BaseModel, Field

from ..ai.base_agent import MoRAGBaseAgent, AgentConfig
from ..ai.models import ConfidenceLevel

logger = structlog.get_logger(__name__)


class StructuredEntity(BaseModel):
    """Entity with structured ID format."""
    
    entity_id: str = Field(
        description="Unique entity ID in format: ent_[name]_[8-char-hex]",
        pattern=r"^ent_[a-zA-Z0-9_]+_[a-f0-9]{8}$"
    )
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional attributes")


class StructuredRelation(BaseModel):
    """Relation with structured ID format."""
    
    relation_id: str = Field(
        description="Unique relation ID in format: rel_[type]_[8-char-hex]",
        pattern=r"^rel_[a-zA-Z0-9_]+_[a-f0-9]{8}$"
    )
    source_entity_id: str = Field(
        description="Source entity ID",
        pattern=r"^ent_[a-zA-Z0-9_]+_[a-f0-9]{8}$"
    )
    target_entity_id: str = Field(
        description="Target entity ID", 
        pattern=r"^ent_[a-zA-Z0-9_]+_[a-f0-9]{8}$"
    )
    relation_type: str = Field(description="Type of relation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    description: str = Field(description="Relation description")


class AdvancedExtractionResult(BaseModel):
    """Result from advanced extraction with structured IDs."""
    
    entities: List[StructuredEntity] = Field(description="Extracted entities with structured IDs")
    relations: List[StructuredRelation] = Field(description="Extracted relations with structured IDs")
    total_entities: int = Field(description="Total number of entities")
    total_relations: int = Field(description="Total number of relations")
    confidence: ConfidenceLevel = Field(description="Overall confidence level")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AdvancedExtractionAgent(MoRAGBaseAgent[AdvancedExtractionResult]):
    """Advanced extraction agent with regex constraints for structured IDs."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the advanced extraction agent.
        
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
        
        # Advanced extraction configuration
        self.entity_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "PRODUCT", 
            "EVENT", "DATE", "QUANTITY", "TECHNOLOGY", "PROCESS"
        ]
        self.relation_types = [
            "SUPPORTS", "ELABORATES", "CONTRADICTS", "SEQUENCE", "COMPARISON",
            "CAUSATION", "PREREQUISITE", "ALTERNATIVE", "HIERARCHY", "LOCATED_IN",
            "PART_OF", "CREATED_BY", "RELATED_TO"
        ]
        self.min_confidence = 0.6
    
    def get_result_type(self) -> Type[AdvancedExtractionResult]:
        """Return the Pydantic model for advanced extraction results.
        
        Returns:
            AdvancedExtractionResult class
        """
        return AdvancedExtractionResult
    
    def get_system_prompt(self) -> str:
        """Return the system prompt for advanced extraction.
        
        Returns:
            The system prompt string
        """
        return f"""You are an advanced entity and relation extraction system with structured ID generation capabilities.

ENTITY TYPES TO EXTRACT:
{', '.join(self.entity_types)}

RELATION TYPES TO EXTRACT:
{', '.join(self.relation_types)}

STRUCTURED ID REQUIREMENTS:
1. Entity IDs must follow format: ent_[name]_[8-char-hex]
   - Example: ent_Apple_Inc_a1b2c3d4
   - Use underscores for spaces in names
   - Hex part should be lowercase a-f and 0-9

2. Relation IDs must follow format: rel_[type]_[8-char-hex]
   - Example: rel_LOCATED_IN_f4e3d2c1
   - Use relation type in uppercase
   - Hex part should be lowercase a-f and 0-9

EXTRACTION GUIDELINES:
1. Extract entities and relations with high accuracy
2. Generate unique, consistent IDs for each entity and relation
3. Ensure all IDs follow the exact regex patterns
4. Provide confidence scores between 0.0 and 1.0
5. Minimum confidence threshold: {self.min_confidence}
6. Link relations to entities using their structured IDs

OUTPUT FORMAT:
Return a JSON object with the following structure:
- entities: List of entities with entity_id, name, type, confidence, attributes
- relations: List of relations with relation_id, source_entity_id, target_entity_id, relation_type, confidence, description
- total_entities: Count of entities
- total_relations: Count of relations
- confidence: Overall confidence level (low, medium, high, very_high)
- processing_time: Processing time (optional)
- metadata: Additional metadata

IMPORTANT CONSTRAINTS:
- All entity_id fields MUST match pattern: ^ent_[a-zA-Z0-9_]+_[a-f0-9]{{8}}$
- All relation_id fields MUST match pattern: ^rel_[a-zA-Z0-9_]+_[a-f0-9]{{8}}$
- All source_entity_id and target_entity_id MUST reference valid entity IDs
- Generate realistic 8-character hex suffixes (e.g., a1b2c3d4, f4e3d2c1)
- Ensure ID uniqueness within the extraction"""

    async def extract_with_structured_ids(
        self,
        text: str,
        domain: str = "general",
        min_confidence: Optional[float] = None
    ) -> AdvancedExtractionResult:
        """Extract entities and relations with structured IDs using regex constraints.
        
        Args:
            text: Text to extract from
            domain: Domain context for extraction
            min_confidence: Minimum confidence threshold
            
        Returns:
            AdvancedExtractionResult with structured IDs
        """
        if not text or not text.strip():
            return AdvancedExtractionResult(
                entities=[],
                relations=[],
                total_entities=0,
                total_relations=0,
                confidence=ConfidenceLevel.HIGH,
                metadata={"error": "Empty text", "domain": domain}
            )
        
        # Update configuration for this extraction
        if min_confidence is not None:
            self.min_confidence = min_confidence
        
        self.logger.info(
            "Starting advanced extraction with structured IDs",
            text_length=len(text),
            domain=domain,
            structured_generation=self.is_outlines_available()
        )
        
        # Prepare the extraction prompt
        prompt = self._create_extraction_prompt(text, domain)
        
        try:
            # Use structured generation with Outlines
            result = await self.run(prompt)
            
            # Post-process and validate the result
            result = self._post_process_result(result, text, domain)
            
            self.logger.info(
                "Advanced extraction completed",
                entities_extracted=result.total_entities,
                relations_extracted=result.total_relations,
                confidence=result.confidence,
                used_outlines=self.is_outlines_available()
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Advanced extraction failed", error=str(e))
            # Return a fallback result
            return AdvancedExtractionResult(
                entities=[],
                relations=[],
                total_entities=0,
                total_relations=0,
                confidence=ConfidenceLevel.LOW,
                metadata={
                    "error": str(e),
                    "domain": domain,
                    "fallback": True
                }
            )
    
    def _create_extraction_prompt(self, text: str, domain: str) -> str:
        """Create the extraction prompt for the given text and domain.
        
        Args:
            text: Text to extract from
            domain: Domain context
            
        Returns:
            Formatted prompt string
        """
        return f"""Extract entities and relations with structured IDs from the following {domain} text:

TEXT:
{text}

EXTRACTION PARAMETERS:
- Domain: {domain}
- Minimum confidence: {self.min_confidence}
- Generate structured IDs following the exact patterns
- Ensure all relations reference valid entity IDs

Extract high-quality entities and relations with properly formatted structured IDs."""
    
    def _post_process_result(
        self,
        result: AdvancedExtractionResult,
        original_text: str,
        domain: str
    ) -> AdvancedExtractionResult:
        """Post-process and validate the extraction result.
        
        Args:
            result: Raw extraction result
            original_text: Original input text
            domain: Domain context
            
        Returns:
            Post-processed and validated result
        """
        import re
        
        # Validate entity IDs
        entity_id_pattern = re.compile(r"^ent_[a-zA-Z0-9_]+_[a-f0-9]{8}$")
        valid_entities = []
        valid_entity_ids = set()
        
        for entity in result.entities:
            if entity_id_pattern.match(entity.entity_id) and entity.confidence >= self.min_confidence:
                valid_entities.append(entity)
                valid_entity_ids.add(entity.entity_id)
            else:
                self.logger.warning(
                    "Invalid entity ID or low confidence",
                    entity_id=entity.entity_id,
                    confidence=entity.confidence
                )
        
        # Validate relation IDs and references
        relation_id_pattern = re.compile(r"^rel_[a-zA-Z0-9_]+_[a-f0-9]{8}$")
        valid_relations = []
        
        for relation in result.relations:
            if (relation_id_pattern.match(relation.relation_id) and
                relation.source_entity_id in valid_entity_ids and
                relation.target_entity_id in valid_entity_ids and
                relation.confidence >= self.min_confidence):
                valid_relations.append(relation)
            else:
                self.logger.warning(
                    "Invalid relation ID, entity references, or low confidence",
                    relation_id=relation.relation_id,
                    source_id=relation.source_entity_id,
                    target_id=relation.target_entity_id,
                    confidence=relation.confidence
                )
        
        # Update metadata
        metadata = result.metadata or {}
        metadata.update({
            "domain": domain,
            "original_entity_count": len(result.entities),
            "valid_entity_count": len(valid_entities),
            "original_relation_count": len(result.relations),
            "valid_relation_count": len(valid_relations),
            "text_length": len(original_text),
            "min_confidence_threshold": self.min_confidence,
            "extraction_method": "outlines" if self.is_outlines_available() else "fallback",
            "id_validation": "regex_enforced"
        })
        
        # Create updated result
        return AdvancedExtractionResult(
            entities=valid_entities,
            relations=valid_relations,
            total_entities=len(valid_entities),
            total_relations=len(valid_relations),
            confidence=result.confidence,
            processing_time=result.processing_time,
            metadata=metadata
        )
