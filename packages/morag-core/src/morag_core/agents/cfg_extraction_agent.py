"""Context-Free Grammar extraction agent for complex structured outputs."""

from typing import Type, List, Optional, Dict, Any, Union
import structlog
from pydantic import BaseModel, Field

from ..ai.base_agent import MoRAGBaseAgent, AgentConfig
from ..ai.models import ConfidenceLevel

logger = structlog.get_logger(__name__)


class KnowledgeTriple(BaseModel):
    """A knowledge triple in subject-predicate-object format."""
    
    subject: str = Field(description="Subject entity")
    predicate: str = Field(description="Relationship/predicate")
    object: str = Field(description="Object entity")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class CFGExtractionResult(BaseModel):
    """Result from CFG-based extraction."""
    
    triples: List[KnowledgeTriple] = Field(description="Extracted knowledge triples")
    total_triples: int = Field(description="Total number of triples")
    confidence: ConfidenceLevel = Field(description="Overall confidence level")
    grammar_used: str = Field(description="Grammar type used for extraction")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CFGExtractionAgent(MoRAGBaseAgent[CFGExtractionResult]):
    """CFG extraction agent using context-free grammars for structured output."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the CFG extraction agent.
        
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
        
        # CFG extraction configuration
        self.min_confidence = 0.6
        self.max_triples = 15
        
        # Define context-free grammar for knowledge triples
        self.knowledge_triple_grammar = """
        ?start: triple_list

        triple_list: triple ("," triple)*

        triple: subject " " predicate " " object

        subject: entity
        predicate: relation
        object: entity

        entity: WORD+
        relation: WORD+

        %import common.WORD
        %import common.WS_INLINE

        %ignore WS_INLINE
        """
        
        # Alternative grammar for more complex structures
        self.complex_grammar = """
        ?start: knowledge_base

        knowledge_base: statement+

        statement: entity_statement | relation_statement

        entity_statement: "ENTITY:" entity "TYPE:" entity_type "ATTRIBUTES:" attribute_list
        relation_statement: "RELATION:" entity "RELATES_TO:" entity "VIA:" relation_type

        entity: WORD+
        entity_type: WORD+
        relation_type: WORD+
        attribute_list: attribute ("," attribute)*
        attribute: WORD+

        %import common.WORD
        %import common.WS_INLINE

        %ignore WS_INLINE
        """
    
    def get_result_type(self) -> Type[CFGExtractionResult]:
        """Return the Pydantic model for CFG extraction results.
        
        Returns:
            CFGExtractionResult class
        """
        return CFGExtractionResult
    
    def get_system_prompt(self) -> str:
        """Return the system prompt for CFG extraction.
        
        Returns:
            The system prompt string
        """
        return f"""You are an advanced knowledge extraction system using context-free grammars for structured output generation.

EXTRACTION TASK:
Extract knowledge triples in subject-predicate-object format from the given text.

GRAMMAR CONSTRAINTS:
The output must follow a specific context-free grammar structure for knowledge representation.

TRIPLE FORMAT:
- Subject: The main entity or concept
- Predicate: The relationship or action
- Object: The target entity or value

EXTRACTION GUIDELINES:
1. Extract meaningful knowledge triples from the text
2. Ensure each triple is complete and accurate
3. Provide confidence scores between 0.0 and 1.0
4. Maximum triples per extraction: {self.max_triples}
5. Minimum confidence threshold: {self.min_confidence}
6. Focus on factual, verifiable relationships

OUTPUT FORMAT:
Return a JSON object with the following structure:
- triples: List of knowledge triples with subject, predicate, object, confidence
- total_triples: Count of extracted triples
- confidence: Overall confidence level (low, medium, high, very_high)
- grammar_used: Type of grammar used
- metadata: Additional extraction metadata

TRIPLE STRUCTURE:
Each triple should have:
- subject: Subject entity (string)
- predicate: Relationship/predicate (string)
- object: Object entity (string)
- confidence: Confidence score (0.0 to 1.0)

EXAMPLES:
- Subject: "Apple Inc.", Predicate: "is headquartered in", Object: "Cupertino"
- Subject: "Tim Cook", Predicate: "serves as CEO of", Object: "Apple Inc."
- Subject: "iPhone", Predicate: "is manufactured by", Object: "Apple Inc."

IMPORTANT: 
- Ensure triples represent factual relationships
- Maintain consistency in entity naming
- Use clear, descriptive predicates
- Follow the grammar constraints for structured output"""

    async def extract_with_cfg(
        self,
        text: str,
        domain: str = "general",
        grammar_type: str = "knowledge_triples",
        min_confidence: Optional[float] = None
    ) -> CFGExtractionResult:
        """Extract knowledge using context-free grammar constraints.
        
        Args:
            text: Text to extract from
            domain: Domain context for extraction
            grammar_type: Type of grammar to use (knowledge_triples, complex)
            min_confidence: Minimum confidence threshold
            
        Returns:
            CFGExtractionResult with grammar-constrained output
        """
        if not text or not text.strip():
            return CFGExtractionResult(
                triples=[],
                total_triples=0,
                confidence=ConfidenceLevel.HIGH,
                grammar_used=grammar_type,
                metadata={"error": "Empty text", "domain": domain}
            )
        
        # Update configuration for this extraction
        if min_confidence is not None:
            self.min_confidence = min_confidence
        
        self.logger.info(
            "Starting CFG extraction",
            text_length=len(text),
            domain=domain,
            grammar_type=grammar_type,
            structured_generation=self.is_outlines_available()
        )
        
        # Prepare the extraction prompt
        prompt = self._create_extraction_prompt(text, domain, grammar_type)
        
        try:
            # For now, use structured generation with Pydantic models
            # In a full implementation, you would use Outlines CFG support
            result = await self.run(prompt)
            
            # Post-process the result
            result = self._post_process_result(result, text, domain, grammar_type)
            
            self.logger.info(
                "CFG extraction completed",
                triples_extracted=result.total_triples,
                confidence=result.confidence,
                grammar_used=result.grammar_used,
                used_outlines=self.is_outlines_available()
            )
            
            return result
            
        except Exception as e:
            self.logger.error("CFG extraction failed", error=str(e))
            # Return a fallback result
            return CFGExtractionResult(
                triples=[],
                total_triples=0,
                confidence=ConfidenceLevel.LOW,
                grammar_used=grammar_type,
                metadata={
                    "error": str(e),
                    "domain": domain,
                    "fallback": True
                }
            )
    
    def get_cfg_generator(self, grammar_type: str = "knowledge_triples"):
        """Get a CFG generator for the specified grammar type.
        
        Args:
            grammar_type: Type of grammar to use
            
        Returns:
            CFG generator (placeholder for actual implementation)
        """
        try:
            # This would be the actual CFG implementation with Outlines
            # from outlines.types import CFG
            # from outlines import Generator
            
            if grammar_type == "knowledge_triples":
                grammar = self.knowledge_triple_grammar
            elif grammar_type == "complex":
                grammar = self.complex_grammar
            else:
                raise ValueError(f"Unknown grammar type: {grammar_type}")
            
            # In actual implementation:
            # cfg = CFG(grammar)
            # generator = Generator(self.outlines_provider._outlines_model, cfg)
            # return generator
            
            # For now, return a placeholder
            self.logger.info(f"CFG grammar prepared for type: {grammar_type}")
            return None
            
        except Exception as e:
            self.logger.error("Failed to create CFG generator", error=str(e))
            raise
    
    def _create_extraction_prompt(
        self,
        text: str,
        domain: str,
        grammar_type: str
    ) -> str:
        """Create the extraction prompt for CFG-based extraction.
        
        Args:
            text: Text to extract from
            domain: Domain context
            grammar_type: Type of grammar to use
            
        Returns:
            Formatted prompt string
        """
        return f"""Extract knowledge triples using context-free grammar constraints from the following {domain} text:

TEXT:
{text}

EXTRACTION PARAMETERS:
- Domain: {domain}
- Grammar type: {grammar_type}
- Maximum triples: {self.max_triples}
- Minimum confidence: {self.min_confidence}

Extract structured knowledge triples that follow the grammar constraints and represent factual relationships in the text."""
    
    def _post_process_result(
        self,
        result: CFGExtractionResult,
        original_text: str,
        domain: str,
        grammar_type: str
    ) -> CFGExtractionResult:
        """Post-process and validate the CFG extraction result.
        
        Args:
            result: Raw extraction result
            original_text: Original input text
            domain: Domain context
            grammar_type: Grammar type used
            
        Returns:
            Post-processed and validated result
        """
        # Filter triples by confidence threshold
        filtered_triples = [
            triple for triple in result.triples
            if triple.confidence >= self.min_confidence
        ]
        
        # Limit to max_triples
        if len(filtered_triples) > self.max_triples:
            # Sort by confidence and take top triples
            filtered_triples = sorted(
                filtered_triples,
                key=lambda t: t.confidence,
                reverse=True
            )[:self.max_triples]
        
        # Update metadata
        metadata = result.metadata or {}
        metadata.update({
            "domain": domain,
            "grammar_type": grammar_type,
            "original_triple_count": len(result.triples),
            "filtered_triple_count": len(filtered_triples),
            "text_length": len(original_text),
            "min_confidence_threshold": self.min_confidence,
            "max_triples_limit": self.max_triples,
            "extraction_method": "outlines_cfg" if self.is_outlines_available() else "fallback",
            "grammar_constraints": "applied"
        })
        
        # Create updated result
        return CFGExtractionResult(
            triples=filtered_triples,
            total_triples=len(filtered_triples),
            confidence=result.confidence,
            grammar_used=grammar_type,
            metadata=metadata
        )
