"""Fact extraction agent."""

from typing import Type, List, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import FactExtractionResult, ExtractedFact, ConfidenceLevel

logger = structlog.get_logger(__name__)


class FactExtractionAgent(BaseAgent[FactExtractionResult]):
    """Agent specialized for extracting structured facts from text."""
    
    def _get_default_config(self) -> AgentConfig:
        """Get default configuration for fact extraction."""
        return AgentConfig(
            name="fact_extraction",
            description="Extracts structured facts from text content",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                include_context=True,
                output_format="json",
                strict_json=True,
                include_confidence=True,
                min_confidence=0.5,
            ),
            agent_config={
                "max_facts": 20,
                "focus_on_actionable": True,
                "include_technical_details": True,
                "filter_generic_advice": True,
            }
        )
    
    # Template is now loaded automatically from global prompts.yaml
    
    def get_result_type(self) -> Type[FactExtractionResult]:
        """Get the result type for fact extraction."""
        return FactExtractionResult
    
    async def extract_facts(
        self,
        text: str,
        domain: str = "general",
        query_context: Optional[str] = None,
        max_facts: Optional[int] = None
    ) -> FactExtractionResult:
        """Extract facts from text.
        
        Args:
            text: Text to extract facts from
            domain: Domain context for extraction
            query_context: Optional query context for relevance
            max_facts: Maximum number of facts to extract
            
        Returns:
            Fact extraction result
        """
        if not text or not text.strip():
            return FactExtractionResult(
                facts=[],
                total_facts=0,
                confidence=ConfidenceLevel.HIGH,
                domain=domain,
                metadata={"error": "Empty text"}
            )
        
        # Update config for this extraction
        if max_facts:
            self.config.set_agent_config("max_facts", max_facts)
        
        self.logger.info(
            "Starting fact extraction",
            text_length=len(text),
            domain=domain,
            has_query_context=query_context is not None
        )
        
        try:
            result = await self.execute(
                text,
                domain=domain,
                query_context=query_context
            )
            
            self.logger.info(
                "Fact extraction completed",
                facts_extracted=result.total_facts,
                confidence=result.confidence
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Fact extraction failed", error=str(e))
            raise
    
    async def extract_facts_batch(
        self,
        texts: List[str],
        domain: str = "general",
        query_context: Optional[str] = None
    ) -> List[FactExtractionResult]:
        """Extract facts from multiple texts.
        
        Args:
            texts: List of texts to process
            domain: Domain context
            query_context: Optional query context
            
        Returns:
            List of fact extraction results
        """
        if not texts:
            return []
        
        self.logger.info(f"Starting batch fact extraction for {len(texts)} texts")
        
        results = await self.batch_execute(
            texts,
            domain=domain,
            query_context=query_context
        )
        
        self.logger.info(f"Batch fact extraction completed for {len(results)} texts")
        return results
