"""Fact extraction agent using Outlines for guaranteed structured output."""

from typing import Any, Dict, List, Optional, Type

from ..ai.base_agent import AgentConfig, MoRAGBaseAgent
from ..ai.models import ConfidenceLevel, ExtractedFact, FactExtractionResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FactExtractionAgent(MoRAGBaseAgent[FactExtractionResult]):
    """Fact extraction agent with guaranteed structured output using Outlines."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the fact extraction agent.

        Args:
            config: Agent configuration
        """
        if config is None:
            config = AgentConfig(
                model="google-gla:gemini-1.5-flash",
                temperature=0.1,
                outlines_provider="gemini",
            )

        super().__init__(config)

        # Fact extraction specific configuration
        self.fact_types = [
            "statistical",
            "causal",
            "technical",
            "definition",
            "procedural",
            "declarative",
            "regulatory",
            "temporal",
            "comparative",
        ]
        self.max_facts = 20
        self.min_confidence = 0.5
        self.focus_on_actionable = True
        self.include_technical_details = True
        self.filter_generic_advice = True

    def get_result_type(self) -> Type[FactExtractionResult]:
        """Return the Pydantic model for fact extraction results.

        Returns:
            FactExtractionResult class
        """
        return FactExtractionResult

    def get_system_prompt(self) -> str:
        """Return the system prompt for fact extraction.

        Returns:
            The system prompt string
        """
        return f"""You are an expert fact extraction system. Your task is to identify and extract structured, self-contained facts from text with high accuracy.

FACT TYPES TO EXTRACT:
{', '.join(self.fact_types)}

EXTRACTION GUIDELINES:
1. Extract facts that are complete and self-contained statements
2. Focus on actionable, specific, and verifiable information
3. Provide confidence scores between 0.0 and 1.0
4. Include relevant keywords and technical terms
5. Avoid generic advice or common knowledge
6. Maximum facts per extraction: {self.max_facts}
7. Minimum confidence threshold: {self.min_confidence}

FACT QUALITY CRITERIA:
- Facts should be specific and detailed
- Include quantitative information when available
- Capture cause-and-effect relationships
- Preserve technical terminology
- Maintain context and nuance

OUTPUT FORMAT:
Return a JSON object with the following structure:
- facts: List of extracted facts with fact_text, fact_type, confidence, structured_metadata, keywords, source_text
- total_facts: Total count of facts
- confidence: Overall confidence level (low, medium, high, very_high)
- domain: Domain context
- language: Language of the text
- metadata: Additional extraction metadata

FACT STRUCTURE:
Each fact should have:
- fact_text: Complete, self-contained fact statement
- fact_type: Type of fact from the supported types
- confidence: Confidence score (0.0 to 1.0)
- structured_metadata: Additional structured information
- keywords: List of relevant technical terms
- source_text: Original text span (optional)

IMPORTANT:
- Ensure facts are self-contained and meaningful
- Focus on information that adds value
- Maintain high quality standards
- Use appropriate fact types for the content"""

    async def extract_facts(
        self,
        text: str,
        domain: str = "general",
        query_context: Optional[str] = None,
        max_facts: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> FactExtractionResult:
        """Extract facts from text with guaranteed structured output.

        Args:
            text: Text to extract facts from
            domain: Domain context for extraction
            query_context: Optional query context for relevance
            max_facts: Maximum number of facts to extract
            min_confidence: Minimum confidence threshold

        Returns:
            FactExtractionResult with guaranteed structure
        """
        if not text or not text.strip():
            return FactExtractionResult(
                facts=[],
                total_facts=0,
                confidence=ConfidenceLevel.HIGH,
                domain=domain,
                language="en",
                metadata={"error": "Empty text", "domain": domain},
            )

        # Update configuration for this extraction
        if max_facts is not None:
            self.max_facts = max_facts
        if min_confidence is not None:
            self.min_confidence = min_confidence

        self.logger.info(
            "Starting fact extraction with Outlines",
            text_length=len(text),
            domain=domain,
            has_query_context=query_context is not None,
            structured_generation=self.is_outlines_available(),
        )

        # Prepare the extraction prompt
        prompt = self._create_extraction_prompt(text, domain, query_context)

        try:
            # Use structured generation with Outlines
            result = await self.run(prompt)

            # Post-process the result
            result = self._post_process_result(result, text, domain)

            self.logger.info(
                "Fact extraction completed",
                facts_extracted=result.total_facts,
                confidence=result.confidence,
                used_outlines=self.is_outlines_available(),
            )

            return result

        except Exception as e:
            self.logger.error("Fact extraction failed", error=str(e))
            # Return a fallback result
            return FactExtractionResult(
                facts=[],
                total_facts=0,
                confidence=ConfidenceLevel.LOW,
                domain=domain,
                language="en",
                metadata={"error": str(e), "domain": domain, "fallback": True},
            )

    async def extract_facts_batch(
        self,
        texts: List[str],
        domain: str = "general",
        query_context: Optional[str] = None,
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

        self.logger.info(
            "Starting batch fact extraction", batch_size=len(texts), domain=domain
        )

        results = []
        for i, text in enumerate(texts):
            try:
                result = await self.extract_facts(text, domain, query_context)
                results.append(result)

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(texts)} texts")

            except Exception as e:
                self.logger.warning(
                    "Failed to extract facts from text", text_index=i, error=str(e)
                )
                # Add fallback result
                results.append(
                    FactExtractionResult(
                        facts=[],
                        total_facts=0,
                        confidence=ConfidenceLevel.LOW,
                        domain=domain,
                        language="en",
                        metadata={"error": str(e), "text_index": i},
                    )
                )

        self.logger.info(f"Batch fact extraction completed for {len(results)} texts")
        return results

    def _create_extraction_prompt(
        self, text: str, domain: str, query_context: Optional[str] = None
    ) -> str:
        """Create the extraction prompt for the given text and domain.

        Args:
            text: Text to extract facts from
            domain: Domain context
            query_context: Optional query context

        Returns:
            Formatted prompt string
        """
        prompt = f"""Extract structured facts from the following {domain} text:

TEXT:
{text}

EXTRACTION PARAMETERS:
- Domain: {domain}
- Maximum facts: {self.max_facts}
- Minimum confidence: {self.min_confidence}
- Focus on actionable information: {self.focus_on_actionable}
- Include technical details: {self.include_technical_details}"""

        if query_context:
            prompt += f"""
- Query context: {query_context}
- Prioritize facts relevant to the query context"""

        prompt += """

Extract high-quality, self-contained facts that provide valuable information about the content."""

        return prompt

    def _post_process_result(
        self, result: FactExtractionResult, original_text: str, domain: str
    ) -> FactExtractionResult:
        """Post-process the extraction result.

        Args:
            result: Raw extraction result
            original_text: Original input text
            domain: Domain context

        Returns:
            Post-processed result
        """
        # Filter facts by confidence threshold
        filtered_facts = [
            fact for fact in result.facts if fact.confidence >= self.min_confidence
        ]

        # Limit to max_facts
        if len(filtered_facts) > self.max_facts:
            # Sort by confidence and take top facts
            filtered_facts = sorted(
                filtered_facts, key=lambda f: f.confidence, reverse=True
            )[: self.max_facts]

        # Update metadata
        metadata = result.metadata or {}
        metadata.update(
            {
                "domain": domain,
                "original_fact_count": len(result.facts),
                "filtered_fact_count": len(filtered_facts),
                "text_length": len(original_text),
                "min_confidence_threshold": self.min_confidence,
                "max_facts_limit": self.max_facts,
                "extraction_method": "outlines"
                if self.is_outlines_available()
                else "fallback",
            }
        )

        # Create updated result
        return FactExtractionResult(
            facts=filtered_facts,
            total_facts=len(filtered_facts),
            confidence=result.confidence,
            domain=domain,
            language=result.language,
            metadata=metadata,
        )
