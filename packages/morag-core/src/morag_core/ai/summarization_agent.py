"""PydanticAI agent for text summarization."""

import asyncio
from typing import Type, List, Optional, Dict, Any
import structlog

from .base_agent import MoRAGBaseAgent
from .models import SummaryResult, ConfidenceLevel

logger = structlog.get_logger(__name__)


class SummarizationAgent(MoRAGBaseAgent[SummaryResult]):
    """PydanticAI agent for intelligent text summarization."""
    
    def __init__(self, max_summary_length: int = 1000, **kwargs):
        """Initialize the summarization agent.
        
        Args:
            max_summary_length: Maximum length of generated summaries
            **kwargs: Additional arguments passed to base agent
        """
        super().__init__(**kwargs)
        self.max_summary_length = max_summary_length
        self.logger = logger.bind(agent="summarization")
    
    def get_result_type(self) -> Type[SummaryResult]:
        return SummaryResult
    
    def get_system_prompt(self) -> str:
        return f"""You are an expert text summarization agent. Your task is to create high-quality, informative summaries of text content.

Your summaries should be:
- Concise yet comprehensive, capturing the main ideas and key information
- Well-structured with clear, coherent sentences
- Factually accurate and faithful to the original content
- Appropriate for the content type and context
- Within the specified length limit (typically {self.max_summary_length} characters)

For each summary, provide:
1. summary: The main summary text that captures the essence of the content
2. key_points: 3-7 bullet points highlighting the most important information
3. confidence: Your confidence in the summary quality (low, medium, high, very_high)
4. word_count: Approximate word count of the summary
5. compression_ratio: Ratio of summary length to original text length

Guidelines for creating summaries:
- Focus on the main themes, arguments, and conclusions
- Include important facts, figures, and specific details when relevant
- Maintain the original tone and perspective when appropriate
- Use clear, accessible language
- Ensure the summary can stand alone and be understood without the original text
- For technical content, preserve key terminology and concepts
- For narrative content, capture the main storyline and key events
- For analytical content, include main arguments and supporting evidence

Adapt your summarization style based on content type:
- Academic/Research: Focus on methodology, findings, and conclusions
- News/Articles: Lead with key facts and follow with supporting details
- Technical Documentation: Emphasize procedures, requirements, and outcomes
- Business Content: Highlight decisions, strategies, and implications
- Creative Content: Capture themes, characters, and narrative elements"""
    
    async def summarize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        style: str = "concise",
        context: Optional[str] = None
    ) -> SummaryResult:
        """Summarize text with specified parameters.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length (overrides default)
            style: Summarization style (concise, detailed, bullet, abstract)
            context: Additional context for summarization
            
        Returns:
            SummaryResult with summary and metadata
        """
        if not text or not text.strip():
            return SummaryResult(
                summary="",
                key_points=[],
                confidence=ConfidenceLevel.HIGH,
                word_count=0,
                compression_ratio=0.0
            )
        
        effective_max_length = max_length or self.max_summary_length
        
        self.logger.info(
            "Starting text summarization",
            text_length=len(text),
            max_length=effective_max_length,
            style=style,
            has_context=context is not None
        )
        
        try:
            # Build the summarization prompt
            prompt = self._build_summarization_prompt(
                text, effective_max_length, style, context
            )
            
            # Generate summary using the agent
            result = await self.run(prompt)
            
            # Calculate compression ratio
            original_length = len(text)
            summary_length = len(result.summary)
            result.compression_ratio = summary_length / original_length if original_length > 0 else 0.0
            
            self.logger.info(
                "Text summarization completed",
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=result.compression_ratio,
                key_points_count=len(result.key_points),
                confidence=result.confidence
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Text summarization failed", error=str(e), error_type=type(e).__name__)
            raise
    
    async def summarize_document(
        self,
        text: str,
        title: Optional[str] = None,
        document_type: Optional[str] = None,
        max_length: Optional[int] = None
    ) -> SummaryResult:
        """Summarize a document with document-specific context.
        
        Args:
            text: Document text to summarize
            title: Document title for context
            document_type: Type of document (article, report, manual, etc.)
            max_length: Maximum summary length
            
        Returns:
            SummaryResult with document summary
        """
        context_parts = []
        if title:
            context_parts.append(f"Document Title: {title}")
        if document_type:
            context_parts.append(f"Document Type: {document_type}")
        
        context = "\n".join(context_parts) if context_parts else None
        
        return await self.summarize_text(
            text=text,
            max_length=max_length,
            style="detailed",
            context=context
        )
    
    async def summarize_chunks(
        self,
        chunks: List[str],
        max_length: Optional[int] = None,
        preserve_structure: bool = True
    ) -> SummaryResult:
        """Summarize multiple text chunks into a unified summary.
        
        Args:
            chunks: List of text chunks to summarize
            max_length: Maximum summary length
            preserve_structure: Whether to preserve chunk structure in summary
            
        Returns:
            SummaryResult with unified summary
        """
        if not chunks:
            return SummaryResult(
                summary="",
                key_points=[],
                confidence=ConfidenceLevel.HIGH,
                word_count=0,
                compression_ratio=0.0
            )
        
        # For small number of chunks, combine and summarize
        if len(chunks) <= 3:
            combined_text = "\n\n".join(chunks)
            return await self.summarize_text(
                text=combined_text,
                max_length=max_length,
                style="detailed"
            )
        
        # For larger number of chunks, use hierarchical summarization
        return await self._hierarchical_summarization(chunks, max_length, preserve_structure)
    
    async def _hierarchical_summarization(
        self,
        chunks: List[str],
        max_length: Optional[int],
        preserve_structure: bool
    ) -> SummaryResult:
        """Perform hierarchical summarization for large documents."""
        effective_max_length = max_length or self.max_summary_length
        
        # First, summarize each chunk individually
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                chunk_summary = await self.summarize_text(
                    text=chunk,
                    max_length=effective_max_length // len(chunks),
                    style="concise"
                )
                chunk_summaries.append(chunk_summary.summary)
            except Exception as e:
                self.logger.warning(f"Failed to summarize chunk {i}", error=str(e))
                # Use truncated original text as fallback
                chunk_summaries.append(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        
        # Then, combine and summarize the chunk summaries
        combined_summaries = "\n\n".join(chunk_summaries)
        
        return await self.summarize_text(
            text=combined_summaries,
            max_length=effective_max_length,
            style="detailed",
            context="This is a hierarchical summary combining multiple document sections."
        )
    
    def _build_summarization_prompt(
        self,
        text: str,
        max_length: int,
        style: str,
        context: Optional[str]
    ) -> str:
        """Build the summarization prompt."""
        style_instructions = {
            "concise": "Create a concise, factual summary focusing on the main points",
            "detailed": "Create a comprehensive summary that captures key details and nuances",
            "bullet": "Create a structured summary with clear bullet points for key information",
            "abstract": "Create an abstract-style summary suitable for academic or professional use"
        }
        
        instruction = style_instructions.get(style, style_instructions["concise"])
        
        prompt_parts = [
            f"{instruction}.",
            f"Target length: approximately {max_length} characters.",
            ""
        ]
        
        if context:
            prompt_parts.extend([
                "Additional Context:",
                context,
                ""
            ])
        
        prompt_parts.extend([
            "Text to summarize:",
            text
        ])
        
        return "\n".join(prompt_parts)
