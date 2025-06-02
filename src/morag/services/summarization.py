"""Enhanced summarization service with CRAG-inspired techniques."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import structlog
import re
from pathlib import Path

from morag.core.config import settings
from morag.core.exceptions import ProcessingError, ExternalServiceError, RateLimitError
from morag.services.embedding import gemini_service, SummaryResult

logger = structlog.get_logger()

class SummaryStrategy(Enum):
    """Summary generation strategies."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"
    CONTEXTUAL = "contextual"

class DocumentType(Enum):
    """Document types for specialized summarization."""
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    BUSINESS = "business"
    NEWS = "news"
    GENERAL = "general"

@dataclass
class SummaryConfig:
    """Configuration for summary generation."""
    strategy: SummaryStrategy = SummaryStrategy.ABSTRACTIVE
    max_length: int = 150
    min_length: int = 50
    style: str = "concise"
    focus_areas: List[str] = field(default_factory=list)
    context_window: int = 2000
    quality_threshold: float = 0.7
    enable_refinement: bool = True
    preserve_structure: bool = False
    document_type: Optional[DocumentType] = None

@dataclass
class SummaryQuality:
    """Quality metrics for generated summaries."""
    coherence: float = 0.0
    completeness: float = 0.0
    conciseness: float = 0.0
    relevance: float = 0.0
    readability: float = 0.0
    overall: float = 0.0

@dataclass
class EnhancedSummaryResult:
    """Enhanced summary result with quality metrics."""
    summary: str
    token_count: int
    model: str
    strategy: SummaryStrategy
    quality: SummaryQuality
    config: SummaryConfig
    processing_time: float = 0.0
    refinement_iterations: int = 0

class ContentAnalyzer:
    """Analyzes content to determine optimal summarization strategy."""
    
    def __init__(self):
        self.academic_keywords = {
            'abstract', 'methodology', 'results', 'conclusion', 'hypothesis',
            'experiment', 'analysis', 'research', 'study', 'findings'
        }
        self.technical_keywords = {
            'implementation', 'algorithm', 'function', 'method', 'procedure',
            'configuration', 'installation', 'setup', 'documentation'
        }
        self.business_keywords = {
            'strategy', 'revenue', 'profit', 'market', 'customer', 'sales',
            'budget', 'forecast', 'meeting', 'decision', 'action'
        }
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze content to determine characteristics."""
        words = text.lower().split()
        word_count = len(words)
        
        # Calculate document type scores
        academic_score = len([w for w in words if w in self.academic_keywords]) / word_count
        technical_score = len([w for w in words if w in self.technical_keywords]) / word_count
        business_score = len([w for w in words if w in self.business_keywords]) / word_count
        
        # Determine document type
        scores = {
            DocumentType.ACADEMIC: academic_score,
            DocumentType.TECHNICAL: technical_score,
            DocumentType.BUSINESS: business_score
        }
        
        doc_type = max(scores, key=scores.get) if max(scores.values()) > 0.01 else DocumentType.GENERAL
        
        # Analyze structure
        has_headers = bool(re.search(r'^#+\s', text, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE))
        has_numbers = bool(re.search(r'^\s*\d+\.', text, re.MULTILINE))
        
        # Calculate complexity
        avg_sentence_length = self._calculate_avg_sentence_length(text)
        complexity = min(1.0, avg_sentence_length / 20.0)  # Normalize to 0-1
        
        return {
            'document_type': doc_type,
            'word_count': word_count,
            'has_structure': has_headers or has_lists or has_numbers,
            'complexity': complexity,
            'type_scores': scores
        }
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)

class EnhancedSummarizationService:
    """Enhanced summarization service with CRAG-inspired techniques."""
    
    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
        self.gemini_service = gemini_service
    
    async def generate_summary(
        self,
        text: str,
        config: Optional[SummaryConfig] = None
    ) -> EnhancedSummaryResult:
        """Generate enhanced summary with quality assessment."""
        import time
        start_time = time.time()
        
        if config is None:
            config = self._create_adaptive_config(text)
        
        logger.info("Generating enhanced summary",
                   strategy=config.strategy.value,
                   max_length=config.max_length,
                   document_type=config.document_type.value if config.document_type else None)
        
        try:
            # Generate initial summary based on strategy
            if config.strategy == SummaryStrategy.EXTRACTIVE:
                summary = await self._extractive_summary(text, config)
            elif config.strategy == SummaryStrategy.ABSTRACTIVE:
                summary = await self._abstractive_summary(text, config)
            elif config.strategy == SummaryStrategy.HYBRID:
                summary = await self._hybrid_summary(text, config)
            elif config.strategy == SummaryStrategy.HIERARCHICAL:
                summary = await self._hierarchical_summary(text, config)
            elif config.strategy == SummaryStrategy.CONTEXTUAL:
                summary = await self._contextual_summary(text, config)
            else:
                summary = await self._abstractive_summary(text, config)
            
            # Assess quality
            quality = self._assess_quality(text, summary, config)
            
            # Refine if needed
            refinement_iterations = 0
            if config.enable_refinement and quality.overall < config.quality_threshold:
                summary, refinement_iterations = await self._refine_summary(
                    text, summary, config, quality
                )
                quality = self._assess_quality(text, summary, config)
            
            processing_time = time.time() - start_time
            
            return EnhancedSummaryResult(
                summary=summary,
                token_count=len(summary.split()),
                model=self.gemini_service.generation_model,
                strategy=config.strategy,
                quality=quality,
                config=config,
                processing_time=processing_time,
                refinement_iterations=refinement_iterations
            )
            
        except Exception as e:
            logger.error("Enhanced summary generation failed", error=str(e))
            raise ProcessingError(f"Summary generation failed: {str(e)}")
    
    def _create_adaptive_config(self, text: str) -> SummaryConfig:
        """Create adaptive configuration based on content analysis."""
        analysis = self.content_analyzer.analyze_content(text)
        
        # Determine strategy based on content
        if analysis['word_count'] > 2000:
            strategy = SummaryStrategy.HIERARCHICAL
        elif analysis['has_structure']:
            strategy = SummaryStrategy.HYBRID
        else:
            strategy = SummaryStrategy.ABSTRACTIVE
        
        # Adaptive length based on content
        base_length = min(200, max(50, analysis['word_count'] // 10))
        max_length = int(base_length * (1 + analysis['complexity']))
        
        return SummaryConfig(
            strategy=strategy,
            max_length=max_length,
            min_length=max(30, max_length // 3),
            document_type=analysis['document_type'],
            preserve_structure=analysis['has_structure']
        )
    
    async def _abstractive_summary(self, text: str, config: SummaryConfig) -> str:
        """Generate abstractive summary using Gemini."""
        prompt = self._build_enhanced_prompt(text, config)
        
        result = await self.gemini_service.generate_summary(
            text=prompt,
            max_length=config.max_length,
            style=config.style
        )
        
        return result.summary
    
    async def _extractive_summary(self, text: str, config: SummaryConfig) -> str:
        """Generate extractive summary by selecting key sentences."""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 3:
            return text
        
        # Score sentences based on various factors
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, text, i, len(sentences))
            scored_sentences.append((score, sentence))
        
        # Select top sentences
        scored_sentences.sort(reverse=True)
        target_sentences = max(2, min(len(sentences) // 3, config.max_length // 20))
        
        selected = [sent for _, sent in scored_sentences[:target_sentences]]
        return ' '.join(selected)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 2]
    
    def _score_sentence(self, sentence: str, full_text: str, position: int, total: int) -> float:
        """Score a sentence for extractive summarization."""
        words = sentence.lower().split()
        
        # Position score (beginning and end are important)
        pos_score = 1.0 if position < 2 or position >= total - 2 else 0.5
        
        # Length score (prefer medium-length sentences)
        length_score = min(1.0, len(words) / 15.0) if len(words) < 30 else 0.5
        
        # Keyword score (prefer sentences with important words)
        important_words = {'important', 'key', 'main', 'significant', 'crucial', 'essential'}
        keyword_score = len([w for w in words if w in important_words]) * 0.2
        
        return pos_score + length_score + keyword_score

    async def _hybrid_summary(self, text: str, config: SummaryConfig) -> str:
        """Generate hybrid summary combining extractive and abstractive approaches."""
        # First, get key sentences using extractive method
        extractive_summary = await self._extractive_summary(text, config)

        # Then, refine with abstractive summarization
        refined_config = SummaryConfig(
            strategy=SummaryStrategy.ABSTRACTIVE,
            max_length=config.max_length,
            style=config.style,
            document_type=config.document_type
        )

        return await self._abstractive_summary(extractive_summary, refined_config)

    async def _hierarchical_summary(self, text: str, config: SummaryConfig) -> str:
        """Generate hierarchical summary for long documents."""
        # Split into chunks
        chunks = self._split_into_chunks(text, config.context_window)

        if len(chunks) <= 1:
            return await self._abstractive_summary(text, config)

        # Summarize each chunk
        chunk_summaries = []
        chunk_config = SummaryConfig(
            strategy=SummaryStrategy.ABSTRACTIVE,
            max_length=config.max_length // len(chunks),
            style=config.style
        )

        for chunk in chunks:
            summary = await self._abstractive_summary(chunk, chunk_config)
            chunk_summaries.append(summary)

        # Combine and summarize the summaries
        combined = ' '.join(chunk_summaries)
        return await self._abstractive_summary(combined, config)

    async def _contextual_summary(self, text: str, config: SummaryConfig) -> str:
        """Generate context-aware summary with focus areas."""
        if not config.focus_areas:
            return await self._abstractive_summary(text, config)

        # Create context-aware prompt
        focus_text = ', '.join(config.focus_areas)
        enhanced_prompt = f"""
        Summarize the following text with special focus on: {focus_text}

        Text: {text}
        """

        result = await self.gemini_service.generate_summary(
            text=enhanced_prompt,
            max_length=config.max_length,
            style=config.style
        )

        return result.summary

    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately equal size."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def _build_enhanced_prompt(self, text: str, config: SummaryConfig) -> str:
        """Build enhanced prompt based on document type and configuration."""
        base_instruction = self._get_type_specific_instruction(config.document_type)

        style_modifiers = {
            "concise": "Keep it brief and to the point.",
            "detailed": "Provide comprehensive coverage of key points.",
            "bullet": "Use bullet points for clarity.",
            "abstract": "Write in academic abstract style."
        }

        style_instruction = style_modifiers.get(config.style, style_modifiers["concise"])

        prompt = f"""
        {base_instruction}
        {style_instruction}

        Target length: {config.max_length} words (minimum {config.min_length} words).
        """

        if config.focus_areas:
            prompt += f"\nPay special attention to: {', '.join(config.focus_areas)}"

        if config.preserve_structure:
            prompt += "\nPreserve the logical structure and organization of the content."

        prompt += f"\n\nText to summarize:\n{text}\n\nSummary:"

        return prompt

    def _get_type_specific_instruction(self, doc_type: Optional[DocumentType]) -> str:
        """Get document type-specific summarization instructions."""
        instructions = {
            DocumentType.ACADEMIC: "Summarize this academic content focusing on methodology, key findings, and conclusions.",
            DocumentType.TECHNICAL: "Summarize this technical content focusing on key procedures, configurations, and important details.",
            DocumentType.BUSINESS: "Summarize this business content focusing on decisions, actions, outcomes, and strategic implications.",
            DocumentType.NEWS: "Summarize this news content focusing on who, what, when, where, and why.",
            DocumentType.GENERAL: "Create a clear, informative summary of the main ideas and key points."
        }

        return instructions.get(doc_type, instructions[DocumentType.GENERAL])

    def _assess_quality(self, original_text: str, summary: str, config: SummaryConfig) -> SummaryQuality:
        """Assess the quality of generated summary."""
        # Basic quality metrics
        coherence = self._assess_coherence(summary)
        completeness = self._assess_completeness(original_text, summary)
        conciseness = self._assess_conciseness(summary, config)
        relevance = self._assess_relevance(original_text, summary)
        readability = self._assess_readability(summary)

        # Calculate overall score
        overall = (coherence + completeness + conciseness + relevance + readability) / 5.0

        return SummaryQuality(
            coherence=coherence,
            completeness=completeness,
            conciseness=conciseness,
            relevance=relevance,
            readability=readability,
            overall=overall
        )

    def _assess_coherence(self, summary: str) -> float:
        """Assess coherence of the summary."""
        sentences = self._split_into_sentences(summary)

        if len(sentences) <= 1:
            return 1.0

        # Simple coherence check based on sentence transitions
        coherence_score = 0.6  # Base score

        # Check for transition words
        transition_words = {'however', 'therefore', 'furthermore', 'additionally', 'consequently'}
        has_transitions = any(word in summary.lower() for word in transition_words)

        if has_transitions:
            coherence_score += 0.4

        # Check for repeated key terms (indicates coherence)
        words = summary.lower().split()
        important_words = [w for w in words if len(w) > 4]
        if len(set(important_words)) < len(important_words) * 0.8:  # Some repetition
            coherence_score += 0.1

        return min(1.0, coherence_score)

    def _assess_completeness(self, original: str, summary: str) -> float:
        """Assess completeness of the summary."""
        original_words = set(original.lower().split())
        summary_words = set(summary.lower().split())

        # Remove common stop words for better assessment
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        original_words -= stop_words
        summary_words -= stop_words

        if not original_words:
            return 1.0

        # Calculate overlap ratio
        overlap = len(original_words & summary_words)
        return min(1.0, overlap / len(original_words) * 2)  # Scale up since summaries are shorter

    def _assess_conciseness(self, summary: str, config: SummaryConfig) -> float:
        """Assess conciseness of the summary."""
        word_count = len(summary.split())

        if word_count <= config.min_length:
            return 0.5  # Too short
        elif word_count <= config.max_length:
            return 1.0  # Perfect length
        else:
            # Penalize for being too long
            excess = word_count - config.max_length
            penalty = min(0.5, excess / config.max_length)
            return max(0.0, 1.0 - penalty)

    def _assess_relevance(self, original: str, summary: str) -> float:
        """Assess relevance of summary to original text."""
        # Simple relevance check based on key terms
        original_words = original.lower().split()
        summary_words = summary.lower().split()

        # Find important words (longer words are often more important)
        important_original = [w for w in original_words if len(w) > 5]
        important_summary = [w for w in summary_words if len(w) > 5]

        if not important_original:
            return 1.0

        overlap = len(set(important_original) & set(important_summary))
        return min(1.0, overlap / len(set(important_original)) * 1.5)

    def _assess_readability(self, summary: str) -> float:
        """Assess readability of the summary."""
        sentences = self._split_into_sentences(summary)

        if not sentences:
            return 0.0

        # Calculate average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Optimal sentence length is around 15-20 words
        if 10 <= avg_sentence_length <= 25:
            return 1.0
        elif avg_sentence_length < 10:
            return 0.7  # Too short
        else:
            return max(0.3, 1.0 - (avg_sentence_length - 25) / 50)  # Too long

    async def _refine_summary(
        self,
        original_text: str,
        summary: str,
        config: SummaryConfig,
        quality: SummaryQuality
    ) -> tuple[str, int]:
        """Refine summary based on quality assessment."""
        max_iterations = 2
        current_summary = summary

        for iteration in range(max_iterations):
            # Identify main quality issues
            issues = []
            if quality.coherence < 0.6:
                issues.append("improve logical flow and coherence")
            if quality.completeness < 0.6:
                issues.append("include more key information")
            if quality.conciseness < 0.6:
                issues.append("adjust length appropriately")
            if quality.readability < 0.6:
                issues.append("improve readability and clarity")

            if not issues:
                break

            # Create refinement prompt
            refinement_prompt = f"""
            Please improve the following summary by addressing these issues: {', '.join(issues)}.

            Original text: {original_text[:1000]}...

            Current summary: {current_summary}

            Improved summary:
            """

            try:
                result = await self.gemini_service.generate_summary(
                    text=refinement_prompt,
                    max_length=config.max_length,
                    style=config.style
                )
                current_summary = result.summary

                # Re-assess quality
                quality = self._assess_quality(original_text, current_summary, config)

                if quality.overall >= config.quality_threshold:
                    break

            except Exception as e:
                logger.warning("Summary refinement failed", iteration=iteration, error=str(e))
                break

        return current_summary, iteration + 1

# Global instance
enhanced_summarization_service = EnhancedSummarizationService()
