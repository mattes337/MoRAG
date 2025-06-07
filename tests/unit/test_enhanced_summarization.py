"""Tests for enhanced summarization service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from morag_services.processing import (
    EnhancedSummarizationService,
    SummaryConfig,
    SummaryStrategy,
    DocumentType,
    ContentAnalyzer,
    SummaryQuality,
    EnhancedSummaryResult
)
from morag_services.embedding import SummaryResult

@pytest.fixture
def content_analyzer():
    """Create ContentAnalyzer instance."""
    return ContentAnalyzer()

@pytest.fixture
def summarization_service():
    """Create EnhancedSummarizationService instance."""
    with patch('morag.services.summarization.gemini_service') as mock_gemini:
        mock_gemini.generation_model = "gemini-2.0-flash-001"
        mock_gemini.generate_summary = AsyncMock(return_value=SummaryResult(
            summary="Test summary",
            token_count=10,
            model="gemini-2.0-flash-001"
        ))
        service = EnhancedSummarizationService()
        service.gemini_service = mock_gemini
        return service

@pytest.fixture
def mock_gemini_service():
    """Mock Gemini service."""
    with patch('morag.services.summarization.gemini_service') as mock:
        mock.generate_summary = AsyncMock(return_value=SummaryResult(
            summary="Test summary",
            token_count=10,
            model="gemini-2.0-flash-001"
        ))
        mock.generation_model = "gemini-2.0-flash-001"
        yield mock

class TestContentAnalyzer:
    """Test ContentAnalyzer functionality."""
    
    def test_analyze_academic_content(self, content_analyzer):
        """Test analysis of academic content."""
        text = """
        This research study presents a novel methodology for analyzing data.
        The experiment was conducted with a hypothesis that results would show
        significant findings. Our analysis reveals important conclusions.
        """
        
        result = content_analyzer.analyze_content(text)
        
        assert result['document_type'] == DocumentType.ACADEMIC
        assert result['word_count'] > 0
        assert 'complexity' in result
        assert 'type_scores' in result
    
    def test_analyze_technical_content(self, content_analyzer):
        """Test analysis of technical content."""
        text = """
        To implement this algorithm, follow these procedures:
        1. Configure the system settings
        2. Install the required dependencies
        3. Run the setup function
        The documentation provides detailed implementation steps.
        """
        
        result = content_analyzer.analyze_content(text)
        
        assert result['document_type'] == DocumentType.TECHNICAL
        assert result['has_structure'] is True
    
    def test_analyze_business_content(self, content_analyzer):
        """Test analysis of business content."""
        text = """
        The quarterly revenue exceeded our forecast by 15%.
        Market analysis shows strong customer demand.
        Strategic decisions were made to increase sales budget.
        Action items include expanding to new markets.
        """
        
        result = content_analyzer.analyze_content(text)
        
        assert result['document_type'] == DocumentType.BUSINESS
    
    def test_analyze_general_content(self, content_analyzer):
        """Test analysis of general content."""
        text = """
        This is a simple text about everyday topics.
        It contains general information without specific
        technical, academic, or business terminology.
        """
        
        result = content_analyzer.analyze_content(text)
        
        assert result['document_type'] == DocumentType.GENERAL
    
    def test_calculate_avg_sentence_length(self, content_analyzer):
        """Test average sentence length calculation."""
        text = "Short sentence. This is a longer sentence with more words."
        
        avg_length = content_analyzer._calculate_avg_sentence_length(text)
        
        assert avg_length > 0
        assert isinstance(avg_length, float)

class TestSummaryConfig:
    """Test SummaryConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SummaryConfig()
        
        assert config.strategy == SummaryStrategy.ABSTRACTIVE
        assert config.max_length == 150
        assert config.min_length == 50
        assert config.style == "concise"
        assert config.focus_areas == []
        assert config.quality_threshold == 0.7
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SummaryConfig(
            strategy=SummaryStrategy.HYBRID,
            max_length=200,
            style="detailed",
            focus_areas=["methodology", "results"],
            document_type=DocumentType.ACADEMIC
        )
        
        assert config.strategy == SummaryStrategy.HYBRID
        assert config.max_length == 200
        assert config.style == "detailed"
        assert "methodology" in config.focus_areas
        assert config.document_type == DocumentType.ACADEMIC

class TestEnhancedSummarizationService:
    """Test EnhancedSummarizationService functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_summary_abstractive(self, summarization_service, mock_gemini_service):
        """Test abstractive summary generation."""
        text = "This is a test document with some content to summarize."
        config = SummaryConfig(strategy=SummaryStrategy.ABSTRACTIVE)
        
        result = await summarization_service.generate_summary(text, config)
        
        assert isinstance(result, EnhancedSummaryResult)
        assert result.summary == "Test summary"
        assert result.strategy == SummaryStrategy.ABSTRACTIVE
        assert isinstance(result.quality, SummaryQuality)
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_generate_summary_extractive(self, summarization_service):
        """Test extractive summary generation."""
        text = """
        This is the first sentence with important information.
        This is the second sentence with key details.
        This is the third sentence with crucial data.
        This is the fourth sentence with significant findings.
        """
        config = SummaryConfig(strategy=SummaryStrategy.EXTRACTIVE)
        
        result = await summarization_service.generate_summary(text, config)
        
        assert isinstance(result, EnhancedSummaryResult)
        assert result.strategy == SummaryStrategy.EXTRACTIVE
        assert len(result.summary) > 0
    
    @pytest.mark.asyncio
    async def test_generate_summary_hybrid(self, summarization_service, mock_gemini_service):
        """Test hybrid summary generation."""
        text = """
        This is the first sentence with important information.
        This is the second sentence with key details.
        This is the third sentence with crucial data.
        This is the fourth sentence with significant findings.
        """
        config = SummaryConfig(strategy=SummaryStrategy.HYBRID)
        
        result = await summarization_service.generate_summary(text, config)
        
        assert isinstance(result, EnhancedSummaryResult)
        assert result.strategy == SummaryStrategy.HYBRID
        # Hybrid strategy should call the service (via the fixture mock)
    
    @pytest.mark.asyncio
    async def test_generate_summary_hierarchical(self, summarization_service, mock_gemini_service):
        """Test hierarchical summary generation."""
        # Create long text that will be split into chunks
        text = " ".join(["This is sentence number {}.".format(i) for i in range(100)])
        config = SummaryConfig(
            strategy=SummaryStrategy.HIERARCHICAL,
            context_window=50  # Small window to force chunking
        )
        
        result = await summarization_service.generate_summary(text, config)
        
        assert isinstance(result, EnhancedSummaryResult)
        assert result.strategy == SummaryStrategy.HIERARCHICAL
        # Hierarchical strategy should process the text (via the fixture mock)
    
    @pytest.mark.asyncio
    async def test_generate_summary_contextual(self, summarization_service, mock_gemini_service):
        """Test contextual summary generation."""
        text = "This document discusses methodology and results of the research."
        config = SummaryConfig(
            strategy=SummaryStrategy.CONTEXTUAL,
            focus_areas=["methodology", "results"]
        )
        
        result = await summarization_service.generate_summary(text, config)
        
        assert isinstance(result, EnhancedSummaryResult)
        assert result.strategy == SummaryStrategy.CONTEXTUAL
        # Contextual strategy should process the text (via the fixture mock)
    
    def test_create_adaptive_config(self, summarization_service):
        """Test adaptive configuration creation."""
        # Test with academic content
        academic_text = "This research study presents methodology and analysis of results."
        config = summarization_service._create_adaptive_config(academic_text)
        
        assert isinstance(config, SummaryConfig)
        assert config.max_length > 0
        assert config.min_length > 0
        
        # Test with long content (should use hierarchical)
        long_text = " ".join(["word"] * 3000)
        config_long = summarization_service._create_adaptive_config(long_text)
        
        assert config_long.strategy == SummaryStrategy.HIERARCHICAL
    
    def test_split_into_sentences(self, summarization_service):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = summarization_service._split_into_sentences(text)
        
        assert len(sentences) == 4
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
    
    def test_score_sentence(self, summarization_service):
        """Test sentence scoring for extractive summarization."""
        sentence = "This is an important sentence with key information."
        full_text = "Some context. " + sentence + " More context."
        
        score = summarization_service._score_sentence(sentence, full_text, 1, 3)
        
        assert isinstance(score, float)
        assert score > 0
    
    def test_split_into_chunks(self, summarization_service):
        """Test text chunking."""
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = summarization_service._split_into_chunks(text, 25)
        
        assert len(chunks) == 4  # 100 words / 25 words per chunk
        assert all(len(chunk.split()) <= 25 for chunk in chunks)
    
    def test_build_enhanced_prompt(self, summarization_service):
        """Test enhanced prompt building."""
        text = "Test content"
        config = SummaryConfig(
            document_type=DocumentType.ACADEMIC,
            style="detailed",
            focus_areas=["methodology"],
            preserve_structure=True
        )
        
        prompt = summarization_service._build_enhanced_prompt(text, config)
        
        assert "academic" in prompt.lower()
        assert "comprehensive" in prompt.lower()  # "detailed" becomes "comprehensive coverage"
        assert "methodology" in prompt
        assert "structure" in prompt.lower()
        assert text in prompt
    
    def test_get_type_specific_instruction(self, summarization_service):
        """Test document type-specific instructions."""
        academic_instruction = summarization_service._get_type_specific_instruction(DocumentType.ACADEMIC)
        technical_instruction = summarization_service._get_type_specific_instruction(DocumentType.TECHNICAL)
        business_instruction = summarization_service._get_type_specific_instruction(DocumentType.BUSINESS)
        
        assert "methodology" in academic_instruction.lower()
        assert "procedure" in technical_instruction.lower()
        assert "decision" in business_instruction.lower()

class TestQualityAssessment:
    """Test quality assessment functionality."""
    
    def test_assess_quality(self, summarization_service):
        """Test overall quality assessment."""
        original = "This is a comprehensive document with important information and key details."
        summary = "This document contains important information and key details."
        config = SummaryConfig()
        
        quality = summarization_service._assess_quality(original, summary, config)
        
        assert isinstance(quality, SummaryQuality)
        assert 0 <= quality.coherence <= 1
        assert 0 <= quality.completeness <= 1
        assert 0 <= quality.conciseness <= 1
        assert 0 <= quality.relevance <= 1
        assert 0 <= quality.readability <= 1
        assert 0 <= quality.overall <= 1
    
    def test_assess_coherence(self, summarization_service):
        """Test coherence assessment."""
        coherent_summary = "First point. However, there is another consideration. Therefore, we conclude."
        incoherent_summary = "Random sentence. Unrelated thought. Another topic."
        
        coherent_score = summarization_service._assess_coherence(coherent_summary)
        incoherent_score = summarization_service._assess_coherence(incoherent_summary)
        
        assert coherent_score > incoherent_score
    
    def test_assess_completeness(self, summarization_service):
        """Test completeness assessment."""
        original = "important research methodology analysis results conclusion"
        complete_summary = "important research methodology results conclusion"
        incomplete_summary = "research results"
        
        complete_score = summarization_service._assess_completeness(original, complete_summary)
        incomplete_score = summarization_service._assess_completeness(original, incomplete_summary)
        
        assert complete_score > incomplete_score
    
    def test_assess_conciseness(self, summarization_service):
        """Test conciseness assessment."""
        config = SummaryConfig(min_length=10, max_length=20)
        
        perfect_summary = " ".join(["word"] * 15)  # Perfect length
        too_short_summary = " ".join(["word"] * 5)   # Too short
        too_long_summary = " ".join(["word"] * 30)   # Too long
        
        perfect_score = summarization_service._assess_conciseness(perfect_summary, config)
        short_score = summarization_service._assess_conciseness(too_short_summary, config)
        long_score = summarization_service._assess_conciseness(too_long_summary, config)
        
        assert perfect_score == 1.0
        assert short_score < perfect_score
        assert long_score < perfect_score
    
    def test_assess_relevance(self, summarization_service):
        """Test relevance assessment."""
        original = "important research methodology analysis results conclusion"
        relevant_summary = "important research methodology results"
        irrelevant_summary = "completely different topic words"
        
        relevant_score = summarization_service._assess_relevance(original, relevant_summary)
        irrelevant_score = summarization_service._assess_relevance(original, irrelevant_summary)
        
        assert relevant_score > irrelevant_score
    
    def test_assess_readability(self, summarization_service):
        """Test readability assessment."""
        readable_summary = "This is a clear and readable summary with appropriate sentence length."
        unreadable_summary = "Word. Another. Short."
        
        readable_score = summarization_service._assess_readability(readable_summary)
        unreadable_score = summarization_service._assess_readability(unreadable_summary)
        
        assert readable_score > unreadable_score

@pytest.mark.asyncio
async def test_summary_refinement():
    """Test summary refinement functionality."""
    with patch('morag.services.summarization.gemini_service') as mock_gemini:
        mock_gemini.generate_summary = AsyncMock(return_value=SummaryResult(
            summary="Improved summary with better quality.",
            token_count=8,
            model="gemini-2.0-flash-001"
        ))
        mock_gemini.generation_model = "gemini-2.0-flash-001"

        summarization_service = EnhancedSummarizationService()

        original_text = "This is the original text with important information."
        poor_summary = "Bad summary."
        config = SummaryConfig(quality_threshold=0.8)

        # Create poor quality assessment
        quality = SummaryQuality(
            coherence=0.5,
            completeness=0.4,
            conciseness=0.6,
            relevance=0.5,
            readability=0.5,
            overall=0.5
        )

        refined_summary, iterations = await summarization_service._refine_summary(
            original_text, poor_summary, config, quality
        )

        assert isinstance(refined_summary, str)
        assert iterations > 0
        assert mock_gemini.generate_summary.called
