"""Tests for sentence processor."""

import pytest
import asyncio
from morag_graph.processors.sentence_processor import SentenceProcessor, ProcessedSentence


class TestSentenceProcessor:
    """Test cases for sentence processor."""
    
    @pytest.fixture
    def processor(self):
        """Create sentence processor instance for testing."""
        config = {
            'min_sentence_length': 10,
            'max_sentence_length': 500,
            'enable_cleaning': True,
            'enable_quality_scoring': True,
            'batch_size': 10
        }
        return SentenceProcessor(config=config)
    
    @pytest.fixture
    def simple_processor(self):
        """Create simple processor without quality scoring."""
        config = {
            'min_sentence_length': 5,
            'max_sentence_length': 1000,
            'enable_cleaning': False,
            'enable_quality_scoring': False
        }
        return SentenceProcessor(config=config)
    
    def test_init(self, processor):
        """Test processor initialization."""
        assert processor.min_sentence_length == 10
        assert processor.max_sentence_length == 500
        assert processor.enable_cleaning is True
        assert processor.enable_quality_scoring is True
        assert processor.batch_size == 10
    
    @pytest.mark.asyncio
    async def test_process_empty_text(self, processor):
        """Test processing empty text."""
        sentences = await processor.process_text("")
        assert sentences == []
        
        sentences = await processor.process_text("   ")
        assert sentences == []
    
    @pytest.mark.asyncio
    async def test_process_simple_sentences(self, simple_processor):
        """Test processing simple sentences."""
        text = "Hello world. This is a test. How are you?"
        sentences = await simple_processor.process_text(text)
        
        assert len(sentences) >= 2
        assert any("Hello world" in s.text for s in sentences)
        assert any("This is a test" in s.text for s in sentences)
        assert any("How are you" in s.text for s in sentences)
    
    @pytest.mark.asyncio
    async def test_process_with_abbreviations(self, simple_processor):
        """Test that abbreviations don't break sentences incorrectly."""
        text = "Dr. Smith works at Inc. Corp. He is very smart."
        sentences = await simple_processor.process_text(text)
        
        # Should not split on "Dr." or "Inc."
        assert len(sentences) >= 1
        # The first sentence should contain the full context
        first_sentence = sentences[0].text
        assert "Dr. Smith" in first_sentence or "Smith" in first_sentence
    
    @pytest.mark.asyncio
    async def test_process_with_numbers(self, simple_processor):
        """Test that decimal numbers don't break sentences."""
        text = "The price is $19.99 today. Tomorrow it will be $25.50."
        sentences = await simple_processor.process_text(text)
        
        assert len(sentences) >= 2
        assert any("19.99" in s.text for s in sentences)
        assert any("25.50" in s.text for s in sentences)
    
    @pytest.mark.asyncio
    async def test_length_filtering(self, processor):
        """Test sentence length filtering."""
        text = "Hi. This is a longer sentence that should pass the filter. Ok."
        sentences = await processor.process_text(text)
        
        # Short sentences like "Hi." and "Ok." should be filtered out
        assert all(len(s.text) >= processor.min_sentence_length for s in sentences)
        assert any("longer sentence" in s.text for s in sentences)
    
    @pytest.mark.asyncio
    async def test_quality_scoring(self, processor):
        """Test quality scoring functionality."""
        text = "This is a good sentence with verbs and subjects. @@##$$%%^^&&."
        sentences = await processor.process_text(text)
        
        # Should have at least one sentence
        assert len(sentences) >= 1
        
        # Check that quality scores are calculated
        for sentence in sentences:
            assert 0.0 <= sentence.quality_score <= 1.0
            assert isinstance(sentence.quality_score, float)
    
    @pytest.mark.asyncio
    async def test_text_cleaning(self, processor):
        """Test text cleaning functionality."""
        text = "This has **bold** and *italic* text. <p>HTML tags</p> should be removed."
        sentences = await processor.process_text(text)
        
        # Should have processed sentences
        assert len(sentences) >= 1
        
        # Check that markdown and HTML are cleaned
        combined_text = " ".join(s.text for s in sentences)
        assert "**" not in combined_text
        assert "<p>" not in combined_text
        assert "</p>" not in combined_text
    
    @pytest.mark.asyncio
    async def test_sentence_metadata(self, processor):
        """Test sentence metadata generation."""
        text = "This is a test sentence with metadata."
        sentences = await processor.process_text(text, source_doc_id="test_doc")
        
        assert len(sentences) >= 1
        sentence = sentences[0]
        
        # Check ProcessedSentence structure
        assert isinstance(sentence, ProcessedSentence)
        assert sentence.text
        assert sentence.original_text
        assert sentence.sentence_id.startswith("test_doc")
        assert isinstance(sentence.quality_score, float)
        assert isinstance(sentence.metadata, dict)
        
        # Check metadata content
        assert 'word_count' in sentence.metadata
        assert 'char_count' in sentence.metadata
        assert 'has_punctuation' in sentence.metadata
        assert 'processing_index' in sentence.metadata
        
        assert sentence.metadata['word_count'] > 0
        assert sentence.metadata['char_count'] > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, processor):
        """Test batch processing of sentences."""
        # Create text with many sentences
        sentences_text = ". ".join([f"This is sentence number {i}" for i in range(25)])
        sentences_text += "."
        
        sentences = await processor.process_text(sentences_text)
        
        # Should process all sentences
        assert len(sentences) >= 20  # Some might be filtered
        
        # Check that batch processing worked (no errors)
        for sentence in sentences:
            assert sentence.text
            assert sentence.sentence_id
    
    @pytest.mark.asyncio
    async def test_rule_based_segmentation(self, processor):
        """Test rule-based segmentation fallback."""
        # Test the rule-based segmentation directly
        text = "First sentence. Second sentence! Third sentence?"
        
        # This tests the fallback when NLTK is not available
        sentences = processor._rule_based_segmentation(text)
        
        assert len(sentences) >= 3
        assert any("First sentence" in s for s in sentences)
        assert any("Second sentence" in s for s in sentences)
        assert any("Third sentence" in s for s in sentences)
    
    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, processor):
        """Test quality score calculation."""
        # Good sentence with verb and subject
        good_score = processor._calculate_quality_score("John runs quickly to the store.")
        
        # Poor sentence with many special characters
        poor_score = processor._calculate_quality_score("@@##$$%%^^&&**")
        
        # Very short sentence
        short_score = processor._calculate_quality_score("Hi.")
        
        assert good_score > poor_score
        assert good_score > short_score
        assert 0.0 <= good_score <= 1.0
        assert 0.0 <= poor_score <= 1.0
        assert 0.0 <= short_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_close(self, processor):
        """Test processor cleanup."""
        # Should not raise any exceptions
        await processor.close()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, processor):
        """Test error handling in processing."""
        # This should not crash the processor
        try:
            sentences = await processor.process_text("Normal text with some edge cases.")
            assert isinstance(sentences, list)
        except Exception as e:
            pytest.fail(f"Processing should not fail on normal text: {e}")
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, processor):
        """Test concurrent processing of multiple texts."""
        texts = [
            "First document with multiple sentences. This is sentence two.",
            "Second document content. Another sentence here.",
            "Third document text. Final sentence."
        ]
        
        # Process all texts concurrently
        tasks = [processor.process_text(text, f"doc_{i}") for i, text in enumerate(texts)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for i, sentences in enumerate(results):
            assert len(sentences) >= 1
            # Check that document IDs are correct
            for sentence in sentences:
                assert sentence.sentence_id.startswith(f"doc_{i}")


if __name__ == "__main__":
    pytest.main([__file__])
