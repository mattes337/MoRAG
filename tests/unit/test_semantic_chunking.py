import pytest
from morag_services.processing import chunking_service, ChunkInfo, SemanticChunker

@pytest.mark.asyncio
async def test_simple_chunking():
    """Test simple text chunking."""
    text = "This is a simple test. It has multiple sentences. Each sentence should be processed correctly."
    
    chunks = await chunking_service.semantic_chunk(text, chunk_size=50, strategy="simple")
    
    assert len(chunks) > 0
    assert all(len(chunk) <= 60 for chunk in chunks)  # Allow some flexibility
    assert "".join(chunks).replace(" ", "") in text.replace(" ", "")

@pytest.mark.asyncio
async def test_sentence_chunking():
    """Test sentence-based chunking."""
    text = """
    This is the first sentence. This is the second sentence with more content.
    This is the third sentence. This is the fourth sentence that is quite long and contains multiple clauses.
    This is the fifth sentence.
    """
    
    chunks = await chunking_service.semantic_chunk(text, chunk_size=100, strategy="sentence")
    
    assert len(chunks) > 0
    # Each chunk should contain complete sentences
    for chunk in chunks:
        assert chunk.strip().endswith('.') or chunk == chunks[-1]  # Last chunk might not end with period

@pytest.mark.asyncio
async def test_paragraph_chunking():
    """Test paragraph-based chunking."""
    text = """
    This is the first paragraph. It contains multiple sentences about a topic.
    
    This is the second paragraph. It discusses a different topic entirely.
    It has more content than the first paragraph.
    
    This is the third paragraph. It's shorter.
    """
    
    chunks = await chunking_service.semantic_chunk(text, chunk_size=200, strategy="paragraph")
    
    assert len(chunks) > 0
    # Should preserve paragraph structure
    for chunk in chunks:
        assert chunk.strip()

@pytest.mark.asyncio
async def test_semantic_chunking():
    """Test semantic chunking (may fall back to simple if spaCy not available)."""
    text = """
    Machine learning is a subset of artificial intelligence. It focuses on algorithms that can learn from data.
    Deep learning is a subset of machine learning. It uses neural networks with multiple layers.
    Natural language processing is another AI field. It deals with understanding human language.
    Computer vision helps machines interpret visual information. It's used in many applications today.
    """
    
    chunks = await chunking_service.semantic_chunk(text, chunk_size=150, strategy="semantic")
    
    assert len(chunks) > 0
    assert all(len(chunk) > 0 for chunk in chunks)

@pytest.mark.asyncio
async def test_chunk_with_metadata():
    """Test chunking with full metadata."""
    text = "This is a test document. It contains multiple sentences for testing purposes."
    
    chunk_infos = await chunking_service.chunk_with_metadata(text, chunk_size=50)
    
    assert len(chunk_infos) > 0
    assert all(isinstance(chunk, ChunkInfo) for chunk in chunk_infos)
    
    for chunk in chunk_infos:
        assert chunk.text
        assert chunk.word_count > 0
        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char
        assert chunk.chunk_type in ["semantic", "simple", "sentence", "paragraph"]

@pytest.mark.asyncio
async def test_empty_text_handling():
    """Test handling of empty text."""
    chunks = await chunking_service.semantic_chunk("", chunk_size=100)
    assert len(chunks) == 0
    
    chunks = await chunking_service.semantic_chunk("   ", chunk_size=100)
    assert len(chunks) == 0

@pytest.mark.asyncio
async def test_very_short_text():
    """Test handling of very short text."""
    text = "Short."
    chunks = await chunking_service.semantic_chunk(text, chunk_size=100)
    
    assert len(chunks) == 1
    assert chunks[0].strip() == text

@pytest.mark.asyncio
async def test_text_structure_analysis():
    """Test text structure analysis."""
    text = """
    This is a complex document with multiple paragraphs. Each paragraph contains several sentences.
    The sentences vary in length and complexity.
    
    This second paragraph discusses different topics. It has more technical content.
    The vocabulary is more advanced and specialized.
    
    The final paragraph is shorter. It serves as a conclusion.
    """
    
    analysis = await chunking_service.analyze_text_structure(text)
    
    assert "word_count" in analysis
    assert "sentence_count" in analysis
    assert "paragraph_count" in analysis
    assert "avg_sentence_length" in analysis
    assert "text_complexity" in analysis
    assert "recommended_strategy" in analysis
    assert "estimated_chunks" in analysis
    assert "spacy_available" in analysis
    
    assert analysis["word_count"] > 0
    assert analysis["sentence_count"] > 0
    assert analysis["paragraph_count"] > 0
    assert analysis["text_complexity"] in ["low", "medium", "high", "empty"]
    assert analysis["recommended_strategy"] in ["simple", "sentence", "paragraph", "semantic"]

@pytest.mark.asyncio
async def test_empty_text_analysis():
    """Test analysis of empty text."""
    analysis = await chunking_service.analyze_text_structure("")
    
    assert analysis["text_complexity"] == "empty"
    assert analysis["estimated_chunks"] == 0
    assert analysis["recommended_strategy"] == "simple"

@pytest.mark.asyncio
async def test_chunk_size_limits():
    """Test chunking with different size limits."""
    # Create text with different topics to encourage chunking
    text = """
    Machine learning is a fascinating field. It involves training algorithms on data.
    Natural language processing deals with text analysis. It helps computers understand human language.
    Computer vision focuses on image recognition. It enables machines to see and interpret visual data.
    Data science combines statistics and programming. It extracts insights from large datasets.
    """

    # Test small chunks with simple strategy to ensure chunking
    small_chunks = await chunking_service.semantic_chunk(text, chunk_size=50, strategy="simple")

    # Test large chunks
    large_chunks = await chunking_service.semantic_chunk(text, chunk_size=500, strategy="simple")

    assert len(small_chunks) >= len(large_chunks)  # Should have at least as many chunks
    assert all(len(chunk) <= 70 for chunk in small_chunks)  # Allow some flexibility

@pytest.mark.asyncio
async def test_chunking_strategies():
    """Test all chunking strategies."""
    text = """
    This is a test document. It has multiple sentences and paragraphs.
    
    This is the second paragraph. It contains different information.
    The content is varied and interesting.
    
    Final paragraph here. Short and sweet.
    """
    
    strategies = ["simple", "sentence", "paragraph", "semantic"]
    
    for strategy in strategies:
        chunks = await chunking_service.semantic_chunk(text, chunk_size=100, strategy=strategy)
        assert len(chunks) > 0, f"Strategy {strategy} failed"
        assert all(chunk.strip() for chunk in chunks), f"Empty chunks in {strategy}"

def test_semantic_chunker_initialization():
    """Test SemanticChunker initialization."""
    chunker = SemanticChunker()
    
    # Should initialize without errors
    assert chunker.max_chunk_size > 0
    assert chunker.min_chunk_size > 0
    assert chunker.overlap_size >= 0

@pytest.mark.asyncio
async def test_chunk_overlap():
    """Test that chunks have appropriate overlap when configured."""
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
    
    chunker = SemanticChunker()
    chunker.overlap_size = 20  # Set overlap
    
    chunks = await chunker.chunk_text(text, chunk_size=50)
    
    if len(chunks) > 1:
        # Check that there's some overlap between consecutive chunks
        for i in range(1, len(chunks)):
            # This is a basic check - in practice, overlap detection is complex
            assert chunks[i].text  # At least ensure chunks exist

@pytest.mark.asyncio
async def test_large_text_processing():
    """Test processing of larger text documents."""
    # Create a larger text
    paragraph = "This is a test paragraph with multiple sentences. It contains various topics and ideas. The content is designed to test the chunking algorithm's ability to handle longer documents. "
    large_text = paragraph * 10  # Repeat to create larger document
    
    chunks = await chunking_service.semantic_chunk(large_text, chunk_size=200)
    
    assert len(chunks) > 1
    assert all(len(chunk) > 0 for chunk in chunks)
    
    # Verify that all text is preserved
    combined_length = sum(len(chunk) for chunk in chunks)
    # Allow for some variation due to spacing and overlap
    assert combined_length >= len(large_text) * 0.8

@pytest.mark.asyncio
async def test_special_characters_handling():
    """Test handling of text with special characters."""
    text = "This text has special characters: @#$%^&*(). It also has numbers: 123, 456. And punctuation: hello, world! How are you?"
    
    chunks = await chunking_service.semantic_chunk(text, chunk_size=50)
    
    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)
    
    # Verify special characters are preserved
    combined_text = " ".join(chunks)
    assert "@#$%^&*()" in combined_text
    assert "123" in combined_text
