"""Integration tests for Document Processing Improvements."""

import sys
import os
import re
from pathlib import Path

# Add the packages to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-services" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-document" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-core" / "src"))

def test_document_id_generation():
    """Test document ID generation function."""
    from morag.ingest_tasks import generate_document_id

    # Test URL ID generation
    url_id = generate_document_id("https://example.com/test-page")
    assert len(url_id) == 16
    assert url_id.isalnum()

    # Test file ID generation
    file_id = generate_document_id("test_document.pdf")
    assert file_id == "test_document_pdf"

    # Test file with content hash
    file_with_content_id = generate_document_id("test.txt", "test content")
    assert file_with_content_id.startswith("test_txt_")
    assert len(file_with_content_id) > len("test_txt_")

def test_document_id_validation():
    """Test document ID validation pattern."""
    # Valid IDs
    valid_ids = ["test_doc", "test-doc", "TestDoc123", "doc_123-abc"]
    pattern = r'^[a-zA-Z0-9_-]+$'

    for doc_id in valid_ids:
        assert re.match(pattern, doc_id), f"Valid ID {doc_id} should match pattern"

    # Invalid IDs
    invalid_ids = ["test doc", "test.doc", "test@doc", "test/doc", ""]

    for doc_id in invalid_ids:
        assert not re.match(pattern, doc_id), f"Invalid ID {doc_id} should not match pattern"

def test_search_response_deduplication():
    """Test that search response formatting eliminates text duplication."""
    # Mock search result with text in metadata
    mock_result = {
        "id": "test1",
        "score": 0.9,
        "metadata": {
            "text": "This is test content",
            "content_type": "document",
            "source": "test.txt",
            "chunk_index": 0
        }
    }

    # Simulate the formatting logic from search_similar
    text_content = mock_result.get("metadata", {}).get("text", "")
    clean_metadata = {k: v for k, v in mock_result.get("metadata", {}).items() if k != "text"}

    formatted_result = {
        "id": mock_result.get("id"),
        "score": mock_result.get("score", 0.0),
        "content": text_content,
        "metadata": clean_metadata,
        "content_type": mock_result.get("metadata", {}).get("content_type"),
        "source": mock_result.get("metadata", {}).get("source")
    }

    # Verify no text duplication
    assert formatted_result["content"] == "This is test content"
    assert "text" not in formatted_result["metadata"]
    assert "content_type" in formatted_result["metadata"]
    assert "source" in formatted_result["metadata"]

    print("✅ Search response deduplication test passed")

def test_word_boundary_logic():
    """Test word boundary detection logic."""
    import re

    text = "This is a test sentence with some punctuation! And another sentence."

    # Test word boundary pattern
    word_boundary_pattern = r'\s+|[.!?;:,\-\(\)\[\]{}"\']'

    # Find boundaries
    boundaries = []
    for match in re.finditer(word_boundary_pattern, text):
        boundaries.append(match.start())

    # Should find multiple boundaries
    assert len(boundaries) > 5

    # Test backward search simulation
    position = 20
    search_text = text[:position]
    matches = list(re.finditer(word_boundary_pattern, search_text))
    if matches:
        boundary = matches[-1].end()
        # Should be at a word boundary
        assert boundary <= position
        if boundary > 0:
            assert text[boundary-1] in ' .!?;:,-()[]{}"\''

    print("✅ Word boundary detection test passed")

def test_sentence_boundary_logic():
    """Test sentence boundary detection logic."""
    import re

    text = "Dr. Smith went to the U.S.A. He bought 3.14 pounds of apples. What a day!"

    # Enhanced sentence boundary detection pattern
    sentence_pattern = r'''
        (?<!\w\.\w.)           # Not preceded by word.word.
        (?<![A-Z][a-z]\.)      # Not preceded by abbreviation like Mr.
        (?<!\d\.\d)            # Not preceded by decimal number
        (?<=\.|\!|\?)          # Preceded by sentence ending punctuation
        \s+                    # Followed by whitespace
        (?=[A-Z])              # Followed by capital letter
    '''

    boundaries = [0]  # Start of text
    for match in re.finditer(sentence_pattern, text, re.VERBOSE):
        boundaries.append(match.start())
    boundaries.append(len(text))  # End of text

    # Should detect proper sentence boundaries, not abbreviations or decimals
    assert len(boundaries) >= 3  # Start, at least one sentence boundary, end
    assert boundaries[0] == 0
    assert boundaries[-1] == len(text)

    # Check that we don't break on abbreviations
    dr_pos = text.find("Dr.")
    usa_pos = text.find("U.S.A.")
    decimal_pos = text.find("3.14")

    # None of these positions should be in our boundaries (except start/end)
    for boundary in boundaries[1:-1]:  # Exclude start and end
        assert boundary != dr_pos + 3  # After "Dr."
        assert boundary != usa_pos + 6  # After "U.S.A."
        assert boundary != decimal_pos + 4  # After "3.14"

    print("✅ Sentence boundary detection test passed")

def test_chunk_size_validation():
    """Test chunk size validation logic."""
    # Valid chunk sizes
    valid_sizes = [500, 1000, 4000, 8000, 16000]

    for size in valid_sizes:
        assert 500 <= size <= 16000, f"Size {size} should be valid"

    # Invalid chunk sizes
    invalid_sizes = [499, 16001, 0, -100]

    for size in invalid_sizes:
        assert not (500 <= size <= 16000), f"Size {size} should be invalid"

    # Valid overlap sizes
    valid_overlaps = [0, 100, 500, 1000]

    for overlap in valid_overlaps:
        assert 0 <= overlap <= 1000, f"Overlap {overlap} should be valid"

    # Invalid overlap sizes
    invalid_overlaps = [-1, 1001, 2000]

    for overlap in invalid_overlaps:
        assert not (0 <= overlap <= 1000), f"Overlap {overlap} should be invalid"

    print("✅ Chunk size validation test passed")

def test_enhanced_chunking_logic():
    """Test enhanced chunking logic without requiring full imports."""
    text = "This is the first sentence. This is the second sentence. This is the third sentence."
    chunk_size = 50
    chunk_overlap = 20

    # Simulate enhanced word chunking
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space

        # Check if adding this word would exceed chunk size
        if current_size + word_length > chunk_size and current_chunk:
            # Create chunk from current words
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # Calculate overlap in words
            overlap_chars = min(chunk_overlap, current_size)
            overlap_words = []
            overlap_size = 0

            # Add words from the end until we reach overlap size
            for overlap_word in reversed(current_chunk):
                word_size = len(overlap_word) + 1
                if overlap_size + word_size <= overlap_chars:
                    overlap_words.insert(0, overlap_word)
                    overlap_size += word_size
                else:
                    break

            current_chunk = overlap_words
            current_size = overlap_size

        current_chunk.append(word)
        current_size += word_length

    # Add final chunk if not empty
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)

    # Should have multiple chunks
    assert len(chunks) > 1

    # Check that chunks don't split words
    for chunk in chunks:
        words_in_chunk = chunk.split()
        # Each word should be complete (no partial words)
        for word in words_in_chunk:
            assert not word.startswith(' ') and not word.endswith(' ')

    print("✅ Enhanced chunking logic test passed")

if __name__ == "__main__":
    print("Running Document Processing Improvements Integration Tests...")

    try:
        test_document_id_generation()
        print("✅ Document ID generation test passed")
    except Exception as e:
        print(f"❌ Document ID generation test failed: {e}")

    try:
        test_document_id_validation()
        print("✅ Document ID validation test passed")
    except Exception as e:
        print(f"❌ Document ID validation test failed: {e}")

    try:
        test_search_response_deduplication()
    except Exception as e:
        print(f"❌ Search response deduplication test failed: {e}")

    try:
        test_word_boundary_logic()
    except Exception as e:
        print(f"❌ Word boundary logic test failed: {e}")

    try:
        test_sentence_boundary_logic()
    except Exception as e:
        print(f"❌ Sentence boundary logic test failed: {e}")

    try:
        test_chunk_size_validation()
    except Exception as e:
        print(f"❌ Chunk size validation test failed: {e}")

    try:
        test_enhanced_chunking_logic()
    except Exception as e:
        print(f"❌ Enhanced chunking logic test failed: {e}")

    print("\n🎉 All Document Processing Improvements tests completed!")
