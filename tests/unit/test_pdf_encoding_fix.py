"""Tests for PDF text encoding fixes."""

from morag_core.utils import clean_pdf_text_encoding, normalize_text_encoding


class TestPDFEncodingFix:
    """Test PDF text encoding fixes."""

    def test_clean_pdf_text_encoding_soft_hyphen(self):
        """Test cleaning of soft hyphens and related issues."""
        # Test the specific reported issue
        input_text = "extrem  ange\u00ad schlagen"
        expected = "extrem angeschlagen"  # The soft hyphen joins "ange" and "schlagen", spaces are normalized
        result = clean_pdf_text_encoding(input_text)
        assert result == expected

    def test_clean_pdf_text_encoding_soft_hyphen_variations(self):
        """Test various soft hyphen scenarios."""
        test_cases = [
            ("ange\u00ad ", "ange"),  # Soft hyphen at end with space
            ("ange\u00ad", "ange"),   # Soft hyphen at end
            ("word\u00ad next", "wordnext"),  # Soft hyphen joins words
            ("hyphen\u00adated", "hyphenated"),  # Soft hyphen within word
            ("multi\u00adple\u00ad hyphens", "multiplehyphens"),  # Multiple soft hyphens
        ]

        for input_text, expected in test_cases:
            result = clean_pdf_text_encoding(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_clean_pdf_text_encoding_unicode_normalization(self):
        """Test Unicode normalization."""
        # Test various Unicode issues
        test_cases = [
            ("café", "café"),  # Should normalize properly
            ("naïve", "naïve"),  # Should handle diacritics
            ("résumé", "résumé"),  # Should preserve accents
        ]

        for input_text, expected in test_cases:
            result = clean_pdf_text_encoding(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_clean_pdf_text_encoding_ligatures(self):
        """Test ligature fixes."""
        test_cases = [
            ("\ufb01le", "file"),  # fi ligature
            ("\ufb02ow", "flow"),  # fl ligature
            ("\ufb00", "ff"),      # ff ligature
            ("\ufb03", "ffi"),     # ffi ligature
            ("\ufb04", "ffl"),     # ffl ligature
        ]

        for input_text, expected in test_cases:
            result = clean_pdf_text_encoding(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_clean_pdf_text_encoding_quotes_and_dashes(self):
        """Test quote and dash normalization."""
        test_cases = [
            ("\u201cHello\u201d", '"Hello"'),  # Smart quotes
            ("\u2018world\u2019", "'world'"),  # Smart single quotes
            ("em\u2014dash", "em-dash"),       # Em dash
            ("en\u2013dash", "en-dash"),       # En dash
        ]

        for input_text, expected in test_cases:
            result = clean_pdf_text_encoding(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_clean_pdf_text_encoding_common_artifacts(self):
        """Test common PDF encoding artifacts."""
        # Using escape sequences to avoid encoding issues
        test_cases = [
            ("\u00e2\u0080\u0099", "'"),    # â€™ -> '
            ("\u00e2\u0080\u009c", '"'),    # â€œ -> "
            ("\u00e2\u0080\u009d", '"'),    # â€ -> "
            ("\u00e2\u0080\u00a6", "..."), # â€¦ -> ...
            ("\u00e2\u0080\u0094", "-"),   # â€" -> -
        ]

        for input_text, expected in test_cases:
            result = clean_pdf_text_encoding(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_clean_pdf_text_encoding_whitespace_cleanup(self):
        """Test whitespace cleanup."""
        test_cases = [
            ("multiple   spaces", "multiple spaces"),
            ("word-\n word", "wordword"),  # Hyphenated line break
            ("text\n\n\n\nmore", "text\n\nmore"),  # Multiple newlines
            ("  leading and trailing  ", "leading and trailing"),
        ]

        for input_text, expected in test_cases:
            result = clean_pdf_text_encoding(input_text)
            assert result == expected, f"Failed for input: {repr(input_text)}"

    def test_clean_pdf_text_encoding_zero_width_characters(self):
        """Test removal of zero-width characters."""
        test_cases = [
            ("word\u200bword", "wordword"),  # Zero-width space
            ("word\u200cword", "wordword"),  # Zero-width non-joiner
            ("word\u200dword", "wordword"),  # Zero-width joiner
            ("word\ufeffword", "wordword"),  # Byte order mark
            ("word\u00adword", "wordword"),  # Soft hyphen
        ]

        for input_text, expected in test_cases:
            result = clean_pdf_text_encoding(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_normalize_text_encoding_with_bytes(self):
        """Test normalize_text_encoding with byte input."""
        # Test UTF-8 bytes
        utf8_bytes = "Hello world".encode('utf-8')
        result = normalize_text_encoding(utf8_bytes)
        assert result == "Hello world"

        # Test Latin-1 bytes
        latin1_bytes = "café".encode('latin-1')
        result = normalize_text_encoding(latin1_bytes)
        assert "caf" in result  # Should decode successfully

    def test_normalize_text_encoding_with_string(self):
        """Test normalize_text_encoding with string input."""
        # Test with problematic PDF text
        input_text = "extrem  ange\u00ad schlagen"
        result = normalize_text_encoding(input_text)
        assert "angeschlagen" in result
        assert "\u00ad" not in result

    def test_normalize_text_encoding_empty_input(self):
        """Test normalize_text_encoding with empty input."""
        assert normalize_text_encoding("") == ""
        assert normalize_text_encoding(None) == ""
        assert normalize_text_encoding(b"") == ""

    def test_normalize_text_encoding_error_handling(self):
        """Test normalize_text_encoding error handling."""
        # Test with invalid input that might cause errors
        result = normalize_text_encoding(123)  # Invalid type
        assert isinstance(result, str)

    def test_clean_pdf_text_encoding_complex_example(self):
        """Test with a complex real-world example."""
        input_text = """
        This is a PDF docu\u00adment with various encoding issues.
        It contains \u201csmart quotes\u201d and em\u2014dashes.
        There are also \ufb01le names and \ufb02ow charts.
        Some text has   multiple   spaces.



        And excessive newlines.
        """

        result = clean_pdf_text_encoding(input_text)

        # Check that issues are fixed
        assert "\u00ad" not in result
        assert "document" in result
        assert "\u201c" not in result
        assert '"smart quotes"' in result
        assert "\u2014" not in result
        assert "em-dash" in result
        assert "\ufb01" not in result
        assert "file" in result
        assert "\ufb02" not in result
        assert "flow" in result
        assert "   " not in result  # Multiple spaces should be cleaned
        assert "\n\n\n\n" not in result  # Excessive newlines should be cleaned

    def test_clean_pdf_text_encoding_preserves_structure(self):
        """Test that cleaning preserves important text structure."""
        input_text = """# Title

        This is a paragraph with some text.

        ## Subtitle

        Another paragraph here."""

        result = clean_pdf_text_encoding(input_text)

        # Should preserve markdown structure
        assert "# Title" in result
        assert "## Subtitle" in result
        assert result.count("\n\n") >= 2  # Should preserve paragraph breaks

    def test_soft_hyphen_word_joining(self):
        """Test soft hyphen word joining from the reported issue."""
        test_cases = [
            ("ange\u00ad schlagen", "angeschlagen"),
            ("word\u00adpart", "wordpart"),
            ("hyph\u00adenated", "hyphenated"),
            ("com\u00adpound", "compound"),
        ]

        for input_text, expected in test_cases:
            result = clean_pdf_text_encoding(input_text)
            assert result == expected, f"Failed for text: {input_text}"
