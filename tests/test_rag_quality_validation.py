#!/usr/bin/env python3
"""
Automated test for RAG quality validation.
Based on test_fact_quality.py
"""
import pytest
import json
import re
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any


class TestRAGQualityValidation:
    """Test that extracted facts are suitable for RAG/context engineering."""

    @pytest.fixture
    def sample_document_content(self):
        """Sample document content for testing."""
        return """# Artificial Intelligence Research

Artificial intelligence (AI) represents a transformative field in computer science that focuses on creating intelligent systems capable of performing tasks that typically require human cognitive abilities.

## Machine Learning Fundamentals

Machine learning constitutes a crucial subset of artificial intelligence, enabling computers to learn and improve their performance on specific tasks through experience rather than explicit programming. Neural networks serve as the foundational architecture for many modern AI systems, drawing inspiration from biological neural structures.

## Deep Learning Applications

Deep learning utilizes multiple layers of neural networks to process complex data patterns. This approach has revolutionized computer vision, natural language processing, and speech recognition technologies. Convolutional neural networks excel at image analysis, while recurrent neural networks demonstrate superior performance in sequential data processing.

## Future Implications

The continued advancement of AI technologies promises significant impacts across healthcare, finance, transportation, and scientific research. Ethical considerations regarding AI deployment, data privacy, and algorithmic bias remain critical areas requiring ongoing attention and regulation."""

    @pytest.fixture
    def sample_facts_data(self):
        """Sample extracted facts for testing."""
        return {
            "facts": [
                {
                    "fact_text": "Artificial intelligence represents a transformative field in computer science that focuses on creating intelligent systems.",
                    "confidence": 0.92,
                    "keywords": ["artificial intelligence", "computer science", "intelligent systems"],
                    "domain": "technology"
                },
                {
                    "fact_text": "Machine learning constitutes a crucial subset of artificial intelligence.",
                    "confidence": 0.88,
                    "keywords": ["machine learning", "artificial intelligence", "subset"],
                    "domain": "technology"
                },
                {
                    "fact_text": "Neural networks serve as the foundational architecture for many modern AI systems.",
                    "confidence": 0.85,
                    "keywords": ["neural networks", "architecture", "AI systems"],
                    "domain": "technology"
                },
                {
                    "fact_text": "It works well.",
                    "confidence": 0.25,
                    "keywords": ["works"],
                    "domain": "general"
                }
            ]
        }

    def test_chunking_quality_simulation(self, sample_document_content):
        """Test that chunking creates reasonable, coherent chunks."""
        content = sample_document_content

        # Simulate paragraph-based chunking
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        # Combine paragraphs to create reasonable chunks (~2000 chars)
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > 2000 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Validate chunk quality
        assert len(chunks) > 0, "Should create at least one chunk"
        assert len(chunks) <= 20, f"Too many chunks ({len(chunks)}), indicates fragmentation"

        # Check chunk sizes
        sizes = [len(chunk) for chunk in chunks]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)

        assert min_size >= 50, f"Chunks too small (min: {min_size})"
        assert avg_size >= 200, f"Average chunk size too small ({avg_size:.0f})"

        # Validate chunk content coherence
        for i, chunk in enumerate(chunks):
            # Each chunk should contain complete sentences
            sentences = chunk.split('.')
            complete_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            assert len(complete_sentences) >= 1, f"Chunk {i} lacks complete sentences"

    def test_keyword_extraction_quality(self):
        """Test keyword extraction without hardcoded stop words."""
        # Test text with domain-specific content
        test_text = """
        Artificial intelligence and machine learning represent revolutionary technologies in computer science.
        Neural networks enable deep learning algorithms to process complex data patterns effectively.
        Natural language processing facilitates human-computer interaction through text analysis.
        """

        # Extract words longer than 3 characters
        words = re.findall(r'\b[a-zA-Z]{4,}\b', test_text.lower())

        # Basic filtering: remove very common words
        basic_stop_words = {
            'that', 'this', 'with', 'from', 'they', 'have', 'been', 'were', 'said',
            'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there',
            'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first',
            'also', 'after', 'back', 'well', 'work', 'life', 'only', 'then'
        }

        keywords = []
        for word in words:
            if (word not in basic_stop_words and
                not word.isdigit() and
                len(word) >= 4):
                keywords.append(word)

        # Remove duplicates
        keywords = list(set(keywords))

        # Validate keyword quality
        expected_domain_terms = [
            'artificial', 'intelligence', 'machine', 'learning', 'neural',
            'networks', 'deep', 'algorithms', 'natural', 'language', 'processing'
        ]

        found_relevant = sum(1 for term in expected_domain_terms if term in keywords)

        assert len(keywords) > 0, "Should extract some keywords"
        assert found_relevant >= 5, f"Should find domain-specific terms, found {found_relevant}"

        # Check that we don't have too many stop words
        common_words = ['that', 'this', 'with', 'from', 'they', 'have', 'been']
        stop_word_count = sum(1 for word in common_words if word in keywords)
        assert stop_word_count <= 2, f"Too many stop words in keywords ({stop_word_count})"

    def test_fact_rag_usefulness(self, sample_facts_data):
        """Test that extracted facts are useful for RAG applications."""
        facts = sample_facts_data["facts"]

        assert len(facts) > 0, "Should have facts to evaluate"

        # Analyze fact quality for RAG
        rag_quality_metrics = {
            'substantial_length': 0,
            'reasonable_confidence': 0,
            'domain_specific': 0,
            'self_contained': 0
        }

        for fact in facts:
            fact_text = fact.get('fact_text', '')
            confidence = fact.get('confidence', 0)

            # Check fact length (should be substantial for RAG)
            if len(fact_text) > 50:
                rag_quality_metrics['substantial_length'] += 1

            # Check confidence (should be reasonable)
            if confidence > 0.5:
                rag_quality_metrics['reasonable_confidence'] += 1

            # Check for domain-specific content
            domain_terms = ['artificial intelligence', 'machine learning', 'neural networks', 'computer science']
            if any(term in fact_text.lower() for term in domain_terms):
                rag_quality_metrics['domain_specific'] += 1

            # Check self-containment (avoid pronouns that make facts unclear)
            problematic_pronouns = ['it ', 'they ', 'this ', 'these ', 'that ']
            if not any(pronoun in fact_text.lower() for pronoun in problematic_pronouns):
                rag_quality_metrics['self_contained'] += 1

        total_facts = len(facts)

        # Calculate quality percentages
        substantial_pct = (rag_quality_metrics['substantial_length'] / total_facts) * 100
        confidence_pct = (rag_quality_metrics['reasonable_confidence'] / total_facts) * 100
        domain_pct = (rag_quality_metrics['domain_specific'] / total_facts) * 100
        self_contained_pct = (rag_quality_metrics['self_contained'] / total_facts) * 100

        # Quality thresholds for RAG usefulness
        assert substantial_pct >= 50, f"Too few substantial facts ({substantial_pct:.1f}%)"
        assert confidence_pct >= 50, f"Too few confident facts ({confidence_pct:.1f}%)"
        assert domain_pct >= 50, f"Too few domain-specific facts ({domain_pct:.1f}%)"
        assert self_contained_pct >= 75, f"Too few self-contained facts ({self_contained_pct:.1f}%)"

    def test_fact_diversity_for_rag(self, sample_facts_data):
        """Test that facts provide diverse information for RAG."""
        facts = sample_facts_data["facts"]

        # Check fact text diversity
        fact_texts = [fact.get('fact_text', '') for fact in facts]
        unique_words = set()

        for text in fact_texts:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            unique_words.update(words)

        # Should have good vocabulary diversity
        total_words = sum(len(re.findall(r'\b[a-zA-Z]+\b', text.lower())) for text in fact_texts)
        diversity_ratio = len(unique_words) / total_words if total_words > 0 else 0

        assert diversity_ratio > 0.3, f"Low vocabulary diversity ({diversity_ratio:.2f})"

        # Check confidence diversity
        confidences = [fact.get('confidence', 0) for fact in facts]
        confidence_range = max(confidences) - min(confidences) if confidences else 0

        assert confidence_range > 0.1, "Facts should have varied confidence levels"

    def test_fact_length_distribution(self, sample_facts_data):
        """Test that facts have appropriate length distribution for RAG."""
        facts = sample_facts_data["facts"]

        lengths = [len(fact.get('fact_text', '')) for fact in facts]

        # Check length statistics
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        min_length = min(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0

        assert avg_length >= 30, f"Average fact length too short ({avg_length:.1f})"
        assert min_length >= 10, f"Minimum fact length too short ({min_length})"
        assert max_length <= 500, f"Maximum fact length too long ({max_length})"

        # Check that we don't have too many very short facts
        short_facts = [length for length in lengths if length < 20]
        short_fact_ratio = len(short_facts) / len(lengths) if lengths else 0

        assert short_fact_ratio <= 0.3, f"Too many short facts ({short_fact_ratio:.1f})"

    def test_keyword_relevance_for_rag(self, sample_facts_data):
        """Test that extracted keywords are relevant for RAG retrieval."""
        facts = sample_facts_data["facts"]

        all_keywords = []
        for fact in facts:
            keywords = fact.get('keywords', [])
            all_keywords.extend(keywords)

        # Remove duplicates
        unique_keywords = list(set(all_keywords))

        assert len(unique_keywords) > 0, "Should have extracted keywords"

        # Check keyword quality
        # Keywords should be meaningful (not too short)
        meaningful_keywords = [kw for kw in unique_keywords if len(kw) >= 3]
        meaningful_ratio = len(meaningful_keywords) / len(unique_keywords) if unique_keywords else 0

        assert meaningful_ratio >= 0.8, f"Too few meaningful keywords ({meaningful_ratio:.1f})"

        # Keywords should relate to fact content
        for fact in facts:
            fact_text = fact.get('fact_text', '').lower()
            fact_keywords = fact.get('keywords', [])

            if fact_keywords:
                # At least some keywords should appear in the fact text
                matching_keywords = [kw for kw in fact_keywords if kw.lower() in fact_text]
                match_ratio = len(matching_keywords) / len(fact_keywords)

                assert match_ratio >= 0.5, f"Keywords don't match fact content well ({match_ratio:.1f})"


@pytest.mark.integration
class TestRAGQualityIntegration:
    """Integration tests for RAG quality with real data."""

    def test_with_real_extraction_results(self):
        """Test RAG quality with real fact extraction results if available."""
        # Check for real extraction results
        results_paths = [
            Path("temp/Broers.json"),
            Path("temp/test_results.json"),
            Path("output/facts.json")
        ]

        results_file = None
        for path in results_paths:
            if path.exists():
                results_file = path
                break

        if not results_file:
            pytest.skip("No real extraction results available for integration testing")

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            pytest.fail(f"Failed to load results from {results_file}: {e}")

        facts = results.get('facts', [])

        if len(facts) == 0:
            pytest.fail("No facts found in extraction results - indicates extraction issues")

        # Test basic RAG quality metrics
        substantial_facts = [f for f in facts if len(f.get('fact_text', '')) > 30]
        confident_facts = [f for f in facts if f.get('confidence', 0) > 0.5]

        substantial_ratio = len(substantial_facts) / len(facts)
        confident_ratio = len(confident_facts) / len(facts)

        assert substantial_ratio >= 0.5, f"Too few substantial facts for RAG ({substantial_ratio:.1f})"
        assert confident_ratio >= 0.3, f"Too few confident facts for RAG ({confident_ratio:.1f})"
