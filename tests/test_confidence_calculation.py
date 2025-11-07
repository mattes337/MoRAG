#!/usr/bin/env python3
"""
Automated test for confidence calculation issues in fact generation.
Based on test_confidence_issue.py
"""
import pytest
import requests
import json
import tempfile
import os
from pathlib import Path


class TestConfidenceCalculation:
    """Test confidence calculation with various settings."""

    @pytest.fixture
    def api_url(self):
        """API endpoint for stage execution."""
        return "http://localhost:8000/api/v1/stages/execute-all"

    @pytest.fixture
    def test_document(self):
        """Create a test document with substantial content."""
        content = """# Artificial Intelligence and Machine Learning

Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.

## Key Concepts

Neural networks are computing systems inspired by biological neural networks. Deep learning uses multiple layers of neural networks to process data. Natural language processing (NLP) enables computers to understand human language. Computer vision allows machines to interpret and understand visual information.

## Applications

AI and ML are used in various industries including healthcare, finance, transportation, and entertainment. In healthcare, AI helps doctors diagnose diseases more accurately. In finance, machine learning algorithms detect fraudulent transactions. Self-driving cars use computer vision and sensor data to navigate safely.

## Future Prospects

The field of AI continues to evolve rapidly with new breakthroughs in areas like generative AI, reinforcement learning, and quantum computing. These advances promise to revolutionize how we work, communicate, and solve complex problems.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            return f.name

    def test_lenient_confidence_settings(self, api_url, test_document):
        """Test that lenient settings extract facts with reasonable confidence."""
        try:
            with open(test_document, "rb") as f:
                files = {"file": ("test.md", f, "text/markdown")}
                data = {
                    "stages": '["markdown-conversion", "chunker", "fact-generator"]',
                    "stage_configs": json.dumps({
                        "fact-generator": {
                            "min_confidence": 0.1,  # Very low threshold
                            "strict_validation": False,
                            "allow_vague_language": True,
                            "require_entities": False,
                            "min_fact_length": 5
                        }
                    })
                }

                response = requests.post(api_url, files=files, data=data, timeout=180)

                assert response.status_code == 200, f"Request failed: {response.status_code}"

                result = response.json()
                facts_extracted = self._extract_facts_from_response(result)

                # With lenient settings, we should extract some facts
                assert len(facts_extracted) > 0, "No facts extracted despite lenient settings"

                # Check that facts have reasonable confidence scores
                confidences = [fact.get("extraction_confidence", 0) for fact in facts_extracted]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                assert avg_confidence > 0, "Average confidence is zero"
                assert any(conf > 0.1 for conf in confidences), "No facts meet minimum confidence threshold"

        finally:
            os.unlink(test_document)

    def test_default_confidence_settings(self, api_url, test_document):
        """Test that default settings work appropriately."""
        try:
            with open(test_document, "rb") as f:
                files = {"file": ("test.md", f, "text/markdown")}
                data = {
                    "stages": '["markdown-conversion", "chunker", "fact-generator"]'
                    # No stage_configs - using defaults
                }

                response = requests.post(api_url, files=files, data=data, timeout=180)

                assert response.status_code == 200, f"Request failed: {response.status_code}"

                result = response.json()
                facts_extracted = self._extract_facts_from_response(result)

                # Default settings should still extract some facts from good content
                # If this fails, it indicates confidence calculation issues
                if len(facts_extracted) == 0:
                    pytest.fail("No facts extracted with default settings - confidence calculation may be flawed")

                # Check confidence distribution
                confidences = [fact.get("extraction_confidence", 0) for fact in facts_extracted]
                high_confidence_facts = [conf for conf in confidences if conf > 0.7]

                # At least some facts should have reasonable confidence
                assert len(high_confidence_facts) > 0 or len(facts_extracted) > 5, \
                    "Either high confidence facts or sufficient quantity expected"

        finally:
            os.unlink(test_document)

    def test_confidence_distribution(self, api_url, test_document):
        """Test that confidence scores are distributed reasonably."""
        try:
            with open(test_document, "rb") as f:
                files = {"file": ("test.md", f, "text/markdown")}
                data = {
                    "stages": '["markdown-conversion", "chunker", "fact-generator"]',
                    "stage_configs": json.dumps({
                        "fact-generator": {
                            "min_confidence": 0.2,
                            "strict_validation": False,
                            "allow_vague_language": True,
                            "require_entities": False,
                            "min_fact_length": 10
                        }
                    })
                }

                response = requests.post(api_url, files=files, data=data, timeout=180)

                assert response.status_code == 200, f"Request failed: {response.status_code}"

                result = response.json()
                facts_extracted = self._extract_facts_from_response(result)

                if len(facts_extracted) == 0:
                    pytest.skip("No facts extracted - cannot test confidence distribution")

                confidences = [fact.get("extraction_confidence", 0) for fact in facts_extracted]

                # Check that confidences are not all the same (indicating proper calculation)
                unique_confidences = set(confidences)
                assert len(unique_confidences) > 1, "All facts have identical confidence - calculation may be broken"

                # Check that confidences are in valid range
                assert all(0 <= conf <= 1 for conf in confidences), "Confidence scores outside valid range [0,1]"

                # Check that we don't have suspiciously low confidences for all facts
                avg_confidence = sum(confidences) / len(confidences)
                assert avg_confidence > 0.1, f"Average confidence too low: {avg_confidence}"

        finally:
            os.unlink(test_document)

    def _extract_facts_from_response(self, result):
        """Extract facts from API response."""
        for stage in result.get("stages_executed", []):
            if stage.get("stage_type") == "fact-generator":
                output_files = stage.get("output_files", [])
                for file_info in output_files:
                    if "facts" in file_info.get("filename", ""):
                        content = file_info.get("content")
                        if content:
                            facts_data = json.loads(content)
                            return facts_data.get("facts", [])
        return []


@pytest.mark.integration
class TestConfidenceCalculationIntegration:
    """Integration tests for confidence calculation."""

    def test_confidence_with_real_document(self):
        """Test confidence calculation with a real document if available."""
        # Check if Broers.md exists for testing
        broers_path = Path("temp/Broers.md")
        if not broers_path.exists():
            pytest.skip("Broers.md not available for integration testing")

        api_url = "http://localhost:8000/api/v1/stages/execute-all"

        with open(broers_path, "rb") as f:
            files = {"file": ("Broers.md", f, "text/markdown")}
            data = {
                "stages": '["markdown-conversion", "chunker", "fact-generator"]',
                "stage_configs": json.dumps({
                    "fact-generator": {
                        "min_confidence": 0.3,
                        "strict_validation": False,
                        "allow_vague_language": True,
                        "require_entities": False,
                        "min_fact_length": 10
                    }
                })
            }

            response = requests.post(api_url, files=files, data=data, timeout=180)

            assert response.status_code == 200, f"Request failed: {response.status_code}"

            result = response.json()
            facts_extracted = self._extract_facts_from_response(result)

            # A substantial document like Broers.md should produce many facts
            # If it doesn't, confidence calculation is likely broken
            assert len(facts_extracted) > 10, \
                f"Expected many facts from substantial document, got {len(facts_extracted)}"

    def _extract_facts_from_response(self, result):
        """Extract facts from API response."""
        for stage in result.get("stages_executed", []):
            if stage.get("stage_type") == "fact-generator":
                output_files = stage.get("output_files", [])
                for file_info in output_files:
                    if "facts" in file_info.get("filename", ""):
                        content = file_info.get("content")
                        if content:
                            facts_data = json.loads(content)
                            return facts_data.get("facts", [])
        return []
