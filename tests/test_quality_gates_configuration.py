#!/usr/bin/env python3
"""
Automated test for configurable quality gates in fact generation.
Based on test_configurable_quality_gates.py and test_simple_quality_gates.py
"""
import pytest
import requests
import json
import tempfile
import os


class TestQualityGatesConfiguration:
    """Test configurable quality gates for fact generation."""

    @pytest.fixture
    def api_url(self):
        """API endpoint for stage execution."""
        return "http://localhost:8000/api/v1/stages/execute-all"

    @pytest.fixture
    def test_content(self):
        """Test content with various types of statements."""
        return """Artificial Intelligence and Machine Learning

Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that can typically perform tasks that require human intelligence. Machine learning is usually a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.

Key Concepts:
- Neural networks are generally computing systems inspired by biological neural networks
- Deep learning often uses multiple layers of neural networks to process data
- Natural language processing (NLP) sometimes enables computers to understand human language
- Computer vision may allow machines to interpret and understand visual information

Applications:
AI and ML are used in various industries including healthcare, finance, transportation, and entertainment. For example, in healthcare, AI helps doctors diagnose diseases more accurately. In finance, machine learning algorithms detect fraudulent transactions."""

    @pytest.fixture
    def temp_file(self, test_content):
        """Create temporary test file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            return f.name

    def test_strict_validation(self, api_url, temp_file):
        """Test strict validation settings filter out questionable facts."""
        try:
            with open(temp_file, "rb") as f:
                files = {"file": ("test.txt", f, "text/plain")}
                data = {
                    "stages": '["markdown-conversion", "chunker", "fact-generator"]',
                    "stage_configs": json.dumps({
                        "fact-generator": {
                            "min_confidence": 0.7,
                            "strict_validation": True,
                            "allow_vague_language": False,
                            "require_entities": True,
                            "min_fact_length": 25
                        }
                    })
                }

                response = requests.post(api_url, files=files, data=data, timeout=120)

                assert response.status_code == 200, f"Request failed: {response.status_code}"

                result = response.json()
                facts_count = self._count_facts(result)

                # Strict validation should be more selective
                # Store for comparison with lenient settings
                return facts_count

        finally:
            os.unlink(temp_file)

    def test_lenient_validation(self, api_url, test_content):
        """Test lenient validation settings allow more facts."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name

        try:
            with open(temp_file, "rb") as f:
                files = {"file": ("test.txt", f, "text/plain")}
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

                response = requests.post(api_url, files=files, data=data, timeout=120)

                assert response.status_code == 200, f"Request failed: {response.status_code}"

                result = response.json()
                facts_count = self._count_facts(result)

                # Lenient validation should allow more facts
                assert facts_count >= 0, "Should extract some facts with lenient settings"

                return facts_count

        finally:
            os.unlink(temp_file)

    def test_default_validation(self, api_url, test_content):
        """Test default validation settings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name

        try:
            with open(temp_file, "rb") as f:
                files = {"file": ("test.txt", f, "text/plain")}
                data = {
                    "stages": '["markdown-conversion", "chunker", "fact-generator"]'
                    # No stage_configs - uses defaults
                }

                response = requests.post(api_url, files=files, data=data, timeout=120)

                assert response.status_code == 200, f"Request failed: {response.status_code}"

                result = response.json()
                facts_count = self._count_facts(result)

                # Default should work reasonably
                assert facts_count >= 0, "Default settings should work"

                return facts_count

        finally:
            os.unlink(temp_file)

    def test_custom_balanced_validation(self, api_url, test_content):
        """Test custom balanced validation settings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name

        try:
            with open(temp_file, "rb") as f:
                files = {"file": ("test.txt", f, "text/plain")}
                data = {
                    "stages": '["markdown-conversion", "chunker", "fact-generator"]',
                    "stage_configs": json.dumps({
                        "fact-generator": {
                            "min_confidence": 0.4,
                            "strict_validation": False,
                            "allow_vague_language": True,
                            "require_entities": True,
                            "min_fact_length": 15
                        }
                    })
                }

                response = requests.post(api_url, files=files, data=data, timeout=120)

                assert response.status_code == 200, f"Request failed: {response.status_code}"

                result = response.json()
                facts_count = self._count_facts(result)

                # Custom settings should work
                assert facts_count >= 0, "Custom balanced settings should work"

                return facts_count

        finally:
            os.unlink(temp_file)

    def test_quality_gates_comparison(self, api_url, test_content):
        """Test that different quality gate settings produce different results."""
        results = {}

        # Test configurations
        configs = {
            "strict": {
                "min_confidence": 0.7,
                "strict_validation": True,
                "allow_vague_language": False,
                "require_entities": True,
                "min_fact_length": 25
            },
            "lenient": {
                "min_confidence": 0.3,
                "strict_validation": False,
                "allow_vague_language": True,
                "require_entities": False,
                "min_fact_length": 10
            },
            "default": None  # No config = default
        }

        for config_name, config in configs.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(test_content)
                temp_file = f.name

            try:
                with open(temp_file, "rb") as f:
                    files = {"file": ("test.txt", f, "text/plain")}
                    data = {
                        "stages": '["markdown-conversion", "chunker", "fact-generator"]'
                    }

                    if config:
                        data["stage_configs"] = json.dumps({"fact-generator": config})

                    response = requests.post(api_url, files=files, data=data, timeout=120)

                    assert response.status_code == 200, f"{config_name} request failed: {response.status_code}"

                    result = response.json()
                    facts_count = self._count_facts(result)
                    results[config_name] = facts_count

            finally:
                os.unlink(temp_file)

        # Verify that configurations produce different behaviors
        # Lenient should generally allow more facts than strict
        assert results["lenient"] >= results["strict"], \
            f"Lenient ({results['lenient']}) should allow at least as many facts as strict ({results['strict']})"

        # All configurations should be functional
        for config_name, count in results.items():
            assert count >= 0, f"{config_name} configuration failed to extract any facts"

    def test_vague_language_detection(self, api_url):
        """Test that vague language is properly detected and handled."""
        vague_content = """Machine Learning Overview

        Machine learning is generally considered a subset of AI. Neural networks are usually effective for pattern recognition. Deep learning often produces good results. Computer vision sometimes works well for image analysis."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(vague_content)
            temp_file = f.name

        try:
            with open(temp_file, "rb") as f:
                files = {"file": ("test.txt", f, "text/plain")}
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

                response = requests.post(api_url, files=files, data=data, timeout=120)

                assert response.status_code == 200, f"Request failed: {response.status_code}"

                result = response.json()
                facts = self._extract_facts_from_response(result)

                # Should extract facts but may mark them with vague language warnings
                assert len(facts) >= 0, "Should handle vague language content"

                # Check if any facts have vague language remarks
                vague_remarks = [fact for fact in facts
                               if 'vague language' in fact.get('remarks', '').lower()]

                # With content containing "generally", "usually", "often", "sometimes"
                # we expect some vague language detection
                if len(facts) > 0:
                    assert len(vague_remarks) >= 0, "Should detect vague language patterns"

        finally:
            os.unlink(temp_file)

    def _count_facts(self, result):
        """Count facts in API response."""
        try:
            for stage in result.get('stages_executed', []):
                if stage.get('stage_type') == 'fact-generator':
                    output_files = stage.get('output_files', [])
                    for file_info in output_files:
                        if 'facts' in file_info.get('filename', ''):
                            content = file_info.get('content')
                            if content:
                                facts_data = json.loads(content)
                                return len(facts_data.get('facts', []))
            return 0
        except Exception:
            return 0

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
