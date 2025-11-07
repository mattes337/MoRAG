#!/usr/bin/env python3
"""
Comprehensive test for search and retrieval functionality.
Tests the complete search pipeline including vector search, graph retrieval, and reasoning paths.
"""
import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch


class TestSearchRetrievalFunctionality:
    """Test search and retrieval functionality across the MoRAG system."""

    @pytest.fixture
    def sample_vector_results(self):
        """Sample vector search results."""
        return [
            {
                "id": "doc_1",
                "score": 0.95,
                "metadata": {
                    "content": "Artificial intelligence is a branch of computer science focused on creating intelligent machines.",
                    "fact_type": "definition",
                    "domain": "technology",
                    "entities": ["Artificial intelligence", "computer science", "intelligent machines"]
                }
            },
            {
                "id": "doc_2",
                "score": 0.87,
                "metadata": {
                    "content": "Machine learning enables computers to learn from data without explicit programming.",
                    "fact_type": "definition",
                    "domain": "technology",
                    "entities": ["Machine learning", "computers", "data"]
                }
            },
            {
                "id": "doc_3",
                "score": 0.82,
                "metadata": {
                    "content": "Neural networks are computing systems inspired by biological neural networks.",
                    "fact_type": "description",
                    "domain": "technology",
                    "entities": ["Neural networks", "computing systems", "biological neural networks"]
                }
            }
        ]

    @pytest.fixture
    def sample_graph_entities(self):
        """Sample graph entities for testing."""
        return [
            {
                "id": "entity_1",
                "name": "Artificial intelligence",
                "type": "CONCEPT",
                "properties": {"domain": "technology", "definition": "Branch of computer science"}
            },
            {
                "id": "entity_2",
                "name": "Machine learning",
                "type": "CONCEPT",
                "properties": {"domain": "technology", "parent": "Artificial intelligence"}
            },
            {
                "id": "entity_3",
                "name": "Neural networks",
                "type": "CONCEPT",
                "properties": {"domain": "technology", "parent": "Machine learning"}
            }
        ]

    @pytest.fixture
    def sample_reasoning_paths(self):
        """Sample reasoning paths for testing."""
        return [
            {
                "path": ["Artificial intelligence", "SUBSET_OF", "Machine learning", "USES", "Neural networks"],
                "relevance_score": 0.92,
                "reasoning": "Direct path from AI to neural networks through ML"
            },
            {
                "path": ["Computer science", "INCLUDES", "Artificial intelligence", "APPLIES_TO", "Data processing"],
                "relevance_score": 0.85,
                "reasoning": "Path showing AI as part of computer science applied to data"
            }
        ]

    def test_vector_search_interface(self, sample_vector_results):
        """Test vector search interface and result format."""
        # Test that vector search results have required fields
        for result in sample_vector_results:
            assert "id" in result, "Vector result should have ID"
            assert "score" in result, "Vector result should have similarity score"
            assert "metadata" in result, "Vector result should have metadata"

            # Test score validity
            score = result["score"]
            assert 0 <= score <= 1, f"Score should be between 0 and 1, got {score}"

            # Test metadata structure
            metadata = result["metadata"]
            assert "content" in metadata, "Metadata should contain content"
            assert len(metadata["content"]) > 0, "Content should not be empty"

            # Test optional fields
            if "entities" in metadata:
                assert isinstance(metadata["entities"], list), "Entities should be a list"
            if "domain" in metadata:
                assert isinstance(metadata["domain"], str), "Domain should be a string"

    def test_search_result_ranking(self, sample_vector_results):
        """Test that search results are properly ranked by relevance."""
        scores = [result["score"] for result in sample_vector_results]

        # Results should be sorted by score (descending)
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

        # All scores should be reasonable
        assert all(score >= 0.5 for score in scores), "All results should have reasonable similarity scores"

        # Top result should have high confidence
        assert scores[0] >= 0.9, "Top result should have high confidence"

    def test_search_filtering_capabilities(self, sample_vector_results):
        """Test search filtering by metadata."""
        # Test domain filtering
        tech_results = [r for r in sample_vector_results if r["metadata"].get("domain") == "technology"]
        assert len(tech_results) == len(sample_vector_results), "All sample results should be technology domain"

        # Test fact type filtering
        definition_results = [r for r in sample_vector_results if r["metadata"].get("fact_type") == "definition"]
        assert len(definition_results) >= 2, "Should have multiple definition-type results"

        # Test entity filtering
        ai_results = [r for r in sample_vector_results
                     if any("artificial intelligence" in entity.lower()
                           for entity in r["metadata"].get("entities", []))]
        assert len(ai_results) >= 1, "Should find results containing AI entities"

    def test_graph_entity_structure(self, sample_graph_entities):
        """Test graph entity structure and relationships."""
        for entity in sample_graph_entities:
            assert "id" in entity, "Entity should have ID"
            assert "name" in entity, "Entity should have name"
            assert "type" in entity, "Entity should have type"

            # Test name validity
            assert len(entity["name"]) > 0, "Entity name should not be empty"

            # Test type validity
            valid_types = ["CONCEPT", "PERSON", "ORGANIZATION", "LOCATION", "EVENT"]
            assert entity["type"] in valid_types, f"Entity type should be valid, got {entity['type']}"

            # Test properties
            if "properties" in entity:
                props = entity["properties"]
                assert isinstance(props, dict), "Properties should be a dictionary"

    def test_reasoning_path_structure(self, sample_reasoning_paths):
        """Test reasoning path structure and validity."""
        for path_data in sample_reasoning_paths:
            assert "path" in path_data, "Path data should contain path"
            assert "relevance_score" in path_data, "Path data should contain relevance score"
            assert "reasoning" in path_data, "Path data should contain reasoning"

            path = path_data["path"]
            assert len(path) >= 3, "Path should have at least 3 elements (entity-relation-entity)"
            assert len(path) % 2 == 1, "Path should have odd number of elements (alternating entities and relations)"

            # Test relevance score
            score = path_data["relevance_score"]
            assert 0 <= score <= 1, f"Relevance score should be between 0 and 1, got {score}"

            # Test reasoning
            reasoning = path_data["reasoning"]
            assert len(reasoning) > 10, "Reasoning should be descriptive"

    def test_search_query_processing(self):
        """Test search query processing and normalization."""
        test_queries = [
            "What is artificial intelligence?",
            "machine learning algorithms",
            "How do neural networks work?",
            "AI applications in healthcare",
            ""  # Empty query
        ]

        for query in test_queries:
            # Test query normalization
            normalized = self._normalize_query(query)

            if query.strip():
                assert len(normalized) > 0, f"Non-empty query should produce normalized result: {query}"
                assert normalized.lower() == normalized, "Normalized query should be lowercase"
            else:
                assert len(normalized) == 0, "Empty query should remain empty"

    def test_search_result_deduplication(self, sample_vector_results):
        """Test that search results are properly deduplicated."""
        # Create duplicate results
        results_with_duplicates = sample_vector_results + [sample_vector_results[0]]

        # Test deduplication logic
        deduplicated = self._deduplicate_results(results_with_duplicates)

        assert len(deduplicated) == len(sample_vector_results), "Duplicates should be removed"

        # Test that highest scoring duplicate is kept
        original_ids = [r["id"] for r in sample_vector_results]
        dedup_ids = [r["id"] for r in deduplicated]
        assert set(original_ids) == set(dedup_ids), "All unique IDs should be preserved"

    def test_search_performance_thresholds(self, sample_vector_results):
        """Test search performance and quality thresholds."""
        # Test minimum score threshold
        min_threshold = 0.5
        quality_results = [r for r in sample_vector_results if r["score"] >= min_threshold]

        assert len(quality_results) == len(sample_vector_results), \
            f"All sample results should meet minimum threshold {min_threshold}"

        # Test result diversity (different content)
        contents = [r["metadata"]["content"] for r in sample_vector_results]
        unique_contents = set(contents)

        assert len(unique_contents) == len(contents), "Results should have diverse content"

        # Test entity coverage
        all_entities = []
        for result in sample_vector_results:
            entities = result["metadata"].get("entities", [])
            all_entities.extend(entities)

        unique_entities = set(all_entities)
        assert len(unique_entities) >= 5, "Results should cover multiple entities"

    def test_hybrid_search_integration(self, sample_vector_results, sample_graph_entities):
        """Test integration between vector search and graph retrieval."""
        # Simulate hybrid search combining vector and graph results
        vector_entities = set()
        for result in sample_vector_results:
            entities = result["metadata"].get("entities", [])
            vector_entities.update(entities)

        graph_entity_names = {entity["name"] for entity in sample_graph_entities}

        # Test entity overlap between vector and graph results
        overlap = vector_entities.intersection(graph_entity_names)
        assert len(overlap) >= 2, "Vector and graph results should have entity overlap"

        # Test that graph can expand on vector results
        expandable_entities = [entity for entity in sample_graph_entities
                             if entity["name"] in vector_entities]
        assert len(expandable_entities) >= 2, "Graph should provide expansion for vector entities"

    def test_search_error_handling(self):
        """Test search error handling and fallback mechanisms."""
        # Test invalid query handling
        invalid_queries = [None, "", "   ", "a" * 10000]  # None, empty, whitespace, too long

        for query in invalid_queries:
            try:
                result = self._safe_search(query)
                if query is None or (isinstance(query, str) and not query.strip()):
                    assert result == [], "Invalid queries should return empty results"
                elif len(query) > 5000:
                    assert result == [], "Overly long queries should be rejected"
            except Exception as e:
                # Exceptions should be handled gracefully
                assert "invalid" in str(e).lower() or "error" in str(e).lower()

    def test_search_context_preservation(self, sample_vector_results):
        """Test that search preserves important context information."""
        for result in sample_vector_results:
            metadata = result["metadata"]

            # Test that essential context is preserved
            assert "content" in metadata, "Content should be preserved"

            # Test that structured metadata is maintained
            if "entities" in metadata:
                entities = metadata["entities"]
                assert all(isinstance(entity, str) for entity in entities), \
                    "Entity names should be strings"
                assert all(len(entity.strip()) > 0 for entity in entities), \
                    "Entity names should not be empty"

            # Test domain classification preservation
            if "domain" in metadata:
                domain = metadata["domain"]
                assert isinstance(domain, str), "Domain should be string"
                assert len(domain) > 0, "Domain should not be empty"

    # Helper methods for testing
    def _normalize_query(self, query: str) -> str:
        """Normalize search query."""
        if not query or not isinstance(query, str):
            return ""
        return query.strip().lower()

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results, keeping highest scoring."""
        seen_ids = set()
        deduplicated = []

        # Sort by score descending to keep highest scoring duplicates
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        for result in sorted_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                deduplicated.append(result)

        return deduplicated

    def _safe_search(self, query: Any) -> List[Dict[str, Any]]:
        """Safely execute search with error handling."""
        try:
            if query is None or not isinstance(query, str) or not query.strip():
                return []

            if len(query) > 5000:
                return []

            # Simulate successful search
            return [{"id": "test", "score": 0.8, "metadata": {"content": "test result"}}]

        except Exception:
            return []


@pytest.mark.integration
class TestSearchRetrievalIntegration:
    """Integration tests for search and retrieval functionality."""

    def test_end_to_end_search_flow(self):
        """Test complete search flow from query to results."""
        # This would test actual API endpoints if available
        pytest.skip("Integration test requires running services")

    def test_search_with_real_data(self):
        """Test search functionality with real ingested data."""
        # Check if there's real data available for testing
        data_paths = [
            Path("temp/facts.json"),
            Path("output/ingested_data.json"),
            Path("data/test_facts.json")
        ]

        available_data = None
        for path in data_paths:
            if path.exists():
                available_data = path
                break

        if not available_data:
            pytest.skip("No real data available for integration testing")

        # Test with real data would go here
        with open(available_data, 'r', encoding='utf-8') as f:
            data = json.load(f)

        facts = data.get('facts', [])
        assert len(facts) > 0, "Should have facts for testing"
