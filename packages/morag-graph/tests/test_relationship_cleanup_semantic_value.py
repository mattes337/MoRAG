"""Test relationship cleanup with semantic value assessment."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from morag_graph.maintenance.relationship_cleanup import (
    RelationshipCleanupResult,
    RelationshipCleanupService,
)


@pytest.fixture
def mock_neo4j_storage():
    """Mock Neo4j storage."""
    storage = MagicMock()
    storage._connection_ops = MagicMock()
    storage._connection_ops._execute_query = AsyncMock()
    return storage


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = MagicMock()
    client.generate_text = AsyncMock()
    return client


@pytest.fixture
def relationship_cleanup(mock_neo4j_storage):
    """Create RelationshipCleanupService instance."""
    from morag_graph.maintenance.relationship_cleanup import RelationshipCleanupConfig

    config = RelationshipCleanupConfig(
        dry_run=True,
        batch_size=100,
        min_confidence=0.3,
        remove_unrelated=True,
        remove_generic=True,
        consolidate_similar=True,
        similarity_threshold=0.85,
        job_tag="test",
    )

    cleanup = RelationshipCleanupService(mock_neo4j_storage, config)
    return cleanup


class TestSemanticValueAssessment:
    """Test semantic value assessment in relationship cleanup."""

    @pytest.mark.asyncio
    async def test_llm_identifies_low_value_relationships(
        self, relationship_cleanup, mock_llm_client
    ):
        """Test that LLM identifies low semantic value relationships for removal."""
        relationship_cleanup._llm_client = mock_llm_client

        # Mock relationship types with mix of high and low value
        relationship_types = [
            {"neo4j_type": "TREATS", "count": 50, "avg_confidence": 0.9},
            {"neo4j_type": "TAGGED_WITH", "count": 100, "avg_confidence": 0.7},
            {"neo4j_type": "CAUSES", "count": 30, "avg_confidence": 0.8},
            {"neo4j_type": "RELATED_TO", "count": 200, "avg_confidence": 0.6},
            {"neo4j_type": "IMPROVES", "count": 25, "avg_confidence": 0.85},
        ]

        # Mock LLM response identifying low-value types
        llm_response = json.dumps(
            {
                "remove_types": ["TAGGED_WITH", "RELATED_TO"],
                "merge_pairs": [],
                "reasoning": "TAGGED_WITH and RELATED_TO are generic with low semantic value compared to specific alternatives like TREATS, CAUSES, IMPROVES",
            }
        )
        mock_llm_client.generate_text.return_value = llm_response

        result = await relationship_cleanup._analyze_relationship_types_with_llm(
            relationship_types
        )

        assert "TAGGED_WITH" in result["remove_types"]
        assert "RELATED_TO" in result["remove_types"]
        assert "TREATS" not in result["remove_types"]
        assert "CAUSES" not in result["remove_types"]
        assert "IMPROVES" not in result["remove_types"]

    @pytest.mark.asyncio
    async def test_llm_prefers_specific_over_generic_in_merging(
        self, relationship_cleanup, mock_llm_client
    ):
        """Test that LLM prefers specific relationship types over generic ones in merging."""
        relationship_cleanup._llm_client = mock_llm_client

        relationship_types = [
            {"neo4j_type": "EMPLOYED_BY", "count": 30, "avg_confidence": 0.8},
            {"neo4j_type": "WORKS_AT", "count": 45, "avg_confidence": 0.85},
            {"neo4j_type": "ASSOCIATED_WITH", "count": 60, "avg_confidence": 0.7},
        ]

        # Mock LLM response preferring specific types
        llm_response = json.dumps(
            {
                "remove_types": [],
                "merge_pairs": [
                    {"primary": "WORKS_AT", "merge_into": ["EMPLOYED_BY"]},
                    {"primary": "WORKS_AT", "merge_into": ["ASSOCIATED_WITH"]},
                ],
                "reasoning": "WORKS_AT is more specific than EMPLOYED_BY and ASSOCIATED_WITH",
            }
        )
        mock_llm_client.generate_text.return_value = llm_response

        result = await relationship_cleanup._analyze_relationship_types_with_llm(
            relationship_types
        )

        # Should merge generic types into the more specific one
        merge_pairs = result["merge_pairs"]
        assert len(merge_pairs) >= 1
        assert any(pair["primary"] == "WORKS_AT" for pair in merge_pairs)

    @pytest.mark.asyncio
    async def test_generic_relationship_cleanup_with_specific_alternatives(
        self, relationship_cleanup, mock_llm_client
    ):
        """Test cleanup of generic relationships when specific alternatives exist."""
        relationship_cleanup._llm_client = mock_llm_client

        # Mock entity pairs with multiple relationship types
        mock_query_result = [
            {
                "source_name": "Aspirin",
                "target_name": "Headache",
                "rel_types": ["TAGGED_WITH", "TREATS"],
                "relationships": [],
            },
            {
                "source_name": "Exercise",
                "target_name": "Health",
                "rel_types": ["RELATED_TO", "IMPROVES"],
                "relationships": [],
            },
        ]

        relationship_cleanup.neo4j_storage._connection_ops._execute_query.return_value = (
            mock_query_result
        )

        # Mock LLM responses for generic relationship identification
        def mock_llm_response(prompt):
            if "Aspirin" in prompt and "Headache" in prompt:
                return json.dumps(
                    {
                        "remove_types": ["TAGGED_WITH"],
                        "reasoning": "TAGGED_WITH is generic compared to specific TREATS relationship",
                    }
                )
            elif "Exercise" in prompt and "Health" in prompt:
                return json.dumps(
                    {
                        "remove_types": ["RELATED_TO"],
                        "reasoning": "RELATED_TO is generic compared to specific IMPROVES relationship",
                    }
                )
            return json.dumps({"remove_types": []})

        mock_llm_client.generate_text.side_effect = mock_llm_response

        result = RelationshipCleanupResult()
        await relationship_cleanup._cleanup_generic_relationships_with_specific_alternatives(
            result
        )

        # Should have identified generic relationships for removal
        assert mock_llm_client.generate_text.call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_analysis_minimal_static_rules(self, relationship_cleanup):
        """Test that fallback analysis uses minimal static rules."""
        relationship_types = [
            {"neo4j_type": "UNRELATED", "stored_type": "UNRELATED", "count": 10},
            {"neo4j_type": "TAGGED_WITH", "stored_type": "TAGGED_WITH", "count": 50},
            {"neo4j_type": "TREATS", "stored_type": "TREATS", "count": 30},
            {"neo4j_type": "RELATED_TO", "stored_type": "RELATED_TO", "count": 40},
        ]

        result = relationship_cleanup._analyze_relationship_types_fallback(
            relationship_types
        )

        # Should only remove explicitly meaningless types
        assert "UNRELATED" in result["remove_types"]
        # Should NOT remove generic types without LLM assessment
        assert "TAGGED_WITH" not in result["remove_types"]
        assert "RELATED_TO" not in result["remove_types"]
        # Should have no merge pairs (no static rules)
        assert len(result["merge_pairs"]) == 0

    @pytest.mark.asyncio
    async def test_semantic_similarity_assessment_considers_value(
        self, relationship_cleanup, mock_llm_client
    ):
        """Test that semantic similarity assessment considers semantic value."""
        relationship_cleanup._llm_client = mock_llm_client

        # Mock LLM response considering semantic value
        llm_response = json.dumps(
            {
                "are_similar": True,
                "confidence": 0.9,
                "preferred_type": "TREATS",
                "reason": "Both express treatment relationship, but TREATS is more specific than TAGGED_WITH",
            }
        )
        mock_llm_client.generate_text.return_value = llm_response

        result = await relationship_cleanup._are_semantically_similar_with_llm(
            "TREATS", "TAGGED_WITH"
        )

        assert result is True
        # Verify the prompt includes semantic value considerations
        call_args = mock_llm_client.generate_text.call_args[0][0]
        assert "semantic value" in call_args.lower()
        assert "descriptive information" in call_args.lower()

    def test_no_static_generic_types_defined(self, relationship_cleanup):
        """Test that no static generic types are defined."""
        # Should only have minimal fallback meaningless types
        assert hasattr(relationship_cleanup, "fallback_meaningless_types")
        assert len(relationship_cleanup.fallback_meaningless_types) <= 3
        # Should not have static generic types list
        assert not hasattr(relationship_cleanup, "generic_types")
