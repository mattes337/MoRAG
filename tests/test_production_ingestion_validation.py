#!/usr/bin/env python3
"""
Automated test for production ingestion validation.
Based on validate_production_ingestion.py
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


class TestProductionIngestionValidation:
    """Test production readiness of extracted facts and relationships."""

    @pytest.fixture
    def sample_facts_data(self):
        """Sample facts data for testing."""
        return {
            "facts": [
                {
                    "id": "fact_1",
                    "fact_text": "Artificial intelligence is a branch of computer science that aims to create intelligent machines.",
                    "fact_type": "definition",
                    "confidence": 0.85,
                    "domain": "technology",
                    "structured_metadata": {
                        "primary_entities": [
                            "Artificial intelligence",
                            "computer science",
                            "intelligent machines",
                        ]
                    },
                    "keywords": ["AI", "computer science", "machines"],
                    "created_at": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "fact_2",
                    "fact_text": "It can perform complex tasks efficiently.",
                    "fact_type": "capability",
                    "confidence": 0.45,
                    "domain": "technology",
                    "structured_metadata": {"primary_entities": []},
                    "keywords": ["tasks", "efficient"],
                    "created_at": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "fact_3",
                    "fact_text": "Machine learning enables computers to learn from data without explicit programming.",
                    "fact_type": "definition",
                    "confidence": 0.92,
                    "domain": "technology",
                    "structured_metadata": {
                        "primary_entities": ["Machine learning", "computers", "data"]
                    },
                    "keywords": ["machine learning", "data", "programming"],
                    "created_at": "2024-01-01T00:00:00Z",
                },
            ],
            "relationships": [
                {
                    "relationship_type": "SUBSET_OF",
                    "source_fact_id": "Machine learning",
                    "target_fact_id": "Artificial intelligence",
                    "confidence": 0.88,
                    "description": "Machine learning is a subset of AI",
                    "created_at": "2024-01-01T00:00:00Z",
                },
                {
                    "relationship_type": "RELATED_TO",
                    "source_fact_id": "fact_1",
                    "target_fact_id": "fact_3",
                    "confidence": 0.75,
                    "description": "Both facts relate to AI concepts",
                    "created_at": "2024-01-01T00:00:00Z",
                },
            ],
        }

    def test_fact_self_containment_validation(self, sample_facts_data):
        """Test validation of fact self-containment."""
        facts = sample_facts_data["facts"]

        validation_result = self._validate_facts(facts)
        stats = validation_result["stats"]
        issues = validation_result["issues"]

        # Check that self-contained facts are identified
        assert stats["self_contained"] >= 2, "Should identify self-contained facts"

        # Check that referential facts are flagged
        referential_issues = [issue for issue in issues if "pronouns" in issue.lower()]
        assert len(referential_issues) >= 1, "Should flag facts with pronouns"

    def test_entity_extraction_validation(self, sample_facts_data):
        """Test validation of entity extraction."""
        facts = sample_facts_data["facts"]

        validation_result = self._validate_facts(facts)
        stats = validation_result["stats"]
        issues = validation_result["issues"]

        # Check entity statistics
        assert stats["with_entities"] >= 2, "Should identify facts with entities"
        assert stats["entity_count"] >= 6, "Should count total entities"

        # Check that facts without entities are flagged
        no_entity_issues = [issue for issue in issues if "No entities" in issue]
        assert len(no_entity_issues) >= 1, "Should flag facts without entities"

    def test_confidence_distribution_validation(self, sample_facts_data):
        """Test validation of confidence distribution."""
        facts = sample_facts_data["facts"]

        validation_result = self._validate_facts(facts)
        stats = validation_result["stats"]

        # Check confidence distribution
        conf_dist = stats["confidence_distribution"]
        assert conf_dist["high"] >= 1, "Should have high confidence facts"
        assert conf_dist["medium"] >= 1, "Should have medium confidence facts"
        assert conf_dist["low"] >= 1, "Should have low confidence facts"

        total_facts = conf_dist["high"] + conf_dist["medium"] + conf_dist["low"]
        assert total_facts == len(
            facts
        ), "Confidence distribution should account for all facts"

    def test_relationship_validation(self, sample_facts_data):
        """Test validation of relationships."""
        relationships = sample_facts_data["relationships"]

        validation_result = self._validate_relationships(relationships)
        stats = validation_result["stats"]

        # Check relationship statistics
        assert stats["total_relationships"] == 2, "Should count all relationships"
        assert (
            stats["entity_to_entity"] >= 1
        ), "Should identify entity-to-entity relationships"
        assert stats["fact_to_fact"] >= 1, "Should identify fact-to-fact relationships"

        # Check relationship types
        assert (
            "SUBSET_OF" in stats["relationship_types"]
        ), "Should track relationship types"
        assert (
            "RELATED_TO" in stats["relationship_types"]
        ), "Should track relationship types"

    def test_neo4j_simulation(self, sample_facts_data):
        """Test Neo4j ingestion simulation."""
        neo4j_sim = self._simulate_neo4j_ingestion(sample_facts_data)

        stats = neo4j_sim["stats"]

        # Check node counts
        assert stats["fact_nodes"] == 3, "Should create fact nodes"
        assert stats["entity_nodes"] >= 6, "Should create entity nodes"
        assert stats["total_nodes"] >= 9, "Should create total nodes"
        assert stats["total_relationships"] == 2, "Should create relationships"

        # Check node structure
        nodes = neo4j_sim["nodes"]
        fact_nodes = [n for n in nodes if "Fact" in n["labels"]]
        entity_nodes = [n for n in nodes if "Entity" in n["labels"]]

        assert len(fact_nodes) == 3, "Should have correct number of fact nodes"
        assert len(entity_nodes) >= 6, "Should have entity nodes"

        # Check that fact nodes have required properties
        for node in fact_nodes:
            props = node["properties"]
            assert "id" in props, "Fact nodes should have ID"
            assert "fact_text" in props, "Fact nodes should have text"
            assert "confidence" in props, "Fact nodes should have confidence"

    def test_qdrant_simulation(self, sample_facts_data):
        """Test Qdrant ingestion simulation."""
        qdrant_sim = self._simulate_qdrant_ingestion(sample_facts_data)

        stats = qdrant_sim["stats"]

        # Check chunk statistics
        assert stats["total_chunks"] == 3, "Should create vector chunks"
        assert stats["avg_text_length"] > 0, "Should have text content"
        assert stats["entities_per_chunk"] >= 1, "Should have entities per chunk"

        # Check chunk structure
        chunks = qdrant_sim["chunks"]
        assert len(chunks) == 3, "Should have correct number of chunks"

        for chunk in chunks:
            assert "id" in chunk, "Chunks should have ID"
            assert "text" in chunk, "Chunks should have text"
            assert "metadata" in chunk, "Chunks should have metadata"
            assert "vector" in chunk, "Chunks should have vector placeholder"

            metadata = chunk["metadata"]
            assert "confidence" in metadata, "Metadata should include confidence"
            assert "entities" in metadata, "Metadata should include entities"

    def test_production_readiness_assessment(self, sample_facts_data):
        """Test overall production readiness assessment."""
        fact_validation = self._validate_facts(sample_facts_data["facts"])
        rel_validation = self._validate_relationships(
            sample_facts_data["relationships"]
        )

        total_issues = len(fact_validation["issues"]) + len(rel_validation["issues"])

        # Assess production readiness
        if total_issues == 0:
            readiness = "ready"
        elif total_issues <= 5:
            readiness = "mostly_ready"
        else:
            readiness = "needs_fixes"

        # With our sample data, we expect some issues but not too many
        assert readiness in [
            "ready",
            "mostly_ready",
        ], f"Production readiness: {readiness}, issues: {total_issues}"

    def test_with_json_file(self, sample_facts_data):
        """Test validation with actual JSON file."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_facts_data, f, indent=2)
            temp_file = f.name

        try:
            # Load and validate
            with open(temp_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            fact_validation = self._validate_facts(data.get("facts", []))
            rel_validation = self._validate_relationships(data.get("relationships", []))

            # Should successfully validate
            assert fact_validation["stats"]["total_facts"] == 3
            assert rel_validation["stats"]["total_relationships"] == 2

        finally:
            os.unlink(temp_file)

    def _validate_facts(self, facts: List[Dict]) -> Dict[str, Any]:
        """Validate extracted facts for production readiness."""
        import re

        issues = []
        stats = {
            "total_facts": len(facts),
            "self_contained": 0,
            "with_entities": 0,
            "entity_count": 0,
            "fact_types": {},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
        }

        # Problematic pronouns that make facts referential
        problematic_patterns = [
            r"\bit\b",
            r"\bthey\b",
            r"\bthem\b",
            r"\bthis\b",
            r"\bthese\b",
            r"\bhe\b",
            r"\bshe\b",
            r"\bhis\b",
            r"\bher\b",
        ]

        for i, fact in enumerate(facts):
            fact_text = fact.get("fact_text", "").lower()
            confidence = fact.get("confidence", 0)
            fact_type = fact.get("fact_type", "unknown")

            # Check self-containment
            has_pronouns = any(
                re.search(pattern, fact_text, re.IGNORECASE)
                for pattern in problematic_patterns
            )
            if not has_pronouns:
                stats["self_contained"] += 1
            else:
                issues.append(
                    f"Fact {i+1}: Contains pronouns - '{fact.get('fact_text', '')[:100]}...'"
                )

            # Check entities
            structured_metadata = fact.get("structured_metadata", {})
            primary_entities = structured_metadata.get("primary_entities", [])

            if primary_entities:
                stats["with_entities"] += 1
                stats["entity_count"] += len(primary_entities)
            else:
                issues.append(f"Fact {i+1}: No entities extracted")

            # Count fact types
            stats["fact_types"][fact_type] = stats["fact_types"].get(fact_type, 0) + 1

            # Confidence distribution
            if confidence >= 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif confidence >= 0.5:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1

        return {"stats": stats, "issues": issues}

    def _validate_relationships(self, relationships: List[Dict]) -> Dict[str, Any]:
        """Validate extracted relationships for production readiness."""
        issues = []
        stats = {
            "total_relationships": len(relationships),
            "entity_to_entity": 0,
            "fact_to_fact": 0,
            "relationship_types": {},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
        }

        for rel in relationships:
            rel_type = rel.get("relationship_type", "unknown")
            confidence = rel.get("confidence", 0)
            source = rel.get("source_fact_id", "")
            target = rel.get("target_fact_id", "")

            # Count relationship types
            stats["relationship_types"][rel_type] = (
                stats["relationship_types"].get(rel_type, 0) + 1
            )

            # Determine if entity-to-entity or fact-to-fact
            is_entity_to_entity = (
                not source.startswith("fact_")
                and not target.startswith("fact_")
                and source
                and target
            )

            if is_entity_to_entity:
                stats["entity_to_entity"] += 1
            else:
                stats["fact_to_fact"] += 1

            # Confidence distribution
            if confidence >= 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif confidence >= 0.5:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1

        return {"stats": stats, "issues": issues}

    def _simulate_neo4j_ingestion(self, data: Dict) -> Dict[str, Any]:
        """Simulate Neo4j ingestion."""
        facts = data.get("facts", [])
        relationships = data.get("relationships", [])

        neo4j_nodes = []

        # Create fact nodes
        for fact in facts:
            node = {
                "labels": ["Fact"],
                "properties": {
                    "id": fact.get("id"),
                    "fact_text": fact.get("fact_text"),
                    "fact_type": fact.get("fact_type"),
                    "confidence": fact.get("confidence"),
                },
            }
            neo4j_nodes.append(node)

        # Create entity nodes
        entity_nodes = {}
        for fact in facts:
            structured_metadata = fact.get("structured_metadata", {})
            primary_entities = structured_metadata.get("primary_entities", [])

            for entity_name in primary_entities:
                if entity_name not in entity_nodes:
                    entity_nodes[entity_name] = {
                        "labels": ["Entity"],
                        "properties": {"name": entity_name, "type": "ENTITY"},
                    }

        neo4j_nodes.extend(entity_nodes.values())

        return {
            "nodes": neo4j_nodes,
            "relationships": relationships,
            "stats": {
                "total_nodes": len(neo4j_nodes),
                "fact_nodes": len(facts),
                "entity_nodes": len(entity_nodes),
                "total_relationships": len(relationships),
            },
        }

    def _simulate_qdrant_ingestion(self, data: Dict) -> Dict[str, Any]:
        """Simulate Qdrant ingestion."""
        facts = data.get("facts", [])

        vector_chunks = []
        for fact in facts:
            chunk = {
                "id": fact.get("id"),
                "text": fact.get("fact_text"),
                "metadata": {
                    "confidence": fact.get("confidence"),
                    "entities": fact.get("structured_metadata", {}).get(
                        "primary_entities", []
                    ),
                },
                "vector": f"[simulated vector for {fact.get('id')}]",
            }
            vector_chunks.append(chunk)

        return {
            "chunks": vector_chunks,
            "stats": {
                "total_chunks": len(vector_chunks),
                "avg_text_length": sum(len(chunk["text"]) for chunk in vector_chunks)
                / len(vector_chunks)
                if vector_chunks
                else 0,
                "entities_per_chunk": sum(
                    len(chunk["metadata"]["entities"]) for chunk in vector_chunks
                )
                / len(vector_chunks)
                if vector_chunks
                else 0,
            },
        }
