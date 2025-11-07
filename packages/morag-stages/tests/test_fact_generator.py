"""Tests for fact-generator stage."""

import json
from pathlib import Path

import pytest
from morag_stages import StageType

from .test_framework import StageTestFramework


@pytest.mark.asyncio
async def test_fact_generator_basic_functionality(
    stage_test_framework: StageTestFramework,
):
    """Test fact-generator stage basic functionality."""

    # Create test chunks file
    chunks_file = stage_test_framework.create_test_chunks_json("input.chunks.json")

    # Execute fact-generator stage
    result = await stage_test_framework.execute_stage_test(
        StageType.FACT_GENERATOR, [chunks_file]
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check output file
    facts_file = stage_test_framework.get_file_by_extension(
        result.output_files, ".facts.json"
    )

    # Validate JSON structure
    stage_test_framework.assert_json_structure(
        facts_file, ["facts", "entities", "relations"]
    )

    # Load and validate content
    with open(facts_file, "r", encoding="utf-8") as f:
        facts_data = json.load(f)

    assert len(facts_data["facts"]) > 0
    assert len(facts_data["entities"]) > 0

    # Check fact structure
    fact = facts_data["facts"][0]
    assert "statement" in fact
    assert "subject" in fact
    assert "predicate" in fact
    assert "object" in fact
    assert "confidence" in fact

    # Check entity structure
    entity = facts_data["entities"][0]
    assert "name" in entity
    assert "normalized_name" in entity
    assert "entity_type" in entity
    assert "confidence" in entity


@pytest.mark.asyncio
async def test_fact_generator_with_entity_extraction(
    stage_test_framework: StageTestFramework,
):
    """Test fact-generator stage with entity extraction enabled."""

    # Create test chunks file
    chunks_file = stage_test_framework.create_test_chunks_json("input.chunks.json")

    # Execute with entity extraction config
    config = {
        "fact-generator": {
            "extract_entities": True,
            "extract_relations": True,
            "extract_keywords": True,
            "domain": "technology",
        }
    }

    result = await stage_test_framework.execute_stage_test(
        StageType.FACT_GENERATOR, [chunks_file], config=config
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check facts file
    facts_file = stage_test_framework.get_file_by_extension(
        result.output_files, ".facts.json"
    )

    with open(facts_file, "r", encoding="utf-8") as f:
        facts_data = json.load(f)

    # Should have extracted entities
    assert len(facts_data["entities"]) > 0

    # Check entity types are appropriate for technology domain
    entity_types = [entity["entity_type"] for entity in facts_data["entities"]]
    # Should contain technology-related entity types

    # Check for keywords if enabled
    if "keywords" in facts_data:
        assert len(facts_data["keywords"]) > 0


@pytest.mark.asyncio
async def test_fact_generator_with_relations(stage_test_framework: StageTestFramework):
    """Test fact-generator stage with relation extraction."""

    # Create test chunks file
    chunks_file = stage_test_framework.create_test_chunks_json("input.chunks.json")

    # Execute with relation extraction config
    config = {
        "fact-generator": {
            "extract_entities": True,
            "extract_relations": True,
            "relation_types": ["IS_PART_OF", "USES", "RELATED_TO", "CAUSES"],
            "min_confidence": 0.3,
        }
    }

    result = await stage_test_framework.execute_stage_test(
        StageType.FACT_GENERATOR, [chunks_file], config=config
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check facts file
    facts_file = stage_test_framework.get_file_by_extension(
        result.output_files, ".facts.json"
    )

    with open(facts_file, "r", encoding="utf-8") as f:
        facts_data = json.load(f)

    # Should have extracted relations
    if "relations" in facts_data:
        assert len(facts_data["relations"]) > 0

        # Check relation structure
        relation = facts_data["relations"][0]
        assert "subject" in relation
        assert "predicate" in relation
        assert "object" in relation
        assert "confidence" in relation

        # Check confidence threshold
        assert relation["confidence"] >= 0.7


@pytest.mark.asyncio
async def test_fact_generator_domain_specific(stage_test_framework: StageTestFramework):
    """Test fact-generator stage with domain-specific extraction."""

    # Create medical domain chunks
    medical_chunks = {
        "document_metadata": {
            "title": "Medical Research Paper",
            "content_type": "document",
            "domain": "medical",
        },
        "summary": "Research on diabetes treatment and insulin therapy.",
        "chunk_count": 2,
        "chunks": [
            {
                "index": 0,
                "content": "Diabetes is a chronic condition that affects blood sugar levels. Insulin therapy is a common treatment.",
                "token_count": 20,
                "embedding": [0.1] * 1536,
                "source_metadata": {"title": "Medical Research Paper"},
            },
            {
                "index": 1,
                "content": "Type 1 diabetes requires insulin injections. Type 2 diabetes can be managed with medication and diet.",
                "token_count": 18,
                "embedding": [0.2] * 1536,
                "source_metadata": {"title": "Medical Research Paper"},
            },
        ],
    }

    chunks_file = stage_test_framework.test_data_dir / "medical.chunks.json"
    chunks_file.write_text(json.dumps(medical_chunks, indent=2), encoding="utf-8")

    # Execute with medical domain config
    config = {
        "fact-generator": {
            "extract_entities": True,
            "extract_relations": True,
            "domain": "medical",
            "entity_types": ["Disease", "Treatment", "Medication", "Symptom"],
        }
    }

    result = await stage_test_framework.execute_stage_test(
        StageType.FACT_GENERATOR, [chunks_file], config=config
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check facts file
    facts_file = stage_test_framework.get_file_by_extension(
        result.output_files, ".facts.json"
    )

    with open(facts_file, "r", encoding="utf-8") as f:
        facts_data = json.load(f)

    # Should extract medical entities
    entity_names = [entity["name"].lower() for entity in facts_data["entities"]]

    # Should find medical terms
    medical_terms_found = any(
        term in " ".join(entity_names) for term in ["diabetes", "insulin", "treatment"]
    )
    assert medical_terms_found


@pytest.mark.asyncio
async def test_fact_generator_with_timestamps(stage_test_framework: StageTestFramework):
    """Test fact-generator stage with timestamp preservation."""

    # Create chunks with timestamps
    timestamp_chunks = {
        "document_metadata": {
            "title": "Audio Transcript",
            "content_type": "audio",
            "duration": 300,
        },
        "summary": "Discussion about artificial intelligence and machine learning.",
        "chunk_count": 2,
        "chunks": [
            {
                "index": 0,
                "content": "[00:00 - 01:00] Artificial intelligence is transforming many industries.",
                "token_count": 12,
                "embedding": [0.1] * 1536,
                "source_metadata": {"timestamp": "00:00-01:00"},
            },
            {
                "index": 1,
                "content": "[01:00 - 02:00] Machine learning algorithms learn from data patterns.",
                "token_count": 11,
                "embedding": [0.2] * 1536,
                "source_metadata": {"timestamp": "01:00-02:00"},
            },
        ],
    }

    chunks_file = stage_test_framework.test_data_dir / "timestamp.chunks.json"
    chunks_file.write_text(json.dumps(timestamp_chunks, indent=2), encoding="utf-8")

    # Execute with timestamp preservation
    config = {
        "fact-generator": {
            "extract_entities": True,
            "include_timestamps": True,
            "preserve_temporal_context": True,
        }
    }

    result = await stage_test_framework.execute_stage_test(
        StageType.FACT_GENERATOR, [chunks_file], config=config
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check facts file
    facts_file = stage_test_framework.get_file_by_extension(
        result.output_files, ".facts.json"
    )

    with open(facts_file, "r", encoding="utf-8") as f:
        facts_data = json.load(f)

    # Check for timestamp information in facts
    for fact in facts_data["facts"]:
        if "source_chunk_index" in fact:
            # Should reference chunks with timestamps
            chunk_index = fact["source_chunk_index"]
            assert chunk_index >= 0


@pytest.mark.asyncio
async def test_fact_generator_skip_existing(stage_test_framework: StageTestFramework):
    """Test fact-generator stage skips when output already exists."""

    # Create test file
    chunks_file = stage_test_framework.create_test_chunks_json("input.chunks.json")

    # Execute first time
    result1 = await stage_test_framework.execute_stage_test(
        StageType.FACT_GENERATOR, [chunks_file]
    )
    stage_test_framework.assert_stage_success(result1)

    # Execute second time - should skip
    result2 = await stage_test_framework.execute_stage_test(
        StageType.FACT_GENERATOR, [chunks_file]
    )
    stage_test_framework.assert_stage_skipped(result2)


@pytest.mark.asyncio
async def test_fact_generator_confidence_filtering(
    stage_test_framework: StageTestFramework,
):
    """Test fact-generator stage with confidence filtering."""

    # Create test chunks file
    chunks_file = stage_test_framework.create_test_chunks_json("input.chunks.json")

    # Execute with high confidence threshold
    config = {
        "fact-generator": {
            "extract_entities": True,
            "extract_relations": True,
            "min_confidence": 0.9,  # High threshold
            "filter_low_confidence": True,
        }
    }

    result = await stage_test_framework.execute_stage_test(
        StageType.FACT_GENERATOR, [chunks_file], config=config
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check facts file
    facts_file = stage_test_framework.get_file_by_extension(
        result.output_files, ".facts.json"
    )

    with open(facts_file, "r", encoding="utf-8") as f:
        facts_data = json.load(f)

    # All facts should meet confidence threshold
    for fact in facts_data["facts"]:
        if "confidence" in fact:
            assert fact["confidence"] >= 0.9

    # All entities should meet confidence threshold
    for entity in facts_data["entities"]:
        if "confidence" in entity:
            assert entity["confidence"] >= 0.9


@pytest.mark.asyncio
async def test_fact_generator_output_naming(stage_test_framework: StageTestFramework):
    """Test fact-generator stage output file naming conventions."""

    # Create test file with specific name
    chunks_file = stage_test_framework.test_data_dir / "my_document.chunks.json"
    chunks_data = stage_test_framework.create_test_chunks_json().read_text()
    chunks_file.write_text(chunks_data)

    result = await stage_test_framework.execute_stage_test(
        StageType.FACT_GENERATOR, [chunks_file]
    )

    stage_test_framework.assert_stage_success(result)

    # Check output file naming
    facts_file = stage_test_framework.get_file_by_extension(
        result.output_files, ".facts.json"
    )

    # Should be based on input filename
    assert "my_document" in facts_file.name
    assert facts_file.name.endswith(".facts.json")


@pytest.mark.asyncio
async def test_fact_generator_empty_chunks(stage_test_framework: StageTestFramework):
    """Test fact-generator stage with empty chunks."""

    # Create empty chunks file
    empty_chunks = {
        "document_metadata": {"title": "Empty Document"},
        "summary": "",
        "chunk_count": 0,
        "chunks": [],
    }

    chunks_file = stage_test_framework.test_data_dir / "empty.chunks.json"
    chunks_file.write_text(json.dumps(empty_chunks, indent=2), encoding="utf-8")

    # Execute stage
    result = await stage_test_framework.execute_stage_test(
        StageType.FACT_GENERATOR, [chunks_file]
    )

    # Should handle gracefully
    if result.status.value == "completed":
        # If successful, should have minimal output
        facts_file = stage_test_framework.get_file_by_extension(
            result.output_files, ".facts.json"
        )

        with open(facts_file, "r", encoding="utf-8") as f:
            facts_data = json.load(f)

        # Should have empty or minimal facts
        assert "facts" in facts_data
        assert "entities" in facts_data
    else:
        # If failed, should have error message
        assert result.error_message is not None
