#!/usr/bin/env python3
"""
Graph Output Validator

This script validates the JSON output from the graph extractor test script
to ensure it contains all required fields and follows the expected format.

Usage:
    python scripts/validate_graph_output.py output.json
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def validate_entity(entity: Dict[str, Any], index: int) -> List[str]:
    """Validate a single entity object."""
    errors = []
    required_fields = ["id", "name", "type", "confidence", "source_doc_id"]

    for field in required_fields:
        if field not in entity:
            errors.append(f"Entity {index}: Missing required field '{field}'")

    if "confidence" in entity:
        confidence = entity["confidence"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            errors.append(
                f"Entity {index}: Invalid confidence value {confidence} (must be 0.0-1.0)"
            )

    return errors


def validate_relation(relation: Dict[str, Any], index: int) -> List[str]:
    """Validate a single relation object."""
    errors = []
    required_fields = [
        "id",
        "source_entity_id",
        "target_entity_id",
        "relation_type",
        "confidence",
        "source_doc_id",
    ]

    for field in required_fields:
        if field not in relation:
            errors.append(f"Relation {index}: Missing required field '{field}'")

    if "confidence" in relation:
        confidence = relation["confidence"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            errors.append(
                f"Relation {index}: Invalid confidence value {confidence} (must be 0.0-1.0)"
            )

    return errors


def validate_graph_output(data: Dict[str, Any]) -> List[str]:
    """Validate the complete graph output structure."""
    errors = []

    # Check top-level structure
    required_top_level = [
        "input_file",
        "content_length",
        "extraction_results",
        "analysis",
    ]
    for field in required_top_level:
        if field not in data:
            errors.append(f"Missing top-level field '{field}'")

    # Check extraction_results structure
    if "extraction_results" in data:
        extraction_results = data["extraction_results"]
        required_extraction = ["entities", "relations", "metadata"]

        for field in required_extraction:
            if field not in extraction_results:
                errors.append(f"Missing extraction_results field '{field}'")

        # Validate entities
        if "entities" in extraction_results:
            entities = extraction_results["entities"]
            if not isinstance(entities, list):
                errors.append("extraction_results.entities must be a list")
            else:
                for i, entity in enumerate(entities):
                    errors.extend(validate_entity(entity, i))

        # Validate relations
        if "relations" in extraction_results:
            relations = extraction_results["relations"]
            if not isinstance(relations, list):
                errors.append("extraction_results.relations must be a list")
            else:
                for i, relation in enumerate(relations):
                    errors.extend(validate_relation(relation, i))

    # Check analysis structure
    if "analysis" in data:
        analysis = data["analysis"]
        required_analysis = [
            "entity_count",
            "relation_count",
            "entity_types",
            "relation_types",
        ]

        for field in required_analysis:
            if field not in analysis:
                errors.append(f"Missing analysis field '{field}'")

        # Validate counts match actual data
        if "extraction_results" in data and "entities" in data["extraction_results"]:
            actual_entity_count = len(data["extraction_results"]["entities"])
            if analysis.get("entity_count") != actual_entity_count:
                errors.append(
                    f"Entity count mismatch: analysis says {analysis.get('entity_count')}, actual is {actual_entity_count}"
                )

        if "extraction_results" in data and "relations" in data["extraction_results"]:
            actual_relation_count = len(data["extraction_results"]["relations"])
            if analysis.get("relation_count") != actual_relation_count:
                errors.append(
                    f"Relation count mismatch: analysis says {analysis.get('relation_count')}, actual is {actual_relation_count}"
                )

    return errors


def main():
    """Main validation function."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_graph_output.py output.json")
        sys.exit(1)

    output_file = Path(sys.argv[1])

    if not output_file.exists():
        print(f"[ERROR] File not found: {output_file}")
        sys.exit(1)

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {output_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error reading file {output_file}: {e}")
        sys.exit(1)

    # Validate the structure
    errors = validate_graph_output(data)

    if errors:
        print(f"[ERROR] Validation failed for {output_file}:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
    else:
        print(f"[OK] Validation passed for {output_file}")

        # Print summary
        analysis = data.get("analysis", {})
        print(f"[SUMMARY] Results:")
        print(f"   - Entities: {analysis.get('entity_count', 0)}")
        print(f"   - Relations: {analysis.get('relation_count', 0)}")
        print(f"   - Entity types: {list(analysis.get('entity_types', {}).keys())}")
        print(f"   - Relation types: {list(analysis.get('relation_types', {}).keys())}")

        if "confidence_stats" in analysis:
            conf_stats = analysis["confidence_stats"]
            if "entity_confidence" in conf_stats:
                entity_conf = conf_stats["entity_confidence"]
                print(
                    f"   - Entity confidence: {entity_conf.get('avg', 0):.2f} (range: {entity_conf.get('min', 0):.2f}-{entity_conf.get('max', 0):.2f})"
                )
            if "relation_confidence" in conf_stats:
                relation_conf = conf_stats["relation_confidence"]
                print(
                    f"   - Relation confidence: {relation_conf.get('avg', 0):.2f} (range: {relation_conf.get('min', 0):.2f}-{relation_conf.get('max', 0):.2f})"
                )


if __name__ == "__main__":
    main()
