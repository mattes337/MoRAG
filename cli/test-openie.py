#!/usr/bin/env python3
"""
OpenIE Testing CLI Script

This script tests the OpenIE (Open Information Extraction) functionality in MoRAG.
It provides comprehensive testing of OpenIE services, extractors, and integration.

Usage:
    python cli/test-openie.py [options]

Examples:
    # Test basic OpenIE functionality
    python cli/test-openie.py --test basic

    # Test OpenIE with sample text
    python cli/test-openie.py --test extraction --text "John works at Microsoft. The company was founded by Bill Gates."

    # Test OpenIE integration with graph building
    python cli/test-openie.py --test integration --file sample.txt

    # Test all OpenIE components
    python cli/test-openie.py --test all --verbose

    # Test OpenIE configuration
    python cli/test-openie.py --test config
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(test_name: str, success: bool, duration: float = 0, details: str = ""):
    """Print test result with formatting."""
    status = "✅ PASS" if success else "❌ FAIL"
    duration_str = f" ({duration:.2f}s)" if duration > 0 else ""
    print(f"[{status}] {test_name}{duration_str}")
    if details:
        print(f"    {details}")


async def test_openie_configuration():
    """Test OpenIE configuration and settings."""
    print_section("OpenIE Configuration Test")
    
    try:
        from morag_core.config import get_settings
        
        settings = get_settings()
        
        config_tests = [
            ("OpenIE Enabled", settings.openie_enabled),
            ("OpenIE Implementation", settings.openie_implementation == "stanford"),
            ("Confidence Threshold", 0.0 <= settings.openie_confidence_threshold <= 1.0),
            ("Max Triplets", settings.openie_max_triplets_per_sentence > 0),
            ("Batch Size", settings.openie_batch_size > 0),
            ("Timeout", settings.openie_timeout_seconds > 0),
        ]
        
        all_passed = True
        for test_name, condition in config_tests:
            print_result(test_name, condition)
            if not condition:
                all_passed = False
        
        # Print configuration details
        print(f"\nConfiguration Details:")
        print(f"  - Enabled: {settings.openie_enabled}")
        print(f"  - Implementation: {settings.openie_implementation}")
        print(f"  - Confidence Threshold: {settings.openie_confidence_threshold}")
        print(f"  - Max Triplets per Sentence: {settings.openie_max_triplets_per_sentence}")
        print(f"  - Entity Linking: {settings.openie_enable_entity_linking}")
        print(f"  - Predicate Normalization: {settings.openie_enable_predicate_normalization}")
        print(f"  - Batch Size: {settings.openie_batch_size}")
        print(f"  - Timeout: {settings.openie_timeout_seconds}s")
        
        return all_passed
        
    except ImportError as e:
        print_result("Configuration Import", False, details=f"Import error: {e}")
        return False
    except Exception as e:
        print_result("Configuration Test", False, details=f"Error: {e}")
        return False


async def test_openie_service():
    """Test OpenIE service functionality."""
    print_section("OpenIE Service Test")
    
    try:
        from morag_graph.services.openie_service import OpenIEService
        
        # Test service initialization
        start_time = time.time()
        service = OpenIEService()
        init_time = time.time() - start_time
        print_result("Service Initialization", True, init_time)
        
        # Test service configuration
        config_ok = (
            hasattr(service, 'enabled') and
            hasattr(service, 'implementation') and
            hasattr(service, 'confidence_threshold')
        )
        print_result("Service Configuration", config_ok)
        
        if not service.enabled:
            print_result("Service Status", False, details="OpenIE service is disabled")
            return False
        
        # Test service initialization
        start_time = time.time()
        await service.initialize()
        init_time = time.time() - start_time
        print_result("Service Async Initialization", True, init_time)
        
        # Test triplet extraction
        sample_text = "John works at Microsoft. The company was founded by Bill Gates."
        start_time = time.time()
        triplets = await service.extract_triplets(sample_text)
        extract_time = time.time() - start_time
        
        extraction_success = isinstance(triplets, list)
        print_result("Triplet Extraction", extraction_success, extract_time, 
                    f"Extracted {len(triplets)} triplets")
        
        # Print extracted triplets
        if triplets:
            print(f"\nExtracted Triplets:")
            for i, triplet in enumerate(triplets[:5], 1):  # Show first 5
                print(f"  {i}. {triplet.subject} --[{triplet.predicate}]--> {triplet.object} "
                      f"(confidence: {triplet.confidence:.2f})")
        
        # Test cleanup
        await service.close()
        print_result("Service Cleanup", True)
        
        return extraction_success
        
    except ImportError as e:
        print_result("Service Import", False, details=f"Import error: {e}")
        return False
    except Exception as e:
        print_result("Service Test", False, details=f"Error: {e}")
        return False


async def test_openie_extractor():
    """Test OpenIE extractor functionality."""
    print_section("OpenIE Extractor Test")
    
    try:
        from morag_graph.extractors.openie_extractor import OpenIEExtractor
        from morag_graph.models import Entity
        
        # Test extractor initialization
        config = {
            "min_confidence": 0.6,
            "enable_entity_linking": True,
            "enable_predicate_normalization": True
        }
        
        start_time = time.time()
        extractor = OpenIEExtractor(config)
        init_time = time.time() - start_time
        print_result("Extractor Initialization", True, init_time)
        
        # Test basic relation extraction
        sample_text = """
        John Smith works at Microsoft Corporation in Seattle.
        He is the CEO of the company and lives in Washington.
        Microsoft was founded by Bill Gates in 1975.
        The company develops software products and services.
        """
        
        start_time = time.time()
        relations = await extractor.extract_relations(sample_text, source_doc_id="test_doc")
        extract_time = time.time() - start_time
        
        extraction_success = isinstance(relations, list)
        print_result("Relation Extraction", extraction_success, extract_time,
                    f"Extracted {len(relations)} relations")
        
        # Test full extraction
        sample_entities = [
            Entity(
                id="entity_1",
                name="John Smith",
                canonical_name="john_smith",
                entity_type="PERSON",
                confidence=0.95,
                metadata={"source": "test"}
            ),
            Entity(
                id="entity_2",
                name="Microsoft",
                canonical_name="microsoft",
                entity_type="ORG",
                confidence=0.98,
                metadata={"source": "test"}
            )
        ]
        
        start_time = time.time()
        result = await extractor.extract_full(sample_text, entities=sample_entities, source_doc_id="test_doc")
        full_extract_time = time.time() - start_time
        
        full_success = (
            hasattr(result, 'relations') and
            hasattr(result, 'triplets') and
            hasattr(result, 'metadata')
        )
        print_result("Full Extraction", full_success, full_extract_time,
                    f"Relations: {len(result.relations)}, Triplets: {len(result.triplets)}")
        
        # Print extraction statistics
        if hasattr(result, 'metadata'):
            print(f"\nExtraction Statistics:")
            for key, value in result.metadata.items():
                print(f"  - {key}: {value}")
        
        return extraction_success and full_success
        
    except ImportError as e:
        print_result("Extractor Import", False, details=f"Import error: {e}")
        return False
    except Exception as e:
        print_result("Extractor Test", False, details=f"Error: {e}")
        return False


async def test_openie_integration():
    """Test OpenIE integration with enhanced graph builder."""
    print_section("OpenIE Integration Test")
    
    try:
        from morag_graph.builders.enhanced_graph_builder import EnhancedGraphBuilder
        from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
        
        # Mock Neo4j configuration for testing
        neo4j_config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="test",  # This won't actually connect
            database="test"
        )
        
        # Test enhanced graph builder initialization
        start_time = time.time()
        builder = EnhancedGraphBuilder(
            storage=None,  # Use None for testing without actual connection
            enable_openie=True,
            openie_config={"min_confidence": 0.6}
        )
        init_time = time.time() - start_time
        
        integration_success = hasattr(builder, 'openie_enabled') and hasattr(builder, 'openie_extractor')
        print_result("Enhanced Graph Builder", integration_success, init_time,
                    f"OpenIE enabled: {getattr(builder, 'openie_enabled', False)}")
        
        return integration_success
        
    except ImportError as e:
        print_result("Integration Import", False, details=f"Import error: {e}")
        return False
    except Exception as e:
        print_result("Integration Test", False, details=f"Error: {e}")
        return False


async def test_openie_with_file(file_path: str):
    """Test OpenIE with a file input."""
    print_section(f"OpenIE File Test: {file_path}")
    
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print_result("File Exists", False, details=f"File not found: {file_path}")
            return False
        
        # Read file content
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print_result("File Read", True, details=f"Content length: {len(content)} characters")
        
        # Test with OpenIE extractor
        from morag_graph.extractors.openie_extractor import OpenIEExtractor
        
        extractor = OpenIEExtractor({"min_confidence": 0.6})
        
        start_time = time.time()
        relations = await extractor.extract_relations(content, source_doc_id=file_path)
        extract_time = time.time() - start_time
        
        print_result("File Processing", True, extract_time,
                    f"Extracted {len(relations)} relations from file")
        
        # Show sample relations
        if relations:
            print(f"\nSample Relations from File:")
            for i, relation in enumerate(relations[:3], 1):  # Show first 3
                print(f"  {i}. {relation.subject} --[{relation.predicate}]--> {relation.object}")
        
        return True
        
    except ImportError as e:
        print_result("File Test Import", False, details=f"Import error: {e}")
        return False
    except Exception as e:
        print_result("File Test", False, details=f"Error: {e}")
        return False


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Test OpenIE functionality in MoRAG")
    parser.add_argument("--test", choices=["basic", "config", "service", "extractor", "integration", "extraction", "all"],
                       default="basic", help="Type of test to run")
    parser.add_argument("--text", help="Sample text for extraction testing")
    parser.add_argument("--file", help="File path for file-based testing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 MoRAG OpenIE Testing Suite")
    print("=" * 60)
    
    results = {}
    
    if args.test in ["basic", "config", "all"]:
        results["config"] = await test_openie_configuration()
    
    if args.test in ["basic", "service", "all"]:
        results["service"] = await test_openie_service()
    
    if args.test in ["basic", "extractor", "all"]:
        results["extractor"] = await test_openie_extractor()
    
    if args.test in ["integration", "all"]:
        results["integration"] = await test_openie_integration()
    
    if args.test == "extraction" and args.text:
        # Test with custom text
        print_section("Custom Text Extraction")
        try:
            from morag_graph.extractors.openie_extractor import OpenIEExtractor
            extractor = OpenIEExtractor({"min_confidence": 0.6})
            relations = await extractor.extract_relations(args.text)
            print(f"Extracted {len(relations)} relations from custom text")
            for i, relation in enumerate(relations, 1):
                print(f"  {i}. {relation.subject} --[{relation.predicate}]--> {relation.object}")
            results["custom_text"] = True
        except Exception as e:
            print(f"Custom text extraction failed: {e}")
            results["custom_text"] = False
    
    if args.file:
        results["file"] = await test_openie_with_file(args.file)
    
    # Print summary
    print_section("Test Summary")
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All OpenIE tests passed!")
        return 0
    else:
        print("⚠️  Some OpenIE tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
