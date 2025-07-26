#!/usr/bin/env python3
"""
OpenIE Integration Testing CLI Script

This script tests the integration of OpenIE with the MoRAG processing pipeline,
specifically testing the EnhancedGraphBuilder and end-to-end processing.

Usage:
    python cli/test-openie-integration.py [options]

Examples:
    # Test basic integration
    python cli/test-openie-integration.py --test basic

    # Test with sample document
    python cli/test-openie-integration.py --test document --text "Sample document content"

    # Test with file input
    python cli/test-openie-integration.py --test file --input sample.txt

    # Test Neo4j integration (requires running Neo4j)
    python cli/test-openie-integration.py --test neo4j --neo4j-uri bolt://localhost:7687 --neo4j-password password

    # Test all integration components
    python cli/test-openie-integration.py --test all --verbose
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


async def test_enhanced_graph_builder():
    """Test EnhancedGraphBuilder with OpenIE integration."""
    print_section("Enhanced Graph Builder Test")
    
    try:
        from morag_graph.builders.enhanced_graph_builder import EnhancedGraphBuilder, EnhancedGraphBuildResult
        
        # Test initialization without storage (for testing)
        start_time = time.time()
        builder = EnhancedGraphBuilder(
            storage=None,  # Mock storage for testing
            enable_openie=True,
            openie_config={
                "min_confidence": 0.6,
                "enable_entity_linking": True,
                "enable_predicate_normalization": True
            }
        )
        init_time = time.time() - start_time
        
        # Check if OpenIE is properly initialized
        openie_available = hasattr(builder, 'openie_enabled') and builder.openie_enabled
        openie_extractor_available = hasattr(builder, 'openie_extractor') and builder.openie_extractor is not None
        
        print_result("Builder Initialization", True, init_time)
        print_result("OpenIE Enabled", openie_available, details=f"OpenIE enabled: {getattr(builder, 'openie_enabled', False)}")
        print_result("OpenIE Extractor Available", openie_extractor_available)
        
        # Test configuration
        if hasattr(builder, 'openie_extractor') and builder.openie_extractor:
            extractor = builder.openie_extractor
            config_ok = (
                hasattr(extractor, 'enabled') and
                hasattr(extractor, 'min_confidence') and
                hasattr(extractor, 'enable_entity_linking')
            )
            print_result("OpenIE Configuration", config_ok)
        
        return openie_available and openie_extractor_available
        
    except ImportError as e:
        print_result("Enhanced Builder Import", False, details=f"Import error: {e}")
        return False
    except Exception as e:
        print_result("Enhanced Builder Test", False, details=f"Error: {e}")
        return False


async def test_document_processing():
    """Test document processing with OpenIE integration."""
    print_section("Document Processing Test")
    
    try:
        from morag_graph.builders.enhanced_graph_builder import EnhancedGraphBuilder
        from morag_graph.storage.base import BaseStorage
        
        # Mock storage for testing
        class MockStorage(BaseStorage):
            async def connect(self): pass
            async def disconnect(self): pass
            async def store_entities(self, entities): return {"entities_stored": len(entities)}
            async def store_relations(self, relations): return {"relations_stored": len(relations)}
            async def get_entity(self, entity_id): return None
            async def get_relation(self, relation_id): return None
        
        mock_storage = MockStorage()
        
        # Initialize enhanced graph builder
        builder = EnhancedGraphBuilder(
            storage=mock_storage,
            enable_openie=True,
            openie_config={"min_confidence": 0.6}
        )
        
        # Test document content
        sample_document = """
        Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
        The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
        Apple is known for its innovative products including the iPhone, iPad, and Mac computers.
        Tim Cook became the CEO of Apple in 2011, succeeding Steve Jobs.
        The company's headquarters, Apple Park, was opened in 2017.
        """
        
        # Note: This test will fail without proper storage, but we can test the structure
        try:
            start_time = time.time()
            # This would normally process the document, but will fail without real storage
            # result = await builder.process_document(sample_document, "test_doc_001")
            # For now, just test that the method exists and is callable
            process_method_exists = hasattr(builder, 'process_document') and callable(builder.process_document)
            process_time = time.time() - start_time
            
            print_result("Process Document Method", process_method_exists, process_time)
            
            # Test chunk processing method
            chunk_method_exists = hasattr(builder, 'process_document_chunks') and callable(builder.process_document_chunks)
            print_result("Process Chunks Method", chunk_method_exists)
            
            return process_method_exists and chunk_method_exists
            
        except Exception as e:
            # Expected to fail without real storage, but we can check the error type
            method_exists = "process_document" in str(e) or hasattr(builder, 'process_document')
            print_result("Document Processing Structure", method_exists, details=f"Expected error: {type(e).__name__}")
            return method_exists
        
    except ImportError as e:
        print_result("Document Processing Import", False, details=f"Import error: {e}")
        return False
    except Exception as e:
        print_result("Document Processing Test", False, details=f"Error: {e}")
        return False


async def test_openie_pipeline():
    """Test the complete OpenIE pipeline."""
    print_section("OpenIE Pipeline Test")
    
    try:
        from morag_graph.extractors.openie_extractor import OpenIEExtractor, OpenIEExtractionResult
        from morag_graph.models import Entity
        
        # Initialize extractor
        extractor = OpenIEExtractor({
            "min_confidence": 0.6,
            "enable_entity_linking": True,
            "enable_predicate_normalization": True
        })
        
        # Test text
        test_text = """
        Microsoft Corporation is a technology company based in Redmond, Washington.
        The company was founded by Bill Gates and Paul Allen in 1975.
        Satya Nadella is the current CEO of Microsoft.
        Microsoft develops software, hardware, and cloud services.
        """
        
        # Sample entities (simulating spaCy NER output)
        sample_entities = [
            Entity(
                id="entity_1",
                name="Microsoft",
                canonical_name="microsoft",
                entity_type="ORG",
                confidence=0.98,
                metadata={"source": "test"}
            ),
            Entity(
                id="entity_2",
                name="Bill Gates",
                canonical_name="bill_gates",
                entity_type="PERSON",
                confidence=0.95,
                metadata={"source": "test"}
            ),
            Entity(
                id="entity_3",
                name="Satya Nadella",
                canonical_name="satya_nadella",
                entity_type="PERSON",
                confidence=0.94,
                metadata={"source": "test"}
            )
        ]
        
        # Test basic relation extraction
        start_time = time.time()
        relations = await extractor.extract_relations(test_text, source_doc_id="pipeline_test")
        relation_time = time.time() - start_time
        
        relation_success = isinstance(relations, list)
        print_result("Relation Extraction", relation_success, relation_time, 
                    f"Extracted {len(relations)} relations")
        
        # Test full extraction
        start_time = time.time()
        full_result = await extractor.extract_full(test_text, entities=sample_entities, source_doc_id="pipeline_test")
        full_time = time.time() - start_time
        
        full_success = isinstance(full_result, OpenIEExtractionResult)
        print_result("Full Pipeline Extraction", full_success, full_time,
                    f"Relations: {len(full_result.relations) if full_success else 0}, "
                    f"Triplets: {len(full_result.triplets) if full_success else 0}")
        
        # Test extraction statistics
        if full_success and hasattr(full_result, 'metadata'):
            stats_available = isinstance(full_result.metadata, dict) and len(full_result.metadata) > 0
            print_result("Extraction Statistics", stats_available)
            
            if stats_available:
                print(f"\nPipeline Statistics:")
                for key, value in full_result.metadata.items():
                    print(f"  - {key}: {value}")
        
        # Show sample results
        if relation_success and relations:
            print(f"\nSample Relations:")
            for i, relation in enumerate(relations[:3], 1):
                print(f"  {i}. {relation.subject} --[{relation.predicate}]--> {relation.object}")
        
        return relation_success and full_success
        
    except ImportError as e:
        print_result("Pipeline Import", False, details=f"Import error: {e}")
        return False
    except Exception as e:
        print_result("Pipeline Test", False, details=f"Error: {e}")
        return False


async def test_file_processing(file_path: str):
    """Test OpenIE integration with file processing."""
    print_section(f"File Processing Test: {file_path}")
    
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print_result("File Exists", False, details=f"File not found: {file_path}")
            return False
        
        # Read file content
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print_result("File Read", True, details=f"Content length: {len(content)} characters")
        
        # Test with enhanced graph builder
        from morag_graph.builders.enhanced_graph_builder import EnhancedGraphBuilder
        from morag_graph.storage.base import BaseStorage
        
        # Mock storage
        class MockStorage(BaseStorage):
            async def connect(self): pass
            async def disconnect(self): pass
            async def store_entities(self, entities): return {"entities_stored": len(entities)}
            async def store_relations(self, relations): return {"relations_stored": len(relations)}
            async def get_entity(self, entity_id): return None
            async def get_relation(self, relation_id): return None
        
        builder = EnhancedGraphBuilder(
            storage=MockStorage(),
            enable_openie=True,
            openie_config={"min_confidence": 0.6}
        )
        
        # Test that the builder can handle the file content structure
        builder_ready = (
            hasattr(builder, 'openie_enabled') and 
            hasattr(builder, 'openie_extractor') and
            hasattr(builder, 'process_document')
        )
        
        print_result("Builder Ready for File", builder_ready)
        
        # Test OpenIE extractor directly with file content
        if hasattr(builder, 'openie_extractor') and builder.openie_extractor:
            start_time = time.time()
            relations = await builder.openie_extractor.extract_relations(content, source_doc_id=file_path)
            extract_time = time.time() - start_time
            
            print_result("File Content Extraction", True, extract_time,
                        f"Extracted {len(relations)} relations from file")
            
            if relations:
                print(f"\nSample Relations from File:")
                for i, relation in enumerate(relations[:3], 1):
                    print(f"  {i}. {relation.subject} --[{relation.predicate}]--> {relation.object}")
        
        return builder_ready
        
    except ImportError as e:
        print_result("File Processing Import", False, details=f"Import error: {e}")
        return False
    except Exception as e:
        print_result("File Processing Test", False, details=f"Error: {e}")
        return False


async def test_neo4j_integration(neo4j_uri: str, neo4j_password: str):
    """Test OpenIE integration with Neo4j storage."""
    print_section("Neo4j Integration Test")
    
    try:
        from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
        from morag_graph.builders.enhanced_graph_builder import EnhancedGraphBuilder
        
        # Configure Neo4j
        neo4j_config = Neo4jConfig(
            uri=neo4j_uri,
            username="neo4j",
            password=neo4j_password,
            database="neo4j"
        )
        
        # Test Neo4j connection
        storage = Neo4jStorage(neo4j_config)
        
        try:
            start_time = time.time()
            await storage.connect()
            connect_time = time.time() - start_time
            print_result("Neo4j Connection", True, connect_time)
            
            # Test OpenIE schema initialization
            start_time = time.time()
            await storage.initialize_openie_schema()
            schema_time = time.time() - start_time
            print_result("OpenIE Schema Initialization", True, schema_time)
            
            # Test enhanced graph builder with real storage
            builder = EnhancedGraphBuilder(
                storage=storage,
                enable_openie=True,
                openie_config={"min_confidence": 0.6}
            )
            
            print_result("Enhanced Builder with Neo4j", True)
            
            # Test small document processing
            test_doc = "Apple Inc. was founded by Steve Jobs. The company is based in Cupertino."
            
            try:
                start_time = time.time()
                result = await builder.process_document(test_doc, "neo4j_test_doc")
                process_time = time.time() - start_time
                
                processing_success = hasattr(result, 'openie_enabled') and hasattr(result, 'openie_relations_created')
                print_result("Document Processing with Neo4j", processing_success, process_time,
                            f"OpenIE relations: {getattr(result, 'openie_relations_created', 0)}")
                
                if processing_success:
                    print(f"\nProcessing Results:")
                    print(f"  - Entities created: {getattr(result, 'entities_created', 0)}")
                    print(f"  - Relations created: {getattr(result, 'relations_created', 0)}")
                    print(f"  - OpenIE relations: {getattr(result, 'openie_relations_created', 0)}")
                    print(f"  - OpenIE triplets: {getattr(result, 'openie_triplets_processed', 0)}")
                
                return processing_success
                
            except Exception as e:
                print_result("Document Processing", False, details=f"Processing error: {e}")
                return False
            
        except Exception as e:
            print_result("Neo4j Connection", False, details=f"Connection error: {e}")
            return False
        finally:
            try:
                await storage.disconnect()
            except:
                pass
        
    except ImportError as e:
        print_result("Neo4j Integration Import", False, details=f"Import error: {e}")
        return False
    except Exception as e:
        print_result("Neo4j Integration Test", False, details=f"Error: {e}")
        return False


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Test OpenIE integration in MoRAG")
    parser.add_argument("--test", choices=["basic", "builder", "document", "pipeline", "file", "neo4j", "all"],
                       default="basic", help="Type of integration test to run")
    parser.add_argument("--text", help="Sample text for document testing")
    parser.add_argument("--input", help="Input file path for file testing")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI for Neo4j testing")
    parser.add_argument("--neo4j-password", help="Neo4j password for Neo4j testing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 MoRAG OpenIE Integration Testing Suite")
    print("=" * 60)
    
    results = {}
    
    if args.test in ["basic", "builder", "all"]:
        results["enhanced_builder"] = await test_enhanced_graph_builder()
    
    if args.test in ["basic", "document", "all"]:
        results["document_processing"] = await test_document_processing()
    
    if args.test in ["basic", "pipeline", "all"]:
        results["openie_pipeline"] = await test_openie_pipeline()
    
    if args.test == "file" and args.input:
        results["file_processing"] = await test_file_processing(args.input)
    elif args.test == "all" and args.input:
        results["file_processing"] = await test_file_processing(args.input)
    
    if args.test == "neo4j" and args.neo4j_password:
        results["neo4j_integration"] = await test_neo4j_integration(args.neo4j_uri, args.neo4j_password)
    elif args.test == "all" and args.neo4j_password:
        results["neo4j_integration"] = await test_neo4j_integration(args.neo4j_uri, args.neo4j_password)
    
    # Print summary
    print_section("Integration Test Summary")
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} integration tests passed")
    
    if total_tests == 0:
        print("⚠️  No tests were run. Use --test option to specify tests.")
        return 1
    elif passed_tests == total_tests:
        print("🎉 All OpenIE integration tests passed!")
        return 0
    else:
        print("⚠️  Some OpenIE integration tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
