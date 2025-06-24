#!/usr/bin/env python3
"""
Test script for the ingestion coordinator to verify it creates both
ingest_result.json and ingest_data.json files and performs actual ingestion.
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

try:
    from morag.ingestion_coordinator import IngestionCoordinator, DatabaseConfig, DatabaseType
    from morag_core.models.config import ProcessingResult
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag")
    sys.exit(1)


async def test_ingestion_coordinator():
    """Test the ingestion coordinator with sample content."""
    print("🧪 Testing Ingestion Coordinator")
    print("=" * 60)
    
    # Sample content for testing
    test_content = """
    This is a test document about artificial intelligence and machine learning.
    
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that work and react like humans. Machine Learning (ML) is 
    a subset of AI that provides systems the ability to automatically learn and 
    improve from experience without being explicitly programmed.
    
    Key concepts in AI include:
    - Natural Language Processing (NLP)
    - Computer Vision
    - Robotics
    - Expert Systems
    
    Machine Learning algorithms can be categorized into:
    1. Supervised Learning
    2. Unsupervised Learning  
    3. Reinforcement Learning
    
    The relationship between AI and ML is fundamental to modern technology advancement.
    """
    
    # Create a temporary test file
    test_file = Path("test_sample.txt")
    test_file.write_text(test_content, encoding='utf-8')
    
    try:
        print(f"📄 Test file: {test_file}")
        print(f"📊 Content length: {len(test_content)} characters")
        
        # Configure databases
        database_configs = []
        
        # Add Qdrant if available
        import os
        if os.getenv('QDRANT_HOST') or Path('qdrant').exists():
            database_configs.append(DatabaseConfig(
                type=DatabaseType.QDRANT,
                hostname='localhost',
                port=6333,
                database_name='morag_documents'
            ))
            print("✅ Qdrant database configured")
        
        # Add Neo4j if available and properly configured
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_user = os.getenv('NEO4J_USERNAME')
        neo4j_pass = os.getenv('NEO4J_PASSWORD')
        if neo4j_uri and neo4j_user and neo4j_pass:
            database_configs.append(DatabaseConfig(
                type=DatabaseType.NEO4J,
                hostname=neo4j_uri,
                username=neo4j_user,
                password=neo4j_pass,
                database_name=os.getenv('NEO4J_DATABASE', 'neo4j')
            ))
            print("✅ Neo4j database configured")
        else:
            print("⚠️ Neo4j not configured (missing environment variables)")
        
        if not database_configs:
            print("⚠️ No databases configured, testing file creation only")
        
        # Create a mock processing result
        processing_result = ProcessingResult(
            success=True,
            task_id='test-task-123',
            source_type='text',
            content=test_content,
            processing_time=1.5,
            metadata={'test': True, 'source_type': 'text'}
        )
        
        # Initialize ingestion coordinator
        print("\n🔧 Initializing ingestion coordinator...")
        coordinator = IngestionCoordinator()
        
        # Perform ingestion
        print("🚀 Starting ingestion process...")
        result = await coordinator.ingest_content(
            content=test_content,
            source_path=str(test_file),
            content_type='text',
            metadata={'test_run': True, 'category': 'ai_ml'},
            processing_result=processing_result,
            databases=database_configs,
            chunk_size=500,
            chunk_overlap=50,
            document_id=None,  # Let coordinator generate proper unified ID
            replace_existing=False
        )
        
        print("✅ Ingestion completed!")
        
        # Check results
        print("\n📋 Ingestion Results:")
        print(f"  Ingestion ID: {result['ingestion_id']}")
        print(f"  Document ID: {result['source_info']['document_id']}")
        print(f"  Processing Time: {result['processing_time']:.2f} seconds")
        print(f"  Chunks Created: {result['embeddings_data']['chunk_count']}")
        print(f"  Entities Extracted: {result['graph_data']['entities_count']}")
        print(f"  Relations Extracted: {result['graph_data']['relations_count']}")
        
        # Check file creation
        print("\n📁 Checking output files:")
        result_file = test_file.parent / f"{test_file.stem}.ingest_result.json"
        data_file = test_file.parent / f"{test_file.stem}.ingest_data.json"
        
        if result_file.exists():
            print(f"✅ Ingest result file created: {result_file}")
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
                print(f"   File size: {result_file.stat().st_size} bytes")
                print(f"   Contains: {len(result_data)} top-level keys")
        else:
            print(f"❌ Ingest result file missing: {result_file}")
        
        if data_file.exists():
            print(f"✅ Ingest data file created: {data_file}")
            with open(data_file, 'r', encoding='utf-8') as f:
                data_content = json.load(f)
                print(f"   File size: {data_file.stat().st_size} bytes")
                print(f"   Vector chunks: {len(data_content.get('vector_data', {}).get('chunks', []))}")
                print(f"   Graph entities: {len(data_content.get('graph_data', {}).get('entities', []))}")
                print(f"   Graph relations: {len(data_content.get('graph_data', {}).get('relations', []))}")
        else:
            print(f"❌ Ingest data file missing: {data_file}")
        
        # Check database results
        if 'database_results' in result:
            print("\n💾 Database Results:")
            for db_type, db_result in result['database_results'].items():
                if db_result.get('success'):
                    print(f"✅ {db_type.title()}: Success")
                    if db_type == 'qdrant' and 'points_stored' in db_result:
                        print(f"   Points stored: {db_result['points_stored']}")
                    elif db_type == 'neo4j':
                        if 'chunks_stored' in db_result:
                            print(f"   Chunks stored: {db_result['chunks_stored']}")
                        if 'entities_stored' in db_result:
                            print(f"   Entities stored: {db_result['entities_stored']}")
                        if 'relations_stored' in db_result:
                            print(f"   Relations stored: {db_result['relations_stored']}")
                else:
                    print(f"❌ {db_type.title()}: Failed - {db_result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during ingestion test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        
        # Clean up output files
        for pattern in ["*.ingest_result.json", "*.ingest_data.json"]:
            for file in Path(".").glob(pattern):
                if file.name.startswith("test_sample"):
                    file.unlink()
                    print(f"🧹 Cleaned up: {file}")


async def main():
    """Main function."""
    print("🧪 MoRAG Ingestion Coordinator Test")
    print("=" * 60)
    
    success = await test_ingestion_coordinator()
    
    if success:
        print("\n🎉 Ingestion coordinator test completed successfully!")
        print("✅ Both ingest_result.json and ingest_data.json files should be created")
        print("✅ Actual database ingestion should be performed")
        return 0
    else:
        print("\n💥 Ingestion coordinator test failed!")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
