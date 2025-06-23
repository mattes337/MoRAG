#!/usr/bin/env python3
"""
Shared utilities for resuming CLI processing from intermediate files.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add the project root to the path
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def print_result(key: str, value: str, indent: int = 0):
    """Print a formatted key-value result."""
    spaces = "  " * indent
    print(f"{spaces}📋 {key}: {value}")


async def resume_from_process_result(process_result_data: Dict[str, Any], source_file: str,
                                   content_type: str, use_qdrant: bool = False, use_neo4j: bool = False,
                                   webhook_url: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Resume processing from a process result file."""
    try:
        print_header(f"MoRAG {content_type.title()} Resume from Process Result")
        print_result("Process Result Mode", "✅ Enabled")
        print_result("Source File", source_file)
        
        # Import ingestion coordinator
        from morag.ingestion_coordinator import IngestionCoordinator
        from morag_graph.models.database_config import DatabaseConfig, DatabaseType
        from morag_core.models.config import ProcessingResult
        
        # Create database configurations
        database_configs = []
        if use_qdrant:
            database_configs.append(DatabaseConfig(
                type=DatabaseType.QDRANT,
                hostname=os.getenv('QDRANT_HOST', 'localhost'),
                port=int(os.getenv('QDRANT_PORT', '6333')),
                collection_name=os.getenv('QDRANT_COLLECTION', 'morag_documents')
            ))
        
        if use_neo4j:
            database_configs.append(DatabaseConfig(
                type=DatabaseType.NEO4J,
                hostname=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                username=os.getenv('NEO4J_USERNAME', 'neo4j'),
                password=os.getenv('NEO4J_PASSWORD', 'password'),
                database_name=os.getenv('NEO4J_DATABASE', 'neo4j')
            ))
        
        # Extract content from process result
        content = process_result_data.get('content', '')
        if not content:
            print("❌ Error: No content found in process result file")
            return False
        
        # Create processing result object
        processing_result = ProcessingResult(
            success=True,
            task_id=process_result_data.get('task_id', 'resumed-task'),
            source_type=content_type,
            content=content,
            metadata=process_result_data.get('metadata', {}),
            processing_time=process_result_data.get('processing_time', 0.0)
        )
        
        # Initialize ingestion coordinator
        coordinator = IngestionCoordinator()
        
        # Perform ingestion from process result
        print_section("Starting Ingestion from Process Result")
        result = await coordinator.ingest_content(
            content=content,
            source_path=source_file,
            content_type=content_type,
            metadata=metadata or {},
            processing_result=processing_result,
            databases=database_configs,
            chunk_size=4000,
            chunk_overlap=200,
            document_id=Path(source_file).stem,
            replace_existing=True
        )
        
        print_section("Ingestion Results")
        print_result("Status", "✅ Success")
        print_result("Document ID", result['source_info']['document_id'])
        print_result("Processing Time", f"{result['processing_time']:.2f} seconds")
        print_result("Chunks Created", str(result['embeddings_data']['chunk_count']))
        print_result("Entities Extracted", str(result['graph_data']['entities_count']))
        print_result("Relations Extracted", str(result['graph_data']['relations_count']))
        
        return True
        
    except Exception as e:
        print(f"❌ Error during resume from process result: {e}")
        import traceback
        traceback.print_exc()
        return False


async def resume_from_ingestion_data(ingestion_data: Dict[str, Any], source_file: str,
                                   content_type: str, use_qdrant: bool = False, use_neo4j: bool = False,
                                   webhook_url: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Resume processing from an ingestion data file."""
    try:
        print_header(f"MoRAG {content_type.title()} Resume from Ingestion Data")
        print_result("Ingestion Data Mode", "✅ Enabled")
        print_result("Source File", source_file)
        
        # Import ingestion coordinator
        from morag.ingestion_coordinator import IngestionCoordinator
        from morag_graph.models.database_config import DatabaseConfig, DatabaseType
        
        # Create database configurations
        database_configs = []
        if use_qdrant:
            database_configs.append(DatabaseConfig(
                type=DatabaseType.QDRANT,
                hostname=os.getenv('QDRANT_HOST', 'localhost'),
                port=int(os.getenv('QDRANT_PORT', '6333')),
                collection_name=os.getenv('QDRANT_COLLECTION', 'morag_documents')
            ))
        
        if use_neo4j:
            database_configs.append(DatabaseConfig(
                type=DatabaseType.NEO4J,
                hostname=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                username=os.getenv('NEO4J_USERNAME', 'neo4j'),
                password=os.getenv('NEO4J_PASSWORD', 'password'),
                database_name=os.getenv('NEO4J_DATABASE', 'neo4j')
            ))
        
        # Initialize ingestion coordinator
        coordinator = IngestionCoordinator()
        
        # Initialize databases
        print_section("Initializing Databases")
        await coordinator._initialize_databases(database_configs, ingestion_data.get('vector_data', {}))
        
        # Write to databases using ingestion data
        print_section("Writing to Databases")
        document_id = ingestion_data.get('document_id', 'unknown')
        
        results = await coordinator._write_to_databases(
            database_configs,
            ingestion_data.get('vector_data', {}),
            ingestion_data.get('graph_data', {}),
            document_id,
            replace_existing=True
        )
        
        print_section("Database Write Results")
        for db_name, result in results.items():
            if result.get('success', False):
                print_result(f"{db_name.upper()} Status", "✅ Success")
                if 'documents_written' in result:
                    print_result(f"{db_name.upper()} Documents", str(result['documents_written']))
                if 'entities_written' in result:
                    print_result(f"{db_name.upper()} Entities", str(result['entities_written']))
                if 'relations_written' in result:
                    print_result(f"{db_name.upper()} Relations", str(result['relations_written']))
            else:
                print_result(f"{db_name.upper()} Status", f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during resume from ingestion data: {e}")
        import traceback
        traceback.print_exc()
        return False


def handle_resume_arguments(args, source_file: str, content_type: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Handle --use-process-result and --use-ingestion-data arguments.
    Returns True if resume was handled, False if normal processing should continue.
    """
    import asyncio
    
    # Handle --use-process-result argument
    if hasattr(args, 'use_process_result') and args.use_process_result:
        process_result_file = Path(args.use_process_result)
        if not process_result_file.exists():
            print(f"❌ Error: Process result file not found: {process_result_file}")
            sys.exit(1)
        
        try:
            with open(process_result_file, 'r', encoding='utf-8') as f:
                process_result_data = json.load(f)
            print(f"✅ Using existing process result from: {process_result_file}")
            print("💡 Skipping processing phase, continuing from result file...")
            
            # Continue with ingestion using the process result data
            success = asyncio.run(resume_from_process_result(
                process_result_data, 
                source_file,
                content_type,
                use_qdrant=getattr(args, 'qdrant', False),
                use_neo4j=getattr(args, 'neo4j', False),
                webhook_url=getattr(args, 'webhook_url', None),
                metadata=metadata
            ))
            
            if success:
                print(f"\n🎉 {content_type.title()} processing resumed successfully!")
                sys.exit(0)
            else:
                print(f"\n💥 {content_type.title()} processing resume failed!")
                sys.exit(1)
                
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in process result file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading process result file: {e}")
            sys.exit(1)

    # Handle --use-ingestion-data argument
    if hasattr(args, 'use_ingestion_data') and args.use_ingestion_data:
        ingestion_data_file = Path(args.use_ingestion_data)
        if not ingestion_data_file.exists():
            print(f"❌ Error: Ingestion data file not found: {ingestion_data_file}")
            sys.exit(1)
        
        try:
            with open(ingestion_data_file, 'r', encoding='utf-8') as f:
                ingestion_data = json.load(f)
            print(f"✅ Using existing ingestion data from: {ingestion_data_file}")
            print("💡 Skipping processing and ingestion calculation, starting database writes...")
            
            # Continue with database writes using the ingestion data
            success = asyncio.run(resume_from_ingestion_data(
                ingestion_data,
                source_file,
                content_type,
                use_qdrant=getattr(args, 'qdrant', False),
                use_neo4j=getattr(args, 'neo4j', False),
                webhook_url=getattr(args, 'webhook_url', None),
                metadata=metadata
            ))
            
            if success:
                print(f"\n🎉 {content_type.title()} ingestion resumed successfully!")
                sys.exit(0)
            else:
                print(f"\n💥 {content_type.title()} ingestion resume failed!")
                sys.exit(1)
                
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in ingestion data file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading ingestion data file: {e}")
            sys.exit(1)
    
    # No resume arguments provided, continue with normal processing
    return False
