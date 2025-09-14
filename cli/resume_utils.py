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
load_dotenv(env_path, override=True)


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
    print(f"{spaces}[INFO] {key}: {value}")


async def resume_from_process_result(process_result_data: Dict[str, Any], source_file: str,
                                   content_type: str, use_qdrant: bool = False, use_neo4j: bool = False,
                                   webhook_url: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Resume processing from a process result file."""
    try:
        print_header(f"MoRAG {content_type.title()} Resume from Process Result")
        print_result("Process Result Mode", "[OK] Enabled")
        print_result("Source File", source_file)
        
        # Import ingestion coordinator
        from morag.ingestion_coordinator import IngestionCoordinator
        from morag_graph.models.database_config import DatabaseConfig, DatabaseType
        from morag_core.models.config import ProcessingResult
        
        # Create database configurations
        database_configs = []
        if use_qdrant:
            # Prefer QDRANT_URL if available, otherwise use QDRANT_HOST/PORT
            qdrant_url = os.getenv('QDRANT_URL')
            if qdrant_url:
                database_configs.append(DatabaseConfig(
                    type=DatabaseType.QDRANT,
                    hostname=qdrant_url,  # Store URL in hostname field
                    database_name=os.getenv('QDRANT_COLLECTION_NAME', 'morag_documents')
                ))
            else:
                database_configs.append(DatabaseConfig(
                    type=DatabaseType.QDRANT,
                    hostname=os.getenv('QDRANT_HOST', 'localhost'),
                    port=int(os.getenv('QDRANT_PORT', '6333')),
                    database_name=os.getenv('QDRANT_COLLECTION_NAME', 'morag_documents')
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
            print("[FAIL] Error: No content found in process result file")
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
        print_result("Status", "[OK] Success")
        print_result("Document ID", result['source_info']['document_id'])
        print_result("Processing Time", f"{result['processing_time']:.2f} seconds")
        print_result("Chunks Created", str(result['embeddings_data']['chunk_count']))
        print_result("Entities Extracted", str(result['graph_data']['entities_count']))
        print_result("Relations Extracted", str(result['graph_data']['relations_count']))
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error during resume from process result: {e}")
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
        print_result("Ingestion Data Mode", "[OK] Enabled")
        print_result("Source File", source_file)
        
        # Import ingestion coordinator
        from morag.ingestion_coordinator import IngestionCoordinator
        from morag_graph.models.database_config import DatabaseConfig, DatabaseType
        
        # Create database configurations
        database_configs = []
        if use_qdrant:
            # Prefer QDRANT_URL if available, otherwise use QDRANT_HOST/PORT
            qdrant_url = os.getenv('QDRANT_URL')
            if qdrant_url:
                database_configs.append(DatabaseConfig(
                    type=DatabaseType.QDRANT,
                    hostname=qdrant_url,  # Store URL in hostname field
                    database_name=os.getenv('QDRANT_COLLECTION_NAME', 'morag_documents')
                ))
            else:
                database_configs.append(DatabaseConfig(
                    type=DatabaseType.QDRANT,
                    hostname=os.getenv('QDRANT_HOST', 'localhost'),
                    port=int(os.getenv('QDRANT_PORT', '6333')),
                    database_name=os.getenv('QDRANT_COLLECTION_NAME', 'morag_documents')
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

        # Transform ingestion data back to embeddings_data format
        # Handle both ingest_result.json and ingest_data.json formats
        if 'vector_data' in ingestion_data:
            # This is ingest_data.json format
            vector_data = ingestion_data.get('vector_data', {})
            chunks_data = vector_data.get('chunks', [])

            # Extract chunks, embeddings, and metadata from ingestion data
            chunks = []
            embeddings = []
            chunk_metadata = []

            for chunk_data in chunks_data:
                chunks.append(chunk_data['chunk_text'])
                embeddings.append(chunk_data['embedding'])
                chunk_metadata.append(chunk_data['metadata'])

        elif 'embeddings_data' in ingestion_data:
            # This is ingest_result.json format
            embeddings_data_section = ingestion_data.get('embeddings_data', {})
            chunks_data = embeddings_data_section.get('chunks', [])
            print(f"🔍 DEBUG: Found {len(chunks_data)} chunks in embeddings_data")

            # Extract chunks, embeddings, and metadata from ingest_result format
            chunks = []
            embeddings = []
            chunk_metadata = []

            for chunk_data in chunks_data:
                chunks.append(chunk_data['chunk_text'])
                embeddings.append(chunk_data['embedding'])
                # Create metadata from the chunk data
                metadata = {
                    'chunk_id': chunk_data['chunk_id'],
                    'document_id': ingestion_data.get('source_info', {}).get('document_id', 'unknown'),
                    'chunk_index': chunk_data['chunk_index'],
                    'chunk_text': chunk_data['chunk_text'],
                    'chunk_size': chunk_data['chunk_size'],
                    'created_at': ingestion_data.get('timestamp', ''),
                    **chunk_data.get('metadata', {})
                }
                chunk_metadata.append(metadata)
        else:
            print("[FAIL] Error: Unrecognized ingestion data format")
            return False

        # Get document ID and convert to expected format if needed
        original_document_id = ingestion_data.get('document_id') or ingestion_data.get('source_info', {}).get('document_id', 'unknown')

        # Convert UUID format to doc_ format if needed for Neo4j compatibility
        if original_document_id and not original_document_id.startswith('doc_'):
            # Convert UUID to doc_ format
            document_id = f"doc_{original_document_id}"
        else:
            document_id = original_document_id

        # Update chunk metadata to use the converted document ID
        for meta in chunk_metadata:
            if 'document_id' in meta:
                meta['document_id'] = document_id
            # Also update chunk_id if it contains the old document ID
            if 'chunk_id' in meta and original_document_id in meta['chunk_id']:
                meta['chunk_id'] = meta['chunk_id'].replace(original_document_id, document_id)

        # Also update the chunks data in the embeddings_data format for consistency
        for i, chunk_data in enumerate(chunks_data):
            if 'chunk_id' in chunk_data and original_document_id in chunk_data['chunk_id']:
                chunk_data['chunk_id'] = chunk_data['chunk_id'].replace(original_document_id, document_id)

        # Reconstruct embeddings_data structure expected by _write_to_databases
        embeddings_data = {
            'chunks': chunks,
            'embeddings': embeddings,
            'chunk_metadata': chunk_metadata,
            'document_id': document_id,
            'embedding_dimension': len(embeddings[0]) if embeddings else 768
        }

        # Initialize databases
        print_section("Initializing Databases")
        await coordinator._initialize_databases(database_configs, embeddings_data)

        # Extract graph data for Neo4j and convert to proper model objects
        from morag_graph.models.entity import Entity
        from morag_graph.models.relation import Relation

        if 'graph_data' in ingestion_data:
            # Extract entities and relations data
            if 'vector_data' in ingestion_data:
                # This is ingest_data.json format
                graph_data_section = ingestion_data.get('graph_data', {})
            else:
                # This is ingest_result.json format
                graph_data_section = ingestion_data.get('graph_data', {})

            # Check for enhanced processing data first (preferred)
            enhanced_processing = graph_data_section.get('enhanced_processing', {})
            if enhanced_processing and enhanced_processing.get('entities'):
                entities_data = enhanced_processing.get('entities', [])
                print(f"🔍 DEBUG: Found {len(entities_data)} entities from enhanced processing")
            else:
                # Fallback to basic entities data
                entities_data = graph_data_section.get('entities', [])
                print(f"🔍 DEBUG: Found {len(entities_data)} entities from basic graph_data")

            relations_data = graph_data_section.get('relations', [])
            print(f"🔍 DEBUG: Found {len(relations_data)} relations in graph_data")

            # Check if chunk_entity_mapping exists, if not, create it
            chunk_entity_mapping = graph_data_section.get('chunk_entity_mapping', {})
            if not chunk_entity_mapping:
                print("🔧 DEBUG: chunk_entity_mapping missing, recreating from entities and chunks...")
                chunk_entity_mapping = await _recreate_chunk_entity_mapping(
                    chunks, entities_data, ingestion_data.get('source_info', {}).get('chunk_size', 4000)
                )
                print(f"🔧 DEBUG: Recreated chunk_entity_mapping with {len(chunk_entity_mapping)} chunks containing entities")

            # Convert dictionary data to proper Entity objects
            entities = []
            for entity_dict in entities_data:
                try:
                    # Use string type directly (no enum conversion needed)
                    entity_type = entity_dict.get('type', 'UNKNOWN')
                    if not isinstance(entity_type, str):
                        entity_type = str(entity_type)

                    entity = Entity(
                        id=entity_dict.get('id', ''),
                        name=entity_dict.get('name', ''),
                        type=entity_type,
                        confidence=entity_dict.get('confidence', 1.0),
                        source_doc_id=entity_dict.get('source_doc_id', document_id),
                        attributes=entity_dict.get('attributes', {})
                    )
                    entities.append(entity)
                except Exception as e:
                    print(f"[WARN]  Warning: Failed to create entity from {entity_dict.get('name', 'unknown')}: {e}")
                    continue

            # Convert dictionary data to proper Relation objects
            relations = []
            for relation_dict in relations_data:
                try:
                    # Use string type directly (no enum conversion needed)
                    relation_type = relation_dict.get('relation_type', 'RELATED_TO')
                    if not isinstance(relation_type, str):
                        relation_type = str(relation_type)

                    relation = Relation(
                        id=relation_dict.get('id', ''),
                        source_entity_id=relation_dict.get('source_entity_id', ''),
                        target_entity_id=relation_dict.get('target_entity_id', ''),
                        type=relation_type,
                        confidence=relation_dict.get('confidence', 1.0),
                        source_doc_id=relation_dict.get('source_doc_id', document_id),
                        attributes=relation_dict.get('attributes', {})
                    )
                    relations.append(relation)
                except Exception as e:
                    print(f"[WARN]  Warning: Failed to create relation {relation_dict.get('id', 'unknown')}: {e}")
                    continue

            # Include enhanced processing data and embeddings
            graph_data = {
                'entities': entities,
                'relations': relations,
                'facts': graph_data_section.get('facts', []),
                'relationships': graph_data_section.get('relationships', []),
                'chunk_fact_mapping': graph_data_section.get('chunk_fact_mapping', {}),
                'enhanced_processing': enhanced_processing,
                'entity_embeddings': graph_data_section.get('entity_embeddings', {}),
                'fact_embeddings': graph_data_section.get('fact_embeddings', {}),
                'extraction_metadata': graph_data_section.get('extraction_metadata', {})
            }
        else:
            # No graph data available
            graph_data = {
                'entities': [],
                'relations': [],
                'facts': [],
                'relationships': [],
                'chunk_fact_mapping': {},
                'enhanced_processing': {},
                'entity_embeddings': {},
                'fact_embeddings': {},
                'extraction_metadata': {}
            }

        # Add the recreated chunk_entity_mapping to graph_data if it was created
        if 'chunk_entity_mapping' in locals() and chunk_entity_mapping:
            graph_data['chunk_entity_mapping'] = chunk_entity_mapping

        # Write to databases using ingestion data
        print_section("Writing to Databases")
        # Use the converted document_id from embeddings_data
        document_id = embeddings_data['document_id']

        results = await coordinator._write_to_databases(
            database_configs,
            embeddings_data,
            graph_data,
            document_id,
            replace_existing=True
        )
        
        print_section("Database Write Results")
        for db_name, result in results.items():
            if result.get('success', False):
                print_result(f"{db_name.upper()} Status", "[OK] Success")
                if 'documents_written' in result:
                    print_result(f"{db_name.upper()} Documents", str(result['documents_written']))
                if 'entities_written' in result:
                    print_result(f"{db_name.upper()} Entities", str(result['entities_written']))
                if 'relations_written' in result:
                    print_result(f"{db_name.upper()} Relations", str(result['relations_written']))
            else:
                print_result(f"{db_name.upper()} Status", f"[FAIL] Failed: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error during resume from ingestion data: {e}")
        import traceback
        traceback.print_exc()
        return False


async def _recreate_chunk_entity_mapping(chunks, entities_data, chunk_size=4000):
    """Recreate chunk-entity mapping by finding which entities appear in which chunks.

    Args:
        chunks: List of chunk texts
        entities_data: List of entity dictionaries
        chunk_size: Size of chunks (for logging)

    Returns:
        Dictionary mapping chunk index (as string) to list of entity IDs
    """
    chunk_entity_mapping = {}

    print(f"🔍 Analyzing {len(chunks)} chunks for entity mentions...")

    for chunk_index, chunk_text in enumerate(chunks):
        entities_in_chunk = []
        chunk_text_lower = chunk_text.lower()

        # Check each entity to see if it appears in this chunk
        for entity_dict in entities_data:
            entity_name = entity_dict.get('name', '').lower()
            entity_id = entity_dict.get('id', '')

            if entity_name and entity_name in chunk_text_lower:
                entities_in_chunk.append(entity_id)
                print(f"🔗 Found entity '{entity_dict.get('name', '')}' in chunk {chunk_index}")

        if entities_in_chunk:
            chunk_entity_mapping[str(chunk_index)] = entities_in_chunk
            print(f"[INFO] Chunk {chunk_index} contains {len(entities_in_chunk)} entities")

    print(f"[OK] Found entities in {len(chunk_entity_mapping)} out of {len(chunks)} chunks")
    return chunk_entity_mapping


def auto_detect_resume_files(source_file: str) -> Dict[str, Optional[str]]:
    """
    Auto-detect existing resume files for a source file.

    Args:
        source_file: Path to the source file

    Returns:
        Dictionary with 'process_result' and 'ingestion_data' keys containing file paths or None
    """
    source_path = Path(source_file)
    base_name = source_path.stem
    source_dir = source_path.parent

    # Look for process result file
    process_result_file = source_dir / f"{base_name}.process_result.json"
    process_result_path = process_result_file if process_result_file.exists() else None

    # Look for ingestion data file
    ingestion_data_file = source_dir / f"{base_name}.ingest_data.json"
    ingestion_data_path = ingestion_data_file if ingestion_data_file.exists() else None

    # Also check for ingest_result.json (older format)
    if not ingestion_data_path:
        ingest_result_file = source_dir / f"{base_name}.ingest_result.json"
        if ingest_result_file.exists():
            ingestion_data_path = ingest_result_file

    return {
        'process_result': str(process_result_path) if process_result_path else None,
        'ingestion_data': str(ingestion_data_path) if ingestion_data_path else None
    }


def handle_resume_arguments(args, source_file: str, content_type: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Handle --use-process-result and --use-ingestion-data arguments.
    Also auto-detects existing resume files if no explicit arguments provided.
    Returns True if resume was handled, False if normal processing should continue.
    """
    import asyncio

    # Auto-detect resume files if no explicit arguments provided
    if not hasattr(args, 'use_process_result') or not args.use_process_result:
        if not hasattr(args, 'use_ingestion_data') or not args.use_ingestion_data:
            detected_files = auto_detect_resume_files(source_file)

            # Prefer ingestion data over process result (more complete)
            if detected_files['ingestion_data']:
                print(f"[AUTO-DETECT] Found existing ingestion data: {detected_files['ingestion_data']}")
                args.use_ingestion_data = detected_files['ingestion_data']
            elif detected_files['process_result']:
                print(f"[AUTO-DETECT] Found existing process result: {detected_files['process_result']}")
                args.use_process_result = detected_files['process_result']
    
    # Handle --use-process-result argument
    if hasattr(args, 'use_process_result') and args.use_process_result:
        process_result_file = Path(args.use_process_result)
        if not process_result_file.exists():
            print(f"[FAIL] Error: Process result file not found: {process_result_file}")
            sys.exit(1)
        
        try:
            with open(process_result_file, 'r', encoding='utf-8') as f:
                process_result_data = json.load(f)
            print(f"[OK] Using existing process result from: {process_result_file}")
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
                print(f"\n[SUCCESS] {content_type.title()} processing resumed successfully!")
                sys.exit(0)
            else:
                print(f"\n[ERROR] {content_type.title()} processing resume failed!")
                sys.exit(1)
                
        except json.JSONDecodeError as e:
            print(f"[FAIL] Error: Invalid JSON in process result file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[FAIL] Error reading process result file: {e}")
            sys.exit(1)

    # Handle --use-ingestion-data argument
    if hasattr(args, 'use_ingestion_data') and args.use_ingestion_data:
        ingestion_data_file = Path(args.use_ingestion_data)
        if not ingestion_data_file.exists():
            print(f"[FAIL] Error: Ingestion data file not found: {ingestion_data_file}")
            sys.exit(1)
        
        try:
            with open(ingestion_data_file, 'r', encoding='utf-8') as f:
                ingestion_data = json.load(f)
            print(f"[OK] Using existing ingestion data from: {ingestion_data_file}")
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
                print(f"\n[SUCCESS] {content_type.title()} ingestion resumed successfully!")
                sys.exit(0)
            else:
                print(f"\n[ERROR] {content_type.title()} ingestion resume failed!")
                sys.exit(1)
                
        except json.JSONDecodeError as e:
            print(f"[FAIL] Error: Invalid JSON in ingestion data file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[FAIL] Error reading ingestion data file: {e}")
            sys.exit(1)
    
    # No resume arguments provided, continue with normal processing
    return False
