#!/usr/bin/env python3
"""
MoRAG Image Processing CLI Script

Supports both processing (immediate results) and ingestion (background + storage) modes
with graph entity/relation extraction and dual database storage.

Usage:
    python test-image.py <image_file> [options]

Processing Mode (immediate results):
    python test-image.py my-image.jpg
    python test-image.py screenshot.png
    python test-image.py diagram.gif

Ingestion Mode (background processing + storage):
    python test-image.py my-image.jpg --ingest
    python test-image.py screenshot.png --ingest --qdrant --neo4j
    python test-image.py diagram.png --ingest --metadata '{"type": "screenshot"}'

Options:
    --ingest                    Enable ingestion mode (background processing + storage)
    --qdrant                   Store in Qdrant vector database (ingestion mode only)
    --neo4j                    Store in Neo4j graph database (ingestion mode only)
    --webhook-url URL          Webhook URL for completion notifications (ingestion mode only)
    --metadata JSON            Additional metadata as JSON string (ingestion mode only)
    --help                     Show this help message
"""

import sys
import os
import asyncio
import json
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any
import requests

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

try:
    from morag_image import ImageProcessor
    from morag_image.processor import ImageConfig
    from morag_core.interfaces.processor import ProcessingConfig
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-image")
    sys.exit(1)

# Import common schema and graph extraction
try:
    from common_schema import (
        IntermediateJSON, MarkdownGenerator, Entity, Relation,
        ContentType, ProcessingMode, create_processing_metadata, get_output_paths
    )
    from graph_extraction import extract_and_ingest
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    print("Make sure the common schema and graph extraction modules are available.")
    sys.exit(1)


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


async def test_image_processing(
    image_file: Path,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Test image processing functionality with graph extraction."""
    print_header("MoRAG Image Processing Test")
    
    if not image_file.exists():
        print(f"[FAIL] Error: Image file not found: {image_file}")
        return False
    
    print_result("Input File", str(image_file))
    print_result("File Size", f"{image_file.stat().st_size / 1024:.2f} KB")
    print_result("File Extension", image_file.suffix.lower())
    
    start_time = time.time()
    
    try:
        # Initialize image processor (no API key for basic functionality)
        processor = ImageProcessor()
        print_result("Image Processor", "[OK] Initialized successfully")

        # Create image configuration (disable OCR since tesseract is not installed)
        image_config = ImageConfig(
            generate_caption=False,  # Requires API key
            extract_text=False,  # Requires tesseract
            extract_metadata=True,
            resize_max_dimension=1024
        )
        print_result("Image Config", "[OK] Created successfully")

        print_section("Processing Image File")
        print("[PROCESSING] Starting image processing...")

        # Process the image file
        result = await processor.process_image(image_file, image_config)

        if not result or not hasattr(result, 'processing_time'):
            print("[FAIL] Image processing failed!")
            return False

        print("[OK] Image processing completed successfully!")
        processing_time = time.time() - start_time

        print_section("Processing Results")
        print_result("Status", "[OK] Success")
        print_result("Processing Time", f"{result.processing_time:.2f} seconds")

        # Prepare text content for graph extraction
        text_content = ""
        if result.caption:
            text_content += f"Image Caption: {result.caption}\n"
        if result.extracted_text:
            text_content += f"Extracted Text: {result.extracted_text}\n"
        
        # Add metadata as text
        metadata = result.metadata
        text_content += f"Image Dimensions: {metadata.width}x{metadata.height} pixels\n"
        text_content += f"Image Format: {metadata.format}\n"
        if metadata.camera_make:
            text_content += f"Camera: {metadata.camera_make}"
            if metadata.camera_model:
                text_content += f" {metadata.camera_model}"
            text_content += "\n"
        if metadata.creation_time:
            text_content += f"Created: {metadata.creation_time}\n"

        # Extract entities and relations
        print_section("Graph Extraction")
        print("[PROCESSING] Extracting entities and relations...")
        
        entities, relations = [], []
        if text_content.strip():
            try:
                from graph_extraction import GraphExtractionService
                extraction_service = GraphExtractionService()
                entities, relations = await extraction_service.extract_entities_and_relations(
                    text=text_content,
                    doc_id=f"image_{image_file.stem}",
                    context=f"Image analysis from {image_file.name}"
                )
                print_result("Entities Extracted", f"{len(entities)}")
                print_result("Relations Extracted", f"{len(relations)}")
            except Exception as e:
                print(f"[WARN] Warning: Graph extraction failed: {e}")
        else:
            print_result("Text Content", "No text content available for extraction")

        # Create intermediate JSON
        print_section("Creating Intermediate Files")
        
        # Create processing metadata
        proc_metadata = create_processing_metadata(
            content_type=ContentType.IMAGE,
            source_path=str(image_file),
            processing_time=processing_time,
            mode=ProcessingMode.PROCESSING,
            model_info={
                'generate_caption': False,
                'extract_text': False,
                'extract_metadata': True,
                'resize_max_dimension': 1024
            },
            options={
                'width': metadata.width,
                'height': metadata.height,
                'format': metadata.format,
                'file_size': metadata.file_size
            }
        )
        
        # Create intermediate JSON
        intermediate = IntermediateJSON(
            content_type=ContentType.IMAGE.value,
            source_path=str(image_file),
            title=f"Image Analysis: {image_file.name}",
            text_content=text_content,
            metadata=proc_metadata,
            entities=entities,
            relations=relations,
            image_info={
                'caption': result.caption,
                'extracted_text': result.extracted_text,
                'width': metadata.width,
                'height': metadata.height,
                'format': metadata.format,
                'mode': metadata.mode,
                'file_size': metadata.file_size,
                'has_exif': metadata.has_exif,
                'creation_time': metadata.creation_time,
                'camera_make': metadata.camera_make,
                'camera_model': metadata.camera_model,
                'confidence_scores': result.confidence_scores
            },
            custom_metadata=custom_metadata
        )
        
        # Get output paths
        output_paths = get_output_paths(image_file, ProcessingMode.PROCESSING)
        
        # Save intermediate JSON
        intermediate.to_json(output_paths['intermediate_json'])
        print_result("Intermediate JSON", str(output_paths['intermediate_json']))
        
        # Save processing result
        with open(output_paths['result_json'], 'w', encoding='utf-8') as f:
            json.dump({
                'mode': 'processing',
                'success': True,
                'processing_time': processing_time,
                'text_content_length': len(text_content),
                'entities_count': len(entities),
                'relations_count': len(relations),
                'caption': result.caption,
                'extracted_text': result.extracted_text,
                'image_metadata': {
                    'width': metadata.width,
                    'height': metadata.height,
                    'format': metadata.format,
                    'mode': metadata.mode,
                    'file_size': metadata.file_size,
                    'has_exif': metadata.has_exif,
                    'creation_time': metadata.creation_time,
                    'camera_make': metadata.camera_make,
                    'camera_model': metadata.camera_model
                },
                'confidence_scores': result.confidence_scores,
                'custom_metadata': custom_metadata,
                'file_path': str(image_file)
            }, f, indent=2, ensure_ascii=False)
        
        print_result("Processing Result", str(output_paths['result_json']))

        # Display preview information
        print_section("Image Information")
        if result.caption:
            print_result("Caption", result.caption)
        else:
            print_result("Caption", "Not generated (requires API key)")

        if result.extracted_text:
            print_result("Extracted Text", result.extracted_text)
        else:
            print_result("Extracted Text", "None found (OCR disabled)")

        print_section("Image Metadata")
        print_result("Width", f"{metadata.width}px")
        print_result("Height", f"{metadata.height}px")
        print_result("Format", metadata.format)
        print_result("Mode", metadata.mode)
        print_result("File Size", f"{metadata.file_size / 1024:.2f} KB")
        print_result("Has EXIF", "[OK] Yes" if metadata.has_exif else "[FAIL] No")

        if metadata.camera_make:
            print_result("Camera Make", metadata.camera_make)
        if metadata.camera_model:
            print_result("Camera Model", metadata.camera_model)
        if metadata.creation_time:
            print_result("Creation Time", metadata.creation_time)

        if entities:
            print_section("Entities Preview (first 5)")
            for entity in entities[:5]:
                print_result(f"Entity", f"{entity.name} ({entity.type}) - confidence: {entity.confidence:.2f}")

        if relations:
            print_section("Relations Preview (first 5)")
            entity_map = {e.id: e.name for e in entities}
            for relation in relations[:5]:
                source_name = entity_map.get(relation.source_entity_id, "Unknown")
                target_name = entity_map.get(relation.target_entity_id, "Unknown")
                print_result(f"Relation", f"{source_name} --[{relation.type}]--> {target_name} (confidence: {relation.confidence:.2f})")

        if result.confidence_scores:
            print_section("Confidence Scores")
            for key, score in result.confidence_scores.items():
                print_result(key, f"{score:.3f}")

        return True

    except Exception as e:
        print(f"[FAIL] Error during image processing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_image_ingestion(
    image_file: Path,
    webhook_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    use_qdrant: bool = True,
    use_neo4j: bool = False,
    qdrant_collection_name: Optional[str] = None,
    neo4j_database_name: Optional[str] = None
) -> bool:
    """Test image ingestion functionality with graph extraction and dual database storage."""
    print_header("MoRAG Image Ingestion Test")

    if not image_file.exists():
        print(f"[FAIL] Error: Image file not found: {image_file}")
        return False

    print_result("Input File", str(image_file))
    print_result("File Size", f"{image_file.stat().st_size / 1024:.2f} KB")
    print_result("File Extension", image_file.suffix.lower())
    print_result("Webhook URL", webhook_url or "Not provided")
    print_result("Custom Metadata", json.dumps(metadata, indent=2) if metadata else "None")
    print_result("Qdrant Storage", "[OK] Enabled" if use_qdrant else "[FAIL] Disabled")
    print_result("Neo4j Storage", "[OK] Enabled" if use_neo4j else "[FAIL] Disabled")

    start_time = time.time()

    try:
        print("[PROCESSING] Starting image processing and ingestion...")

        # Initialize image processor
        processor = ImageProcessor()
        
        # Create image configuration
        image_config = ImageConfig(
            generate_caption=False,  # Requires API key
            extract_text=False,  # Requires tesseract
            extract_metadata=True,
            resize_max_dimension=1024
        )

        # Process the image file
        result = await processor.process_image(image_file, image_config)

        if not result or not hasattr(result, 'processing_time'):
            print("[FAIL] Image processing failed!")
            return False

        print("[OK] Image processing completed successfully!")
        processing_time = time.time() - start_time
        print_result("Processing Time", f"{result.processing_time:.2f} seconds")

        # Prepare text content for graph extraction
        text_content = ""
        if result.caption:
            text_content += f"Image Caption: {result.caption}\n"
        if result.extracted_text:
            text_content += f"Extracted Text: {result.extracted_text}\n"
        
        # Add metadata as text
        img_metadata = result.metadata
        text_content += f"Image Dimensions: {img_metadata.width}x{img_metadata.height} pixels\n"
        text_content += f"Image Format: {img_metadata.format}\n"
        if img_metadata.camera_make:
            text_content += f"Camera: {img_metadata.camera_make}"
            if img_metadata.camera_model:
                text_content += f" {img_metadata.camera_model}"
            text_content += "\n"
        if img_metadata.creation_time:
            text_content += f"Created: {img_metadata.creation_time}\n"

        # Extract entities and relations
        print_section("Graph Extraction")
        print("[PROCESSING] Extracting entities and relations...")
        
        entities, relations = [], []
        if text_content.strip():
            try:
                from graph_extraction import GraphExtractionService
                extraction_service = GraphExtractionService()
                entities, relations = await extraction_service.extract_entities_and_relations(
                    text=text_content,
                    doc_id=f"image_{image_file.stem}",
                    context=f"Image analysis from {image_file.name}"
                )
                print_result("Entities Extracted", f"{len(entities)}")
                print_result("Relations Extracted", f"{len(relations)}")
            except Exception as e:
                print(f"[WARN] Warning: Graph extraction failed: {e}")
        else:
            print_result("Text Content", "No text content available for extraction")

        # Create intermediate files
        print_section("Creating Intermediate Files")
        
        # Create processing metadata
        proc_metadata = create_processing_metadata(
            content_type=ContentType.IMAGE,
            source_path=str(image_file),
            processing_time=processing_time,
            mode=ProcessingMode.INGESTION,
            model_info={
                'generate_caption': False,
                'extract_text': False,
                'extract_metadata': True,
                'resize_max_dimension': 1024
            },
            options={
                'width': img_metadata.width,
                'height': img_metadata.height,
                'format': img_metadata.format,
                'file_size': img_metadata.file_size,
                'use_qdrant': use_qdrant,
                'use_neo4j': use_neo4j
            }
        )
        
        # Create intermediate JSON
        intermediate = IntermediateJSON(
            content_type=ContentType.IMAGE.value,
            source_path=str(image_file),
            title=f"Image Analysis: {image_file.name}",
            text_content=text_content,
            metadata=proc_metadata,
            entities=entities,
            relations=relations,
            image_info={
                'caption': result.caption,
                'extracted_text': result.extracted_text,
                'width': img_metadata.width,
                'height': img_metadata.height,
                'format': img_metadata.format,
                'mode': img_metadata.mode,
                'file_size': img_metadata.file_size,
                'has_exif': img_metadata.has_exif,
                'creation_time': img_metadata.creation_time,
                'camera_make': img_metadata.camera_make,
                'camera_model': img_metadata.camera_model,
                'confidence_scores': result.confidence_scores
            },
            custom_metadata=metadata
        )
        
        # Get output paths
        output_paths = get_output_paths(image_file, ProcessingMode.INGESTION)
        
        # Save intermediate JSON
        intermediate.to_json(output_paths['intermediate_json'])
        print_result("Intermediate JSON", str(output_paths['intermediate_json']))
        
        # Generate and save intermediate markdown
        MarkdownGenerator.save_markdown(intermediate, output_paths['intermediate_md'])
        print_result("Intermediate Markdown", str(output_paths['intermediate_md']))

        # Database ingestion
        print_section("Database Ingestion")
        
        ingestion_results = {'qdrant': None, 'neo4j': None}
        
        if (use_qdrant or use_neo4j) and text_content.strip():
            print("[PROCESSING] Starting database ingestion...")
            
            # Prepare metadata for ingestion
            ingestion_metadata = {
                "source_type": "image",
                "source_path": str(image_file),
                "processing_time": result.processing_time,
                "text_content_length": len(text_content),
                "entities_count": len(entities),
                "relations_count": len(relations),
                "width": img_metadata.width,
                "height": img_metadata.height,
                "format": img_metadata.format,
                "file_size": img_metadata.file_size,
                "has_caption": bool(result.caption),
                "has_extracted_text": bool(result.extracted_text),
                **(metadata or {})
            }
            
            try:
                # Prepare database configurations
                qdrant_config = None
                neo4j_config = None

                if use_qdrant:
                    collection_name = qdrant_collection_name or os.getenv('MORAG_QDRANT_COLLECTION', 'morag_images')
                    qdrant_config = {
                        'host': 'localhost',
                        'port': 6333,
                        'collection_name': collection_name
                    }

                if use_neo4j:
                    db_name = neo4j_database_name or os.getenv('NEO4J_DATABASE', 'neo4j')
                    neo4j_config = {
                        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                        'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
                        'password': os.getenv('NEO4J_PASSWORD', 'password'),
                        'database': db_name
                    }

                graph_results = await extract_and_ingest(
                    text_content=text_content,
                    doc_id=f"image_{image_file.stem}",
                    context=f"Image analysis from {image_file.name}",
                    use_qdrant=use_qdrant,
                    use_neo4j=use_neo4j,
                    qdrant_config=qdrant_config,
                    neo4j_config=neo4j_config,
                    metadata=ingestion_metadata
                )
                
                ingestion_results = graph_results.get('ingestion', {})
                
                if use_qdrant and ingestion_results.get('qdrant'):
                    if ingestion_results['qdrant'].get('success'):
                        print_result("Qdrant Ingestion", "[OK] Success")
                        print_result("Qdrant Chunks", str(ingestion_results['qdrant'].get('chunks_count', 0)))
                    else:
                        print_result("Qdrant Ingestion", f"[FAIL] Failed: {ingestion_results['qdrant'].get('error', 'Unknown error')}")
                
                if use_neo4j and ingestion_results.get('neo4j'):
                    if ingestion_results['neo4j'].get('success'):
                        print_result("Neo4j Ingestion", "[OK] Success")
                        print_result("Neo4j Entities", str(ingestion_results['neo4j'].get('entities_stored', 0)))
                        print_result("Neo4j Relations", str(ingestion_results['neo4j'].get('relations_stored', 0)))
                    else:
                        print_result("Neo4j Ingestion", f"[FAIL] Failed: {ingestion_results['neo4j'].get('error', 'Unknown error')}")
                        
            except Exception as e:
                print(f"[WARN] Warning: Database ingestion failed: {e}")
                ingestion_results = {'error': str(e)}
        elif not text_content.strip():
            print_result("Database Ingestion", "Skipped - no text content available")
        
        print("[OK] Image ingestion completed successfully!")

        print_section("Ingestion Results")
        print_result("Status", "[OK] Success")
        print_result("Total Processing Time", f"{processing_time:.2f} seconds")
        print_result("Text Content Length", str(len(text_content)))
        print_result("Entities Extracted", str(len(entities)))
        print_result("Relations Extracted", str(len(relations)))

        # Save ingestion result
        with open(output_paths['result_json'], 'w', encoding='utf-8') as f:
            json.dump({
                'mode': 'ingestion',
                'success': True,
                'processing_time': processing_time,
                'text_content_length': len(text_content),
                'entities_count': len(entities),
                'relations_count': len(relations),
                'use_qdrant': use_qdrant,
                'use_neo4j': use_neo4j,
                'webhook_url': webhook_url,
                'ingestion_results': ingestion_results,
                'image_info': {
                    'caption': result.caption,
                    'extracted_text': result.extracted_text,
                    'width': img_metadata.width,
                    'height': img_metadata.height,
                    'format': img_metadata.format,
                    'file_size': img_metadata.file_size,
                    'confidence_scores': result.confidence_scores
                },
                'metadata': ingestion_metadata,
                'file_path': str(image_file)
            }, f, indent=2, ensure_ascii=False)

        print_section("Output Files")
        print_result("Intermediate JSON", str(output_paths['intermediate_json']))
        print_result("Intermediate Markdown", str(output_paths['intermediate_md']))
        print_result("Ingestion Result", str(output_paths['result_json']))

        return True

    except Exception as e:
        print(f"[FAIL] Error during image ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MoRAG Image Processing Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Processing Mode (immediate results):
    python test-image.py my-image.jpg
    python test-image.py screenshot.png
    python test-image.py diagram.gif

  Ingestion Mode (background processing + storage):
    python test-image.py my-image.jpg --ingest
    python test-image.py screenshot.png --ingest --metadata '{"type": "screenshot"}'
    python test-image.py diagram.png --ingest --webhook-url https://my-app.com/webhook

  Resume from Process Result:
    python test-image.py my-image.jpg --use-process-result my-image.process_result.json

  Resume from Ingestion Data:
    python test-image.py my-image.jpg --use-ingestion-data my-image.ingest_data.json
        """
    )

    parser.add_argument('image_file', help='Path to image file')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (background processing + storage)')
    parser.add_argument('--qdrant', action='store_true',
                       help='Store in Qdrant vector database (ingestion mode only)')
    parser.add_argument('--qdrant-collection', help='Qdrant collection name (default: from environment or morag_images)')
    parser.add_argument('--neo4j', action='store_true',
                       help='Store in Neo4j graph database (ingestion mode only)')
    parser.add_argument('--neo4j-database', help='Neo4j database name (default: from environment or neo4j)')
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')
    parser.add_argument('--language', required=True, help='Language code for processing (e.g., en, de, fr) - MANDATORY for consistent processing')
    parser.add_argument('--use-process-result', help='Skip processing and use existing process result file (e.g., my-file.process_result.json)')
    parser.add_argument('--use-ingestion-data', help='Skip processing and ingestion calculation, use existing ingestion data file (e.g., my-file.ingest_data.json)')

    args = parser.parse_args()

    image_file = Path(args.image_file)

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"[FAIL] Error: Invalid JSON in metadata: {e}")
            sys.exit(1)

    # Handle resume arguments
    from resume_utils import handle_resume_arguments
    handle_resume_arguments(args, str(image_file), 'image', metadata)

    try:
        if args.ingest:
            # Default to Qdrant if no database flags specified
            use_qdrant = args.qdrant or (not args.qdrant and not args.neo4j)
            use_neo4j = args.neo4j
            
            # Ingestion mode
            success = asyncio.run(test_image_ingestion(
                image_file,
                webhook_url=args.webhook_url,
                metadata=metadata,
                use_qdrant=use_qdrant,
                use_neo4j=use_neo4j,
                qdrant_collection_name=args.qdrant_collection,
                neo4j_database_name=args.neo4j_database
            ))
            if success:
                print("\n[SUCCESS] Image ingestion test completed successfully!")
                print("💡 Check the output files for detailed results and debugging information.")
                sys.exit(0)
            else:
                print("\n[ERROR] Image ingestion test failed!")
                sys.exit(1)
        else:
            # Processing mode
            success = asyncio.run(test_image_processing(
                image_file,
                custom_metadata=metadata
            ))
            if success:
                print("\n[SUCCESS] Image processing test completed successfully!")
                print("💡 Check the output files for detailed results and debugging information.")
                sys.exit(0)
            else:
                print("\n[ERROR] Image processing test failed!")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n[STOP]  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
