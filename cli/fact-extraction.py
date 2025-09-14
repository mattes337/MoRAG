#!/usr/bin/env python3
"""
Fact Extraction CLI Script

This script extracts structured facts from documents using the MoRAG fact extraction system.
It supports various input formats and can store results in both Neo4j and Qdrant databases.

Usage:
    python fact-extraction.py input.md --neo4j --qdrant
    python fact-extraction.py input.pdf --domain research --min-confidence 0.8
    python fact-extraction.py input.txt --output facts.json --verbose
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-graph" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-services" / "src"))

try:
    from morag_graph.extraction.fact_extractor import FactExtractor
    from morag_graph.extraction.fact_graph_builder import FactGraphBuilder
    from morag_graph.models.fact import Fact, FactRelation
    from morag_graph.models.document_chunk import DocumentChunk
    from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
    from morag_graph.storage.qdrant_storage import QdrantStorage, QdrantConfig
    # Import embedding service separately to avoid complex dependency chains
    try:
        from morag_embedding import GeminiEmbeddingService
    except ImportError:
        print("⚠️  Warning: GeminiEmbeddingService not available. Vector storage will be disabled.")
        GeminiEmbeddingService = None
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the required packages:")
    print("  pip install -e packages/morag-graph")
    print("  pip install -e packages/morag-embedding")
    sys.exit(1)

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def extract_facts_from_file(
    input_file: Path,
    domain: str = "general",
    min_confidence: float = 0.7,
    max_facts: int = 20,
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Extract facts from a file.
    
    Args:
        input_file: Path to input file
        domain: Domain context for extraction
        min_confidence: Minimum confidence threshold
        max_facts: Maximum facts per chunk
        model: LLM model to use
        api_key: API key for LLM service
        verbose: Show detailed output
        
    Returns:
        Dictionary containing extraction results
    """
    # Read the input file
    try:
        content = input_file.read_text(encoding='utf-8')
    except Exception as e:
        logger.error("Failed to read input file", file=str(input_file), error=str(e))
        return {}
    
    if verbose:
        print(f"📄 Processing file: {input_file.name}")
        print(f"📝 Content length: {len(content)} characters")
        print(f"🎯 Domain: {domain}")
        print(f"🎚️ Min confidence: {min_confidence}")
        print(f"📊 Max facts: {max_facts}")
    
    # Initialize fact extractor
    fact_extractor = FactExtractor(
        model_id=model,
        api_key=api_key or os.getenv('GEMINI_API_KEY'),
        min_confidence=min_confidence,
        max_facts_per_chunk=max_facts,
        domain=domain
    )
    
    # Create document chunk
    document_id = f"doc_{input_file.stem}"
    chunk_id = f"{document_id}_chunk_0"
    
    # Extract facts
    try:
        facts = await fact_extractor.extract_facts(
            chunk_text=content,
            chunk_id=chunk_id,
            document_id=document_id,
            context={
                'domain': domain,
                'language': 'en',
                'source_file_path': str(input_file),
                'source_file_name': input_file.name
            }
        )
        
        if verbose:
            print(f"✅ Extracted {len(facts)} facts")
            for fact in facts:
                entities = fact.structured_metadata.primary_entities if fact.structured_metadata and fact.structured_metadata.primary_entities else ["N/A"]
                entities_str = ", ".join(entities[:2])  # Show first 2 entities
                print(f"  • {fact.fact_text[:50]}... ({fact.fact_type}) - entities: {entities_str} - confidence: {fact.extraction_confidence:.2f}")
        
        # Extract relationships if we have multiple facts
        relationships = []
        if len(facts) > 1:
            try:
                fact_graph_builder = FactGraphBuilder(
                    model_id=model,
                    api_key=api_key or os.getenv('GEMINI_API_KEY')
                )
                
                fact_graph = await fact_graph_builder.build_fact_graph(facts)
                if hasattr(fact_graph, 'relationships'):
                    relationships = fact_graph.relationships
                
                if verbose:
                    print(f"✅ Extracted {len(relationships)} relationships")
                    for rel in relationships:
                        print(f"  • {rel.source_fact_id} --[{rel.relationship_type}]--> {rel.target_fact_id}")
                        
            except Exception as e:
                logger.warning("Failed to extract relationships", error=str(e))
        
        # Convert to serializable format
        facts_data = [fact.to_dict() for fact in facts]
        relationships_data = [
            {
                'id': rel.id,
                'source_fact_id': rel.source_fact_id,
                'target_fact_id': rel.target_fact_id,
                'relationship_type': rel.relationship_type,
                'confidence': rel.confidence,
                'description': rel.description,
                'created_at': rel.created_at.isoformat()
            }
            for rel in relationships
        ]
        
        return {
            'source_file': str(input_file),
            'document_id': document_id,
            'model': model,
            'domain': domain,
            'extraction_timestamp': datetime.utcnow().isoformat(),
            'facts': facts_data,
            'relationships': relationships_data,
            'statistics': {
                'facts_extracted': len(facts),
                'relationships_created': len(relationships),
                'chunks_processed': 1,
                'fact_types': {}
            }
        }
        
    except Exception as e:
        logger.error("Fact extraction failed", error=str(e))
        if verbose:
            import traceback
            traceback.print_exc()
        return {}


async def store_in_neo4j(
    facts: List[Fact],
    relationships: List[FactRelation],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    verbose: bool = False
) -> bool:
    """Store facts and relationships in Neo4j.
    
    Args:
        facts: List of facts to store
        relationships: List of relationships to store
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        verbose: Show detailed output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config = Neo4jConfig(
            uri=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password
        )
        
        storage = Neo4jStorage(config)
        await storage.connect()
        
        if verbose:
            print("✅ Connected to Neo4j")
        
        # Store facts and relationships using fact extraction service
        fact_service = FactExtractionService(
            neo4j_storage=storage,
            enable_vector_storage=False
        )
        
        # Create document chunks for the facts
        chunks = []
        for fact in facts:
            chunk = DocumentChunk(
                id=fact.source_chunk_id,
                document_id=fact.source_document_id,
                chunk_index=0,
                text="",  # Not needed for storage
                metadata={}
            )
            chunks.append(chunk)
        
        # Store using the service
        result = await fact_service.extract_and_store_facts(chunks)
        
        if verbose:
            stats = result.get('statistics', {})
            print(f"✅ Stored {stats.get('facts_stored', 0)} facts")
            print(f"✅ Stored {stats.get('relationships_created', 0)} relationships")
        
        await storage.disconnect()
        return True
        
    except Exception as e:
        logger.error("Failed to store in Neo4j", error=str(e))
        if verbose:
            import traceback
            traceback.print_exc()
        return False


async def store_in_qdrant(
    facts: List[Fact],
    qdrant_url: str,
    qdrant_api_key: Optional[str],
    collection_name: str,
    verbose: bool = False
) -> bool:
    """Store facts in Qdrant vector database.
    
    Args:
        facts: List of facts to store
        qdrant_url: Qdrant connection URL
        qdrant_api_key: Qdrant API key (optional)
        collection_name: Collection name
        verbose: Show detailed output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config = QdrantConfig(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name
        )
        
        storage = QdrantStorage(config)
        await storage.connect()
        
        if verbose:
            print("✅ Connected to Qdrant")
        
        # Initialize embedding service
        embedding_service = GeminiEmbeddingService(
            api_key=os.getenv('GEMINI_API_KEY')
        )
        
        # Initialize fact vector operations
        fact_vector_ops = FactVectorOperations(
            client=storage.client,
            collection_name=f"{collection_name}_facts",
            embedding_service=embedding_service
        )
        
        # Store facts as vectors
        point_ids = await fact_vector_ops.store_facts_batch(facts)
        
        if verbose:
            print(f"✅ Stored {len(point_ids)} fact vectors")
        
        await storage.disconnect()
        return True
        
    except Exception as e:
        logger.error("Failed to store in Qdrant", error=str(e))
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def save_results(results: Dict[str, Any], output_file: Path, verbose: bool = False) -> bool:
    """Save extraction results to JSON file.
    
    Args:
        results: Extraction results
        output_file: Output file path
        verbose: Show detailed output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"💾 Results saved to: {output_file}")
            stats = results.get('statistics', {})
            print(f"📊 Statistics:")
            print(f"   • Facts: {stats.get('facts_extracted', 0)}")
            print(f"   • Relationships: {stats.get('relationships_created', 0)}")
        
        return True
        
    except Exception as e:
        logger.error("Failed to save results", file=str(output_file), error=str(e))
        return False


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract structured facts from documents using MoRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python fact-extraction.py document.md                           # Extract facts with default settings
  python fact-extraction.py document.pdf --domain research       # Extract facts for research domain
  python fact-extraction.py document.txt --neo4j                 # Store in Neo4j database
  python fact-extraction.py document.md --qdrant                 # Store in Qdrant vector database
  python fact-extraction.py document.md --neo4j --qdrant         # Store in both databases
  python fact-extraction.py document.md --min-confidence 0.8     # Higher confidence threshold
  python fact-extraction.py document.md --max-facts 30           # More facts per chunk
  python fact-extraction.py document.md --output results.json    # Custom output file
  python fact-extraction.py document.md --verbose                # Show detailed output
"""
    )

    # Required arguments
    parser.add_argument(
        "input_file",
        help="Path to input file (supports .txt, .md, .pdf, etc.)"
    )

    # Extraction options
    parser.add_argument(
        "--domain",
        type=str,
        default="general",
        help="Domain context for extraction (default: general)"
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7)"
    )

    parser.add_argument(
        "--max-facts",
        type=int,
        default=20,
        help="Maximum facts per chunk (default: 20)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="LLM model to use (default: gemini-2.0-flash)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for LLM service (defaults to GEMINI_API_KEY env var)"
    )

    # Storage options
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Store results in Neo4j database"
    )

    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )

    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )

    parser.add_argument(
        "--neo4j-password",
        type=str,
        help="Neo4j password (defaults to NEO4J_PASSWORD env var)"
    )

    parser.add_argument(
        "--qdrant",
        action="store_true",
        help="Store results in Qdrant vector database"
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant connection URL (default: http://localhost:6333)"
    )

    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        help="Qdrant API key (optional)"
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default="morag_facts",
        help="Qdrant collection name (default: morag_facts)"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: input_file.facts.json)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    print("🧠 MoRAG Fact Extraction")
    print("=" * 60)

    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return 1

    if not input_file.is_file():
        print(f"❌ Input path is not a file: {input_file}")
        return 1

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.with_suffix('.facts.json')

    # Validate arguments
    if args.min_confidence < 0.0 or args.min_confidence > 1.0:
        print("❌ Minimum confidence must be between 0.0 and 1.0")
        return 1

    if args.max_facts < 1:
        print("❌ Maximum facts must be at least 1")
        return 1

    # Check API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No API key provided. Set GEMINI_API_KEY environment variable or use --api-key")
        return 1

    # Check database credentials if needed
    if args.neo4j:
        neo4j_password = args.neo4j_password or os.getenv('NEO4J_PASSWORD')
        if not neo4j_password:
            print("❌ No Neo4j password provided. Set NEO4J_PASSWORD environment variable or use --neo4j-password")
            return 1

    try:
        print(f"📄 Processing: {input_file.name}")
        print(f"🤖 Using model: {args.model}")
        print(f"🎯 Domain: {args.domain}")
        print(f"🎚️ Min confidence: {args.min_confidence}")
        print(f"📊 Max facts: {args.max_facts}")
        print(f"💾 Output file: {output_file.name}")
        print("=" * 60)

        # Extract facts
        results = await extract_facts_from_file(
            input_file=input_file,
            domain=args.domain,
            min_confidence=args.min_confidence,
            max_facts=args.max_facts,
            model=args.model,
            api_key=api_key,
            verbose=args.verbose
        )

        if not results:
            print("❌ Fact extraction failed")
            return 1

        # Save results to file
        if not save_results(results, output_file, args.verbose):
            print("❌ Failed to save results")
            return 1

        # Convert results to objects for database storage
        facts = [Fact(**fact_data) for fact_data in results['facts']]
        relationships = [FactRelation(**rel_data) for rel_data in results['relationships']]

        # Store in databases if requested
        if args.neo4j:
            print("\n🔗 Storing in Neo4j...")
            success = await store_in_neo4j(
                facts=facts,
                relationships=relationships,
                neo4j_uri=args.neo4j_uri,
                neo4j_user=args.neo4j_user,
                neo4j_password=neo4j_password,
                verbose=args.verbose
            )
            if not success:
                print("❌ Failed to store in Neo4j")
                return 1

        if args.qdrant:
            print("\n🔍 Storing in Qdrant...")
            success = await store_in_qdrant(
                facts=facts,
                qdrant_url=args.qdrant_url,
                qdrant_api_key=args.qdrant_api_key,
                collection_name=args.collection_name,
                verbose=args.verbose
            )
            if not success:
                print("❌ Failed to store in Qdrant")
                return 1

        print("=" * 60)
        print("✅ Fact extraction completed successfully!")
        print(f"📁 Results saved to: {output_file}")

        stats = results.get('statistics', {})
        print(f"📊 Extracted {stats.get('facts_extracted', 0)} facts and {stats.get('relationships_created', 0)} relationships")

        return 0

    except Exception as e:
        logger.error("Fact extraction failed", error=str(e))
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
