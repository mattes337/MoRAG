"""Example of using the enhanced graph builder with OpenIE integration."""

import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate enhanced graph building with OpenIE."""
    try:
        # Import components
        from morag_graph.storage import Neo4jStorage, Neo4jConfig
        from morag_graph.builders import EnhancedGraphBuilder, EnhancedGraphBuildResult
        from morag_core.config import LLMConfig
        
        logger.info("Starting OpenIE enhanced extraction example")
        
        # Sample document content
        sample_content = """
        Dr. Sarah Johnson works as a senior researcher at Stanford University. 
        She specializes in artificial intelligence and machine learning.
        Stanford University was founded in 1885 by Leland Stanford.
        The university is located in Stanford, California.
        Dr. Johnson has published over 50 research papers on neural networks.
        She collaborates with researchers from MIT and Carnegie Mellon University.
        Her latest research focuses on transformer architectures for natural language processing.
        The research is funded by the National Science Foundation.
        """
        
        # Configure Neo4j storage
        neo4j_config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",  # Change this to your Neo4j password
            database="neo4j"
        )
        
        # Configure LLM for traditional extraction
        llm_config = LLMConfig(
            provider="gemini",
            model="gemini-1.5-flash",
            api_key="your-api-key-here"  # Set your API key
        )
        
        # Configure OpenIE
        openie_config = {
            "min_confidence": 0.6,
            "enable_entity_linking": True,
            "enable_predicate_normalization": True,
            "batch_size": 50
        }
        
        # Initialize storage
        storage = Neo4jStorage(neo4j_config)
        await storage.connect()
        
        # Initialize enhanced graph builder
        builder = EnhancedGraphBuilder(
            storage=storage,
            llm_config=llm_config,
            enable_openie=True,
            openie_config=openie_config
        )
        
        logger.info("Enhanced graph builder initialized")
        
        # Process the document
        logger.info("Processing document with enhanced extraction...")
        result = await builder.process_document(
            content=sample_content,
            document_id="example_document_001",
            metadata={
                "title": "AI Research Example",
                "author": "Example Author",
                "source": "example"
            }
        )
        
        # Display results
        logger.info("Processing completed!")
        logger.info(f"Document ID: {result.document_id}")
        logger.info(f"Processing time: {result.processing_time:.2f} seconds")
        logger.info(f"Entities created: {result.entities_created}")
        logger.info(f"Total relations created: {result.relations_created}")
        
        if result.openie_enabled:
            logger.info("OpenIE Results:")
            logger.info(f"  - OpenIE relations: {result.openie_relations_created}")
            logger.info(f"  - Triplets processed: {result.openie_triplets_processed}")
            logger.info(f"  - Entity matches: {result.openie_entity_matches}")
            logger.info(f"  - Normalized predicates: {result.openie_normalized_predicates}")
            
            if result.openie_metadata:
                logger.info("  - OpenIE metadata:")
                for key, value in result.openie_metadata.items():
                    logger.info(f"    - {key}: {value}")
        else:
            logger.info("OpenIE was not enabled or available")
        
        if result.errors:
            logger.warning("Errors encountered:")
            for error in result.errors:
                logger.warning(f"  - {error}")
        
        # Demonstrate querying the results
        logger.info("\nQuerying stored entities...")
        entities = await storage.get_all_entities()
        logger.info(f"Total entities in storage: {len(entities)}")
        
        for i, entity in enumerate(entities[:5]):  # Show first 5
            logger.info(f"Entity {i+1}: {entity.name} ({entity.entity_type}) - confidence: {entity.confidence:.2f}")
        
        logger.info("\nQuerying stored relations...")
        relations = await storage.get_all_relations()
        logger.info(f"Total relations in storage: {len(relations)}")
        
        for i, relation in enumerate(relations[:5]):  # Show first 5
            logger.info(
                f"Relation {i+1}: {relation.subject} --[{relation.predicate}]--> {relation.object} "
                f"(confidence: {relation.confidence:.2f})"
            )
            if "extraction_method" in relation.metadata:
                logger.info(f"  - Extraction method: {relation.metadata['extraction_method']}")
        
        # Get storage statistics
        logger.info("\nStorage statistics:")
        stats = await storage.get_statistics()
        for key, value in stats.items():
            logger.info(f"  - {key}: {value}")
        
        await storage.disconnect()
        logger.info("✅ Enhanced extraction example completed successfully")
        
    except ImportError as e:
        logger.error(f"Required components not available: {e}")
        logger.info("Make sure OpenIE dependencies are installed and configured")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        logger.exception("Full error details:")


async def chunk_processing_example():
    """Demonstrate chunk-based processing with OpenIE."""
    try:
        from morag_graph.models import DocumentChunk
        from morag_graph.storage import Neo4jStorage, Neo4jConfig
        from morag_graph.builders import EnhancedGraphBuilder
        from morag_core.config import LLMConfig
        
        logger.info("Starting chunk processing example")
        
        # Create sample chunks
        chunks = [
            DocumentChunk(
                id="chunk_1",
                text="Dr. Sarah Johnson works at Stanford University. She is a researcher in AI.",
                chunk_type="paragraph",
                metadata={"section": "introduction"}
            ),
            DocumentChunk(
                id="chunk_2", 
                text="Stanford University was founded in 1885. It is located in California.",
                chunk_type="paragraph",
                metadata={"section": "background"}
            ),
            DocumentChunk(
                id="chunk_3",
                text="Dr. Johnson collaborates with MIT researchers. Her work focuses on neural networks.",
                chunk_type="paragraph",
                metadata={"section": "research"}
            )
        ]
        
        # Configure components (same as above)
        neo4j_config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j", 
            password="password",
            database="neo4j"
        )
        
        llm_config = LLMConfig(
            provider="gemini",
            model="gemini-1.5-flash",
            api_key="your-api-key-here"
        )
        
        storage = Neo4jStorage(neo4j_config)
        await storage.connect()
        
        builder = EnhancedGraphBuilder(
            storage=storage,
            llm_config=llm_config,
            enable_openie=True
        )
        
        # Process chunks
        logger.info(f"Processing {len(chunks)} chunks...")
        result = await builder.process_document_chunks(
            chunks=chunks,
            document_id="chunked_document_001",
            metadata={"processing_type": "chunked"}
        )
        
        logger.info("Chunk processing completed!")
        logger.info(f"Chunks processed: {result.chunks_processed}")
        logger.info(f"Entities created: {result.entities_created}")
        logger.info(f"Relations created: {result.relations_created}")
        logger.info(f"OpenIE relations: {result.openie_relations_created}")
        
        await storage.disconnect()
        logger.info("✅ Chunk processing example completed")
        
    except Exception as e:
        logger.error(f"Chunk processing example failed: {e}")


if __name__ == "__main__":
    print("OpenIE Enhanced Extraction Example")
    print("=" * 50)
    
    # Run main example
    asyncio.run(main())
    
    print("\n" + "=" * 50)
    print("Chunk Processing Example")
    print("=" * 50)
    
    # Run chunk processing example
    asyncio.run(chunk_processing_example())
