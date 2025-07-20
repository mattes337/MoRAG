#!/usr/bin/env python3
"""
Example demonstrating hybrid episode mapping with contextual processing.

This example shows how to use the enhanced DocumentEpisodeMapper to create
both granular chunk-level episodes and contextual processing that preserves
document-level understanding while enabling fine-grained retrieval.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from morag_core.models import Document, DocumentChunk, DocumentMetadata
from morag_graph.graphiti import (
    GraphitiConfig, 
    DocumentEpisodeMapper,
    EpisodeStrategy,
    ContextLevel,
    create_hybrid_episode_mapper,
    create_contextual_chunk_mapper
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_document() -> Document:
    """Create a sample document with multiple chunks for demonstration."""
    
    # Create document metadata
    metadata = DocumentMetadata(
        title="AI Research Paper: Transformer Architecture",
        source_name="transformer_paper.pdf",
        source_type="academic_paper",
        mime_type="application/pdf"
    )
    
    # Create document
    document = Document(metadata=metadata)
    
    # Add sample chunks representing different sections
    chunks_data = [
        {
            "content": """
            Abstract: This paper introduces the Transformer architecture, a novel neural network 
            architecture based solely on attention mechanisms. The Transformer achieves superior 
            performance on machine translation tasks while being more parallelizable than 
            recurrent neural networks.
            """,
            "section": "Abstract"
        },
        {
            "content": """
            Introduction: Recurrent neural networks, long short-term memory networks, and gated 
            recurrent neural networks have been firmly established as state of the art approaches 
            in sequence modeling and transduction problems. However, these models are inherently 
            sequential, which precludes parallelization within training examples.
            """,
            "section": "Introduction"
        },
        {
            "content": """
            Model Architecture: The Transformer follows the overall architecture of encoder-decoder 
            models using stacked self-attention and point-wise, fully connected layers for both 
            the encoder and decoder. The encoder is composed of a stack of N=6 identical layers.
            """,
            "section": "Model Architecture"
        },
        {
            "content": """
            Attention Mechanism: An attention function can be described as mapping a query and a 
            set of key-value pairs to an output. The output is computed as a weighted sum of the 
            values, where the weight assigned to each value is computed by a compatibility function.
            """,
            "section": "Attention Mechanism"
        },
        {
            "content": """
            Experimental Results: We evaluate our model on two machine translation tasks: 
            WMT 2014 English-to-German and WMT 2014 English-to-French. Our model achieves 
            28.4 BLEU on the English-to-German translation task, establishing a new state-of-the-art.
            """,
            "section": "Results"
        }
    ]
    
    # Add chunks to document
    for i, chunk_data in enumerate(chunks_data):
        chunk = DocumentChunk(
            document_id=document.id,
            content=chunk_data["content"].strip(),
            chunk_index=i,
            section=chunk_data["section"]
        )
        document.chunks.append(chunk)
    
    # Set raw text
    document.raw_text = "\n\n".join([chunk.content for chunk in document.chunks])
    
    return document


async def demonstrate_hybrid_mapping():
    """Demonstrate hybrid episode mapping strategy."""
    
    print("🔄 Creating sample document...")
    document = create_sample_document()
    print(f"✅ Created document with {len(document.chunks)} chunks")
    
    # Configure Graphiti (you'll need to set your actual credentials)
    config = GraphitiConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        openai_api_key="your-openai-key"  # Replace with actual key
    )
    
    print("\n🔄 Creating hybrid episode mapper...")
    mapper = create_hybrid_episode_mapper(
        config=config,
        context_level=ContextLevel.RICH
    )
    
    print("✅ Mapper created with HYBRID strategy and RICH context level")
    
    try:
        print("\n🔄 Mapping document using hybrid strategy...")
        result = await mapper.map_document_hybrid(
            document=document,
            episode_name_prefix="transformer_paper",
            source_description="Academic paper on Transformer architecture"
        )
        
        print(f"✅ Hybrid mapping completed!")
        print(f"   Strategy: {result['strategy']}")
        print(f"   Success: {result['success']}")
        print(f"   Total episodes created: {result['total_episodes']}")
        
        # Document episode result
        doc_episode = result.get('document_episode', {})
        if doc_episode.get('success'):
            print(f"   📄 Document episode: {doc_episode['episode_name']}")
            print(f"      Content length: {doc_episode['content_length']} chars")
        
        # Chunk episodes results
        chunk_episodes = result.get('chunk_episodes', [])
        successful_chunks = [ep for ep in chunk_episodes if ep.get('success')]
        print(f"   📝 Chunk episodes: {len(successful_chunks)}/{len(chunk_episodes)} successful")
        
        for i, chunk_ep in enumerate(successful_chunks[:3]):  # Show first 3
            print(f"      Chunk {i+1}: {chunk_ep['episode_name']}")
            print(f"         Context length: {len(chunk_ep.get('contextual_summary', ''))}")
            print(f"         Enhanced content: {chunk_ep.get('enhanced_content_length', 0)} chars")
        
        if len(successful_chunks) > 3:
            print(f"      ... and {len(successful_chunks) - 3} more chunks")
            
    except Exception as e:
        print(f"❌ Hybrid mapping failed: {e}")
        return False
    
    return True


async def demonstrate_contextual_chunks_only():
    """Demonstrate contextual chunks strategy without document episode."""
    
    print("\n" + "="*60)
    print("🔄 Demonstrating CONTEXTUAL_CHUNKS strategy...")
    
    document = create_sample_document()
    
    # Configure for contextual chunks only
    config = GraphitiConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j", 
        neo4j_password="password",
        openai_api_key="your-openai-key"
    )
    
    mapper = create_contextual_chunk_mapper(
        config=config,
        context_level=ContextLevel.COMPREHENSIVE
    )
    
    print("✅ Created CONTEXTUAL_CHUNKS mapper with COMPREHENSIVE context")
    
    try:
        print("\n🔄 Mapping chunks with comprehensive context...")
        results = await mapper.map_document_chunks_to_contextual_episodes(
            document=document,
            chunk_episode_prefix="transformer_contextual"
        )
        
        successful_results = [r for r in results if r.get('success')]
        print(f"✅ Contextual chunk mapping completed!")
        print(f"   Episodes created: {len(successful_results)}/{len(results)}")
        
        for i, result in enumerate(successful_results):
            print(f"\n   📝 Chunk {i+1}: {result['episode_name']}")
            print(f"      Context level: {result.get('context_level', 'unknown')}")
            print(f"      Contextual summary: {result['contextual_summary'][:100]}...")
            print(f"      Enhanced content: {result.get('enhanced_content_length', 0)} chars")
            
    except Exception as e:
        print(f"❌ Contextual chunks mapping failed: {e}")
        return False
    
    return True


async def demonstrate_strategy_comparison():
    """Compare different episode strategies."""
    
    print("\n" + "="*60)
    print("🔄 Comparing different episode strategies...")
    
    document = create_sample_document()
    config = GraphitiConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password", 
        openai_api_key="your-openai-key"
    )
    
    strategies = [
        (EpisodeStrategy.DOCUMENT_ONLY, ContextLevel.MINIMAL),
        (EpisodeStrategy.CHUNK_ONLY, ContextLevel.STANDARD),
        (EpisodeStrategy.CONTEXTUAL_CHUNKS, ContextLevel.RICH),
        (EpisodeStrategy.HYBRID, ContextLevel.COMPREHENSIVE)
    ]
    
    for strategy, context_level in strategies:
        print(f"\n🔄 Testing {strategy.value} with {context_level.value} context...")
        
        mapper = DocumentEpisodeMapper(
            config=config,
            strategy=strategy,
            context_level=context_level,
            enable_ai_summarization=True
        )
        
        try:
            if strategy == EpisodeStrategy.DOCUMENT_ONLY:
                result = await mapper.map_document_to_episode(document)
                episodes_count = 1 if result.get('success') else 0
                
            elif strategy == EpisodeStrategy.CHUNK_ONLY:
                results = await mapper.map_document_chunks_to_episodes(document)
                episodes_count = len([r for r in results if r.get('success')])
                
            elif strategy == EpisodeStrategy.CONTEXTUAL_CHUNKS:
                results = await mapper.map_document_chunks_to_contextual_episodes(document)
                episodes_count = len([r for r in results if r.get('success')])
                
            elif strategy == EpisodeStrategy.HYBRID:
                result = await mapper.map_document_hybrid(document)
                episodes_count = result.get('total_episodes', 0)
            
            print(f"   ✅ {strategy.value}: {episodes_count} episodes created")
            
        except Exception as e:
            print(f"   ❌ {strategy.value}: Failed - {e}")


async def main():
    """Main demonstration function."""
    
    print("🚀 Hybrid Episode Mapping Demonstration")
    print("="*60)
    
    # Note: This example requires actual Graphiti credentials
    print("⚠️  Note: This example requires valid Graphiti/Neo4j credentials")
    print("   Update the GraphitiConfig with your actual connection details")
    print("   before running this example.")
    
    # Demonstrate hybrid mapping
    success1 = await demonstrate_hybrid_mapping()
    
    # Demonstrate contextual chunks only
    success2 = await demonstrate_contextual_chunks_only()
    
    # Compare strategies
    await demonstrate_strategy_comparison()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("✅ All demonstrations completed successfully!")
    else:
        print("⚠️  Some demonstrations failed - check your Graphiti configuration")
    
    print("\n📋 Key Benefits of Hybrid Approach:")
    print("   • Granular chunk-level episodes for precise retrieval")
    print("   • Rich contextual summaries preserve document understanding")
    print("   • Configurable context levels for different use cases")
    print("   • AI-powered summarization for better semantic understanding")
    print("   • Both document-level and chunk-level episodes available")


if __name__ == "__main__":
    asyncio.run(main())
