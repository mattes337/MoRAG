#!/usr/bin/env python3
"""
Configuration debugging script for MoRAG.

This script outputs the configured and used configuration for embedding, 
splitting, chunking, etc. to help identify why settings are not working as expected.
"""

import os
import sys
from pathlib import Path

# Add the packages to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-services" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-document" / "src"))

from morag_core.config import get_settings
from morag_core.interfaces.converter import ChunkingStrategy


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_config_item(name: str, configured_value, used_value=None, description: str = ""):
    """Print a configuration item with comparison."""
    status = "✅" if used_value is None or configured_value == used_value else "❌"
    print(f"{status} {name:<30} = {configured_value}")
    if description:
        print(f"   {'Description:':<28} {description}")
    if used_value is not None and configured_value != used_value:
        print(f"   {'Actually used:':<28} {used_value}")
    print()


def check_environment_variables():
    """Check environment variables."""
    print_section("Environment Variables")
    
    # Core environment variables
    env_vars = [
        ("GEMINI_API_KEY", "Google AI API key for LLM operations"),
        ("QDRANT_HOST", "Qdrant vector database host"),
        ("QDRANT_PORT", "Qdrant vector database port"),
        ("QDRANT_COLLECTION_NAME", "Qdrant collection name"),
        ("MORAG_API_HOST", "API server host"),
        ("MORAG_API_PORT", "API server port"),
        ("MORAG_DEBUG", "Debug mode"),
        ("MORAG_LOG_LEVEL", "Logging level"),
        ("MORAG_DEFAULT_CHUNK_SIZE", "Default chunk size for documents"),
        ("MORAG_DEFAULT_CHUNK_OVERLAP", "Default chunk overlap"),
        ("MORAG_MAX_TOKENS_PER_CHUNK", "Maximum tokens per chunk"),
        ("MORAG_DEFAULT_CHUNKING_STRATEGY", "Default chunking strategy"),
        ("MORAG_ENABLE_PAGE_BASED_CHUNKING", "Enable page-based chunking"),
        ("MORAG_MAX_PAGE_CHUNK_SIZE", "Maximum page chunk size"),
        ("MORAG_EMBEDDING_BATCH_SIZE", "Embedding batch size"),
        ("MORAG_ENABLE_BATCH_EMBEDDING", "Enable batch embedding"),
    ]
    
    for var_name, description in env_vars:
        value = os.getenv(var_name)
        if value is not None:
            print(f"✅ {var_name:<35} = {value}")
        else:
            print(f"❌ {var_name:<35} = NOT SET")
        print(f"   {'Description:':<33} {description}")
        print()


def check_settings_configuration():
    """Check settings configuration."""
    print_section("Settings Configuration")
    
    try:
        settings = get_settings()
        
        # API Configuration
        print("API Configuration:")
        print_config_item("api_host", settings.api_host, description="API server host")
        print_config_item("api_port", settings.api_port, description="API server port")
        print_config_item("api_workers", settings.api_workers, description="Number of API workers")
        
        # Gemini Configuration
        print("Gemini Configuration:")
        print_config_item("gemini_api_key", "***" if settings.gemini_api_key else None, description="Gemini API key (masked)")
        print_config_item("gemini_model", settings.gemini_model, description="Gemini generation model")
        print_config_item("gemini_embedding_model", settings.gemini_embedding_model, description="Gemini embedding model")
        print_config_item("gemini_vision_model", settings.gemini_vision_model, description="Gemini vision model")
        
        # Embedding Configuration
        print("Embedding Configuration:")
        print_config_item("embedding_batch_size", settings.embedding_batch_size, description="Batch size for embeddings")
        print_config_item("enable_batch_embedding", settings.enable_batch_embedding, description="Enable batch embedding")
        
        # Document Processing Configuration
        print("Document Processing Configuration:")
        print_config_item("default_chunk_size", settings.default_chunk_size, description="Default chunk size")
        print_config_item("default_chunk_overlap", settings.default_chunk_overlap, description="Default chunk overlap")
        print_config_item("max_tokens_per_chunk", settings.max_tokens_per_chunk, description="Maximum tokens per chunk")
        print_config_item("default_chunking_strategy", settings.default_chunking_strategy, description="Default chunking strategy")
        print_config_item("enable_page_based_chunking", settings.enable_page_based_chunking, description="Enable page-based chunking")
        print_config_item("max_page_chunk_size", settings.max_page_chunk_size, description="Maximum page chunk size")
        
        # Qdrant Configuration
        print("Qdrant Configuration:")
        print_config_item("qdrant_host", settings.qdrant_host, description="Qdrant host")
        print_config_item("qdrant_port", settings.qdrant_port, description="Qdrant port")
        print_config_item("qdrant_collection_name", settings.qdrant_collection_name, description="Qdrant collection name")
        print_config_item("qdrant_api_key", "***" if settings.qdrant_api_key else None, description="Qdrant API key (masked)")
        
        # Logging Configuration
        print("Logging Configuration:")
        print_config_item("log_level", settings.log_level, description="Logging level")
        print_config_item("log_format", settings.log_format, description="Log format")
        print_config_item("debug", settings.debug, description="Debug mode")
        print_config_item("environment", settings.environment, description="Environment")
        
    except Exception as e:
        print(f"❌ Error loading settings: {e}")


def check_chunking_strategies():
    """Check available chunking strategies."""
    print_section("Available Chunking Strategies")
    
    try:
        strategies = list(ChunkingStrategy)
        for strategy in strategies:
            print(f"✅ {strategy.value:<20} - {strategy.name}")
        print()
        
        # Check if PAGE strategy is available
        if ChunkingStrategy.PAGE in strategies:
            print("✅ PAGE chunking strategy is available")
        else:
            print("❌ PAGE chunking strategy is NOT available")
            
    except Exception as e:
        print(f"❌ Error checking chunking strategies: {e}")


def check_implementation_status():
    """Check implementation status of key features."""
    print_section("Implementation Status")
    
    # Check if page-based chunking is implemented
    try:
        from morag_document.converters.base import BaseConverter
        converter = BaseConverter()
        
        if hasattr(converter, '_chunk_by_pages'):
            print("✅ Page-based chunking implementation found")
        else:
            print("❌ Page-based chunking implementation NOT found")
            
        if hasattr(converter, '_find_word_boundary'):
            print("✅ Word boundary preservation implementation found")
        else:
            print("❌ Word boundary preservation implementation NOT found")
            
    except Exception as e:
        print(f"❌ Error checking document converter: {e}")
    
    # Check if batch embedding is implemented
    try:
        from morag_services.embedding import GeminiEmbeddingService
        
        if hasattr(GeminiEmbeddingService, 'generate_embeddings_batch'):
            print("✅ Batch embedding implementation found")
        else:
            print("❌ Batch embedding implementation NOT found")
            
    except Exception as e:
        print(f"❌ Error checking embedding service: {e}")


def main():
    """Main function."""
    print("MoRAG Configuration Debug Report")
    print(f"Generated at: {os.popen('date').read().strip()}")
    
    check_environment_variables()
    check_settings_configuration()
    check_chunking_strategies()
    check_implementation_status()
    
    print_section("Summary")
    print("This report shows the current configuration state of MoRAG.")
    print("Look for ❌ markers to identify configuration issues.")
    print("\nKey things to check:")
    print("1. Environment variables are properly set with MORAG_ prefix")
    print("2. Settings are loaded correctly from environment")
    print("3. All chunking strategies are available")
    print("4. Implementation features are properly loaded")
    print("\nIf you see issues, check:")
    print("- .env file has correct variable names")
    print("- Python path includes all MoRAG packages")
    print("- All dependencies are installed")


if __name__ == "__main__":
    main()
