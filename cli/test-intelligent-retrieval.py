#!/usr/bin/env python3
"""Test CLI script for the intelligent entity retrieval using direct code calls."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the packages to the path
sys.path.insert(
    0, str(Path(__file__).parent.parent / "packages" / "morag-reasoning" / "src")
)
sys.path.insert(
    0, str(Path(__file__).parent.parent / "packages" / "morag-graph" / "src")
)
sys.path.insert(
    0, str(Path(__file__).parent.parent / "packages" / "morag-vector" / "src")
)

from morag_graph.storage.neo4j_storage import Neo4jConfig, Neo4jStorage
from morag_graph.storage.qdrant_storage import QdrantConfig, QdrantStorage
from morag_reasoning import (
    IntelligentRetrievalRequest,
    IntelligentRetrievalResponse,
    IntelligentRetrievalService,
    LLMClient,
    LLMConfig,
)

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
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class IntelligentRetrievalTester:
    """Test client for intelligent entity retrieval using direct code calls."""

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = None,
        qdrant_host: str = None,
        qdrant_port: int = None,
        qdrant_collection: str = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = None,
    ):
        """Initialize the tester.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
            qdrant_collection: Qdrant collection name
            gemini_api_key: Gemini API key
            gemini_model: Gemini model name
        """
        # Use environment variables with fallbacks
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        # Parse Qdrant URL if provided, otherwise use host/port
        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url and not qdrant_host:
            # Extract host and port from URL
            from urllib.parse import urlparse

            parsed = urlparse(qdrant_url)
            self.qdrant_host = parsed.hostname
            self.qdrant_port = parsed.port or (
                443 if parsed.scheme == "https" else 6333
            )
            self.qdrant_use_ssl = parsed.scheme == "https"
        else:
            self.qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
            self.qdrant_port = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))
            self.qdrant_use_ssl = False

        self.qdrant_collection = qdrant_collection or os.getenv(
            "MORAG_QDRANT_COLLECTION", "morag_chunks"
        )
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = gemini_model or os.getenv(
            "MORAG_GEMINI_MODEL", "gemini-1.5-flash"
        )

        self.service = None

    async def initialize(self) -> bool:
        """Initialize the intelligent retrieval service.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize LLM client
            if not self.gemini_api_key:
                logger.error("Gemini API key not provided")
                return False

            llm_config = LLMConfig(api_key=self.gemini_api_key, model=self.gemini_model)
            llm_client = LLMClient(llm_config)

            # Initialize Neo4j storage
            neo4j_config = Neo4jConfig(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                database=self.neo4j_database,
            )
            neo4j_storage = Neo4jStorage(neo4j_config)
            await neo4j_storage.connect()

            # Initialize Qdrant storage
            qdrant_config = QdrantConfig(
                host=self.qdrant_host,
                port=self.qdrant_port,
                collection_name=self.qdrant_collection,
                api_key=os.getenv("QDRANT_API_KEY"),
                use_ssl=self.qdrant_use_ssl,
            )
            qdrant_storage = QdrantStorage(qdrant_config)
            await qdrant_storage.connect()

            # Initialize intelligent retrieval service
            self.service = IntelligentRetrievalService(
                llm_client=llm_client,
                neo4j_storage=neo4j_storage,
                qdrant_storage=qdrant_storage,
            )

            logger.info("Intelligent retrieval service initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize service", error=str(e))
            return False

    async def check_health(self) -> Dict[str, Any]:
        """Check the health of the intelligent retrieval components.

        Returns:
            Health status information
        """
        if not self.service:
            return {"status": "unhealthy", "error": "Service not initialized"}

        health = {"status": "healthy", "services": {}}

        try:
            # Check Neo4j connection
            try:
                await self.service.neo4j_storage.health_check()
                health["services"]["neo4j"] = True
            except Exception as e:
                health["services"]["neo4j"] = False
                health["neo4j_error"] = str(e)

            # Check Qdrant connection
            try:
                await self.service.qdrant_storage.health_check()
                health["services"]["qdrant"] = True
            except Exception as e:
                health["services"]["qdrant"] = False
                health["qdrant_error"] = str(e)

            # Check LLM client
            health["services"]["llm"] = self.service.llm_client is not None

            # Overall status
            if not all(health["services"].values()):
                health["status"] = "degraded"

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)

        return health

    async def get_endpoint_info(self) -> Dict[str, Any]:
        """Get information about the intelligent retrieval service.

        Returns:
            Service information
        """
        return {
            "name": "Intelligent Entity-Based Retrieval",
            "description": "Performs intelligent retrieval using entity identification, recursive graph traversal, and fact extraction",
            "version": "1.0.0",
            "features": [
                "Entity identification from user queries",
                "Recursive graph path following with LLM decisions",
                "Key fact extraction with source tracking",
                "Configurable iteration limits and thresholds",
                "Support for multiple Neo4j databases and Qdrant collections",
            ],
            "parameters": {
                "query": "User query/prompt (required)",
                "max_iterations": "Maximum recursive iterations (default: 5)",
                "max_entities_per_iteration": "Max entities to explore per iteration (default: 10)",
                "max_paths_per_entity": "Max paths to consider per entity (default: 5)",
                "max_depth": "Maximum path depth (default: 3)",
                "min_relevance_threshold": "Minimum relevance threshold (default: 0.3)",
                "include_debug_info": "Include debug information (default: false)",
                "neo4j_database": "Neo4j database name (optional)",
                "qdrant_collection": "Qdrant collection name (optional)",
                "language": "Language for processing (optional)",
            },
        }

    async def test_intelligent_retrieval(
        self,
        query: str,
        max_iterations: int = 3,
        max_entities_per_iteration: int = 5,
        max_paths_per_entity: int = 3,
        max_depth: int = 2,
        min_relevance_threshold: float = 0.3,
        include_debug_info: bool = True,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Test the intelligent retrieval service.

        Args:
            query: User query to test
            max_iterations: Maximum recursive iterations
            max_entities_per_iteration: Max entities to explore per iteration
            max_paths_per_entity: Max paths to consider per entity
            max_depth: Maximum path depth
            min_relevance_threshold: Minimum relevance threshold
            include_debug_info: Include debug information
            language: Language for processing

        Returns:
            Response from the service
        """
        if not self.service:
            return {"success": False, "error": "Service not initialized"}

        try:
            # Create request
            request = IntelligentRetrievalRequest(
                query=query,
                max_iterations=max_iterations,
                max_entities_per_iteration=max_entities_per_iteration,
                max_paths_per_entity=max_paths_per_entity,
                max_depth=max_depth,
                min_relevance_threshold=min_relevance_threshold,
                include_debug_info=include_debug_info,
                neo4j_database=self.neo4j_database,
                qdrant_collection=self.qdrant_collection,
                language=language,
            )

            # Run intelligent retrieval
            response = await self.service.retrieve_intelligently(request)

            return {"success": True, "data": response.model_dump()}

        except Exception as e:
            logger.error("Intelligent retrieval test failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def cleanup(self):
        """Clean up resources."""
        if self.service:
            try:
                if hasattr(self.service.neo4j_storage, "close"):
                    await self.service.neo4j_storage.close()
                if hasattr(self.service.qdrant_storage, "close"):
                    await self.service.qdrant_storage.close()
            except Exception as e:
                logger.warning("Error during cleanup", error=str(e))

    def print_results(self, results: Dict[str, Any], verbose: bool = False):
        """Print test results in a formatted way.

        Args:
            results: Results from the test
            verbose: Whether to print verbose output
        """
        if not results.get("success"):
            print(f"[FAIL] Test failed: {results.get('error', 'Unknown error')}")
            if "status_code" in results:
                print(f"   Status code: {results['status_code']}")
            return

        data = results["data"]
        print(f"[OK] Test successful!")
        print(f"   Query ID: {data['query_id']}")
        print(f"   Query: {data['query']}")
        print(f"   Processing time: {data['processing_time_ms']:.2f}ms")
        print(f"   LLM calls made: {data['llm_calls_made']}")
        print(f"   Total iterations: {data['total_iterations']}")
        print(f"   Entities explored: {data['total_entities_explored']}")
        print(f"   Chunks retrieved: {data['total_chunks_retrieved']}")
        print(f"   Key facts found: {len(data['key_facts'])}")
        print(f"   Confidence score: {data['confidence_score']:.2f}")
        print(f"   Completeness score: {data['completeness_score']:.2f}")

        if data["initial_entities"]:
            print(f"   Initial entities: {', '.join(data['initial_entities'])}")

        # Print key facts
        if data["key_facts"]:
            print("\n[INFO] Key Facts:")
            for i, fact in enumerate(data["key_facts"], 1):
                print(f"   {i}. {fact['fact']}")
                print(f"      Type: {fact['fact_type']}")
                print(f"      Confidence: {fact['confidence']:.2f}")
                print(f"      Relevance: {fact['relevance_to_query']:.2f}")
                print(f"      Sources: {len(fact['sources'])} documents")
                if verbose and fact["sources"]:
                    for j, source in enumerate(
                        fact["sources"][:2], 1
                    ):  # Show first 2 sources
                        print(f"        Source {j}: {source['document_name']}")
                print()

        # Print iteration details if verbose
        if verbose and data["iterations"]:
            print("\n[PROCESSING] Iteration Details:")
            for iteration in data["iterations"]:
                print(f"   Iteration {iteration['iteration']}:")
                print(
                    f"     Entities explored: {', '.join(iteration['entities_explored'])}"
                )
                print(f"     Paths found: {len(iteration['paths_found'])}")
                print(f"     Paths followed: {len(iteration['paths_followed'])}")
                print(f"     Chunks retrieved: {iteration['chunks_retrieved']}")
                if iteration["llm_stop_reason"]:
                    print(f"     Stop reason: {iteration['llm_stop_reason']}")
                print()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test intelligent entity retrieval using direct code calls"
    )
    parser.add_argument("query", help="Query to test")

    # Database configuration
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD", "password"),
        help="Neo4j password",
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database",
    )
    parser.add_argument(
        "--qdrant-host",
        help="Qdrant host (auto-detected from QDRANT_URL if not specified)",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        help="Qdrant port (auto-detected from QDRANT_URL if not specified)",
    )
    parser.add_argument(
        "--qdrant-collection",
        default=os.getenv("MORAG_QDRANT_COLLECTION", "morag_chunks"),
        help="Qdrant collection",
    )

    # LLM configuration
    parser.add_argument(
        "--gemini-api-key",
        default=os.getenv("GEMINI_API_KEY"),
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.getenv("MORAG_GEMINI_MODEL", "gemini-1.5-flash"),
        help="Gemini model",
    )

    # Retrieval parameters
    parser.add_argument(
        "--max-iterations", type=int, default=3, help="Maximum iterations"
    )
    parser.add_argument(
        "--max-entities", type=int, default=5, help="Max entities per iteration"
    )
    parser.add_argument("--max-paths", type=int, default=3, help="Max paths per entity")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum path depth")
    parser.add_argument(
        "--min-relevance", type=float, default=0.3, help="Minimum relevance threshold"
    )
    parser.add_argument("--language", help="Language for processing")

    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--check-health", action="store_true", help="Check service health"
    )
    parser.add_argument("--output-json", help="Save results to JSON file")

    args = parser.parse_args()

    # Initialize tester
    tester = IntelligentRetrievalTester(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_collection=args.qdrant_collection,
        gemini_api_key=args.gemini_api_key,
        gemini_model=args.gemini_model,
    )

    try:
        # Initialize service
        print("🔧 Initializing intelligent retrieval service...")
        if not await tester.initialize():
            print("[FAIL] Failed to initialize service")
            return 1

        # Check health if requested
        if args.check_health:
            print("🏥 Checking service health...")
            health = await tester.check_health()
            print(f"Health status: {health.get('status', 'unknown')}")
            if health.get("services"):
                for service, status in health["services"].items():
                    status_icon = "[OK]" if status else "[FAIL]"
                    print(
                        f"  {status_icon} {service}: {'available' if status else 'unavailable'}"
                    )
            if health.get("error"):
                print(f"Error: {health['error']}")
            print()

        # Run the test
        print(f"🧠 Testing intelligent retrieval with query: '{args.query}'")
        print(f"   Max iterations: {args.max_iterations}")
        print(f"   Max entities per iteration: {args.max_entities}")
        print(f"   Max paths per entity: {args.max_paths}")
        print(f"   Max depth: {args.max_depth}")
        print(f"   Min relevance threshold: {args.min_relevance}")
        if args.language:
            print(f"   Language: {args.language}")
        print()

        # Test entity search to debug the issue
        print("🔍 Testing entity search in database...")
        try:
            # Get a sample of entities from the database
            all_entities = await tester.service.neo4j_storage.get_all_entities()
            print(f"   Found {len(all_entities)} total entities in database:")
            # Show a sample of different entity types
            entity_types = {}
            for entity in all_entities:
                if entity.type not in entity_types:
                    entity_types[entity.type] = []
                entity_types[entity.type].append(entity)

            # Show up to 2 entities per type, max 5 types
            shown_count = 0
            for entity_type, entities in list(entity_types.items())[:5]:
                for entity in entities[:2]:
                    if shown_count < 10:  # Limit total shown entities
                        print(f"     - {entity.name} (type: {entity.type})")
                        shown_count += 1
        except Exception as e:
            print(f"   Error testing entity search: {e}")
        print()

        results = await tester.test_intelligent_retrieval(
            query=args.query,
            max_iterations=args.max_iterations,
            max_entities_per_iteration=args.max_entities,
            max_paths_per_entity=args.max_paths,
            max_depth=args.max_depth,
            min_relevance_threshold=args.min_relevance,
            include_debug_info=args.verbose,
            language=args.language,
        )

        # Print results
        tester.print_results(results, verbose=args.verbose)

        # Save to JSON if requested
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output_json}")

        return 0 if results.get("success") else 1

    finally:
        await tester.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
