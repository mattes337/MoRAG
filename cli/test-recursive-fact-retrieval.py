#!/usr/bin/env python3
"""Test CLI script for recursive fact retrieval using direct code calls."""

import asyncio
import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import structlog
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env file in the project root (parent directory of cli/)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    # Only show this in verbose mode - we'll add it to the verbose output later
else:
    # Fallback to current directory
    load_dotenv()

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-reasoning" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-graph" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-vector" / "src"))

from morag_reasoning import (
    RecursiveFactRetrievalService,
    RecursiveFactRetrievalRequest,
    RecursiveFactRetrievalResponse,
    LLMClient,
    LLMConfig
)
from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
from morag_graph.storage.qdrant_storage import QdrantStorage, QdrantConfig

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


class RecursiveFactRetrievalTester:
    """Test client for recursive fact retrieval using direct code calls."""

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
        stronger_gemini_model: str = None
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
            gemini_model: Gemini model name for GTA/FCA
            stronger_gemini_model: Stronger Gemini model for final synthesis
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
            self.qdrant_port = parsed.port or (443 if parsed.scheme == "https" else 6333)
            self.qdrant_use_ssl = parsed.scheme == "https"
        else:
            self.qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
            self.qdrant_port = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))
            self.qdrant_use_ssl = False

        self.qdrant_collection = qdrant_collection or os.getenv("MORAG_QDRANT_COLLECTION", "morag_chunks")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = gemini_model or os.getenv("MORAG_GEMINI_MODEL", "gemini-1.5-flash")
        self.stronger_gemini_model = stronger_gemini_model or os.getenv("MORAG_STRONGER_GEMINI_MODEL", "gemini-1.5-pro")

        self.service = None
    
    async def initialize(self) -> bool:
        """Initialize the recursive fact retrieval service.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize LLM clients
            if not self.gemini_api_key:
                logger.error("Gemini API key not provided")
                return False

            # Regular LLM client for GTA/FCA
            llm_config = LLMConfig(
                api_key=self.gemini_api_key,
                model=self.gemini_model
            )
            llm_client = LLMClient(llm_config)

            # Stronger LLM client for final synthesis
            stronger_llm_config = LLMConfig(
                api_key=self.gemini_api_key,
                model=self.stronger_gemini_model
            )
            stronger_llm_client = LLMClient(stronger_llm_config)

            # Initialize Neo4j storage
            neo4j_config = Neo4jConfig(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                database=self.neo4j_database
            )
            neo4j_storage = Neo4jStorage(neo4j_config)
            await neo4j_storage.connect()

            # Initialize Qdrant storage
            qdrant_config = QdrantConfig(
                host=self.qdrant_host,
                port=self.qdrant_port,
                collection_name=self.qdrant_collection,
                api_key=os.getenv("QDRANT_API_KEY"),
                use_ssl=self.qdrant_use_ssl
            )
            qdrant_storage = QdrantStorage(qdrant_config)
            await qdrant_storage.connect()

            # Initialize recursive fact retrieval service
            self.service = RecursiveFactRetrievalService(
                llm_client=llm_client,
                neo4j_storage=neo4j_storage,
                qdrant_storage=qdrant_storage,
                stronger_llm_client=stronger_llm_client
            )

            logger.info("Recursive fact retrieval service initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize service", error=str(e))
            return False

    async def check_health(self) -> Dict[str, Any]:
        """Check the health of the recursive fact retrieval components.

        Returns:
            Health status information
        """
        if not self.service:
            return {
                "status": "unhealthy",
                "error": "Service not initialized"
            }

        health = {
            "status": "healthy",
            "services": {}
        }

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

            # Check LLM clients
            health["services"]["llm_client"] = self.service.llm_client is not None
            health["services"]["stronger_llm_client"] = self.service.stronger_llm_client is not None

            # Overall status
            if not all(health["services"].values()):
                health["status"] = "degraded"

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)

        return health
    
    async def test_recursive_fact_retrieval(
        self,
        user_query: str,
        max_depth: int = 3,
        decay_rate: float = 0.2,
        max_facts_per_node: int = 5,
        min_fact_score: float = 0.1,
        max_total_facts: int = 50,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Test the recursive fact retrieval service.

        Args:
            user_query: User query to test
            max_depth: Maximum traversal depth
            decay_rate: Rate of score decay per depth level
            max_facts_per_node: Maximum facts to extract per node
            min_fact_score: Minimum score threshold for facts
            max_total_facts: Maximum total facts to collect
            language: Language for processing

        Returns:
            Response from the service
        """
        if not self.service:
            return {
                "success": False,
                "error": "Service not initialized"
            }

        try:
            # Create request
            request = RecursiveFactRetrievalRequest(
                user_query=user_query,
                max_depth=max_depth,
                decay_rate=decay_rate,
                max_facts_per_node=max_facts_per_node,
                min_fact_score=min_fact_score,
                max_total_facts=max_total_facts,
                neo4j_database=self.neo4j_database,
                qdrant_collection=self.qdrant_collection,
                language=language
            )

            # Run recursive fact retrieval
            response = await self.service.retrieve_facts_recursively(request)

            return {
                "success": True,
                "data": response.model_dump()
            }

        except Exception as e:
            logger.error("Recursive fact retrieval test failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }

    async def cleanup(self):
        """Clean up resources."""
        if self.service:
            try:
                if hasattr(self.service.neo4j_storage, 'close'):
                    await self.service.neo4j_storage.close()
                if hasattr(self.service.qdrant_storage, 'close'):
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
            print(f"❌ Test failed: {results.get('error', 'Unknown error')}")
            if "status_code" in results:
                print(f"   Status code: {results['status_code']}")
            return
        
        data = results["data"]
        print(f"✅ Test successful!")
        print(f"   Query ID: {data['query_id']}")
        print(f"   Query: {data['user_query']}")
        print(f"   Processing time: {data['processing_time_ms']:.2f}ms")
        print(f"   Initial entities: {', '.join(data['initial_entities'])}")
        print(f"   Nodes explored: {data['total_nodes_explored']}")
        print(f"   Max depth reached: {data['max_depth_reached']}")
        print(f"   Raw facts extracted: {data['total_raw_facts']}")
        print(f"   Scored facts: {data['total_scored_facts']}")
        print(f"   Final facts: {len(data['final_facts'])}")
        print(f"   LLM calls - GTA: {data['gta_llm_calls']}, FCA: {data['fca_llm_calls']}, Final: {data['final_llm_calls']}")
        print(f"   Confidence score: {data['confidence_score']:.2f}")
        
        # Print traversal steps
        if data['traversal_steps'] and verbose:
            print("\n🔄 Traversal Steps:")
            for step in data['traversal_steps']:
                print(f"   Depth {step['depth']}: {step['node_name']} (ID: {step['node_id']})")
                print(f"     Facts extracted: {step['facts_extracted']}")
                print(f"     Decision: {step['next_nodes_decision']}")
                if step['reasoning']:
                    print(f"     Reasoning: {step['reasoning'][:100]}...")
                print()
        
        # Print final facts
        if data['final_facts']:
            print("\n📋 Final Facts:")
            for i, fact in enumerate(data['final_facts'][:10], 1):  # Show top 10 facts
                print(f"   {i}. {fact['fact_text']}")
                print(f"      Score: {fact['final_decayed_score']:.3f} (original: {fact['score']:.3f})")
                print(f"      Depth: {fact['extracted_from_depth']}")
                print(f"      Source: {fact['source_description']}")
                print()
        
        # Print final answer
        print(f"\n💡 Final Answer:")
        print(f"   {data['final_answer']}")


def show_environment_config():
    """Display current environment configuration."""
    print("🔧 Environment Configuration:")

    # Show which .env file was loaded
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        print(f"   📄 Loaded from: {env_path}")
    else:
        print("   📄 Loaded from: current directory or system environment")

    print(f"   GEMINI_API_KEY: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Not set'}")
    print(f"   MORAG_GEMINI_MODEL: {os.getenv('MORAG_GEMINI_MODEL', 'gemini-1.5-flash')}")
    print(f"   NEO4J_URI: {os.getenv('NEO4J_URI', 'bolt://localhost:7687')}")
    print(f"   NEO4J_USERNAME: {os.getenv('NEO4J_USERNAME', 'neo4j')}")
    print(f"   NEO4J_DATABASE: {os.getenv('NEO4J_DATABASE', 'neo4j')}")

    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        print(f"   QDRANT_URL: {qdrant_url}")
    else:
        print(f"   QDRANT_HOST: {os.getenv('QDRANT_HOST', 'localhost')}")
        print(f"   QDRANT_PORT: {os.getenv('QDRANT_PORT', '6333')}")

    print(f"   MORAG_QDRANT_COLLECTION: {os.getenv('MORAG_QDRANT_COLLECTION', 'morag_chunks')}")
    print()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test recursive fact retrieval using direct code calls",
        epilog="Environment variables are loaded from .env file in project root. "
               "Command line arguments override environment variables."
    )
    parser.add_argument("query", nargs="?", help="Query to test (optional when using --show-config)")

    # Database configuration
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                       help=f"Neo4j URI (env: NEO4J_URI, current: {os.getenv('NEO4J_URI', 'bolt://localhost:7687')})")
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USERNAME", "neo4j"),
                       help=f"Neo4j username (env: NEO4J_USERNAME, current: {os.getenv('NEO4J_USERNAME', 'neo4j')})")
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "password"),
                       help="Neo4j password (env: NEO4J_PASSWORD)")
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"),
                       help=f"Neo4j database (env: NEO4J_DATABASE, current: {os.getenv('NEO4J_DATABASE', 'neo4j')})")
    parser.add_argument("--qdrant-host",
                       help="Qdrant host (auto-detected from QDRANT_URL if not specified)")
    parser.add_argument("--qdrant-port", type=int,
                       help="Qdrant port (auto-detected from QDRANT_URL if not specified)")
    parser.add_argument("--qdrant-collection", default=os.getenv("MORAG_QDRANT_COLLECTION", "morag_chunks"),
                       help=f"Qdrant collection (env: MORAG_QDRANT_COLLECTION, current: {os.getenv('MORAG_QDRANT_COLLECTION', 'morag_chunks')})")

    # LLM configuration
    parser.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY"),
                       help=f"Gemini API key (env: GEMINI_API_KEY, {'✅ set' if os.getenv('GEMINI_API_KEY') else '❌ not set'})")
    parser.add_argument("--gemini-model", default=os.getenv("MORAG_GEMINI_MODEL", "gemini-1.5-flash"),
                       help=f"Gemini model for GTA/FCA (env: MORAG_GEMINI_MODEL, current: {os.getenv('MORAG_GEMINI_MODEL', 'gemini-1.5-flash')})")
    parser.add_argument("--stronger-gemini-model", default=os.getenv("MORAG_STRONGER_GEMINI_MODEL", "gemini-1.5-pro"),
                       help=f"Stronger Gemini model for final synthesis (env: MORAG_STRONGER_GEMINI_MODEL, current: {os.getenv('MORAG_STRONGER_GEMINI_MODEL', 'gemini-1.5-pro')})")

    # Retrieval parameters
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum traversal depth")
    parser.add_argument("--decay-rate", type=float, default=0.2, help="Rate of score decay per depth level")
    parser.add_argument("--max-facts-per-node", type=int, default=5, help="Maximum facts to extract per node")
    parser.add_argument("--min-fact-score", type=float, default=0.1, help="Minimum score threshold for facts")
    parser.add_argument("--max-total-facts", type=int, default=50, help="Maximum total facts to collect")
    parser.add_argument("--language", help="Language for processing")

    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--check-health", action="store_true", help="Check service health")
    parser.add_argument("--output-json", help="Save results to JSON file")
    parser.add_argument("--show-config", action="store_true", help="Show current environment configuration and exit")

    args = parser.parse_args()

    # Show configuration if requested
    if args.show_config:
        show_environment_config()
        return 0

    # Ensure query is provided when not showing config
    if not args.query:
        parser.error("Query is required unless using --show-config")

    # Initialize tester
    tester = RecursiveFactRetrievalTester(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_collection=args.qdrant_collection,
        gemini_api_key=args.gemini_api_key,
        gemini_model=args.gemini_model,
        stronger_gemini_model=args.stronger_gemini_model
    )

    try:
        # Show environment configuration if verbose
        if args.verbose:
            show_environment_config()

        # Initialize service
        print("🔧 Initializing recursive fact retrieval service...")
        if not await tester.initialize():
            print("❌ Failed to initialize service")
            return 1

        # Check health if requested
        if args.check_health:
            print("🏥 Checking service health...")
            health = await tester.check_health()
            print(f"Health status: {health.get('status', 'unknown')}")
            if health.get('services'):
                for service, status in health['services'].items():
                    status_icon = "✅" if status else "❌"
                    print(f"  {status_icon} {service}: {'available' if status else 'unavailable'}")
            if health.get('error'):
                print(f"Error: {health['error']}")
            print()

        # Run the test
        print(f"🧠 Testing recursive fact retrieval with query: '{args.query}'")
        print(f"   Max depth: {args.max_depth}")
        print(f"   Decay rate: {args.decay_rate}")
        print(f"   Max facts per node: {args.max_facts_per_node}")
        print(f"   Min fact score: {args.min_fact_score}")
        print(f"   Max total facts: {args.max_total_facts}")
        if args.language:
            print(f"   Language: {args.language}")
        print()

        results = await tester.test_recursive_fact_retrieval(
            user_query=args.query,
            max_depth=args.max_depth,
            decay_rate=args.decay_rate,
            max_facts_per_node=args.max_facts_per_node,
            min_fact_score=args.min_fact_score,
            max_total_facts=args.max_total_facts,
            language=args.language
        )

        # Print results
        tester.print_results(results, verbose=args.verbose)

        # Save to JSON if requested
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output_json}")

        return 0 if results.get("success") else 1

    finally:
        await tester.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
