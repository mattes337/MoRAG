#!/usr/bin/env python3
"""CLI script to test multi-hop reasoning via MoRAG API.

This script tests the multi-hop reasoning functionality through the MoRAG API
endpoints, providing a realistic test of the integrated system.

Usage:
    python test_reasoning_api.py "How are Apple's AI research efforts related to their partnership with universities?"
    
Or with specific parameters:
    python test_reasoning_api.py "query" --api-url http://localhost:8000 --verbose
    
Options:
    --api-url        MoRAG API base URL (default: http://localhost:8000)
    --strategy       Reasoning strategy (forward_chaining, backward_chaining, bidirectional)
    --max-paths      Maximum number of paths to discover (default: 50)
    --max-iterations Maximum context refinement iterations (default: 5)
    --verbose        Show detailed reasoning output
    --output         Save results to JSON file
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx


async def check_api_health(api_url: str, verbose: bool = False) -> bool:
    """Check if the MoRAG API is available and healthy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/health")
            
            if response.status_code == 200:
                health_data = response.json()
                if verbose:
                    print(f"✅ API Health: {health_data.get('status', 'unknown')}")
                    print(f"   Version: {health_data.get('version', 'unknown')}")
                return True
            else:
                print(f"❌ API health check failed: HTTP {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Cannot connect to API at {api_url}: {e}")
        return False


async def check_reasoning_status(api_url: str, verbose: bool = False) -> bool:
    """Check if multi-hop reasoning is available."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/reasoning/status")
            
            if response.status_code == 200:
                status_data = response.json()
                reasoning_available = status_data.get("reasoning_available", False)
                
                if verbose:
                    print(f"🧠 Reasoning Available: {reasoning_available}")
                    components = status_data.get("components", {})
                    for component, available in components.items():
                        status_icon = "✅" if available else "❌"
                        print(f"   {component}: {status_icon}")
                    
                    config = status_data.get("configuration", {})
                    if config:
                        print(f"   Max paths: {config.get('max_paths_default', 'unknown')}")
                        print(f"   Max iterations: {config.get('max_iterations_default', 'unknown')}")
                        print(f"   Strategies: {config.get('supported_strategies', [])}")
                
                return reasoning_available
            else:
                print(f"❌ Reasoning status check failed: HTTP {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Cannot check reasoning status: {e}")
        return False


async def test_multi_hop_reasoning_api(
    query: str,
    api_url: str,
    strategy: str = "forward_chaining",
    max_paths: int = 50,
    max_iterations: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """Test multi-hop reasoning via API."""
    
    if verbose:
        print(f"🧠 Testing multi-hop reasoning via API")
        print(f"📝 Query: {query}")
        print(f"🔄 Strategy: {strategy}")
        print(f"📊 Max paths: {max_paths}")
        print(f"🔁 Max iterations: {max_iterations}")
        print("=" * 60)
    
    results = {
        "query": query,
        "strategy": strategy,
        "api_url": api_url,
        "reasoning_paths": [],
        "refined_context": {},
        "performance": {},
        "success": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Test path finding
            if verbose:
                print("🔍 Step 1: Finding reasoning paths...")
            
            start_time = time.time()
            
            path_request = {
                "query": query,
                "start_entities": ["Apple Inc.", "AI research", "Stanford University"],
                "strategy": strategy,
                "max_paths": max_paths
            }
            
            response = await client.post(
                f"{api_url}/reasoning/find-paths",
                json=path_request
            )
            
            if response.status_code != 200:
                error_msg = f"Path finding failed: HTTP {response.status_code}"
                if verbose:
                    try:
                        error_detail = response.json()
                        error_msg += f" - {error_detail.get('detail', 'Unknown error')}"
                    except:
                        error_msg += f" - {response.text}"
                print(f"❌ {error_msg}")
                results["error"] = error_msg
                return results
            
            path_data = response.json()
            path_finding_time = time.time() - start_time
            
            reasoning_paths = path_data.get("paths", [])
            
            if verbose:
                print(f"✅ Found {len(reasoning_paths)} reasoning paths in {path_finding_time:.2f}s")
                for i, path in enumerate(reasoning_paths[:3]):  # Show top 3
                    print(f"   Path {i+1}: Score {path.get('relevance_score', 0):.2f}")
                    print(f"      Entities: {' -> '.join(path.get('entities', []))}")
                    if path.get('reasoning'):
                        print(f"      Reasoning: {path['reasoning']}")
            
            results["reasoning_paths"] = reasoning_paths
            
            # Step 2: Test context refinement
            if verbose:
                print("\n🔄 Step 2: Testing context refinement...")
            
            start_time = time.time()
            
            # Create initial context from paths
            initial_context = {
                "entities": {},
                "relations": [],
                "documents": [],
                "paths": reasoning_paths[:5]  # Use top 5 paths
            }
            
            # Add entities from paths
            for path in reasoning_paths[:3]:
                for entity in path.get("entities", []):
                    initial_context["entities"][entity] = {"type": "UNKNOWN"}
            
            refinement_request = {
                "query": query,
                "initial_context": initial_context,
                "max_iterations": max_iterations
            }
            
            response = await client.post(
                f"{api_url}/reasoning/refine-context",
                json=refinement_request
            )
            
            if response.status_code != 200:
                error_msg = f"Context refinement failed: HTTP {response.status_code}"
                if verbose:
                    try:
                        error_detail = response.json()
                        error_msg += f" - {error_detail.get('detail', 'Unknown error')}"
                    except:
                        error_msg += f" - {response.text}"
                print(f"❌ {error_msg}")
                # Don't fail completely if context refinement fails
                results["refinement_error"] = error_msg
            else:
                refinement_data = response.json()
                refinement_time = time.time() - start_time
                
                refined_context = refinement_data.get("refined_context", {})
                
                if verbose:
                    print(f"✅ Context refinement completed in {refinement_time:.2f}s")
                    print(f"   Iterations used: {refined_context.get('iterations_used', 0)}")
                    print(f"   Final entities: {len(refined_context.get('entities', {}))}")
                    print(f"   Final documents: {len(refined_context.get('documents', []))}")
                    
                    final_analysis = refined_context.get('final_analysis')
                    if final_analysis:
                        print(f"   Final confidence: {final_analysis.get('confidence', 0):.2f}")
                        print(f"   Context sufficient: {final_analysis.get('is_sufficient', False)}")
                
                results["refined_context"] = refined_context
                results["performance"]["refinement_time"] = refinement_time
            
            results["performance"]["path_finding_time"] = path_finding_time
            results["performance"]["total_time"] = path_finding_time + results["performance"].get("refinement_time", 0)
            
            results["success"] = True
            
            if verbose:
                print("\n" + "=" * 60)
                print("✅ Multi-hop reasoning API test completed successfully!")
                print(f"📊 Total time: {results['performance']['total_time']:.2f}s")
        
    except httpx.TimeoutException:
        error_msg = "API request timed out"
        print(f"❌ {error_msg}")
        results["error"] = error_msg
    except Exception as e:
        error_msg = f"Error during API test: {e}"
        print(f"❌ {error_msg}")
        results["error"] = error_msg
        if verbose:
            import traceback
            traceback.print_exc()
    
    return results


async def test_reasoning_query_endpoint(
    query: str,
    api_url: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """Test the unified reasoning query endpoint."""
    
    if verbose:
        print(f"🔍 Testing unified reasoning query endpoint...")
        print(f"📝 Query: {query}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.time()
            
            request_data = {
                "query": query,
                "use_reasoning": True,
                "reasoning_strategy": "bidirectional",
                "max_reasoning_paths": 20
            }
            
            response = await client.post(
                f"{api_url}/reasoning/query",
                json=request_data
            )
            
            query_time = time.time() - start_time
            
            if response.status_code != 200:
                error_msg = f"Reasoning query failed: HTTP {response.status_code}"
                if verbose:
                    try:
                        error_detail = response.json()
                        error_msg += f" - {error_detail.get('detail', 'Unknown error')}"
                    except:
                        error_msg += f" - {response.text}"
                print(f"❌ {error_msg}")
                return {"success": False, "error": error_msg}
            
            result_data = response.json()
            
            if verbose:
                print(f"✅ Reasoning query completed in {query_time:.2f}s")
                print(f"   Answer: {result_data.get('answer', 'No answer')[:200]}...")
                print(f"   Sources: {len(result_data.get('sources', []))}")
                print(f"   Reasoning paths used: {len(result_data.get('reasoning_paths', []))}")
                
                reasoning_metadata = result_data.get('reasoning_metadata', {})
                if reasoning_metadata:
                    print(f"   Strategy used: {reasoning_metadata.get('strategy', 'unknown')}")
                    print(f"   Paths evaluated: {reasoning_metadata.get('paths_evaluated', 0)}")
                    print(f"   Context iterations: {reasoning_metadata.get('context_iterations', 0)}")
            
            return {
                "success": True,
                "query_time": query_time,
                "answer": result_data.get("answer", ""),
                "sources": result_data.get("sources", []),
                "reasoning_paths": result_data.get("reasoning_paths", []),
                "reasoning_metadata": result_data.get("reasoning_metadata", {})
            }
    
    except Exception as e:
        error_msg = f"Error during reasoning query test: {e}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}


def save_results(results: Dict[str, Any], output_file: Path, verbose: bool = False) -> bool:
    """Save test results to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"💾 Results saved to: {output_file}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error saving results to {output_file}: {e}")
        return False


async def main():
    """Main function to run API-based multi-hop reasoning test."""
    parser = argparse.ArgumentParser(
        description="Test multi-hop reasoning via MoRAG API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python test_reasoning_api.py "How are Apple's AI research efforts related to their partnership with universities?"
  python test_reasoning_api.py "What connects Steve Jobs to iPhone development?" --strategy bidirectional
  python test_reasoning_api.py "How does AI research influence product development?" --verbose --output results.json
"""
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="Multi-hop reasoning query to test"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="MoRAG API base URL (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["forward_chaining", "backward_chaining", "bidirectional"],
        default="forward_chaining",
        help="Reasoning strategy to use (default: forward_chaining)"
    )
    
    parser.add_argument(
        "--max-paths",
        type=int,
        default=50,
        help="Maximum number of paths to discover (default: 50)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum context refinement iterations (default: 5)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed reasoning output"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        "--test-unified",
        action="store_true",
        help="Test the unified reasoning query endpoint instead of individual components"
    )
    
    args = parser.parse_args()
    
    print("🧠 MoRAG Multi-Hop Reasoning API Test")
    print("=" * 60)
    
    # Check API health
    if not await check_api_health(args.api_url, args.verbose):
        return 1
    
    # Check reasoning availability
    if not await check_reasoning_status(args.api_url, args.verbose):
        print("❌ Multi-hop reasoning is not available")
        return 1
    
    print("=" * 60)
    
    try:
        if args.test_unified:
            # Test unified reasoning endpoint
            results = await test_reasoning_query_endpoint(
                query=args.query,
                api_url=args.api_url,
                verbose=args.verbose
            )
        else:
            # Test individual reasoning components
            results = await test_multi_hop_reasoning_api(
                query=args.query,
                api_url=args.api_url,
                strategy=args.strategy,
                max_paths=args.max_paths,
                max_iterations=args.max_iterations,
                verbose=args.verbose
            )
        
        if not results.get("success", False):
            print("❌ Multi-hop reasoning API test failed")
            return 1
        
        # Save results if requested
        if args.output:
            output_file = Path(args.output)
            if save_results(results, output_file, args.verbose):
                print(f"📁 Results saved to: {output_file}")
            else:
                return 1
        
        print("✅ Multi-hop reasoning API test completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
