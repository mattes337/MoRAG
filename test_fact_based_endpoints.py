#!/usr/bin/env python3
"""
Test script for fact-based endpoints and CLI tools.

This script tests the updated endpoints to ensure they work correctly with the new fact-based system.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import httpx
import structlog

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = structlog.get_logger(__name__)


class FactBasedEndpointTester:
    """Test class for fact-based endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.test_results = []
    
    async def test_enhanced_query_fact_retrieval(self) -> Dict[str, Any]:
        """Test the enhanced query endpoint with fact-based retrieval."""
        logger.info("Testing enhanced query endpoint with fact-based retrieval")
        
        test_data = {
            "query": "What are the benefits of machine learning?",
            "use_fact_retrieval": True,
            "max_depth": 2,
            "max_total_facts": 20,
            "facts_only": False,
            "language": "en"
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v2/query",
                json=test_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate fact-based response structure
                assert "facts" in result, "Response should contain facts"
                assert "final_answer" in result, "Response should contain final_answer"
                assert "traversal_steps" in result, "Response should contain traversal_steps"
                assert "gta_llm_calls" in result, "Response should contain gta_llm_calls"
                
                return {
                    "test": "enhanced_query_fact_retrieval",
                    "status": "PASS",
                    "facts_count": len(result.get("facts", [])),
                    "has_final_answer": result.get("final_answer") is not None,
                    "processing_time_ms": result.get("processing_time_ms", 0)
                }
            else:
                return {
                    "test": "enhanced_query_fact_retrieval",
                    "status": "FAIL",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "test": "enhanced_query_fact_retrieval",
                "status": "ERROR",
                "error": str(e)
            }
    
    async def test_intelligent_retrieval_facts(self) -> Dict[str, Any]:
        """Test the intelligent retrieval endpoint with facts."""
        logger.info("Testing intelligent retrieval endpoint with facts")
        
        test_data = {
            "user_query": "How does artificial intelligence work?",
            "max_depth": 2,
            "max_total_facts": 15,
            "facts_only": True,
            "language": "en"
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v2/intelligent-query/facts",
                json=test_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate fact-based response structure
                assert "final_facts" in result, "Response should contain final_facts"
                assert "traversal_steps" in result, "Response should contain traversal_steps"
                assert "initial_entities" in result, "Response should contain initial_entities"
                
                return {
                    "test": "intelligent_retrieval_facts",
                    "status": "PASS",
                    "facts_count": len(result.get("final_facts", [])),
                    "entities_count": len(result.get("initial_entities", [])),
                    "processing_time_ms": result.get("processing_time_ms", 0)
                }
            else:
                return {
                    "test": "intelligent_retrieval_facts",
                    "status": "FAIL",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "test": "intelligent_retrieval_facts",
                "status": "ERROR",
                "error": str(e)
            }
    
    async def test_reasoning_fact_based(self) -> Dict[str, Any]:
        """Test the reasoning endpoint with fact-based approach."""
        logger.info("Testing reasoning endpoint with fact-based approach")
        
        test_data = {
            "query": "What is the relationship between AI and machine learning?",
            "max_depth": 2,
            "max_facts": 20,
            "facts_only": False,
            "language": "en"
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/reasoning/query/facts/simple",
                json=test_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate fact-based response structure
                assert "final_facts" in result, "Response should contain final_facts"
                assert "final_answer" in result, "Response should contain final_answer"
                
                return {
                    "test": "reasoning_fact_based",
                    "status": "PASS",
                    "facts_count": len(result.get("final_facts", [])),
                    "has_final_answer": result.get("final_answer") is not None,
                    "processing_time_ms": result.get("processing_time_ms", 0)
                }
            else:
                return {
                    "test": "reasoning_fact_based",
                    "status": "FAIL",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "test": "reasoning_fact_based",
                "status": "ERROR",
                "error": str(e)
            }
    
    async def test_health_checks(self) -> Dict[str, Any]:
        """Test health check endpoints."""
        logger.info("Testing health check endpoints")
        
        health_endpoints = [
            "/api/v2/intelligent-query/health",
            "/api/v2/recursive-fact-retrieval/health",
            "/reasoning/status"
        ]
        
        results = []
        for endpoint in health_endpoints:
            try:
                response = await self.client.get(f"{self.base_url}{endpoint}")
                results.append({
                    "endpoint": endpoint,
                    "status": "PASS" if response.status_code == 200 else "FAIL",
                    "response_code": response.status_code
                })
            except Exception as e:
                results.append({
                    "endpoint": endpoint,
                    "status": "ERROR",
                    "error": str(e)
                })
        
        return {
            "test": "health_checks",
            "status": "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL",
            "results": results
        }
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all tests and return results."""
        logger.info("Starting fact-based endpoint tests")
        
        tests = [
            self.test_enhanced_query_fact_retrieval,
            self.test_intelligent_retrieval_facts,
            self.test_reasoning_fact_based,
            self.test_health_checks
        ]
        
        results = []
        for test in tests:
            try:
                result = await test()
                results.append(result)
                logger.info(f"Test {result['test']}: {result['status']}")
            except Exception as e:
                logger.error(f"Test {test.__name__} failed with exception", error=str(e))
                results.append({
                    "test": test.__name__,
                    "status": "ERROR",
                    "error": str(e)
                })
        
        await self.client.aclose()
        return results


async def main():
    """Main test function."""
    print("🧪 Testing Fact-Based Endpoints")
    print("=" * 50)
    
    # Check if server is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/docs")
            if response.status_code != 200:
                print("❌ MoRAG server is not running on localhost:8000")
                print("   Please start the server with: python -m morag.server")
                return 1
    except Exception:
        print("❌ Cannot connect to MoRAG server on localhost:8000")
        print("   Please start the server with: python -m morag.server")
        return 1
    
    print("✅ Server is running")
    
    # Run tests
    tester = FactBasedEndpointTester()
    results = await tester.run_all_tests()
    
    # Print results
    print("\n📊 Test Results")
    print("=" * 50)
    
    passed = 0
    failed = 0
    errors = 0
    
    for result in results:
        status_emoji = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
        print(f"{status_emoji} {result['test']}: {result['status']}")
        
        if result["status"] == "PASS":
            passed += 1
            # Show additional info for successful tests
            if "facts_count" in result:
                print(f"   Facts extracted: {result['facts_count']}")
            if "processing_time_ms" in result:
                print(f"   Processing time: {result['processing_time_ms']:.0f}ms")
        elif result["status"] == "FAIL":
            failed += 1
            if "error" in result:
                print(f"   Error: {result['error']}")
        else:
            errors += 1
            if "error" in result:
                print(f"   Error: {result['error']}")
    
    print(f"\n📈 Summary: {passed} passed, {failed} failed, {errors} errors")
    
    # Save detailed results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Detailed results saved to test_results.json")
    
    return 0 if failed == 0 and errors == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)
