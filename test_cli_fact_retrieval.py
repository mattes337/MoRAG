#!/usr/bin/env python3
"""
Test script for CLI fact-based retrieval.

This script tests the updated CLI test-prompt.py script with fact-based retrieval.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def test_cli_fact_retrieval():
    """Test the CLI script with fact-based retrieval."""
    print("🧪 Testing CLI Fact-Based Retrieval")
    print("=" * 50)
    
    # Check if required environment variables are set
    required_vars = ["GEMINI_API_KEY", "NEO4J_URI", "QDRANT_HOST"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("   Please set these variables in your .env file")
        return 1
    
    print("✅ Environment variables are set")
    
    # Test queries
    test_queries = [
        {
            "query": "What are the benefits of machine learning?",
            "description": "Simple fact extraction query"
        },
        {
            "query": "How does artificial intelligence relate to machine learning?",
            "description": "Relationship-based query"
        }
    ]
    
    cli_script = project_root / "cli" / "test-prompt.py"
    if not cli_script.exists():
        print(f"❌ CLI script not found: {cli_script}")
        return 1
    
    print("✅ CLI script found")
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"   Query: {test_case['query']}")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            output_file = tmp_file.name
        
        try:
            # Run CLI script with fact-based retrieval
            cmd = [
                sys.executable,
                str(cli_script),
                "--neo4j",
                "--qdrant", 
                "--use-fact-retrieval",
                "--output", output_file,
                test_case["query"]
            ]
            
            print(f"   Running: {' '.join(cmd)}")
            
            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=project_root
            )
            
            if result.returncode == 0:
                print("   ✅ CLI execution successful")
                
                # Check if output file was created and contains valid JSON
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r') as f:
                            output_data = json.load(f)
                        
                        # Validate output structure
                        if "results" in output_data:
                            cli_results = output_data["results"]
                            
                            # Check for fact-based results
                            if cli_results.get("method") == "fact_retrieval":
                                facts_count = len(cli_results.get("facts", []))
                                has_final_answer = cli_results.get("final_result") is not None
                                processing_time = cli_results.get("performance", {}).get("total_time_seconds", 0)
                                
                                print(f"   📊 Facts extracted: {facts_count}")
                                print(f"   💬 Has final answer: {has_final_answer}")
                                print(f"   ⏱️ Processing time: {processing_time:.2f}s")
                                
                                results.append({
                                    "test": f"cli_test_{i}",
                                    "query": test_case["query"],
                                    "status": "PASS",
                                    "facts_count": facts_count,
                                    "has_final_answer": has_final_answer,
                                    "processing_time_seconds": processing_time
                                })
                            else:
                                print("   ❌ Output does not indicate fact-based retrieval")
                                results.append({
                                    "test": f"cli_test_{i}",
                                    "query": test_case["query"],
                                    "status": "FAIL",
                                    "error": "Method is not fact_retrieval"
                                })
                        else:
                            print("   ❌ Invalid output structure")
                            results.append({
                                "test": f"cli_test_{i}",
                                "query": test_case["query"],
                                "status": "FAIL",
                                "error": "Invalid output structure"
                            })
                            
                    except json.JSONDecodeError as e:
                        print(f"   ❌ Invalid JSON output: {e}")
                        results.append({
                            "test": f"cli_test_{i}",
                            "query": test_case["query"],
                            "status": "FAIL",
                            "error": f"Invalid JSON: {e}"
                        })
                else:
                    print("   ❌ Output file not created")
                    results.append({
                        "test": f"cli_test_{i}",
                        "query": test_case["query"],
                        "status": "FAIL",
                        "error": "Output file not created"
                    })
            else:
                print(f"   ❌ CLI execution failed (exit code: {result.returncode})")
                print(f"   STDOUT: {result.stdout}")
                print(f"   STDERR: {result.stderr}")
                results.append({
                    "test": f"cli_test_{i}",
                    "query": test_case["query"],
                    "status": "FAIL",
                    "error": f"Exit code {result.returncode}: {result.stderr}"
                })
                
        except subprocess.TimeoutExpired:
            print("   ❌ CLI execution timed out")
            results.append({
                "test": f"cli_test_{i}",
                "query": test_case["query"],
                "status": "FAIL",
                "error": "Execution timed out"
            })
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            results.append({
                "test": f"cli_test_{i}",
                "query": test_case["query"],
                "status": "ERROR",
                "error": str(e)
            })
        finally:
            # Clean up temporary file
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    # Print summary
    print("\n📊 CLI Test Results")
    print("=" * 50)
    
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    
    for result in results:
        status_emoji = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
        print(f"{status_emoji} {result['test']}: {result['status']}")
        if result["status"] != "PASS" and "error" in result:
            print(f"   Error: {result['error']}")
    
    print(f"\n📈 Summary: {passed} passed, {failed} failed, {errors} errors")
    
    # Save results
    with open("cli_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Detailed results saved to cli_test_results.json")
    
    return 0 if failed == 0 and errors == 0 else 1


def main():
    """Main function."""
    try:
        return test_cli_fact_retrieval()
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
