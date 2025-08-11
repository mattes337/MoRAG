#!/usr/bin/env python3
"""Performance testing script for MoRAG document processing."""

import asyncio
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import structlog
from morag_document.processor import DocumentProcessor
from morag_core.config import Settings

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


class PerformanceTester:
    """Test and measure document processing performance."""
    
    def __init__(self):
        self.settings = Settings()
        self.processor = DocumentProcessor()
        self.results = []
    
    async def test_document_processing(self, file_path: str, test_name: str = "default") -> Dict[str, Any]:
        """Test document processing performance."""
        logger.info("Starting performance test", test_name=test_name, file_path=file_path)
        
        # Get file info
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        # Test with different configurations
        configurations = [
            {
                "name": "baseline",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "chunking_strategy": "page"
            },
            {
                "name": "optimized_small_chunks",
                "chunk_size": 2000,
                "chunk_overlap": 100,
                "chunking_strategy": "page"
            },
            {
                "name": "optimized_large_chunks",
                "chunk_size": 4000,
                "chunk_overlap": 200,
                "chunking_strategy": "page"
            },
            {
                "name": "semantic_chunking",
                "chunk_size": 3000,
                "chunk_overlap": 150,
                "chunking_strategy": "semantic"
            }
        ]
        
        test_results = {
            "file_path": file_path,
            "file_size_mb": file_size,
            "test_name": test_name,
            "configurations": []
        }
        
        for config in configurations:
            logger.info("Testing configuration", config=config["name"])
            
            start_time = time.time()
            
            try:
                # Process document with current configuration
                result = await self.processor.process_file(
                    file_path=file_path,
                    chunk_size=config["chunk_size"],
                    chunk_overlap=config["chunk_overlap"],
                    chunking_strategy=config["chunking_strategy"]
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                config_result = {
                    "config_name": config["name"],
                    "processing_time": processing_time,
                    "chunks_count": len(result.document.chunks),
                    "success": result.success,
                    "chunk_size": config["chunk_size"],
                    "chunk_overlap": config["chunk_overlap"],
                    "chunking_strategy": config["chunking_strategy"],
                    "performance_metrics": {
                        "time_per_chunk": processing_time / len(result.document.chunks) if result.document.chunks else 0,
                        "chunks_per_second": len(result.document.chunks) / processing_time if processing_time > 0 else 0,
                        "mb_per_second": file_size / processing_time if processing_time > 0 else 0
                    }
                }
                
                logger.info("Configuration completed", 
                           config=config["name"],
                           processing_time=processing_time,
                           chunks_count=len(result.document.chunks))
                
            except Exception as e:
                config_result = {
                    "config_name": config["name"],
                    "processing_time": None,
                    "chunks_count": 0,
                    "success": False,
                    "error": str(e),
                    "chunk_size": config["chunk_size"],
                    "chunk_overlap": config["chunk_overlap"],
                    "chunking_strategy": config["chunking_strategy"]
                }
                
                logger.error("Configuration failed", config=config["name"], error=str(e))
            
            test_results["configurations"].append(config_result)
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        return test_results
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance test results."""
        successful_configs = [c for c in results["configurations"] if c["success"]]
        
        if not successful_configs:
            return {"error": "No successful configurations"}
        
        # Find best performing configuration
        best_config = min(successful_configs, key=lambda x: x["processing_time"])
        worst_config = max(successful_configs, key=lambda x: x["processing_time"])
        
        # Calculate averages
        avg_time = sum(c["processing_time"] for c in successful_configs) / len(successful_configs)
        avg_chunks = sum(c["chunks_count"] for c in successful_configs) / len(successful_configs)
        
        analysis = {
            "file_info": {
                "file_path": results["file_path"],
                "file_size_mb": results["file_size_mb"]
            },
            "best_configuration": {
                "name": best_config["config_name"],
                "processing_time": best_config["processing_time"],
                "chunks_count": best_config["chunks_count"],
                "performance_metrics": best_config["performance_metrics"]
            },
            "worst_configuration": {
                "name": worst_config["config_name"],
                "processing_time": worst_config["processing_time"],
                "chunks_count": worst_config["chunks_count"]
            },
            "averages": {
                "avg_processing_time": avg_time,
                "avg_chunks_count": avg_chunks
            },
            "improvement_potential": {
                "time_saved": worst_config["processing_time"] - best_config["processing_time"],
                "percentage_improvement": ((worst_config["processing_time"] - best_config["processing_time"]) / worst_config["processing_time"]) * 100
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if best_config["processing_time"] < 3.0:
            analysis["recommendations"].append("Performance is good (< 3 seconds)")
        elif best_config["processing_time"] < 5.0:
            analysis["recommendations"].append("Performance is acceptable (< 5 seconds)")
        else:
            analysis["recommendations"].append("Performance needs improvement (> 5 seconds)")
        
        if best_config["config_name"] == "optimized_large_chunks":
            analysis["recommendations"].append("Use larger chunks for better performance")
        elif best_config["config_name"] == "optimized_small_chunks":
            analysis["recommendations"].append("Use smaller chunks for this document type")
        
        return analysis
    
    def print_results(self, analysis: Dict[str, Any]):
        """Print performance analysis results."""
        print("\n" + "="*60)
        print("  MoRAG Performance Test Results")
        print("="*60)
        
        print(f"\n📄 File: {analysis['file_info']['file_path']}")
        print(f"📊 Size: {analysis['file_info']['file_size_mb']:.2f} MB")
        
        print(f"\n🏆 Best Configuration: {analysis['best_configuration']['name']}")
        print(f"⏱️  Processing Time: {analysis['best_configuration']['processing_time']:.2f} seconds")
        print(f"📝 Chunks Generated: {analysis['best_configuration']['chunks_count']}")
        print(f"🚀 Performance: {analysis['best_configuration']['performance_metrics']['mb_per_second']:.2f} MB/s")
        
        print(f"\n📈 Improvement Potential:")
        print(f"⏰ Time Saved: {analysis['improvement_potential']['time_saved']:.2f} seconds")
        print(f"📊 Improvement: {analysis['improvement_potential']['percentage_improvement']:.1f}%")
        
        print(f"\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")
        
        print("\n" + "="*60)


async def main():
    """Main function to run performance tests."""
    if len(sys.argv) < 2:
        print("Usage: python test-performance.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    
    tester = PerformanceTester()
    
    try:
        # Run performance test
        results = await tester.test_document_processing(file_path, "performance_test")
        
        # Analyze results
        analysis = tester.analyze_results(results)
        
        # Print results
        tester.print_results(analysis)
        
        # Save detailed results to file
        import json
        output_file = f"{file_path}_performance_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                "detailed_results": results,
                "analysis": analysis
            }, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {output_file}")
        
    except Exception as e:
        logger.error("Performance test failed", error=str(e))
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
