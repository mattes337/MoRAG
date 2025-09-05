"""Integration tests for base components working together."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Any

from tests.utils.mocks import (
    MockStorage,
    MockEmbeddingService, 
    MockProcessor,
    MockTaskManager
)


class TestBaseComponentsIntegration:
    """Test integration between base components."""
    
    @pytest.fixture
    async def integrated_services(self):
        """Create integrated services for testing."""
        services = {
            "storage": MockStorage(),
            "embedding": MockEmbeddingService(embedding_dim=384),
            "processor": MockProcessor(supported_formats=[".txt", ".md"]),
            "task_manager": MockTaskManager()
        }
        
        # Connect storage
        await services["storage"].connect()
        
        yield services
        
        # Cleanup
        await services["storage"].disconnect()
    
    @pytest.fixture
    def sample_documents(self, tmp_path):
        """Create sample documents for integration testing."""
        docs = {}
        
        docs['doc1'] = tmp_path / "document1.txt"
        docs['doc1'].write_text("This is the first document for testing integration.")
        
        docs['doc2'] = tmp_path / "document2.md"
        docs['doc2'].write_text("# Second Document\n\nThis is markdown content for testing.")
        
        docs['doc3'] = tmp_path / "document3.txt" 
        docs['doc3'].write_text("Third document with different content to test processing.")
        
        return docs
    
    async def test_document_processing_workflow(self, integrated_services, sample_documents):
        """Test complete document processing workflow."""
        storage = integrated_services["storage"]
        embedding_service = integrated_services["embedding"]
        processor = integrated_services["processor"]
        task_manager = integrated_services["task_manager"]
        
        # Process documents
        processed_docs = []
        for doc_name, doc_path in sample_documents.items():
            # 1. Process document
            process_result = await processor.process(doc_path)
            assert process_result["success"] is True
            
            # 2. Generate embedding for content
            embedding = await embedding_service.generate_embedding(process_result["content"])
            assert len(embedding) == 384
            
            # 3. Store in vector storage
            vector_id = f"vector_{doc_name}"
            await storage.store_vector(vector_id, embedding, {
                "file_path": str(doc_path),
                "content": process_result["content"],
                "processed_at": "2024-01-01"
            })
            
            # 4. Create processing task
            task_id = await task_manager.create_task(
                "document_processing",
                {"doc_path": str(doc_path), "vector_id": vector_id}
            )
            
            processed_docs.append({
                "doc_name": doc_name,
                "doc_path": doc_path,
                "vector_id": vector_id,
                "task_id": task_id,
                "embedding": embedding,
                "content": process_result["content"]
            })
        
        # Verify all documents were processed
        assert len(processed_docs) == 3
        
        # Test vector search
        query_text = "document testing"
        query_embedding = await embedding_service.generate_embedding(query_text)
        search_results = await storage.search_vectors(query_embedding, limit=5)
        
        assert len(search_results) > 0
        assert all("result" in str(result["id"]) for result in search_results)
        
        # Test task tracking
        tasks = await task_manager.list_tasks(task_type="document_processing")
        assert len(tasks) == 3
        assert all(task["status"] == "pending" for task in tasks)
        
        # Update task statuses
        for doc in processed_docs:
            await task_manager.update_task_status(
                doc["task_id"], 
                "completed", 
                result={"vector_id": doc["vector_id"]}
            )
        
        # Verify task completion
        completed_tasks = await task_manager.list_tasks(status="completed")
        assert len(completed_tasks) == 3
    
    async def test_batch_processing_integration(self, integrated_services, sample_documents):
        """Test batch processing integration."""
        embedding_service = integrated_services["embedding"]
        processor = integrated_services["processor"]
        storage = integrated_services["storage"]
        
        # Batch process all documents
        doc_paths = list(sample_documents.values())
        process_results = await processor.process_batch(doc_paths)
        
        assert len(process_results) == 3
        assert all(result["success"] for result in process_results)
        
        # Batch generate embeddings
        contents = [result["content"] for result in process_results]
        embeddings = await embedding_service.generate_embeddings(contents)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        
        # Batch store vectors
        for i, (embedding, result) in enumerate(zip(embeddings, process_results)):
            vector_id = f"batch_vector_{i}"
            await storage.store_vector(vector_id, embedding, {
                "file_path": result["file_path"],
                "content": result["content"],
                "batch_index": i
            })
        
        # Verify storage
        health = await storage.health_check()
        assert health["vector_count"] == 3
    
    async def test_error_handling_integration(self, integrated_services, tmp_path):
        """Test error handling across components."""
        storage = integrated_services["storage"]
        embedding_service = integrated_services["embedding"]
        processor = integrated_services["processor"]
        
        # Test processor error handling
        nonexistent_file = tmp_path / "nonexistent.txt"
        process_result = await processor.process(nonexistent_file)
        assert process_result["success"] is False
        
        # Test embedding service error handling
        try:
            await embedding_service.generate_embedding("")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test storage with disconnection
        await storage.disconnect()
        
        try:
            await storage.store("key", "value")
            assert False, "Should have raised ConnectionError"
        except ConnectionError:
            pass  # Expected
        
        # Reconnect for cleanup
        await storage.connect()
    
    async def test_concurrent_operations_integration(self, integrated_services, sample_documents):
        """Test concurrent operations across components."""
        embedding_service = integrated_services["embedding"]
        processor = integrated_services["processor"]
        storage = integrated_services["storage"]
        
        # Create concurrent processing tasks
        async def process_and_embed(doc_path):
            # Process document
            process_result = await processor.process(doc_path)
            if not process_result["success"]:
                return None
            
            # Generate embedding
            embedding = await embedding_service.generate_embedding(process_result["content"])
            
            # Store vector
            vector_id = f"concurrent_{doc_path.name}"
            await storage.store_vector(vector_id, embedding, {
                "file_path": str(doc_path),
                "content": process_result["content"]
            })
            
            return {
                "vector_id": vector_id,
                "embedding": embedding,
                "content": process_result["content"]
            }
        
        # Run concurrent tasks
        tasks = [process_and_embed(doc_path) for doc_path in sample_documents.values()]
        results = await asyncio.gather(*tasks)
        
        # Verify all tasks completed successfully
        assert len(results) == 3
        assert all(result is not None for result in results)
        assert all("vector_id" in result for result in results)
        
        # Verify storage contains all vectors
        health = await storage.health_check()
        assert health["vector_count"] >= 3
    
    async def test_service_health_monitoring_integration(self, integrated_services):
        """Test health monitoring across all services."""
        services = integrated_services
        
        # Check health of all services
        health_checks = {}
        
        # Storage health
        health_checks["storage"] = await services["storage"].health_check()
        assert health_checks["storage"]["status"] == "healthy"
        assert health_checks["storage"]["connected"] is True
        
        # Embedding service health  
        health_checks["embedding"] = await services["embedding"].health_check()
        assert health_checks["embedding"]["status"] == "healthy"
        assert health_checks["embedding"]["embedding_dimension"] == 384
        
        # Overall system health
        overall_health = all(
            health.get("status") == "healthy" 
            for health in health_checks.values()
        )
        assert overall_health is True
    
    async def test_data_consistency_integration(self, integrated_services, sample_documents):
        """Test data consistency across components."""
        storage = integrated_services["storage"]
        embedding_service = integrated_services["embedding"]
        processor = integrated_services["processor"]
        task_manager = integrated_services["task_manager"]
        
        doc_path = list(sample_documents.values())[0]
        
        # Process same document multiple times
        results = []
        for i in range(3):
            # Process document
            process_result = await processor.process(doc_path)
            
            # Generate embedding
            embedding = await embedding_service.generate_embedding(process_result["content"])
            
            results.append({
                "content": process_result["content"],
                "embedding": embedding
            })
        
        # Verify consistency
        base_content = results[0]["content"]
        base_embedding = results[0]["embedding"]
        
        for result in results[1:]:
            assert result["content"] == base_content, "Content should be consistent"
            assert result["embedding"] == base_embedding, "Embeddings should be deterministic"
    
    async def test_scalability_simulation(self, integrated_services, tmp_path):
        """Test system behavior with larger data volumes."""
        storage = integrated_services["storage"]
        embedding_service = integrated_services["embedding"]
        processor = integrated_services["processor"]
        
        # Create many documents
        documents = []
        for i in range(20):
            doc_path = tmp_path / f"scale_test_{i}.txt"
            doc_path.write_text(f"This is scale test document number {i} with unique content.")
            documents.append(doc_path)
        
        # Process all documents
        start_time = asyncio.get_event_loop().time()
        
        processed_count = 0
        for doc_path in documents:
            process_result = await processor.process(doc_path)
            if process_result["success"]:
                embedding = await embedding_service.generate_embedding(process_result["content"])
                await storage.store_vector(f"scale_{processed_count}", embedding, {
                    "content": process_result["content"]
                })
                processed_count += 1
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Verify performance
        assert processed_count == 20
        assert processing_time < 5.0  # Should complete within 5 seconds for mock services
        
        # Check throughput
        throughput = processed_count / processing_time
        assert throughput > 5  # At least 5 documents per second
        
        # Verify final storage state
        health = await storage.health_check()
        assert health["vector_count"] >= 20