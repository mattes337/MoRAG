# Task 14: Gemini API Integration

## Overview
Integrate Google Gemini API for text embeddings and summarization, providing the core AI capabilities for the MoRAG pipeline.

## Prerequisites
- Task 01: Project Setup completed
- Google Cloud account with Gemini API access
- Gemini API key

## Dependencies
- Task 01: Project Setup

## Implementation Steps

### 1. Gemini Service Implementation
Create `src/morag/services/embedding.py`:
```python
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import structlog
import asyncio
import time
from dataclasses import dataclass

from morag.core.config import settings
from morag.core.exceptions import ExternalServiceError, RateLimitError

logger = structlog.get_logger()

@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embedding: List[float]
    token_count: int
    model: str

@dataclass
class SummaryResult:
    """Result of text summarization."""
    summary: str
    token_count: int
    model: str

class GeminiService:
    """Service for interacting with Google Gemini API."""
    
    def __init__(self):
        self.embedding_model = settings.gemini_embedding_model
        self.text_model = settings.gemini_model
        self._configure_api()
    
    def _configure_api(self) -> None:
        """Configure Gemini API with credentials."""
        try:
            genai.configure(api_key=settings.gemini_api_key)
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error("Failed to configure Gemini API", error=str(e))
            raise ExternalServiceError(f"Failed to configure Gemini API: {str(e)}", "gemini")
    
    async def generate_embedding(
        self,
        text: str,
        task_type: str = "retrieval_document"
    ) -> EmbeddingResult:
        """Generate embedding for a single text."""
        try:
            # Use asyncio.to_thread for CPU-bound operations
            result = await asyncio.to_thread(
                self._generate_embedding_sync,
                text,
                task_type
            )
            
            logger.debug("Generated embedding", text_length=len(text), model=self.embedding_model)
            return result
            
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                logger.warning("Rate limit hit for embedding generation", error=str(e))
                raise RateLimitError(f"Gemini API rate limit: {str(e)}")
            
            logger.error("Failed to generate embedding", error=str(e))
            raise ExternalServiceError(f"Embedding generation failed: {str(e)}", "gemini")
    
    def _generate_embedding_sync(self, text: str, task_type: str) -> EmbeddingResult:
        """Synchronous embedding generation."""
        result = genai.embed_content(
            model=f"models/{self.embedding_model}",
            content=text,
            task_type=task_type
        )
        
        return EmbeddingResult(
            embedding=result['embedding'],
            token_count=len(text.split()),  # Approximate token count
            model=self.embedding_model
        )
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        task_type: str = "retrieval_document",
        batch_size: int = 10,
        delay_between_batches: float = 1.0
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts with rate limiting."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            # Process batch concurrently
            tasks = [
                self.generate_embedding(text, task_type)
                for text in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions in batch
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(
                            "Failed to generate embedding in batch",
                            batch_index=i + j,
                            error=str(result)
                        )
                        # Create a zero embedding as fallback
                        batch_results[j] = EmbeddingResult(
                            embedding=[0.0] * 768,  # text-embedding-004 dimension
                            token_count=0,
                            model=self.embedding_model
                        )
                
                results.extend(batch_results)
                
                # Rate limiting delay between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(delay_between_batches)
                    
            except Exception as e:
                logger.error("Batch embedding generation failed", error=str(e))
                # Add fallback embeddings for the entire batch
                for _ in batch:
                    results.append(EmbeddingResult(
                        embedding=[0.0] * 768,
                        token_count=0,
                        model=self.embedding_model
                    ))
        
        logger.info("Batch embedding generation completed", total_texts=len(texts))
        return results
    
    async def generate_summary(
        self,
        text: str,
        max_length: int = 150,
        style: str = "concise"
    ) -> SummaryResult:
        """Generate a summary of the given text."""
        try:
            prompt = self._build_summary_prompt(text, max_length, style)
            
            result = await asyncio.to_thread(
                self._generate_text_sync,
                prompt
            )
            
            logger.debug("Generated summary", original_length=len(text), summary_length=len(result))
            return SummaryResult(
                summary=result,
                token_count=len(result.split()),
                model=self.text_model
            )
            
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                logger.warning("Rate limit hit for summary generation", error=str(e))
                raise RateLimitError(f"Gemini API rate limit: {str(e)}")
            
            logger.error("Failed to generate summary", error=str(e))
            raise ExternalServiceError(f"Summary generation failed: {str(e)}", "gemini")
    
    def _generate_text_sync(self, prompt: str) -> str:
        """Synchronous text generation."""
        model = genai.GenerativeModel(self.text_model)
        response = model.generate_content(prompt)
        return response.text.strip()
    
    def _build_summary_prompt(self, text: str, max_length: int, style: str) -> str:
        """Build prompt for text summarization."""
        style_instructions = {
            "concise": "Create a concise, factual summary",
            "detailed": "Create a detailed summary that captures key points",
            "bullet": "Create a bullet-point summary of main ideas",
            "abstract": "Create an abstract-style summary suitable for academic content"
        }
        
        instruction = style_instructions.get(style, style_instructions["concise"])
        
        return f"""
{instruction} of the following text in approximately {max_length} words or less.
Focus on the main ideas, key facts, and important details.

Text to summarize:
{text}

Summary:
"""
    
    async def generate_summaries_batch(
        self,
        texts: List[str],
        max_length: int = 150,
        style: str = "concise",
        batch_size: int = 5,
        delay_between_batches: float = 2.0
    ) -> List[SummaryResult]:
        """Generate summaries for multiple texts with rate limiting."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            # Process batch sequentially to avoid rate limits
            for text in batch:
                try:
                    result = await self.generate_summary(text, max_length, style)
                    batch_results.append(result)
                except Exception as e:
                    logger.error("Failed to generate summary in batch", error=str(e))
                    # Create fallback summary
                    fallback_summary = text[:max_length] + "..." if len(text) > max_length else text
                    batch_results.append(SummaryResult(
                        summary=fallback_summary,
                        token_count=len(fallback_summary.split()),
                        model=self.text_model
                    ))
                
                # Small delay between individual requests
                await asyncio.sleep(0.5)
            
            results.extend(batch_results)
            
            # Longer delay between batches
            if i + batch_size < len(texts):
                await asyncio.sleep(delay_between_batches)
        
        logger.info("Batch summary generation completed", total_texts=len(texts))
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if Gemini API is accessible."""
        try:
            # Test with a simple embedding request
            test_result = await self.generate_embedding("Health check test")
            
            return {
                "status": "healthy",
                "embedding_model": self.embedding_model,
                "text_model": self.text_model,
                "embedding_dimension": len(test_result.embedding)
            }
            
        except Exception as e:
            logger.error("Gemini health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "embedding_model": self.embedding_model,
                "text_model": self.text_model
            }

# Global instance
gemini_service = GeminiService()
```

### 2. Update Configuration Validation
Update `src/morag/core/config.py` to validate Gemini API key:
```python
# Add this method to the Settings class
def validate_gemini_config(self) -> None:
    """Validate Gemini API configuration."""
    if not self.gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    if not self.gemini_api_key.startswith("AI"):
        raise ValueError("Invalid Gemini API key format")

# Add validation call in __init__ or as a validator
```

### 3. Update Health Check
Update `src/morag/api/routes/health.py`:
```python
# Add import
from morag.services.embedding import gemini_service

# Replace Gemini check in readiness_check:
    # Check Gemini API
    try:
        gemini_health = await gemini_service.health_check()
        services["gemini"] = gemini_health["status"]
    except Exception as e:
        logger.error("Gemini health check failed", error=str(e))
        services["gemini"] = "unhealthy"
```

### 4. Utility Functions
Create `src/morag/utils/text_processing.py`:
```python
from typing import List, Dict, Any
import re

def prepare_text_for_embedding(text: str) -> str:
    """Prepare text for embedding generation."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"\']', ' ', text)
    
    # Trim and ensure reasonable length
    text = text.strip()
    
    # Truncate if too long (Gemini has token limits)
    max_chars = 30000  # Conservative limit
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    return text

def prepare_text_for_summary(text: str) -> str:
    """Prepare text for summarization."""
    # Similar cleaning but preserve more structure
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
    
    text = text.strip()
    
    # Truncate if too long
    max_chars = 25000  # Leave room for prompt
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    return text

def combine_text_and_summary(text: str, summary: str) -> str:
    """Combine summary and original text for embedding."""
    return f"Summary: {summary}\n\nContent: {text}"
```

## Testing Instructions

### 1. Set Up API Key
```bash
# Add to your .env file
GEMINI_API_KEY=your_actual_api_key_here
```

### 2. Test Gemini Service
Create `tests/integration/test_gemini.py`:
```python
import pytest
import asyncio
from morag.services.embedding import gemini_service

@pytest.mark.asyncio
async def test_embedding_generation():
    """Test single embedding generation."""
    text = "This is a test document about machine learning and artificial intelligence."
    
    result = await gemini_service.generate_embedding(text)
    
    assert isinstance(result.embedding, list)
    assert len(result.embedding) == 768  # text-embedding-004 dimension
    assert result.model == "text-embedding-004"
    assert result.token_count > 0

@pytest.mark.asyncio
async def test_batch_embedding_generation():
    """Test batch embedding generation."""
    texts = [
        "First document about AI",
        "Second document about machine learning",
        "Third document about data science"
    ]
    
    results = await gemini_service.generate_embeddings_batch(texts, batch_size=2)
    
    assert len(results) == len(texts)
    for result in results:
        assert len(result.embedding) == 768

@pytest.mark.asyncio
async def test_summary_generation():
    """Test text summarization."""
    text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms
    that can learn from and make predictions or decisions based on data. It involves
    training models on large datasets to identify patterns and relationships.
    The goal is to create systems that can automatically improve their performance
    on a specific task through experience.
    """
    
    result = await gemini_service.generate_summary(text, max_length=50)
    
    assert isinstance(result.summary, str)
    assert len(result.summary) > 0
    assert len(result.summary.split()) <= 60  # Allow some flexibility
    assert result.model == "gemini-pro"

@pytest.mark.asyncio
async def test_health_check():
    """Test Gemini service health check."""
    health = await gemini_service.health_check()
    
    assert health["status"] == "healthy"
    assert health["embedding_dimension"] == 768
```

### 3. Test API Integration
```bash
# Start the API
python src/morag/api/main.py

# Test health check
curl http://localhost:8000/health/ready
```

### 4. Manual Testing Script
Create `scripts/test_gemini.py`:
```python
#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.services.embedding import gemini_service

async def main():
    print("Testing Gemini integration...")
    
    # Test embedding
    text = "This is a test document."
    embedding_result = await gemini_service.generate_embedding(text)
    print(f"Embedding dimension: {len(embedding_result.embedding)}")
    
    # Test summary
    long_text = "Machine learning is a powerful technology..." * 10
    summary_result = await gemini_service.generate_summary(long_text)
    print(f"Summary: {summary_result.summary}")
    
    # Test health
    health = await gemini_service.health_check()
    print(f"Health status: {health}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Success Criteria
- [ ] Gemini API key is properly configured
- [ ] Embedding generation works for single texts
- [ ] Batch embedding generation works with rate limiting
- [ ] Text summarization produces quality summaries
- [ ] Health check reports Gemini as healthy
- [ ] Rate limiting prevents API quota issues
- [ ] Error handling works for API failures
- [ ] Integration tests pass

## Next Steps
- Task 05: Document Parser (uses Gemini for summarization)
- Task 15: Vector Storage (uses Gemini embeddings)
- Task 07: Summary Generation (enhanced implementation)
