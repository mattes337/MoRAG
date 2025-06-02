import google.genai as genai
from google.genai import types
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
        self.api_key = settings.gemini_api_key
        self.embedding_model = "text-embedding-004"
        self.generation_model = "gemini-2.0-flash-001"
        self.max_retries = 3
        self.base_delay = 1.0

        # Configure Gemini client
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Gemini service initialized",
                       embedding_model=self.embedding_model,
                       generation_model=self.generation_model)
        else:
            self.client = None
            logger.warning("Gemini API key not found - service will not work")
    
    async def generate_embedding(
        self,
        text: str,
        task_type: str = "retrieval_document"
    ) -> EmbeddingResult:
        """Generate embedding for a single text."""
        if not self.client:
            raise ExternalServiceError("Gemini client not initialized", "gemini")

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
        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text
        )

        return EmbeddingResult(
            embedding=response.embeddings[0].values,
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
            
            logger.debug("Processing embedding batch", 
                        batch_num=i//batch_size + 1,
                        batch_size=len(batch),
                        total_texts=len(texts))
            
            # Process batch
            batch_results = []
            for text in batch:
                try:
                    result = await self.generate_embedding(text, task_type)
                    batch_results.append(result)
                except Exception as e:
                    logger.error("Failed to generate embedding in batch", error=str(e))
                    # Create a dummy result to maintain order
                    batch_results.append(EmbeddingResult(
                        embedding=[0.0] * 768,  # Default dimension for text-embedding-004
                        token_count=0,
                        model=self.embedding_model
                    ))
            
            results.extend(batch_results)
            
            # Delay between batches to respect rate limits
            if i + batch_size < len(texts):
                await asyncio.sleep(delay_between_batches)
        
        logger.info("Completed batch embedding generation", 
                   total_texts=len(texts),
                   successful_embeddings=len([r for r in results if r.token_count > 0]))
        
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
            
            logger.debug("Generated summary", 
                        original_length=len(text),
                        summary_length=len(result),
                        model=self.generation_model)
            
            return SummaryResult(
                summary=result,
                token_count=len(result.split()),
                model=self.generation_model
            )
            
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                logger.warning("Rate limit hit for summary generation", error=str(e))
                raise RateLimitError(f"Gemini API rate limit: {str(e)}")
            
            logger.error("Failed to generate summary", error=str(e))
            raise ExternalServiceError(f"Summary generation failed: {str(e)}", "gemini")
    
    def _generate_text_sync(self, prompt: str) -> str:
        """Synchronous text generation."""
        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=prompt
        )
        return response.text
    
    def _build_summary_prompt(self, text: str, max_length: int, style: str) -> str:
        """Build prompt for summary generation."""
        style_instructions = {
            "concise": "Create a concise, factual summary",
            "detailed": "Create a detailed summary that captures key points",
            "bullet": "Create a bullet-point summary of key information",
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

    async def health_check(self) -> Dict[str, Any]:
        """Check if Gemini API is accessible."""
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "error": "Gemini client not initialized",
                    "embedding_model": self.embedding_model,
                    "generation_model": self.generation_model
                }

            # Test with a simple embedding request
            test_result = await self.generate_embedding("Health check test")

            return {
                "status": "healthy",
                "embedding_model": self.embedding_model,
                "generation_model": self.generation_model,
                "embedding_dimension": len(test_result.embedding)
            }

        except Exception as e:
            logger.error("Gemini health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "embedding_model": self.embedding_model,
                "generation_model": self.generation_model
            }

# Global instance
gemini_service = GeminiService()
