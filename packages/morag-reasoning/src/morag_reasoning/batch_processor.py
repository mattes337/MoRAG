"""Batch processing utilities for LLM operations."""

import logging
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .llm import LLMClient

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchItem(Generic[T]):
    """Represents an item to be processed in a batch."""
    data: T
    prompt: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchResult(Generic[T, R]):
    """Represents the result of batch processing."""
    item: T
    result: R
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchProcessor(ABC, Generic[T, R]):
    """Abstract base class for batch processing with LLM."""

    def __init__(self, llm_client: LLMClient, batch_size: Optional[int] = None):
        """Initialize batch processor.

        Args:
            llm_client: LLM client for processing
            batch_size: Override default batch size
        """
        self.llm_client = llm_client
        self.batch_size = batch_size or llm_client.config.batch_size
        self.logger = logger

    @abstractmethod
    def create_prompt(self, item: T) -> str:
        """Create a prompt for processing an item.

        Args:
            item: Item to create prompt for

        Returns:
            Prompt string
        """

    @abstractmethod
    def parse_response(self, response: str, item: T) -> R:
        """Parse LLM response for an item.

        Args:
            response: LLM response
            item: Original item

        Returns:
            Parsed result
        """

    async def process_batch(
        self,
        items: List[T],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[BatchResult[T, R]]:
        """Process a batch of items using LLM with memory-aware streaming.

        Args:
            items: Items to process
            max_tokens: Maximum tokens per response
            temperature: Temperature for generation

        Returns:
            List of batch results
        """
        if not items:
            return []

        # Memory-aware streaming processing
        MAX_MEMORY_MB = 500  # Configure based on available resources
        current_memory_usage = 0
        all_results = []

        # Process items in smaller chunks to prevent memory accumulation
        chunk_size = min(self.batch_size, 50)  # Conservative chunk size

        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            chunk_memory = sum(len(str(item)) for item in chunk) / (1024 * 1024)

            # If memory threshold exceeded, flush accumulated results
            if current_memory_usage + chunk_memory > MAX_MEMORY_MB:
                await self._flush_batch_results(all_results)
                current_memory_usage = 0

            # Create prompts for current chunk
            batch_items = []
            for item in chunk:
                try:
                    prompt = self.create_prompt(item)
                    batch_items.append(BatchItem(data=item, prompt=prompt))
                except Exception as e:
                    self.logger.error(f"Failed to create prompt for item: {str(e)}")
                    # Add error result
                    batch_items.append(BatchItem(
                        data=item,
                        prompt="",
                        metadata={"prompt_error": str(e)}
                    ))

            # Extract prompts for LLM processing
            prompts = [item.prompt for item in batch_items if item.prompt]

            if not prompts:
                # All prompts failed for this chunk, add error results
                chunk_results = [
                    BatchResult(
                        item=batch_item.data,
                        result=None,
                        success=False,
                        error_message=batch_item.metadata.get("prompt_error", "Failed to create prompt")
                    )
                    for batch_item in batch_items
                ]
                all_results.extend(chunk_results)
                continue

            try:
                # Process prompts using LLM batch processing
                responses = await self.llm_client.generate_batch(
                    prompts, max_tokens, temperature, len(prompts)
                )

                # Parse responses and create results for this chunk
                chunk_results = []
                for i, batch_item in enumerate(batch_items):
                    if not batch_item.prompt:
                        # This item had a prompt creation error
                        chunk_results.append(BatchResult(
                            item=batch_item.data,
                            result=None,
                            success=False,
                            error_message=batch_item.metadata.get("prompt_error", "Failed to create prompt")
                        ))
                        continue

                    try:
                        # Find corresponding response
                        response_index = len([item for item in batch_items[:i] if item.prompt])
                        if response_index < len(responses):
                            response = responses[response_index]
                            parsed_result = self.parse_response(response, batch_item.data)
                            chunk_results.append(BatchResult(
                                item=batch_item.data,
                                result=parsed_result,
                                success=True
                            ))
                        else:
                            chunk_results.append(BatchResult(
                                item=batch_item.data,
                                result=None,
                                success=False,
                                error_message="No response received from LLM"
                            ))
                    except Exception as e:
                        self.logger.error(f"Failed to parse response for item: {str(e)}")
                        chunk_results.append(BatchResult(
                            item=batch_item.data,
                            result=None,
                            success=False,
                            error_message=f"Failed to parse response: {str(e)}"
                        ))

                all_results.extend(chunk_results)
                current_memory_usage += chunk_memory

            except Exception as e:
                self.logger.error(f"Batch LLM processing failed for chunk: {str(e)}")
                # Return error results for all items in this chunk
                chunk_results = [
                    BatchResult(
                        item=batch_item.data,
                        result=None,
                        success=False,
                        error_message=f"Batch processing failed: {str(e)}"
                    )
                    for batch_item in batch_items
                ]
                all_results.extend(chunk_results)

        return all_results

    async def _flush_batch_results(self, results: List[BatchResult[T, R]]) -> None:
        """Flush accumulated results to free memory.

        This method can be extended to persist results to disk or database
        for very large batch processing scenarios.

        Args:
            results: List of accumulated results to flush
        """
        if results:
            self.logger.debug(f"Flushing {len(results)} accumulated batch results to free memory")
            # In a more advanced implementation, results could be written to disk
            # or streamed to a database here to prevent memory accumulation
            # For now, we rely on the caller to handle the cleared results


class TextAnalysisBatchProcessor(BatchProcessor[str, Dict[str, Any]]):
    """Batch processor for text analysis tasks."""

    def __init__(self, llm_client: LLMClient, analysis_type: str = "general", batch_size: Optional[int] = None):
        """Initialize text analysis batch processor.

        Args:
            llm_client: LLM client for processing
            analysis_type: Type of analysis (entity_extraction, relation_extraction, summarization, etc.)
            batch_size: Override default batch size
        """
        super().__init__(llm_client, batch_size)
        self.analysis_type = analysis_type

    def create_prompt(self, text: str) -> str:
        """Create analysis prompt for text."""
        if self.analysis_type == "entity_extraction":
            return f"""Extract entities from the following text. Return a JSON list of entities with name, type, and confidence:

Text: {text}

Return format: [{{"name": "entity_name", "type": "ENTITY_TYPE", "confidence": 0.9}}]"""

        elif self.analysis_type == "relation_extraction":
            return f"""Extract relations from the following text. Return a JSON list of relations:

Text: {text}

Return format: [{{"source": "entity1", "target": "entity2", "relation": "RELATION_TYPE", "confidence": 0.9}}]"""

        elif self.analysis_type == "summarization":
            return f"""Provide a concise summary of the following text:

Text: {text}

Summary:"""

        else:
            return f"""Analyze the following text for {self.analysis_type}:

Text: {text}

Analysis:"""

    def parse_response(self, response: str, text: str) -> Dict[str, Any]:
        """Parse analysis response."""
        try:
            if self.analysis_type in ["entity_extraction", "relation_extraction"]:
                # Try to parse as JSON
                import json
                return {"analysis": json.loads(response.strip()), "original_text": text}
            else:
                return {"analysis": response.strip(), "original_text": text}
        except Exception as e:
            self.logger.warning(f"Failed to parse JSON response, returning as text: {str(e)}")
            return {"analysis": response.strip(), "original_text": text, "parse_error": str(e)}


class DocumentChunkBatchProcessor(BatchProcessor[Dict[str, Any], Dict[str, Any]]):
    """Batch processor for document chunks."""

    def __init__(self, llm_client: LLMClient, processing_type: str = "extraction", batch_size: Optional[int] = None):
        """Initialize document chunk batch processor.

        Args:
            llm_client: LLM client for processing
            processing_type: Type of processing (extraction, analysis, summarization)
            batch_size: Override default batch size
        """
        super().__init__(llm_client, batch_size)
        self.processing_type = processing_type

    def create_prompt(self, chunk: Dict[str, Any]) -> str:
        """Create processing prompt for document chunk."""
        text = chunk.get("text", "")
        chunk_id = chunk.get("id", "unknown")
        document_id = chunk.get("document_id", "unknown")

        if self.processing_type == "extraction":
            return f"""Extract entities and relations from this document chunk:

Document ID: {document_id}
Chunk ID: {chunk_id}
Text: {text}

Return JSON with entities and relations:
{{"entities": [{{"name": "...", "type": "...", "confidence": 0.9}}], "relations": [{{"source": "...", "target": "...", "relation": "...", "confidence": 0.9}}]}}"""

        elif self.processing_type == "summarization":
            return f"""Summarize this document chunk:

Document ID: {document_id}
Chunk ID: {chunk_id}
Text: {text}

Summary:"""

        else:
            return f"""Analyze this document chunk for {self.processing_type}:

Document ID: {document_id}
Chunk ID: {chunk_id}
Text: {text}

Analysis:"""

    def parse_response(self, response: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Parse chunk processing response."""
        try:
            if self.processing_type == "extraction":
                import json
                parsed = json.loads(response.strip())
                return {
                    "chunk_id": chunk.get("id"),
                    "document_id": chunk.get("document_id"),
                    "entities": parsed.get("entities", []),
                    "relations": parsed.get("relations", []),
                    "original_chunk": chunk
                }
            else:
                return {
                    "chunk_id": chunk.get("id"),
                    "document_id": chunk.get("document_id"),
                    "result": response.strip(),
                    "original_chunk": chunk
                }
        except Exception as e:
            self.logger.warning(f"Failed to parse response, returning as text: {str(e)}")
            return {
                "chunk_id": chunk.get("id"),
                "document_id": chunk.get("document_id"),
                "result": response.strip(),
                "original_chunk": chunk,
                "parse_error": str(e)
            }


async def batch_llm_calls(
    llm_client: LLMClient,
    items: List[T],
    prompt_creator: Callable[[T], str],
    response_parser: Callable[[str, T], R],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    batch_size: Optional[int] = None
) -> List[BatchResult[T, R]]:
    """Utility function to easily batch LLM calls.

    Args:
        llm_client: LLM client for processing
        items: Items to process
        prompt_creator: Function to create prompt from item
        response_parser: Function to parse response for item
        max_tokens: Maximum tokens per response
        temperature: Temperature for generation
        batch_size: Override default batch size

    Returns:
        List of batch results
    """
    class CustomBatchProcessor(BatchProcessor[T, R]):
        def create_prompt(self, item: T) -> str:
            return prompt_creator(item)

        def parse_response(self, response: str, item: T) -> R:
            return response_parser(response, item)

    processor = CustomBatchProcessor(llm_client, batch_size)
    return await processor.process_batch(items, max_tokens, temperature)


async def batch_text_analysis(
    llm_client: LLMClient,
    texts: List[str],
    analysis_type: str = "general",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    batch_size: Optional[int] = None
) -> List[BatchResult[str, Dict[str, Any]]]:
    """Convenience function for batch text analysis.

    Args:
        llm_client: LLM client for processing
        texts: Texts to analyze
        analysis_type: Type of analysis (entity_extraction, relation_extraction, summarization, etc.)
        max_tokens: Maximum tokens per response
        temperature: Temperature for generation
        batch_size: Override default batch size

    Returns:
        List of batch results
    """
    processor = TextAnalysisBatchProcessor(llm_client, analysis_type, batch_size)
    return await processor.process_batch(texts, max_tokens, temperature)


async def batch_document_chunks(
    llm_client: LLMClient,
    chunks: List[Dict[str, Any]],
    processing_type: str = "extraction",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    batch_size: Optional[int] = None
) -> List[BatchResult[Dict[str, Any], Dict[str, Any]]]:
    """Convenience function for batch document chunk processing.

    Args:
        llm_client: LLM client for processing
        chunks: Document chunks to process
        processing_type: Type of processing (extraction, analysis, summarization)
        max_tokens: Maximum tokens per response
        temperature: Temperature for generation
        batch_size: Override default batch size

    Returns:
        List of batch results
    """
    processor = DocumentChunkBatchProcessor(llm_client, processing_type, batch_size)
    return await processor.process_batch(chunks, max_tokens, temperature)
