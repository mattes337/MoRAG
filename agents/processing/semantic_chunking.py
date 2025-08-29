"""Semantic chunking agent for intelligent text segmentation."""

from typing import Type, List, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import SemanticChunkingResult

logger = structlog.get_logger(__name__)


class SemanticChunkingAgent(BaseAgent[SemanticChunkingResult]):
    """Agent specialized for intelligent semantic chunking of text."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="semantic_chunking",
            description="Intelligently chunks text based on semantic boundaries",
            prompt=PromptConfig(output_format="json", strict_json=True),
            agent_config={
                "min_confidence": 0.6,
                "max_chunk_size": 4000,
                "min_chunk_size": 500,
                "strategy": "semantic"
            }
        )

    def get_result_type(self) -> Type[SemanticChunkingResult]:
        return SemanticChunkingResult

    async def chunk_text(
        self,
        text: str,
        max_chunk_size: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
        strategy: Optional[str] = None,
        **kwargs
    ) -> SemanticChunkingResult:
        """Chunk text using semantic analysis.

        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            strategy: Chunking strategy ("semantic", "hybrid", "size-based")
            **kwargs: Additional arguments passed to the agent

        Returns:
            SemanticChunkingResult containing the chunks and boundaries
        """
        logger.info(
            "Starting semantic chunking",
            agent=self.__class__.__name__,
            text_length=len(text),
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            strategy=strategy or "semantic",
            version=self.config.version
        )

        try:
            # Prepare context for chunking
            context = {
                "max_chunk_size": max_chunk_size or 4000,
                "min_chunk_size": min_chunk_size or 500,
                "strategy": strategy or "semantic",
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=text,
                **context
            )

            logger.info(
                "Semantic chunking completed",
                agent=self.__class__.__name__,
                num_chunks=len(result.chunks) if hasattr(result, 'chunks') else 0,
                num_boundaries=len(result.boundaries) if hasattr(result, 'boundaries') else 0,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Semantic chunking failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
