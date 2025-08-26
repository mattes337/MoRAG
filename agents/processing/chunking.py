"""Chunking agent."""

from typing import Type, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import ChunkingResult

logger = structlog.get_logger(__name__)


class ChunkingAgent(BaseAgent[ChunkingResult]):
    """Agent specialized for intelligent text chunking."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="chunking",
            description="Performs intelligent semantic chunking of text",
            prompt=PromptConfig(output_format="json", strict_json=True),
            agent_config={"max_chunk_size": 4000, "min_chunk_size": 500}
        )

    def get_result_type(self) -> Type[ChunkingResult]:
        return ChunkingResult

    async def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        strategy: Optional[str] = None,
        **kwargs
    ) -> ChunkingResult:
        """Chunk text into semantically coherent segments.

        Args:
            text: Input text to chunk
            chunk_size: Target chunk size in characters
            strategy: Chunking strategy (semantic, fixed, paragraph, etc.)
            **kwargs: Additional arguments passed to the agent

        Returns:
            ChunkingResult containing the text chunks
        """
        logger.info(
            "Starting text chunking",
            agent=self.__class__.__name__,
            text_length=len(text),
            chunk_size=chunk_size,
            strategy=strategy,
            version=self.config.version
        )

        try:
            # Prepare context for chunking
            context_data = {
                "chunk_size": chunk_size or 4000,
                "strategy": strategy or "semantic",
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=text,
                result_type=ChunkingResult,
                **context_data
            )

            logger.info(
                "Text chunking completed",
                agent=self.__class__.__name__,
                chunks_created=result.chunk_count if hasattr(result, 'chunk_count') else 0,
                avg_chunk_size=result.avg_chunk_size if hasattr(result, 'avg_chunk_size') else 0,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Text chunking failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
