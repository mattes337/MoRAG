"""Synthesis agent."""

from typing import Type, Optional, List
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import SynthesisResult

logger = structlog.get_logger(__name__)


class SynthesisAgent(BaseAgent[SynthesisResult]):
    """Agent specialized for synthesizing information from multiple sources."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="synthesis",
            description="Synthesizes information from multiple sources",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[SynthesisResult]:
        return SynthesisResult

    async def synthesize(
        self,
        sources: List[str],
        synthesis_type: Optional[str] = None,
        focus: Optional[str] = None,
        **kwargs
    ) -> SynthesisResult:
        """Synthesize information from multiple sources.

        Args:
            sources: List of source texts to synthesize
            synthesis_type: Type of synthesis (comparative, integrative, thematic, etc.)
            focus: Specific focus or theme for synthesis
            **kwargs: Additional arguments passed to the agent

        Returns:
            SynthesisResult containing the synthesized information
        """
        logger.info(
            "Starting synthesis",
            agent=self.__class__.__name__,
            source_count=len(sources),
            synthesis_type=synthesis_type,
            focus=focus,
            version=self.config.version
        )

        try:
            # Prepare context for synthesis
            context_data = {
                "sources": sources,
                "synthesis_type": synthesis_type or "integrative",
                "focus": focus,
                **kwargs
            }

            # Combine sources into input text
            combined_input = f"Synthesis focus: {focus or 'general'}\n\n"
            for i, source in enumerate(sources, 1):
                combined_input += f"Source {i}:\n{source}\n\n"

            # Execute the agent
            result = await self.execute(
                user_input=combined_input,
                result_type=SynthesisResult,
                **context_data
            )

            logger.info(
                "Synthesis completed",
                agent=self.__class__.__name__,
                synthesis_length=len(result.synthesis) if hasattr(result, 'synthesis') else 0,
                sources_integrated=result.sources_integrated if hasattr(result, 'sources_integrated') else 0,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Synthesis failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
