"""Filtering agent."""

from typing import Type, Optional, List, Dict, Any
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import FilteringResult

logger = structlog.get_logger(__name__)


class FilteringAgent(BaseAgent[FilteringResult]):
    """Agent specialized for content filtering."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="filtering",
            description="Filters content based on quality and relevance criteria",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[FilteringResult]:
        return FilteringResult

    async def filter_content(
        self,
        content_items: List[Dict[str, Any]],
        criteria: Optional[Dict[str, Any]] = None,
        filter_type: str = "relevance",
        **kwargs
    ) -> FilteringResult:
        """Filter content based on specified criteria.

        Args:
            content_items: List of content items to filter
            criteria: Filtering criteria and thresholds
            filter_type: Type of filtering (relevance, quality, etc.)
            **kwargs: Additional arguments passed to the agent

        Returns:
            FilteringResult containing filtered content
        """
        logger.info(
            "Starting content filtering",
            agent=self.__class__.__name__,
            items_count=len(content_items),
            filter_type=filter_type,
            has_criteria=criteria is not None,
            version=self.config.version
        )

        try:
            # Prepare context for filtering
            context_data = {
                "content_items": content_items,
                "criteria": criteria or {},
                "filter_type": filter_type,
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=f"Filter {len(content_items)} content items using {filter_type} criteria",
                result_type=FilteringResult,
                **context_data
            )

            logger.info(
                "Content filtering completed",
                agent=self.__class__.__name__,
                original_count=len(content_items),
                filtered_count=len(result.filtered_items),
                filter_type=filter_type,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Content filtering failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
