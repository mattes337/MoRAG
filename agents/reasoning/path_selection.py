"""Path selection agent for reasoning."""

from typing import Type, Optional, List, Dict, Any
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import PathSelectionResult

logger = structlog.get_logger(__name__)


class PathSelectionAgent(BaseAgent[PathSelectionResult]):
    """Agent specialized for selecting optimal reasoning paths."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="path_selection",
            description="Selects optimal paths for multi-hop reasoning",
            prompt=PromptConfig(output_format="json", strict_json=True),
            agent_config={"max_paths": 10, "strategy": "bidirectional"}
        )

    def get_result_type(self) -> Type[PathSelectionResult]:
        return PathSelectionResult

    async def select_paths(
        self,
        start_node: str,
        end_node: str,
        available_paths: Optional[List[Dict[str, Any]]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> PathSelectionResult:
        """Select optimal reasoning paths between nodes.

        Args:
            start_node: Starting point for reasoning
            end_node: Target endpoint for reasoning
            available_paths: Available paths to consider
            constraints: Constraints on path selection
            **kwargs: Additional arguments passed to the agent

        Returns:
            PathSelectionResult containing selected paths
        """
        logger.info(
            "Starting path selection",
            agent=self.__class__.__name__,
            start_node=start_node,
            end_node=end_node,
            available_paths_count=len(available_paths) if available_paths else 0,
            has_constraints=constraints is not None,
            version=self.config.version
        )

        try:
            # Prepare context for path selection
            context_data = {
                "start_node": start_node,
                "end_node": end_node,
                "available_paths": available_paths or [],
                "constraints": constraints or {},
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=f"Select optimal reasoning paths from {start_node} to {end_node}",
                **context_data
            )

            logger.info(
                "Path selection completed",
                agent=self.__class__.__name__,
                selected_paths_count=len(result.selected_paths),
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Path selection failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
