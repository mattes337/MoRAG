"""Classification agent."""

from typing import Type, Optional, List
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import ClassificationResult

logger = structlog.get_logger(__name__)


class ClassificationAgent(BaseAgent[ClassificationResult]):
    """Agent specialized for text classification."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="classification",
            description="Classifies text into predefined categories",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[ClassificationResult]:
        return ClassificationResult

    async def classify_content(
        self,
        content: str,
        categories: Optional[List[str]] = None,
        classification_type: Optional[str] = None,
        **kwargs
    ) -> ClassificationResult:
        """Classify content into categories.

        Args:
            content: Input content to classify
            categories: List of possible categories (if None, LLM will generate)
            classification_type: Type of classification (topic, sentiment, intent, etc.)
            **kwargs: Additional arguments passed to the agent

        Returns:
            ClassificationResult containing the classification
        """
        logger.info(
            "Starting content classification",
            agent=self.__class__.__name__,
            content_length=len(content),
            categories_provided=len(categories) if categories else 0,
            classification_type=classification_type,
            version=self.config.version
        )

        try:
            # Prepare context for classification
            context_data = {
                "categories": categories or [],
                "classification_type": classification_type or "general",
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=content,
                result_type=ClassificationResult,
                **context_data
            )

            logger.info(
                "Content classification completed",
                agent=self.__class__.__name__,
                predicted_category=result.category if hasattr(result, 'category') else "unknown",
                confidence_score=result.confidence_score if hasattr(result, 'confidence_score') else 0,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Content classification failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
