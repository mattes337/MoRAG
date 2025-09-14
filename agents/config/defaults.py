"""Default configurations for agents."""

from typing import Dict
from ..base.config import AgentConfig, PromptConfig, ModelConfig, RetryConfig
from .entity_types import get_agent_entity_types


class DefaultConfigs:
    """Default configurations for all agents."""
    
    @staticmethod
    def get_fact_extraction_config() -> AgentConfig:
        """Get default configuration for fact extraction agent."""
        return AgentConfig(
            name="fact_extraction",
            description="Extracts structured facts from text content",
            model=ModelConfig(
                provider="gemini",
                model="gemini-1.5-flash",
                temperature=0.1,
                max_tokens=4000
            ),
            retry=RetryConfig(
                max_retries=3,
                base_delay=1.0,
                max_delay=60.0
            ),
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                include_context=True,
                output_format="json",
                strict_json=True,
                include_confidence=True,
                min_confidence=0.3
            ),
            agent_config={
                "max_facts": 20,
                "focus_on_actionable": True,
                "include_technical_details": True,
                "filter_generic_advice": True
            }
        )
    
    @staticmethod
    def get_entity_extraction_config() -> AgentConfig:
        """Get default configuration for entity extraction agent."""
        return AgentConfig(
            name="entity_extraction",
            description="Extracts named entities from text content",
            model=ModelConfig(
                provider="gemini",
                model="gemini-1.5-flash",
                temperature=0.1,
                max_tokens=4000
            ),
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
                min_confidence=0.4
            ),
            agent_config={
                "entity_types": get_agent_entity_types("entity_extraction"),
                "include_offsets": True,
                "normalize_entities": True,
                "min_entity_length": 2
            }
        )
    
    @staticmethod
    def get_query_analysis_config() -> AgentConfig:
        """Get default configuration for query analysis agent."""
        return AgentConfig(
            name="query_analysis",
            description="Analyzes user queries to extract intent and entities",
            model=ModelConfig(
                provider="gemini",
                model="gemini-1.5-flash",
                temperature=0.1,
                max_tokens=2000
            ),
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
                min_confidence=0.7
            ),
            agent_config={
                "extract_entities": True,
                "extract_keywords": True,
                "analyze_complexity": True,
                "detect_temporal_context": True
            }
        )
    
    @staticmethod
    def get_summarization_config() -> AgentConfig:
        """Get default configuration for summarization agent."""
        return AgentConfig(
            name="summarization",
            description="Generates high-quality summaries of text content",
            model=ModelConfig(
                provider="gemini",
                model="gemini-1.5-flash",
                temperature=0.2,
                max_tokens=2000
            ),
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True
            ),
            agent_config={
                "max_summary_length": 1000,
                "summary_type": "abstractive",
                "include_key_points": True,
                "compression_ratio": 0.3
            }
        )
    
    @staticmethod
    def get_path_selection_config() -> AgentConfig:
        """Get default configuration for path selection agent."""
        return AgentConfig(
            name="path_selection",
            description="Selects optimal paths for multi-hop reasoning",
            model=ModelConfig(
                provider="gemini",
                model="gemini-1.5-flash",
                temperature=0.1,
                max_tokens=3000
            ),
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True
            ),
            agent_config={
                "max_paths": 10,
                "strategy": "bidirectional",
                "min_relevance_threshold": 0.6,
                "consider_path_length": True
            }
        )
    
    @staticmethod
    def get_relation_extraction_config() -> AgentConfig:
        """Get default configuration for relation extraction agent."""
        return AgentConfig(
            name="relation_extraction",
            description="Extracts semantic relations between entities",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
                include_confidence=True,
            ),
        )

    @staticmethod
    def get_keyword_extraction_config() -> AgentConfig:
        """Get default configuration for keyword extraction agent."""
        return AgentConfig(
            name="keyword_extraction",
            description="Extracts relevant keywords from text",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_content_analysis_config() -> AgentConfig:
        """Get default configuration for content analysis agent."""
        return AgentConfig(
            name="content_analysis",
            description="Analyzes content structure, topics, and complexity",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_sentiment_analysis_config() -> AgentConfig:
        """Get default configuration for sentiment analysis agent."""
        return AgentConfig(
            name="sentiment_analysis",
            description="Analyzes sentiment and emotions in text",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_topic_analysis_config() -> AgentConfig:
        """Get default configuration for topic analysis agent."""
        return AgentConfig(
            name="topic_analysis",
            description="Analyzes and models topics in text",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_decision_making_config() -> AgentConfig:
        """Get default configuration for decision making agent."""
        return AgentConfig(
            name="decision_making",
            description="Makes decisions and evaluates options",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_context_analysis_config() -> AgentConfig:
        """Get default configuration for context analysis agent."""
        return AgentConfig(
            name="context_analysis",
            description="Analyzes context and determines relevance",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_explanation_config() -> AgentConfig:
        """Get default configuration for explanation agent."""
        return AgentConfig(
            name="explanation",
            description="Generates clear explanations for complex topics",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_synthesis_config() -> AgentConfig:
        """Get default configuration for synthesis agent."""
        return AgentConfig(
            name="synthesis",
            description="Synthesizes information from multiple sources",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_chunking_config() -> AgentConfig:
        """Get default configuration for chunking agent."""
        return AgentConfig(
            name="chunking",
            description="Performs intelligent semantic chunking of text",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
            agent_config={
                "max_chunk_size": 4000,
                "min_chunk_size": 500,
                "overlap": 100
            }
        )

    @staticmethod
    def get_classification_config() -> AgentConfig:
        """Get default configuration for classification agent."""
        return AgentConfig(
            name="classification",
            description="Classifies text into predefined categories",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_validation_config() -> AgentConfig:
        """Get default configuration for validation agent."""
        return AgentConfig(
            name="validation",
            description="Validates content quality and accuracy",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_filtering_config() -> AgentConfig:
        """Get default configuration for filtering agent."""
        return AgentConfig(
            name="filtering",
            description="Filters content based on quality and relevance criteria",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_reasoning_config() -> AgentConfig:
        """Get default configuration for reasoning agent."""
        return AgentConfig(
            name="reasoning",
            description="Performs logical reasoning and inference",
            prompt=PromptConfig(
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_response_generation_config() -> AgentConfig:
        """Get default configuration for response generation agent."""
        return AgentConfig(
            name="response_generation",
            description="Generates comprehensive responses to user queries",
            prompt=PromptConfig(
                output_format="json",
                strict_json=True,
            ),
        )

    @staticmethod
    def get_all_configs() -> Dict[str, AgentConfig]:
        """Get all default configurations."""
        return {
            "fact_extraction": DefaultConfigs.get_fact_extraction_config(),
            "entity_extraction": DefaultConfigs.get_entity_extraction_config(),
            "query_analysis": DefaultConfigs.get_query_analysis_config(),
            "summarization": DefaultConfigs.get_summarization_config(),
            "path_selection": DefaultConfigs.get_path_selection_config(),
            "relation_extraction": DefaultConfigs.get_relation_extraction_config(),
            "keyword_extraction": DefaultConfigs.get_keyword_extraction_config(),
            "content_analysis": DefaultConfigs.get_content_analysis_config(),
            "sentiment_analysis": DefaultConfigs.get_sentiment_analysis_config(),
            "topic_analysis": DefaultConfigs.get_topic_analysis_config(),
            "decision_making": DefaultConfigs.get_decision_making_config(),
            "context_analysis": DefaultConfigs.get_context_analysis_config(),
            "explanation": DefaultConfigs.get_explanation_config(),
            "synthesis": DefaultConfigs.get_synthesis_config(),
            "chunking": DefaultConfigs.get_chunking_config(),
            "classification": DefaultConfigs.get_classification_config(),
            "validation": DefaultConfigs.get_validation_config(),
            "filtering": DefaultConfigs.get_filtering_config(),
            "reasoning": DefaultConfigs.get_reasoning_config(),
            "response_generation": DefaultConfigs.get_response_generation_config(),
            # Aliases for incorrectly converted names
            "relationextraction": DefaultConfigs.get_relation_extraction_config(),
            "keywordextraction": DefaultConfigs.get_keyword_extraction_config(),
            "topicanalysis": DefaultConfigs.get_topic_analysis_config(),
            "pathselection": DefaultConfigs.get_path_selection_config(),
            "contextanalysis": DefaultConfigs.get_context_analysis_config(),
            "queryanalysis": DefaultConfigs.get_query_analysis_config(),
            "factextraction": DefaultConfigs.get_fact_extraction_config(),
            "sentimentanalysis": DefaultConfigs.get_sentiment_analysis_config(),
            "decisionmaking": DefaultConfigs.get_decision_making_config(),
            "contentanalysis": DefaultConfigs.get_content_analysis_config(),
            "entityextraction": DefaultConfigs.get_entity_extraction_config(),
            "responsegeneration": DefaultConfigs.get_response_generation_config(),
        }

    @staticmethod
    def get_config_for_agent(agent_name: str) -> AgentConfig:
        """Get default configuration for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Default configuration for the agent

        Raises:
            ValueError: If agent configuration is not found
        """
        all_configs = DefaultConfigs.get_all_configs()
        if agent_name not in all_configs:
            raise ValueError(f"No default configuration found for agent: {agent_name}")
        return all_configs[agent_name]


# Convenience function for agents to use
def get_default_config(agent_name: str) -> AgentConfig:
    """Get default configuration for an agent.

    Args:
        agent_name: Name of the agent

    Returns:
        Default configuration for the agent
    """
    return DefaultConfigs.get_config_for_agent(agent_name)
