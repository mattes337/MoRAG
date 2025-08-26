"""Default configurations for agents."""

from typing import Dict
from ..base.config import AgentConfig, PromptConfig, ModelConfig, RetryConfig


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
                min_confidence=0.5
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
                min_confidence=0.6
            ),
            agent_config={
                "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "PRODUCT"],
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
    def get_all_configs() -> Dict[str, AgentConfig]:
        """Get all default configurations."""
        return {
            "fact_extraction": DefaultConfigs.get_fact_extraction_config(),
            "entity_extraction": DefaultConfigs.get_entity_extraction_config(),
            "query_analysis": DefaultConfigs.get_query_analysis_config(),
            "summarization": DefaultConfigs.get_summarization_config(),
            "path_selection": DefaultConfigs.get_path_selection_config(),
        }
