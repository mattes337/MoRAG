"""Test configuration and fixtures for morag-reasoning tests."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict, Any

from morag_reasoning.llm import LLMClient, LLMConfig
from morag_reasoning.path_selection import PathSelectionAgent, ReasoningPathFinder
from morag_reasoning.iterative_retrieval import IterativeRetriever, RetrievalContext
from morag_graph.operations import GraphPath
from morag_graph.models import Entity, Relation


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def llm_config():
    """Create a test LLM configuration."""
    return LLMConfig(
        provider="gemini",
        model="gemini-1.5-flash",
        api_key="test-api-key",
        temperature=0.1,
        max_tokens=1000,
        max_retries=3
    )


@pytest.fixture
def mock_llm_client(llm_config):
    """Create a mock LLM client."""
    client = LLMClient(llm_config)

    # Mock the generate method
    async def mock_generate(prompt, **kwargs) -> str:
        # Handle both string prompts and message lists
        if isinstance(prompt, list):
            # For message lists, check the content of messages
            prompt_text = " ".join(msg.get("content", "") for msg in prompt if isinstance(msg, dict))
        else:
            prompt_text = str(prompt)

        if "path selection" in prompt_text.lower():
            return '''
            {
              "selected_paths": [
                {
                  "path_id": 1,
                  "relevance_score": 8.5,
                  "confidence": 9.0,
                  "reasoning": "This path directly connects the query entities."
                },
                {
                  "path_id": 2,
                  "relevance_score": 7.2,
                  "confidence": 8.0,
                  "reasoning": "This path provides relevant context."
                }
              ]
            }
            '''
        elif "context analysis" in prompt_text.lower():
            return '''
            {
              "is_sufficient": false,
              "confidence": 6.5,
              "reasoning": "Context provides basic information but lacks specific details.",
              "gaps": [
                {
                  "gap_type": "missing_entity",
                  "description": "Need more information about entity X",
                  "entities_needed": ["entity_x"],
                  "priority": 0.8
                }
              ],
              "suggested_queries": ["What is the relationship between X and Y?"]
            }
            '''
        else:
            return "Mock LLM response"

    client.generate = AsyncMock(side_effect=mock_generate)
    client.generate_from_messages = AsyncMock(side_effect=mock_generate)
    return client


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(id="ent_1", name="Apple Inc.", type="ORG", properties={"industry": "technology"}),
        Entity(id="ent_2", name="Steve Jobs", type="PERSON", properties={"role": "founder"}),
        Entity(id="ent_3", name="iPhone", type="PRODUCT", properties={"category": "smartphone"}),
        Entity(id="ent_4", name="Cupertino", type="LOCATION", properties={"state": "California"}),
    ]


@pytest.fixture
def sample_relations():
    """Create sample relations for testing."""
    return [
        Relation(id="rel_1", type="FOUNDED_BY", source_entity_id="ent_1", target_entity_id="ent_2", properties={}),
        Relation(id="rel_2", type="DEVELOPS", source_entity_id="ent_1", target_entity_id="ent_3", properties={}),
        Relation(id="rel_3", type="LOCATED_IN", source_entity_id="ent_1", target_entity_id="ent_4", properties={}),
    ]


@pytest.fixture
def sample_graph_paths(sample_entities, sample_relations):
    """Create sample graph paths for testing."""
    return [
        GraphPath(
            entities=[sample_entities[0], sample_entities[1]],
            relations=[sample_relations[0]]
        ),
        GraphPath(
            entities=[sample_entities[0], sample_entities[2]],
            relations=[sample_relations[1]]
        ),
        GraphPath(
            entities=[sample_entities[0], sample_entities[3]],
            relations=[sample_relations[2]]
        ),
        GraphPath(
            entities=[sample_entities[0], sample_entities[1], sample_entities[2]],
            relations=[sample_relations[0], sample_relations[1]]
        ),
    ]


@pytest.fixture
def mock_graph_engine():
    """Create a mock graph engine."""
    engine = MagicMock()

    # Mock async methods
    engine.get_entity = AsyncMock(return_value=None)
    engine.get_entity_details = AsyncMock(return_value={"type": "ORG", "name": "Test Entity"})
    engine.get_relations_by_type = AsyncMock(return_value=[])
    engine.find_neighbors = AsyncMock(return_value=[])
    engine.find_shortest_path = AsyncMock(return_value=None)
    engine.traverse = AsyncMock(return_value={"paths": []})

    return engine


@pytest.fixture
def mock_vector_retriever():
    """Create a mock vector retriever."""
    retriever = MagicMock()

    # Mock async methods
    retriever.search = AsyncMock(return_value=[
        {"id": "doc1", "content": "Sample document content", "score": 0.8},
        {"id": "doc2", "content": "Another document", "score": 0.7}
    ])
    retriever.retrieve = AsyncMock(return_value=[
        {"id": "doc1", "content": "Sample document content", "score": 0.8}
    ])

    return retriever


@pytest.fixture
def path_selection_agent(mock_llm_client):
    """Create a path selection agent for testing."""
    return PathSelectionAgent(mock_llm_client, max_paths=5)


@pytest.fixture
def reasoning_path_finder(mock_graph_engine, path_selection_agent):
    """Create a reasoning path finder for testing."""
    return ReasoningPathFinder(mock_graph_engine, path_selection_agent)


@pytest.fixture
def iterative_retriever(mock_llm_client, mock_graph_engine, mock_vector_retriever):
    """Create an iterative retriever for testing."""
    return IterativeRetriever(
        llm_client=mock_llm_client,
        graph_engine=mock_graph_engine,
        vector_retriever=mock_vector_retriever,
        max_iterations=3,
        sufficiency_threshold=0.8
    )


@pytest.fixture
def sample_retrieval_context():
    """Create a sample retrieval context for testing."""
    return RetrievalContext(
        entities={"Apple Inc.": {"type": "ORG", "name": "Apple Inc."}},
        relations=[
            {"subject": "Apple Inc.", "predicate": "FOUNDED_BY", "object": "Steve Jobs"}
        ],
        documents=[
            {"id": "doc1", "content": "Apple Inc. is a technology company"}
        ],
        metadata={"source": "test"}
    )


@pytest.fixture
def complex_query():
    """Create a complex multi-hop query for testing."""
    return "How are Apple's AI research efforts related to their partnership with universities?"


@pytest.fixture
def simple_query():
    """Create a simple query for testing."""
    return "What products does Apple make?"
