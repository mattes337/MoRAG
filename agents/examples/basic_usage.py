#!/usr/bin/env python3
"""
Basic usage examples for MoRAG Agents Framework.

This script demonstrates how to use the various agents for different tasks.
"""

import asyncio
import os
from typing import List

# Set up environment
os.environ["GEMINI_API_KEY"] = "your-api-key-here"  # Replace with actual key

from agents import (
    get_agent,
    create_agent,
    AgentConfig,
    PromptConfig,
    list_agents
)


async def fact_extraction_example():
    """Demonstrate fact extraction."""
    print("\n=== Fact Extraction Example ===")

    # Get fact extraction agent
    fact_agent = get_agent("fact_extraction")

    # Sample text with facts
    text = """
    Ginkgo biloba extract (120-240mg daily) has been shown to improve cognitive function
    in elderly patients with mild cognitive impairment. The standardized extract contains
    24% flavonglycosides and 6% terpenlactones. Clinical studies demonstrate significant
    improvements in memory and attention after 12 weeks of treatment.
    """

    try:
        # Extract facts
        result = await fact_agent.extract_facts(
            text=text,
            domain="medical",
            max_facts=10
        )

        print(f"Extracted {result.total_facts} facts:")
        for i, fact in enumerate(result.facts, 1):
            print(f"{i}. {fact.subject} -> {fact.object}")
            print(f"   Approach: {fact.approach}")
            print(f"   Confidence: {fact.confidence}")
            print(f"   Keywords: {', '.join(fact.keywords)}")
            print()

    except Exception as e:
        print(f"Error: {e}")


async def query_analysis_example():
    """Demonstrate query analysis."""
    print("\n=== Query Analysis Example ===")

    # Get query analysis agent
    query_agent = get_agent("query_analysis")

    # Sample queries
    queries = [
        "What are the benefits of meditation for stress reduction?",
        "How does machine learning compare to traditional programming?",
        "Can you explain the process of photosynthesis?",
        "What is Python?"
    ]

    try:
        for query in queries:
            result = await query_agent.analyze_query(query)

            print(f"Query: {query}")
            print(f"Intent: {result.intent}")
            print(f"Type: {result.query_type}")
            print(f"Complexity: {result.complexity}")
            print(f"Entities: {', '.join(result.entities)}")
            print(f"Keywords: {', '.join(result.keywords)}")
            print(f"Confidence: {result.confidence}")
            print()

    except Exception as e:
        print(f"Error: {e}")


async def entity_extraction_example():
    """Demonstrate entity extraction."""
    print("\n=== Entity Extraction Example ===")

    # Get entity extraction agent
    entity_agent = get_agent("entity_extraction")

    # Sample text
    text = """
    Apple Inc. announced a partnership with Stanford University to research
    machine learning applications in healthcare. The collaboration will focus
    on developing AI algorithms for medical diagnosis and treatment planning.
    """

    try:
        result = await entity_agent.extract_entities(
            text=text,
            domain="technology"
        )

        print(f"Extracted {result.total_entities} entities:")
        for entity in result.entities:
            print(f"- {entity.name} ({entity.entity_type})")
            print(f"  Canonical: {entity.canonical_name}")
            print(f"  Confidence: {entity.confidence}")
            print(f"  Context: {entity.context}")
            print()

    except Exception as e:
        print(f"Error: {e}")


async def batch_processing_example():
    """Demonstrate batch processing."""
    print("\n=== Batch Processing Example ===")

    # Get fact extraction agent
    fact_agent = get_agent("fact_extraction")

    # Multiple texts to process
    texts = [
        "Vitamin D (1000-4000 IU daily) supports bone health and immune function.",
        "Regular exercise (150 minutes weekly) reduces cardiovascular disease risk by 30%.",
        "Green tea contains catechins that have antioxidant properties.",
        "Meditation practice (10-20 minutes daily) can reduce stress and anxiety."
    ]

    try:
        # Process all texts in batch
        results = await fact_agent.extract_facts_batch(
            texts=texts,
            domain="health"
        )

        print(f"Processed {len(results)} texts:")
        for i, result in enumerate(results, 1):
            print(f"\nText {i}: {result.total_facts} facts extracted")
            for fact in result.facts:
                print(f"  - {fact.subject}: {fact.object}")

    except Exception as e:
        print(f"Error: {e}")


async def custom_configuration_example():
    """Demonstrate custom agent configuration."""
    print("\n=== Custom Configuration Example ===")

    # Create custom configuration
    custom_config = AgentConfig(
        name="custom_fact_extraction",
        description="Custom fact extraction with high confidence threshold",
        prompt=PromptConfig(
            domain="medical",
            include_examples=True,
            min_confidence=0.8,
            custom_instructions="Focus only on evidence-based medical facts."
        ),
        agent_config={
            "max_facts": 5,
            "filter_generic_advice": True,
            "focus_on_actionable": True
        }
    )

    # Create agent with custom configuration
    custom_agent = create_agent("fact_extraction", custom_config)

    text = """
    Clinical trials show that omega-3 fatty acids (1-3g daily) reduce inflammation
    markers by 15-25%. The EPA and DHA components are particularly effective for
    cardiovascular health. However, more research is needed to establish optimal dosing.
    """

    try:
        result = await custom_agent.extract_facts(text)

        print("Custom configured fact extraction:")
        print(f"Extracted {result.total_facts} high-confidence facts:")
        for fact in result.facts:
            print(f"- {fact.subject} -> {fact.object} (confidence: {fact.confidence})")

    except Exception as e:
        print(f"Error: {e}")


async def multi_agent_workflow_example():
    """Demonstrate using multiple agents in a workflow."""
    print("\n=== Multi-Agent Workflow Example ===")

    # User query
    user_query = "What are the health benefits of green tea?"

    # Sample document
    document = """
    Green tea (Camellia sinensis) contains high levels of catechins, particularly
    epigallocatechin gallate (EGCG). Studies show that consuming 3-5 cups daily
    (300-400mg catechins) can reduce LDL cholesterol by 5-10% and improve insulin
    sensitivity. The antioxidant properties may also reduce cancer risk by 20-30%
    according to meta-analyses of population studies.
    """

    try:
        # Step 1: Analyze the query
        query_agent = get_agent("query_analysis")
        query_analysis = await query_agent.analyze_query(user_query)

        print(f"Query Analysis:")
        print(f"- Intent: {query_analysis.intent}")
        print(f"- Entities: {', '.join(query_analysis.entities)}")
        print(f"- Keywords: {', '.join(query_analysis.keywords)}")

        # Step 2: Extract relevant facts from document
        fact_agent = get_agent("fact_extraction")
        facts = await fact_agent.extract_facts(
            text=document,
            domain="health",
            query_context=user_query
        )

        print(f"\nExtracted Facts ({facts.total_facts}):")
        for fact in facts.facts:
            print(f"- {fact.subject}: {fact.solution or fact.object}")

        # Step 3: Generate summary
        summary_agent = get_agent("summarization")
        summary = await summary_agent.execute(
            f"Summarize the health benefits of green tea based on these facts: {facts.facts}"
        )

        print(f"\nGenerated Summary:")
        print(summary.summary if hasattr(summary, 'summary') else str(summary))

    except Exception as e:
        print(f"Error in workflow: {e}")


async def main():
    """Run all examples."""
    print("MoRAG Agents Framework - Basic Usage Examples")
    print("=" * 50)

    # List available agents
    print("\nAvailable Agents:")
    agents = list_agents()
    for name, class_name in agents.items():
        print(f"- {name}: {class_name}")

    # Run examples
    await fact_extraction_example()
    await query_analysis_example()
    await entity_extraction_example()
    await batch_processing_example()
    await custom_configuration_example()
    await multi_agent_workflow_example()

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your-api-key-here":
        print("Please set your GEMINI_API_KEY environment variable before running examples.")
        print("export GEMINI_API_KEY=your_actual_api_key")
        exit(1)

    # Run examples
    asyncio.run(main())
