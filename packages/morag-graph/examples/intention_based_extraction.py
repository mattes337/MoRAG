#!/usr/bin/env python3
"""
Example demonstrating intention-based entity and relation extraction.

This example shows how the new intention-based extraction system works:
1. Generates a document intention summary
2. Uses the intention to guide entity and relation type abstraction
3. Produces more abstract, domain-appropriate types

Usage:
    python intention_based_extraction.py --api-key YOUR_GEMINI_API_KEY
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_graph.extraction.entity_extractor import EntityExtractor
from morag_graph.extraction.relation_extractor import RelationExtractor
from morag_graph.extraction.base import LLMConfig


async def generate_intention(content: str, api_key: str, model: str = "gemini-1.5-flash") -> str:
    """Generate document intention summary."""
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        
        prompt = f"""
Analyze the following document and provide a concise intention summary that captures the document's primary purpose and domain.

The intention should be a single sentence that describes what the document aims to achieve or communicate.

Examples:
- For medical content: "Heal the pineal gland for spiritual enlightenment"
- For organizational documents: "Document explaining the structure of the organization/company"
- For technical guides: "Guide for implementing software architecture patterns"
- For educational content: "Teach fundamental concepts of machine learning"

Document content:
{content[:2000]}...

Provide only the intention summary (maximum 200 characters):
"""
        
        response = model_instance.generate_content(prompt)
        intention = response.text.strip()
        
        if len(intention) > 200:
            intention = intention[:197] + "..."
            
        return intention
        
    except Exception as e:
        print(f"Failed to generate intention: {e}")
        return "General document analysis"


async def extract_with_intention(content: str, api_key: str, model: str = "gemini-1.5-flash"):
    """Demonstrate intention-based extraction."""
    
    print("🎯 INTENTION-BASED EXTRACTION DEMO")
    print("=" * 50)
    
    # Step 1: Generate document intention
    print("\n📋 Step 1: Generating document intention...")
    intention = await generate_intention(content, api_key, model)
    print(f"💡 Document intention: {intention}")
    
    # Step 2: Configure extractors
    llm_config = LLMConfig(
        provider="gemini",
        model=model,
        api_key=api_key,
        temperature=0.1
    )
    
    entity_extractor = EntityExtractor(config=llm_config)
    relation_extractor = RelationExtractor(config=llm_config)
    
    # Step 3: Extract entities with intention context
    print("\n🏷️  Step 2: Extracting entities with intention context...")
    entities = await entity_extractor.extract(content, intention=intention)
    
    print(f"📊 Found {len(entities)} entities:")
    for entity in entities:
        print(f"  • {entity.name} ({entity.type}) - {entity.confidence:.2f}")
    
    # Step 4: Extract relations with intention context
    print("\n🔗 Step 3: Extracting relations with intention context...")
    relations = await relation_extractor.extract(content, entities=entities, intention=intention)
    
    print(f"📊 Found {len(relations)} relations:")
    for relation in relations:
        source_name = relation.attributes.get("source_entity_name", "Unknown")
        target_name = relation.attributes.get("target_entity_name", "Unknown")
        print(f"  • {source_name} --[{relation.type}]--> {target_name} ({relation.confidence:.2f})")
    
    # Step 5: Show type abstraction benefits
    print("\n🎨 Step 4: Type abstraction analysis...")
    entity_types = set(entity.type for entity in entities)
    relation_types = set(relation.type for relation in relations)
    
    print(f"📈 Entity types used: {len(entity_types)}")
    for etype in sorted(entity_types):
        count = sum(1 for e in entities if e.type == etype)
        print(f"  • {etype}: {count} entities")
    
    print(f"📈 Relation types used: {len(relation_types)}")
    for rtype in sorted(relation_types):
        count = sum(1 for r in relations if r.type == rtype)
        print(f"  • {rtype}: {count} relations")
    
    return entities, relations, intention


async def compare_with_without_intention(content: str, api_key: str, model: str = "gemini-1.5-flash"):
    """Compare extraction with and without intention."""
    
    print("\n🔄 COMPARISON: WITH vs WITHOUT INTENTION")
    print("=" * 50)
    
    llm_config = LLMConfig(
        provider="gemini",
        model=model,
        api_key=api_key,
        temperature=0.1
    )
    
    entity_extractor = EntityExtractor(config=llm_config)
    relation_extractor = RelationExtractor(config=llm_config)
    
    # Extract without intention
    print("\n❌ Without intention context:")
    entities_without = await entity_extractor.extract(content)
    relations_without = await relation_extractor.extract(content, entities=entities_without)
    
    entity_types_without = set(entity.type for entity in entities_without)
    relation_types_without = set(relation.type for relation in relations_without)
    
    print(f"  📊 Entity types: {len(entity_types_without)} ({', '.join(sorted(entity_types_without))})")
    print(f"  📊 Relation types: {len(relation_types_without)} ({', '.join(sorted(relation_types_without))})")
    
    # Extract with intention
    print("\n✅ With intention context:")
    intention = await generate_intention(content, api_key, model)
    entities_with = await entity_extractor.extract(content, intention=intention)
    relations_with = await relation_extractor.extract(content, entities=entities_with, intention=intention)
    
    entity_types_with = set(entity.type for entity in entities_with)
    relation_types_with = set(relation.type for relation in relations_with)
    
    print(f"  💡 Intention: {intention}")
    print(f"  📊 Entity types: {len(entity_types_with)} ({', '.join(sorted(entity_types_with))})")
    print(f"  📊 Relation types: {len(relation_types_with)} ({', '.join(sorted(relation_types_with))})")
    
    # Show improvement
    print(f"\n📈 Improvement:")
    print(f"  🎯 Entity type reduction: {len(entity_types_without)} → {len(entity_types_with)} ({len(entity_types_without) - len(entity_types_with):+d})")
    print(f"  🎯 Relation type reduction: {len(relation_types_without)} → {len(relation_types_with)} ({len(relation_types_without) - len(relation_types_with):+d})")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Intention-based extraction demo")
    parser.add_argument("--api-key", help="Gemini API key", default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--model", help="Model to use", default="gemini-1.5-flash")
    parser.add_argument("--content", help="Content to analyze", default=None)
    parser.add_argument("--compare", action="store_true", help="Compare with/without intention")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("❌ Error: Please provide a Gemini API key via --api-key or GEMINI_API_KEY environment variable")
        return 1
    
    # Sample content for different domains
    sample_contents = {
        "medical": """
        The pineal gland is a small endocrine gland in the brain that produces melatonin.
        Many spiritual practitioners believe that activating the pineal gland through
        meditation, breathwork, and specific dietary practices can lead to enhanced
        spiritual awareness and enlightenment. This guide provides comprehensive methods
        for healing and activating the pineal gland, including detoxification protocols,
        meditation techniques, and lifestyle changes that support pineal gland function.
        The pineal gland is often called the "third eye" and is associated with
        spiritual vision, intuition, and connection to higher consciousness.
        """,
        
        "organizational": """
        TechCorp Organizational Structure and Hierarchy
        
        Our company is structured with clear hierarchies and reporting lines to ensure
        efficient operations and communication. John Smith serves as Chief Executive Officer,
        overseeing all company operations and strategic direction. The engineering division
        is led by Jane Doe as Chief Technology Officer, who manages all technical teams
        including software development, DevOps, and quality assurance. The marketing
        department is headed by Bob Johnson as Chief Marketing Officer, responsible for
        brand management, digital marketing, and customer acquisition. Sarah Wilson serves
        as Chief Financial Officer, managing financial planning, accounting, and investor
        relations. Each department has clear reporting structures and defined responsibilities.
        """,
        
        "technical": """
        Microservices Architecture Implementation Guide
        
        This document outlines the implementation of a microservices architecture using
        Docker containers, Kubernetes orchestration, and API Gateway patterns. The system
        consists of multiple independent services including user authentication service,
        payment processing service, inventory management service, and notification service.
        Each service communicates through REST APIs and message queues. The API Gateway
        handles routing, authentication, and rate limiting. Kubernetes manages container
        orchestration, scaling, and health monitoring. The system uses PostgreSQL for
        persistent data storage and Redis for caching and session management.
        """
    }
    
    if args.content:
        content = args.content
    else:
        print("📚 Available sample contents:")
        for key, sample in sample_contents.items():
            print(f"  {key}: {sample[:100]}...")
        
        choice = input("\nSelect content type (medical/organizational/technical) or press Enter for medical: ").strip().lower()
        if choice not in sample_contents:
            choice = "medical"
        
        content = sample_contents[choice]
        print(f"\n📖 Using {choice} content sample")
    
    try:
        if args.compare:
            await compare_with_without_intention(content, args.api_key, args.model)
        else:
            await extract_with_intention(content, args.api_key, args.model)
        
        print("\n✅ Demo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
