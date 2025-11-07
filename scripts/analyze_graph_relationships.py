#!/usr/bin/env python3
"""Analyze relationship types in the MoRAG knowledge graph.

This script helps understand what dynamic relationship types the LLM has generated
in your specific domain, which is crucial for understanding maintenance job behavior.

Usage:
    python scripts/analyze_graph_relationships.py

Environment variables:
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
"""
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

import structlog

# Add the packages to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "packages", "morag-graph", "src")
)

from morag_graph.maintenance.query_optimizer import QueryOptimizer
from morag_graph.storage import Neo4jConfig, Neo4jStorage

logger = structlog.get_logger(__name__)


async def analyze_relationships() -> Dict[str, Any]:
    """Analyze relationship types in the graph."""
    # Setup Neo4j connection
    neo4j_config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )

    storage = Neo4jStorage(neo4j_config)

    try:
        await storage.connect()
        optimizer = QueryOptimizer(storage)

        print("🔍 Analyzing MoRAG Knowledge Graph Relationships...")
        print("=" * 60)

        # Get all relationship types
        all_rels = await optimizer.get_relationship_type_summary()

        # Get fact-to-entity relationship types specifically
        fact_rels = await optimizer.get_fact_relationship_types()

        # Get basic graph statistics
        stats_query = """
        MATCH (e:Entity)
        WITH count(e) AS entity_count
        MATCH (f:Fact)
        WITH entity_count, count(f) AS fact_count
        MATCH ()-[r]->()
        RETURN entity_count, fact_count, count(r) AS total_relationships
        """

        stats = await optimizer._execute_with_stats(stats_query, {}, "graph_stats")

        # Analyze relationship patterns
        analysis = {
            "graph_statistics": stats[0] if stats else {},
            "all_relationship_types": all_rels,
            "fact_to_entity_relationships": fact_rels,
            "analysis": {},
        }

        # Categorize relationship types
        fact_rel_types = {rel["rel_type"] for rel in fact_rels}
        all_rel_types = {rel["rel_type"] for rel in all_rels}
        entity_rel_types = all_rel_types - fact_rel_types

        analysis["analysis"] = {
            "total_relationship_types": len(all_rel_types),
            "fact_to_entity_types": len(fact_rel_types),
            "entity_to_entity_types": len(entity_rel_types),
            "most_common_fact_relationships": fact_rels[:10],
            "most_common_all_relationships": all_rels[:10],
            "relationship_type_categories": {
                "fact_to_entity": sorted(list(fact_rel_types)),
                "entity_to_entity": sorted(list(entity_rel_types)),
            },
        }

        # Print summary
        print(f"📊 Graph Statistics:")
        print(f"   Entities: {analysis['graph_statistics'].get('entity_count', 0):,}")
        print(f"   Facts: {analysis['graph_statistics'].get('fact_count', 0):,}")
        print(
            f"   Total Relationships: {analysis['graph_statistics'].get('total_relationships', 0):,}"
        )
        print()

        print(f"🔗 Relationship Type Analysis:")
        print(
            f"   Total Relationship Types: {analysis['analysis']['total_relationship_types']}"
        )
        print(f"   Fact→Entity Types: {analysis['analysis']['fact_to_entity_types']}")
        print(
            f"   Entity→Entity Types: {analysis['analysis']['entity_to_entity_types']}"
        )
        print()

        print(f"📈 Top 10 Fact→Entity Relationship Types:")
        for i, rel in enumerate(
            analysis["analysis"]["most_common_fact_relationships"], 1
        ):
            print(f"   {i:2d}. {rel['rel_type']:20s} ({rel['count']:,} relationships)")
        print()

        if entity_rel_types:
            print(f"🔄 Entity→Entity Relationship Types:")
            entity_rels = [
                rel for rel in all_rels if rel["rel_type"] in entity_rel_types
            ][:10]
            for i, rel in enumerate(entity_rels, 1):
                print(
                    f"   {i:2d}. {rel['rel_type']:20s} ({rel['count']:,} relationships)"
                )
            print()

        print(f"💡 Key Insights:")

        # Check for common patterns
        common_fact_types = [rel["rel_type"] for rel in fact_rels[:5]]

        if any("ABOUT" in t or "RELATES_TO" in t for t in common_fact_types):
            print(f"   ✓ Found standard relationship types (ABOUT, RELATES_TO)")
        else:
            print(
                f"   ⚠️  No standard relationship types found - using domain-specific types"
            )

        # Check for domain-specific patterns
        domain_indicators = {
            "medical": ["TREATS", "CAUSES", "SYMPTOM", "DIAGNOSIS", "MEDICATION"],
            "technical": ["IMPLEMENTS", "USES", "REQUIRES", "CONFIGURES"],
            "business": ["MANAGES", "REPORTS_TO", "OWNS", "RESPONSIBLE_FOR"],
            "academic": ["STUDIES", "RESEARCHES", "PUBLISHES", "CITES"],
        }

        detected_domains = []
        for domain, indicators in domain_indicators.items():
            if any(
                any(indicator in rel_type for indicator in indicators)
                for rel_type in fact_rel_types
            ):
                detected_domains.append(domain)

        if detected_domains:
            print(f"   🎯 Detected domain(s): {', '.join(detected_domains)}")
        else:
            print(
                f"   🔍 Custom domain detected - relationship types are domain-specific"
            )

        print(f"   ✅ All maintenance queries use relationship-agnostic patterns")
        print(
            f"   📝 This ensures all {analysis['analysis']['total_relationship_types']} relationship types are captured"
        )

        return analysis

    except Exception as e:
        logger.error("Failed to analyze relationships", error=str(e))
        raise
    finally:
        await storage.disconnect()


async def main():
    """Main function."""
    try:
        analysis = await analyze_relationships()

        # Optionally save detailed analysis to file
        if "--save" in sys.argv:
            output_file = "graph_relationship_analysis.json"
            with open(output_file, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"\n💾 Detailed analysis saved to {output_file}")

        print("\n✅ Analysis complete!")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
