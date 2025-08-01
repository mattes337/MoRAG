#!/usr/bin/env python3
"""Test script to verify improved entity and relation extraction."""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the packages to the Python path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

from morag_graph.extraction.entity_extractor import EntityExtractor
from morag_graph.extraction.relation_extractor import RelationExtractor


async def test_improved_extraction():
    """Test the improved extraction with a sample German medical text."""
    
    # Sample text from the document that had issues
    test_text = """
    Umweltgifte wie Schwermetalle, Pestizide, Glyphosat, Fluorid, Flammschutzmittel, 
    Mikroplastik können auf schonende Weise ausgeleitet werden.
    
    Silizium Pur ist ein Mittel zur Entgiftung von Aluminium. Normalem Siliziumdioxid 
    hat eine nur sehr geringe Bioverfügbarkeit. Hashimoto ist eine Autoimmunerkrankung 
    der Schilddrüse, die durch Quecksilber verursacht werden kann.
    
    Zeolith enthält über 30 Mineralstoffe und Spurenelemente wie Silizium, Zink, 
    Calcium, Magnesium, Mangan, Selen, Eisen.
    """
    
    print("Testing improved entity and relation extraction...")
    print("=" * 60)
    
    try:
        # Initialize extractors with optimized settings (domain-agnostic)
        entity_extractor = EntityExtractor(
            min_confidence=0.5,
            chunk_size=800,
            extraction_passes=3,
            domain="general",  # Use general domain by default
            language="de"
        )

        relation_extractor = RelationExtractor(
            min_confidence=0.5,
            chunk_size=800,
            extraction_passes=3,
            domain="general",  # Use general domain by default
            language="de"
        )
        
        print("Extracting entities...")
        # Test with automatic domain inference
        entities = await entity_extractor.extract(test_text, auto_infer_domain=True)
        
        print(f"\nFound {len(entities)} entities:")
        for i, entity in enumerate(entities, 1):
            print(f"{i:2d}. {entity.name} ({entity.type}) - confidence: {entity.confidence:.2f}")
            if hasattr(entity, 'attributes') and entity.attributes:
                if 'original_name' in entity.attributes:
                    print(f"     Original: {entity.attributes['original_name']}")
                if 'normalization_confidence' in entity.attributes:
                    print(f"     Normalization confidence: {entity.attributes['normalization_confidence']:.2f}")
        
        print("\nExtracting relations...")
        relations = await relation_extractor.extract(test_text, entities=entities)
        
        print(f"\nFound {len(relations)} relations:")
        for i, relation in enumerate(relations, 1):
            source_entity = relation.attributes.get('source_entity', 'Unknown')
            target_entity = relation.attributes.get('target_entity', 'Unknown')
            print(f"{i:2d}. {source_entity} --[{relation.type}]--> {target_entity}")
            print(f"     Confidence: {relation.confidence:.2f}")
            if relation.context:
                print(f"     Context: {relation.context[:100]}...")
        
        # Test specific improvements
        print("\n" + "=" * 60)
        print("TESTING SPECIFIC IMPROVEMENTS:")
        print("=" * 60)
        
        # Check for multiple entity splitting
        multi_entity_relations = [r for r in relations if ',' in r.attributes.get('source_entity', '')]
        if multi_entity_relations:
            print("❌ ISSUE: Found relations with multiple entities in source_entity field:")
            for rel in multi_entity_relations:
                print(f"   {rel.attributes.get('source_entity', 'Unknown')}")
        else:
            print("✅ FIXED: No relations with multiple entities in single field")
        
        # Check for entity normalization
        normalized_entities = [e for e in entities if hasattr(e, 'attributes') and 'original_name' in e.attributes]
        if normalized_entities:
            print(f"✅ IMPROVEMENT: {len(normalized_entities)} entities were normalized:")
            for entity in normalized_entities[:5]:  # Show first 5
                original = entity.attributes['original_name']
                normalized = entity.name
                if original != normalized:
                    print(f"   '{original}' → '{normalized}'")
        
        # Check for better relation types
        medical_relations = [r for r in relations if r.type in ['AFFECTS', 'DETOXIFIES', 'CAUSES', 'ACCUMULATES_IN', 'PROTECTS']]
        if medical_relations:
            print(f"✅ IMPROVEMENT: Found {len(medical_relations)} medical-specific relation types:")
            for rel in medical_relations[:3]:  # Show first 3
                source = rel.attributes.get('source_entity', 'Unknown')
                target = rel.attributes.get('target_entity', 'Unknown')
                print(f"   {source} --[{rel.type}]--> {target}")
        
        # Summary
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY:")
        print("=" * 60)
        print(f"Total entities extracted: {len(entities)}")
        print(f"Total relations extracted: {len(relations)}")
        print(f"Entities with normalization: {len(normalized_entities)}")
        print(f"Medical-specific relations: {len(medical_relations)}")
        
        # Compare with target numbers
        target_entities = 338
        target_relations = 450
        
        print(f"\nComparison with previous results:")
        print(f"Entities: {len(entities)}/{target_entities} ({len(entities)/target_entities*100:.1f}%)")
        print(f"Relations: {len(relations)}/{target_relations} ({len(relations)/target_relations*100:.1f}%)")
        
        if len(entities) >= target_entities * 0.8 and len(relations) >= target_relations * 0.8:
            print("✅ GOOD: Extraction coverage is within acceptable range")
        else:
            print("⚠️  WARNING: Extraction coverage is below target")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_improved_extraction())
