#!/usr/bin/env python3
"""
Script to convert CUSTOM relations in JSON files to more meaningful relation types.
This script analyzes the context of CUSTOM relations and suggests better relation types.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional


def detect_relation_type_from_context(context: str, source_entity: str, target_entity: str) -> Optional[str]:
    """Detect relation type based on context patterns."""
    context_lower = context.lower()
    
    # German context patterns
    if "rolle war" in context_lower or "spielte" in context_lower:
        return "PLAYED_ROLE"
    elif "darstellte" in context_lower or "verkörperte" in context_lower:
        return "PORTRAYED"
    elif "betrieben habe" in context_lower or "praktiziert" in context_lower:
        return "PRACTICES"
    elif "beschäftigte sich" in context_lower or "engagierte sich" in context_lower:
        return "ENGAGED_IN"
    elif "studierte" in context_lower or "lernte" in context_lower:
        return "STUDIED"
    
    # English context patterns
    elif "played role" in context_lower or "acted as" in context_lower:
        return "PLAYED_ROLE"
    elif "portrayed" in context_lower or "depicted" in context_lower:
        return "PORTRAYED"
    elif "practices" in context_lower or "engaged in" in context_lower:
        return "PRACTICES"
    elif "studied" in context_lower or "learned" in context_lower:
        return "STUDIED"
    
    return None


def convert_custom_relations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert CUSTOM relations to more meaningful types."""
    converted_count = 0
    
    if "relations" in data:
        for relation in data["relations"]:
            if relation.get("type") == "CUSTOM":
                # Get context from attributes
                context = ""
                if "attributes" in relation and "context" in relation["attributes"]:
                    context = relation["attributes"]["context"]
                
                # Try to detect better relation type
                source_entity = relation.get("source_entity_name", "")
                target_entity = relation.get("target_entity_name", "")
                
                new_type = detect_relation_type_from_context(context, source_entity, target_entity)
                
                if new_type:
                    relation["type"] = new_type
                    converted_count += 1
                    print(f"Converted CUSTOM -> {new_type}: {source_entity} -> {target_entity}")
                    print(f"  Context: {context[:100]}...")
                else:
                    # Fallback to RELATED_TO instead of keeping CUSTOM
                    relation["type"] = "RELATED_TO"
                    converted_count += 1
                    print(f"Converted CUSTOM -> RELATED_TO: {source_entity} -> {target_entity}")
    
    print(f"\nTotal conversions: {converted_count}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Convert CUSTOM relations to meaningful types")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("-o", "--output", help="Output file path (default: overwrites input)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be converted without making changes")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return 1
    
    # Load JSON data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return 1
    
    print(f"Processing {input_path}...")
    
    if args.dry_run:
        print("DRY RUN - No changes will be made")
    
    # Convert relations
    converted_data = convert_custom_relations(data)
    
    if not args.dry_run:
        # Save converted data
        output_path = Path(args.output) if args.output else input_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)
            print(f"\nSaved converted data to {output_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())