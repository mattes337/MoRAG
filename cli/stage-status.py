#!/usr/bin/env python3
"""Monitor and check status of stage executions."""

import argparse
import json
from datetime import datetime
from pathlib import Path


def check_stage_outputs(output_dir: Path):
    """Check what stage outputs exist in directory."""

    stage_files = {
        "markdown-conversion": list(output_dir.glob("*.md")),
        "markdown-optimizer": list(output_dir.glob("*.opt.md")),
        "chunker": list(output_dir.glob("*.chunks.json")),
        "fact-generator": list(output_dir.glob("*.facts.json")),
        "ingestor": list(output_dir.glob("*.ingestion.json")),
    }

    print(f"📁 Checking stage outputs in: {output_dir}")
    print("=" * 60)

    stage_descriptions = {
        "markdown-conversion": "Markdown Conversion",
        "markdown-optimizer": "Markdown Optimizer",
        "chunker": "Chunking",
        "fact-generator": "Fact Generation",
        "ingestor": "Ingestion",
    }

    for stage, files in stage_files.items():
        description = stage_descriptions[stage]

        if files:
            print(f"✅ {stage} ({description}): {len(files)} files")
            for file in files[:3]:  # Show first 3 files
                print(f"   📄 {file.name}")
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more")
        else:
            print(f"❌ {stage} ({description}): No outputs found")

    print("=" * 60)


def analyze_stage_file(file_path: Path):
    """Analyze a specific stage output file."""

    print(f"📄 Analyzing: {file_path.name}")
    print("-" * 40)

    if file_path.suffix == ".md":
        # Analyze markdown file
        content = file_path.read_text(encoding="utf-8")

        # Extract metadata
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                import yaml

                try:
                    metadata = yaml.safe_load(parts[1])
                    content_text = parts[2].strip()

                    print(f"📊 Metadata:")
                    for key, value in metadata.items():
                        print(f"   {key}: {value}")

                    print(f"\n📝 Content Stats:")
                    print(f"   Word count: {len(content_text.split())}")
                    print(f"   Character count: {len(content_text)}")
                    print(f"   Line count: {len(content_text.splitlines())}")

                    # Check for timestamps
                    timestamp_count = content_text.count("[") + content_text.count("##")
                    if timestamp_count > 0:
                        print(f"   Timestamps/Headers: ~{timestamp_count}")

                except yaml.YAMLError as e:
                    print(f"   ⚠️  Could not parse metadata: {e}")
        else:
            print(f"📝 Content Stats:")
            print(f"   Word count: {len(content.split())}")
            print(f"   Character count: {len(content)}")
            print(f"   Line count: {len(content.splitlines())}")

    elif file_path.suffix == ".json":
        # Analyze JSON file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"📊 JSON Structure:")
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"   {key}: {len(value)} items")
                    if len(value) > 0 and isinstance(value[0], dict):
                        # Show sample keys from first item
                        sample_keys = list(value[0].keys())[:5]
                        print(f"      Sample keys: {sample_keys}")
                elif isinstance(value, dict):
                    print(f"   {key}: {len(value)} keys")
                    sample_keys = list(value.keys())[:5]
                    print(f"      Keys: {sample_keys}")
                else:
                    print(f"   {key}: {type(value).__name__}")

            # Special analysis for different file types
            if file_path.name.endswith(".chunks.json"):
                analyze_chunks_file(data)
            elif file_path.name.endswith(".facts.json"):
                analyze_facts_file(data)
            elif file_path.name.endswith(".ingestion.json"):
                analyze_ingestion_file(data)

        except json.JSONDecodeError as e:
            print(f"   ❌ Could not parse JSON: {e}")
        except Exception as e:
            print(f"   ❌ Error analyzing file: {e}")


def analyze_chunks_file(data):
    """Analyze chunks JSON file."""
    print(f"\n🧩 Chunks Analysis:")

    if "chunks" in data:
        chunks = data["chunks"]
        print(f"   Total chunks: {len(chunks)}")

        if chunks:
            # Analyze chunk sizes
            chunk_sizes = [len(chunk.get("content", "")) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            print(f"   Average chunk size: {avg_size:.0f} characters")
            print(f"   Size range: {min(chunk_sizes)} - {max(chunk_sizes)} characters")

            # Check for embeddings
            has_embeddings = any("embedding" in chunk for chunk in chunks)
            print(f"   Has embeddings: {'Yes' if has_embeddings else 'No'}")

    if "summary" in data:
        summary = data["summary"]
        print(f"   Summary length: {len(summary)} characters")


def analyze_facts_file(data):
    """Analyze facts JSON file."""
    print(f"\n🔍 Facts Analysis:")

    if "facts" in data:
        facts = data["facts"]
        print(f"   Total facts: {len(facts)}")

        if facts:
            # Analyze fact types
            fact_types = {}
            for fact in facts:
                fact_type = fact.get("type", "unknown")
                fact_types[fact_type] = fact_types.get(fact_type, 0) + 1

            print(f"   Fact types:")
            for fact_type, count in sorted(fact_types.items()):
                print(f"      {fact_type}: {count}")

    if "entities" in data:
        entities = data["entities"]
        print(f"   Total entities: {len(entities)}")

        if entities:
            # Analyze entity types
            entity_types = {}
            for entity in entities:
                entity_type = entity.get("type", "unknown")
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

            print(f"   Entity types:")
            for entity_type, count in sorted(entity_types.items()):
                print(f"      {entity_type}: {count}")

    if "relations" in data:
        relations = data["relations"]
        print(f"   Total relations: {len(relations)}")


def analyze_ingestion_file(data):
    """Analyze ingestion JSON file."""
    print(f"\n💾 Ingestion Analysis:")

    if "success" in data:
        print(f"   Success: {data['success']}")

    if "chunks_ingested" in data:
        print(f"   Chunks ingested: {data['chunks_ingested']}")

    if "entities_ingested" in data:
        print(f"   Entities ingested: {data['entities_ingested']}")

    if "relations_ingested" in data:
        print(f"   Relations ingested: {data['relations_ingested']}")

    if "processing_time" in data:
        print(f"   Processing time: {data['processing_time']:.2f}s")

    if "databases" in data:
        databases = data["databases"]
        print(f"   Target databases: {', '.join(databases)}")


def compare_stage_outputs(output_dir: Path):
    """Compare outputs across different stages."""

    print(f"🔄 Comparing stage outputs in: {output_dir}")
    print("=" * 60)

    # Find all stage output files
    md_files = list(output_dir.glob("*.md"))
    opt_files = list(output_dir.glob("*.opt.md"))
    chunk_files = list(output_dir.glob("*.chunks.json"))
    fact_files = list(output_dir.glob("*.facts.json"))
    ingestion_files = list(output_dir.glob("*.ingestion.json"))

    # Group by base filename
    base_names = set()
    for file_list in [md_files, opt_files, chunk_files, fact_files, ingestion_files]:
        for file in file_list:
            base_name = file.name.split(".")[0]
            base_names.add(base_name)

    for base_name in sorted(base_names):
        print(f"\n📄 {base_name}:")

        # Check which stages have outputs
        stages_completed = []
        if any(
            f.name.startswith(base_name) and f.name.endswith(".md") for f in md_files
        ):
            stages_completed.append("markdown-conversion")
        if any(
            f.name.startswith(base_name) and f.name.endswith(".opt.md")
            for f in opt_files
        ):
            stages_completed.append("markdown-optimizer")
        if any(
            f.name.startswith(base_name) and f.name.endswith(".chunks.json")
            for f in chunk_files
        ):
            stages_completed.append("chunker")
        if any(
            f.name.startswith(base_name) and f.name.endswith(".facts.json")
            for f in fact_files
        ):
            stages_completed.append("fact-generator")
        if any(
            f.name.startswith(base_name) and f.name.endswith(".ingestion.json")
            for f in ingestion_files
        ):
            stages_completed.append("ingestor")

        print(f"   Completed stages: {', '.join(stages_completed)}")
        print(f"   Progress: {len(stages_completed)}/5 stages")


def main():
    parser = argparse.ArgumentParser(description="Check stage execution status")
    parser.add_argument(
        "--output-dir", default="./output", help="Output directory to check"
    )
    parser.add_argument("--file", help="Specific file to analyze")
    parser.add_argument(
        "--compare", action="store_true", help="Compare outputs across stages"
    )

    args = parser.parse_args()

    if args.file:
        analyze_stage_file(Path(args.file))
    elif args.compare:
        compare_stage_outputs(Path(args.output_dir))
    else:
        check_stage_outputs(Path(args.output_dir))


if __name__ == "__main__":
    main()
