# Task 7: Update CLI Scripts and Commands

## Overview
**COMPLETELY REPLACE** all CLI scripts with new stage-based processing using canonical stage names.

## Objectives
- **COMPLETELY REPLACE** CLI interface with new stage-based commands
- Add individual stage execution commands using canonical names
- Implement stage chaining and dependency resolution
- **REMOVE ALL EXISTING CLI SCRIPTS** and replace with new ones
- **NO BACKWARD COMPATIBILITY** - clean slate approach

## Deliverables

### 1. New Stage-Based CLI Interface
```python
# cli/morag-stages.py
import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from morag_stages import StageManager, StageType, PipelineConfig
from morag_stages.models import StageContext

async def execute_stage(args):
    """Execute a single stage."""
    stage_manager = StageManager()

    # Create stage context
    context = StageContext(
        source_path=Path(args.input),
        output_dir=Path(args.output_dir),
        webhook_url=args.webhook_url,
        config=load_config(args.config) if args.config else {}
    )

    # Execute stage
    try:
        stage_type = StageType(args.stage)
    except ValueError:
        print(f"❌ Invalid stage name: {args.stage}")
        print(f"Valid stages: {[s.value for s in StageType]}")
        sys.exit(1)

    result = await stage_manager.execute_stage(stage_type, [Path(args.input)], context)

    if result.status == StageStatus.COMPLETED:
        print(f"✅ Stage {args.stage} completed successfully")
        print(f"Output files: {[str(f) for f in result.output_files]}")
        if args.webhook_url:
            print(f"Webhook notification sent to: {args.webhook_url}")
    else:
        print(f"❌ Stage {args.stage} failed: {result.error_message}")
        sys.exit(1)

async def execute_stage_chain(args):
    """Execute a chain of stages using canonical names."""
    stage_manager = StageManager()

    # Parse stage names
    stage_names = [s.strip() for s in args.stages.split(',')]

    # Convert to stage types
    stage_types = []
    for name in stage_names:
        try:
            stage_types.append(StageType(name))
        except ValueError:
            print(f"❌ Invalid stage name: {name}")
            print(f"Valid stages: {[s.value for s in StageType]}")
            sys.exit(1)

    # Create context
    context = StageContext(
        source_path=Path(args.input),
        output_dir=Path(args.output_dir),
        webhook_url=args.webhook_url,
        config=load_config(args.config) if args.config else {}
    )

    # Execute stage chain
    results = await stage_manager.execute_stage_chain(stage_types, [Path(args.input)], context)

    # Report results
    for stage_type, result in results.items():
        if result.status == StageStatus.COMPLETED:
            print(f"✅ Stage {stage_type.value} completed")
        else:
            print(f"❌ Stage {stage_type.value} failed: {result.error_message}")

async def execute_full_pipeline(args):
    """Execute full pipeline (backward compatibility)."""
    stage_manager = StageManager()

    # Determine which stages to run
    stages = [StageType.MARKDOWN_CONVERSION, StageType.CHUNKER, StageType.FACT_GENERATOR, StageType.INGESTOR]

    if args.optimize:
        stages.insert(1, StageType.MARKDOWN_OPTIMIZER)

    if args.skip_stages:
        skip_stage_names = [s.strip() for s in args.skip_stages.split(',')]
        skip_stages = []
        for name in skip_stage_names:
            try:
                skip_stages.append(StageType(name))
            except ValueError:
                print(f"❌ Invalid stage name to skip: {name}")
                sys.exit(1)
        stages = [s for s in stages if s not in skip_stages]

    # Create context
    context = StageContext(
        source_path=Path(args.input),
        output_dir=Path(args.output_dir),
        webhook_url=args.webhook_url,
        config=load_config(args.config) if args.config else {}
    )

    # Execute pipeline
    results = await stage_manager.execute_stage_chain(stages, [Path(args.input)], context)

    # Report final results
    successful = sum(1 for r in results.values() if r.status == StageStatus.COMPLETED)
    total = len(results)

    print(f"\n📊 Pipeline Results: {successful}/{total} stages completed successfully")

def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description="MoRAG Stage-Based Processing")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single stage execution using canonical names
    stage_parser = subparsers.add_parser("stage", help="Execute a single stage")
    stage_parser.add_argument("stage", choices=["markdown-conversion", "markdown-optimizer", "chunker", "fact-generator", "ingestor"], help="Stage name to execute")
    stage_parser.add_argument("input", help="Input file or previous stage output")
    stage_parser.add_argument("--output-dir", default="./output", help="Output directory")
    stage_parser.add_argument("--webhook-url", help="Webhook URL for notifications")
    stage_parser.add_argument("--config", help="Configuration file path")

    # Stage chain execution using canonical names
    chain_parser = subparsers.add_parser("stages", help="Execute a chain of stages")
    chain_parser.add_argument("stages", help="Comma-separated stage names (e.g., 'markdown-conversion,chunker,fact-generator')")
    chain_parser.add_argument("input", help="Input file")
    chain_parser.add_argument("--output-dir", default="./output", help="Output directory")
    chain_parser.add_argument("--webhook-url", help="Webhook URL for notifications")
    chain_parser.add_argument("--config", help="Configuration file path")

    # Full pipeline (backward compatibility)
    pipeline_parser = subparsers.add_parser("process", help="Execute full pipeline")
    pipeline_parser.add_argument("input", help="Input file")
    pipeline_parser.add_argument("--output-dir", default="./output", help="Output directory")
    pipeline_parser.add_argument("--optimize", action="store_true", help="Include markdown optimization stage")
    pipeline_parser.add_argument("--skip-stages", help="Comma-separated list of stages to skip")
    pipeline_parser.add_argument("--webhook-url", help="Webhook URL for notifications")
    pipeline_parser.add_argument("--config", help="Configuration file path")

    return parser

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    import json
    import yaml

    config_path = Path(config_path)

    if config_path.suffix == '.json':
        with open(config_path) as f:
            return json.load(f)
    elif config_path.suffix in ['.yml', '.yaml']:
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

async def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.command == "stage":
        await execute_stage(args)
    elif args.command == "stages":
        await execute_stage_chain(args)
    elif args.command == "process":
        await execute_full_pipeline(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Updated Individual CLI Scripts
```python
# cli/test-document-stages.py
"""Test document processing using stage-based architecture."""

import asyncio
import sys
from pathlib import Path

from morag_stages import StageManager, StageType
from morag_stages.models import StageContext

async def test_document_stages():
    """Test document processing through all stages."""

    # Test file
    test_file = Path("test_data/sample.pdf")
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)

    stage_manager = StageManager()

    # Create context
    context = StageContext(
        source_path=test_file,
        output_dir=output_dir,
        config={
            'stage1': {
                'include_timestamps': False,
                'preserve_formatting': True
            },
            'stage3': {
                'chunk_strategy': 'semantic',
                'chunk_size': 2000,
                'generate_summary': True
            },
            'stage4': {
                'extract_entities': True,
                'extract_relations': True,
                'domain': 'general'
            },
            'stage5': {
                'databases': ['qdrant'],
                'collection_name': 'test_collection'
            }
        }
    )

    # Execute stages sequentially
    stages = [
        StageType.INPUT_TO_MARKDOWN,
        StageType.CHUNKING,
        StageType.FACT_GENERATION,
        StageType.INGESTION
    ]

    input_files = [test_file]

    for stage in stages:
        print(f"\n🔄 Executing Stage {stage.value}: {stage.name}")

        result = await stage_manager.execute_stage(stage, input_files, context)

        if result.status == StageStatus.COMPLETED:
            print(f"✅ Stage {stage.value} completed successfully")
            print(f"   Output files: {[f.name for f in result.output_files]}")
            print(f"   Execution time: {result.execution_time:.2f}s")

            # Use output files as input for next stage
            input_files = result.output_files
        else:
            print(f"❌ Stage {stage.value} failed: {result.error_message}")
            sys.exit(1)

    print("\n🎉 All stages completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_document_stages())
```

### 3. Stage-Aware Batch Processing
```python
# cli/batch-process-stages.py
"""Batch process multiple files using stage-based architecture."""

import asyncio
import argparse
from pathlib import Path
from typing import List

from morag_stages import StageManager, StageType
from morag_stages.models import StageContext

async def process_file_through_stages(file_path: Path,
                                     stages: List[StageType],
                                     output_dir: Path,
                                     config: dict) -> bool:
    """Process a single file through specified stages."""

    stage_manager = StageManager()

    # Create file-specific output directory
    file_output_dir = output_dir / file_path.stem
    file_output_dir.mkdir(exist_ok=True)

    context = StageContext(
        source_path=file_path,
        output_dir=file_output_dir,
        config=config
    )

    try:
        results = await stage_manager.execute_stage_chain(stages, [file_path], context)

        # Check if all stages completed successfully
        success = all(r.status == StageStatus.COMPLETED for r in results.values())

        if success:
            print(f"✅ {file_path.name}: All stages completed")
        else:
            failed_stages = [s.value for s, r in results.items() if r.status != StageStatus.COMPLETED]
            print(f"❌ {file_path.name}: Failed stages: {failed_stages}")

        return success

    except Exception as e:
        print(f"❌ {file_path.name}: Error: {e}")
        return False

async def batch_process(args):
    """Process multiple files in batch."""

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find all supported files
    supported_extensions = {'.pdf', '.docx', '.txt', '.mp4', '.mp3', '.wav'}
    files = [f for f in input_dir.rglob('*') if f.suffix.lower() in supported_extensions]

    print(f"Found {len(files)} files to process")

    # Parse stages to execute
    if args.stages == 'all':
        stages = [StageType.INPUT_TO_MARKDOWN, StageType.CHUNKING,
                 StageType.FACT_GENERATION, StageType.INGESTION]
    else:
        stage_numbers = [int(s) for s in args.stages.split(',')]
        stages = [StageType(s) for s in stage_numbers]

    # Load configuration
    config = {}
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)

    # Process files
    successful = 0
    failed = 0

    for file_path in files:
        print(f"\n📄 Processing: {file_path.name}")

        success = await process_file_through_stages(
            file_path, stages, output_dir, config
        )

        if success:
            successful += 1
        else:
            failed += 1

    print(f"\n📊 Batch Processing Results:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Batch process files using stages")
    parser.add_argument("input_dir", help="Input directory containing files")
    parser.add_argument("--output-dir", default="./batch_output", help="Output directory")
    parser.add_argument("--stages", default="all", help="Stages to execute (e.g., '1,3,5' or 'all')")
    parser.add_argument("--config", help="Configuration file path")

    args = parser.parse_args()
    asyncio.run(batch_process(args))

if __name__ == "__main__":
    main()
```

### 4. Stage Status and Monitoring
```python
# cli/stage-status.py
"""Monitor and check status of stage executions."""

import argparse
import json
from pathlib import Path
from datetime import datetime

def check_stage_outputs(output_dir: Path):
    """Check what stage outputs exist in directory."""

    stage_files = {
        1: list(output_dir.glob("*.md")),
        2: list(output_dir.glob("*.opt.md")),
        3: list(output_dir.glob("*.chunks.json")),
        4: list(output_dir.glob("*.facts.json")),
        5: list(output_dir.glob("*.ingestion.json"))
    }

    print(f"📁 Checking stage outputs in: {output_dir}")
    print("=" * 60)

    for stage, files in stage_files.items():
        stage_names = {
            1: "Input-to-Markdown",
            2: "Markdown Optimizer",
            3: "Chunking",
            4: "Fact Generation",
            5: "Ingestion"
        }

        if files:
            print(f"✅ Stage {stage} ({stage_names[stage]}): {len(files)} files")
            for file in files[:3]:  # Show first 3 files
                print(f"   📄 {file.name}")
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more")
        else:
            print(f"❌ Stage {stage} ({stage_names[stage]}): No outputs found")

    print("=" * 60)

def analyze_stage_file(file_path: Path):
    """Analyze a specific stage output file."""

    print(f"📄 Analyzing: {file_path.name}")
    print("-" * 40)

    if file_path.suffix == '.md':
        # Analyze markdown file
        content = file_path.read_text(encoding='utf-8')

        # Extract metadata
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                import yaml
                metadata = yaml.safe_load(parts[1])
                content_text = parts[2].strip()

                print(f"📊 Metadata:")
                for key, value in metadata.items():
                    print(f"   {key}: {value}")

                print(f"\n📝 Content Stats:")
                print(f"   Word count: {len(content_text.split())}")
                print(f"   Character count: {len(content_text)}")

    elif file_path.suffix == '.json':
        # Analyze JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"📊 JSON Structure:")
        for key, value in data.items():
            if isinstance(value, list):
                print(f"   {key}: {len(value)} items")
            elif isinstance(value, dict):
                print(f"   {key}: {len(value)} keys")
            else:
                print(f"   {key}: {type(value).__name__}")

def main():
    parser = argparse.ArgumentParser(description="Check stage execution status")
    parser.add_argument("--output-dir", default="./output", help="Output directory to check")
    parser.add_argument("--file", help="Specific file to analyze")

    args = parser.parse_args()

    if args.file:
        analyze_stage_file(Path(args.file))
    else:
        check_stage_outputs(Path(args.output_dir))

if __name__ == "__main__":
    main()
```

## Implementation Steps

1. **REMOVE ALL EXISTING CLI SCRIPTS COMPLETELY**
2. **Create completely new stage-based CLI interface using canonical names**
3. **Add stage chaining and dependency resolution with named stages**
4. **Implement batch processing with named stages**
5. **Add stage status monitoring tools**
6. **Create configuration file support**
7. **Add webhook integration to CLI**
8. **Implement error handling and recovery**
9. **Add progress reporting and logging**
10. **Create new CLI documentation for named stages only**

## Testing Requirements

- Unit tests for CLI argument parsing
- Integration tests for stage execution
- Batch processing tests
- Error handling validation
- Configuration file parsing tests
- Backward compatibility tests

## Files to Update/Create

- `cli/morag-stages.py` (completely new main CLI)
- `cli/test-document-stages.py` (completely new)
- `cli/test-audio-stages.py` (completely new)
- `cli/test-video-stages.py` (completely new)
- `cli/batch-process-stages.py` (completely new)
- `cli/stage-status.py` (completely new)
- **REMOVE** all existing `cli/test-*.py` scripts completely

## Success Criteria

- All CLI scripts work with new stage-based architecture using canonical names
- Individual stages can be executed via CLI using canonical names
- Stage chaining works correctly with dependency resolution
- **ALL OLD CLI SCRIPTS ARE COMPLETELY REMOVED** - no backwards compatibility
- Batch processing supports stage-based execution with named stages
- Error handling and reporting work properly with new interface
