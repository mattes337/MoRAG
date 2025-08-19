#!/usr/bin/env python3
"""MoRAG Stage-Based Processing CLI - Main interface for stage execution."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add packages directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-stages" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-services" / "src"))

from morag_stages import StageManager, StageType, StageStatus
from morag_stages.models import StageContext


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file."""
    if not config_path:
        return {}
    
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        import json
        with open(config_path) as f:
            return json.load(f)
    elif config_path.suffix in ['.yml', '.yaml']:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


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

    try:
        result = await stage_manager.execute_stage(stage_type, [Path(args.input)], context)
        
        if result.status == StageStatus.COMPLETED:
            print(f"✅ Stage {args.stage} completed successfully")
            print(f"📁 Output files: {[str(f) for f in result.output_files]}")
            print(f"⏱️  Execution time: {result.metadata.execution_time:.2f}s")
            if args.webhook_url:
                print(f"🔔 Webhook notification sent to: {args.webhook_url}")
        elif result.status == StageStatus.SKIPPED:
            print(f"⏭️  Stage {args.stage} skipped (outputs already exist)")
        else:
            print(f"❌ Stage {args.stage} failed: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error executing stage {args.stage}: {str(e)}")
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
    
    try:
        # Execute stage chain
        results = await stage_manager.execute_stage_chain(stage_types, [Path(args.input)], context)
        
        # Report results
        successful = 0
        failed = 0
        
        for result in results:
            if result.status == StageStatus.COMPLETED:
                print(f"✅ Stage {result.stage_type.value} completed")
                successful += 1
            elif result.status == StageStatus.SKIPPED:
                print(f"⏭️  Stage {result.stage_type.value} skipped")
                successful += 1
            else:
                print(f"❌ Stage {result.stage_type.value} failed: {result.error_message}")
                failed += 1
        
        print(f"\n📊 Stage Chain Results: {successful}/{len(results)} stages completed successfully")
        
        if failed > 0:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error executing stage chain: {str(e)}")
        sys.exit(1)


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
    
    try:
        # Execute pipeline
        results = await stage_manager.execute_stage_chain(stages, [Path(args.input)], context)
        
        # Report final results
        successful = sum(1 for r in results if r.status == StageStatus.COMPLETED)
        skipped = sum(1 for r in results if r.status == StageStatus.SKIPPED)
        total = len(results)
        
        print(f"\n📊 Pipeline Results: {successful + skipped}/{total} stages completed successfully")
        
        if successful + skipped < total:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error executing pipeline: {str(e)}")
        sys.exit(1)


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
    
    # List available stages
    list_parser = subparsers.add_parser("list", help="List available stages")
    
    return parser


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
    elif args.command == "list":
        print("Available stages:")
        for stage_type in StageType:
            print(f"  - {stage_type.value}")
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
