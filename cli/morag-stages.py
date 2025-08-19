#!/usr/bin/env python3
"""MoRAG Stage-Based Processing CLI - Main interface for stage execution."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add packages directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-stages" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-services" / "src"))

from morag_stages import StageManager, StageType, StageStatus
from morag_stages.models import StageContext



async def execute_stage(args):
    """Execute a single stage."""
    stage_manager = StageManager()
    
    # Create CLI overrides from arguments
    cli_overrides = {}

    # Global LLM overrides
    if hasattr(args, 'llm_model') and args.llm_model:
        cli_overrides['model'] = args.llm_model
    if hasattr(args, 'llm_provider') and args.llm_provider:
        cli_overrides['provider'] = args.llm_provider
    if hasattr(args, 'llm_temperature') and args.llm_temperature is not None:
        cli_overrides['temperature'] = args.llm_temperature
    if hasattr(args, 'llm_max_tokens') and args.llm_max_tokens:
        cli_overrides['max_tokens'] = args.llm_max_tokens

    # Stage-specific overrides
    stage_overrides = {}
    if hasattr(args, 'chunk_size') and args.chunk_size:
        stage_overrides['chunk_size'] = args.chunk_size
    if hasattr(args, 'max_chunk_size') and args.max_chunk_size:
        stage_overrides['max_chunk_size'] = args.max_chunk_size
    if hasattr(args, 'domain') and args.domain:
        stage_overrides['domain'] = args.domain

    # Combine overrides
    config_overrides = {args.stage: {**cli_overrides, **stage_overrides}} if cli_overrides or stage_overrides else {}

    # Create stage context - configuration loaded from environment variables with CLI overrides
    context = StageContext(
        source_path=Path(args.input),
        output_dir=Path(args.output_dir),
        webhook_url=args.webhook_url,
        config=config_overrides
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
    
    # Create CLI overrides from arguments
    cli_overrides = {}

    # Global LLM overrides
    if hasattr(args, 'llm_model') and args.llm_model:
        cli_overrides['model'] = args.llm_model
    if hasattr(args, 'llm_provider') and args.llm_provider:
        cli_overrides['provider'] = args.llm_provider
    if hasattr(args, 'llm_temperature') and args.llm_temperature is not None:
        cli_overrides['temperature'] = args.llm_temperature
    if hasattr(args, 'llm_max_tokens') and args.llm_max_tokens:
        cli_overrides['max_tokens'] = args.llm_max_tokens

    # Apply overrides to all stages in the chain
    config_overrides = {}
    if cli_overrides:
        for stage_name in stage_names:
            config_overrides[stage_name] = cli_overrides.copy()

    # Create context - configuration loaded from environment variables with CLI overrides
    context = StageContext(
        source_path=Path(args.input),
        output_dir=Path(args.output_dir),
        webhook_url=args.webhook_url,
        config=config_overrides
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
    
    # Create context - configuration loaded from environment variables
    context = StageContext(
        source_path=Path(args.input),
        output_dir=Path(args.output_dir),
        webhook_url=args.webhook_url,
        config={}
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

    # LLM configuration overrides
    stage_parser.add_argument("--llm-model", help="Override LLM model (e.g., gemini-1.5-flash)")
    stage_parser.add_argument("--llm-provider", help="Override LLM provider (e.g., gemini)")
    stage_parser.add_argument("--llm-temperature", type=float, help="Override LLM temperature")
    stage_parser.add_argument("--llm-max-tokens", type=int, help="Override LLM max tokens")

    # Stage-specific overrides
    stage_parser.add_argument("--chunk-size", type=int, help="Override chunk size for chunker")
    stage_parser.add_argument("--max-chunk-size", type=int, help="Override max chunk size for markdown optimizer")
    stage_parser.add_argument("--domain", help="Override domain for fact generator")

    # Stage chain execution using canonical names
    chain_parser = subparsers.add_parser("stages", help="Execute a chain of stages")
    chain_parser.add_argument("stages", help="Comma-separated stage names (e.g., 'markdown-conversion,chunker,fact-generator')")
    chain_parser.add_argument("input", help="Input file")
    chain_parser.add_argument("--output-dir", default="./output", help="Output directory")
    chain_parser.add_argument("--webhook-url", help="Webhook URL for notifications")

    # LLM configuration overrides for chain
    chain_parser.add_argument("--llm-model", help="Override LLM model for all stages")
    chain_parser.add_argument("--llm-provider", help="Override LLM provider for all stages")
    chain_parser.add_argument("--llm-temperature", type=float, help="Override LLM temperature for all stages")
    chain_parser.add_argument("--llm-max-tokens", type=int, help="Override LLM max tokens for all stages")
    
    # Full pipeline (backward compatibility)
    pipeline_parser = subparsers.add_parser("process", help="Execute full pipeline")
    pipeline_parser.add_argument("input", help="Input file")
    pipeline_parser.add_argument("--output-dir", default="./output", help="Output directory")
    pipeline_parser.add_argument("--optimize", action="store_true", help="Include markdown optimization stage")
    pipeline_parser.add_argument("--skip-stages", help="Comma-separated list of stages to skip")
    pipeline_parser.add_argument("--webhook-url", help="Webhook URL for notifications")
    
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
