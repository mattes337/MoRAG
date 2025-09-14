#!/usr/bin/env python3
"""Example CLI usage of MoRAG Stages."""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from morag_stages import (
    StageManager, StageType, StageContext, 
    StageError, StageDependencyError
)


async def execute_single_stage(
    stage_name: str,
    input_file: Path,
    output_dir: Path,
    config: Optional[dict] = None,
    webhook_url: Optional[str] = None
) -> bool:
    """Execute a single stage.
    
    Args:
        stage_name: Name of stage to execute
        input_file: Input file path
        output_dir: Output directory
        config: Stage configuration
        webhook_url: Webhook URL for notifications
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert stage name to type
        stage_type = StageType(stage_name)
    except ValueError:
        print(f"❌ Invalid stage name: {stage_name}")
        print(f"Valid stages: {[s.value for s in StageType]}")
        return False
    
    # Create stage manager and context
    manager = StageManager()
    context = StageContext(
        source_path=input_file,
        output_dir=output_dir,
        webhook_url=webhook_url,
        config=config or {}
    )
    
    try:
        print(f"🚀 Executing stage: {stage_name}")
        result = await manager.execute_stage(stage_type, [input_file], context)
        
        if result.success:
            print(f"✅ Stage {stage_name} completed successfully")
            print(f"📁 Output files: {[str(f) for f in result.output_files]}")
            print(f"⏱️  Execution time: {result.metadata.execution_time:.2f}s")
            return True
        elif result.skipped:
            print(f"⏭️  Stage {stage_name} skipped (outputs already exist)")
            return True
        else:
            print(f"❌ Stage {stage_name} failed: {result.error_message}")
            return False
            
    except StageError as e:
        print(f"❌ Stage execution failed: {e}")
        return False


async def execute_stage_chain(
    stage_names: List[str],
    input_file: Path,
    output_dir: Path,
    config: Optional[dict] = None,
    webhook_url: Optional[str] = None
) -> bool:
    """Execute a chain of stages.
    
    Args:
        stage_names: List of stage names to execute
        input_file: Input file path
        output_dir: Output directory
        config: Stage configuration
        webhook_url: Webhook URL for notifications
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert stage names to types
        stage_types = []
        for name in stage_names:
            try:
                stage_types.append(StageType(name))
            except ValueError:
                print(f"❌ Invalid stage name: {name}")
                print(f"Valid stages: {[s.value for s in StageType]}")
                return False
    
        # Create stage manager and context
        manager = StageManager()
        context = StageContext(
            source_path=input_file,
            output_dir=output_dir,
            webhook_url=webhook_url,
            config=config or {}
        )
        
        print(f"🚀 Executing stage chain: {' → '.join(stage_names)}")
        results = await manager.execute_stage_chain(stage_types, [input_file], context)
        
        # Report results
        successful = 0
        skipped = 0
        failed = 0
        
        for result in results:
            if result.success:
                successful += 1
                print(f"✅ {result.stage_type.value}: completed ({result.metadata.execution_time:.2f}s)")
            elif result.skipped:
                skipped += 1
                print(f"⏭️  {result.stage_type.value}: skipped")
            else:
                failed += 1
                print(f"❌ {result.stage_type.value}: failed - {result.error_message}")
        
        print(f"\n📊 Summary: {successful} successful, {skipped} skipped, {failed} failed")
        
        # Get final output files
        if results:
            final_result = results[-1]
            if final_result.success:
                print(f"📁 Final outputs: {[str(f) for f in final_result.output_files]}")
        
        return failed == 0
        
    except (StageError, StageDependencyError) as e:
        print(f"❌ Stage chain execution failed: {e}")
        return False


def load_config(config_file: Optional[Path]) -> dict:
    """Load configuration from file.
    
    Args:
        config_file: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not config_file or not config_file.exists():
        return {}
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  Failed to load config file {config_file}: {e}")
        return {}


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="MoRAG Stages CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Execute single stage
  python cli_example.py stage markdown-conversion input.mp4 --output-dir ./output
  
  # Execute stage chain
  python cli_example.py chain markdown-conversion,chunker,fact-generator input.mp4 --output-dir ./output
  
  # With configuration
  python cli_example.py chain markdown-conversion,chunker input.pdf --config config.json --output-dir ./output
  
  # With webhook notifications
  python cli_example.py stage markdown-conversion input.txt --webhook-url http://localhost:8080/webhook
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Single stage command
    stage_parser = subparsers.add_parser('stage', help='Execute single stage')
    stage_parser.add_argument('stage_name', help='Stage name to execute')
    stage_parser.add_argument('input_file', type=Path, help='Input file path')
    stage_parser.add_argument('--output-dir', type=Path, default=Path('./output'), help='Output directory')
    stage_parser.add_argument('--config', type=Path, help='Configuration file (JSON)')
    stage_parser.add_argument('--webhook-url', help='Webhook URL for notifications')
    
    # Stage chain command
    chain_parser = subparsers.add_parser('chain', help='Execute stage chain')
    chain_parser.add_argument('stages', help='Comma-separated list of stage names')
    chain_parser.add_argument('input_file', type=Path, help='Input file path')
    chain_parser.add_argument('--output-dir', type=Path, default=Path('./output'), help='Output directory')
    chain_parser.add_argument('--config', type=Path, help='Configuration file (JSON)')
    chain_parser.add_argument('--webhook-url', help='Webhook URL for notifications')
    
    # List stages command
    list_parser = subparsers.add_parser('list', help='List available stages')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        print("Available stages:")
        for stage_type in StageType:
            print(f"  - {stage_type.value}")
        return
    
    # Load configuration
    config = load_config(getattr(args, 'config', None))
    
    # Create output directory
    output_dir = getattr(args, 'output_dir', Path('./output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute command
    success = False
    
    if args.command == 'stage':
        success = await execute_single_stage(
            args.stage_name,
            args.input_file,
            output_dir,
            config,
            getattr(args, 'webhook_url', None)
        )
    elif args.command == 'chain':
        stage_names = [s.strip() for s in args.stages.split(',')]
        success = await execute_stage_chain(
            stage_names,
            args.input_file,
            output_dir,
            config,
            getattr(args, 'webhook_url', None)
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
