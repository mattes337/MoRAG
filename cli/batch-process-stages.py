#!/usr/bin/env python3
"""Batch process multiple files using stage-based architecture."""

import asyncio
import argparse
from pathlib import Path
from typing import List

from morag_stages import StageManager, StageType, StageStatus
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
        success = all(r.status in [StageStatus.COMPLETED, StageStatus.SKIPPED] for r in results)
        
        if success:
            print(f"✅ {file_path.name}: All stages completed")
        else:
            failed_stages = [r.stage_type.value for r in results if r.status not in [StageStatus.COMPLETED, StageStatus.SKIPPED]]
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
    supported_extensions = {'.pdf', '.docx', '.txt', '.mp4', '.mp3', '.wav', '.m4a', '.flac'}
    files = [f for f in input_dir.rglob('*') if f.suffix.lower() in supported_extensions]
    
    print(f"Found {len(files)} files to process")
    
    # Parse stages to execute
    if args.stages == 'all':
        stages = [StageType.MARKDOWN_CONVERSION, StageType.CHUNKER, 
                 StageType.FACT_GENERATOR, StageType.INGESTOR]
        if args.optimize:
            stages.insert(1, StageType.MARKDOWN_OPTIMIZER)
    else:
        stage_names = [s.strip() for s in args.stages.split(',')]
        stages = []
        for name in stage_names:
            try:
                stages.append(StageType(name))
            except ValueError:
                print(f"❌ Invalid stage name: {name}")
                print(f"Valid stages: {[s.value for s in StageType]}")
                return
    
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


async def batch_process_with_resume(args):
    """Process multiple files with resume capability."""
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create resume file
    resume_file = output_dir / "batch_progress.json"
    
    # Load existing progress
    processed_files = set()
    if resume_file.exists():
        import json
        with open(resume_file) as f:
            progress_data = json.load(f)
            processed_files = set(progress_data.get('completed', []))
        print(f"📋 Resuming batch processing. {len(processed_files)} files already processed.")
    
    # Find all supported files
    supported_extensions = {'.pdf', '.docx', '.txt', '.mp4', '.mp3', '.wav', '.m4a', '.flac'}
    all_files = [f for f in input_dir.rglob('*') if f.suffix.lower() in supported_extensions]
    
    # Filter out already processed files
    files = [f for f in all_files if str(f.relative_to(input_dir)) not in processed_files]
    
    print(f"Found {len(files)} files to process ({len(all_files)} total, {len(processed_files)} already done)")
    
    # Parse stages to execute
    if args.stages == 'all':
        stages = [StageType.MARKDOWN_CONVERSION, StageType.CHUNKER, 
                 StageType.FACT_GENERATOR, StageType.INGESTOR]
        if args.optimize:
            stages.insert(1, StageType.MARKDOWN_OPTIMIZER)
    else:
        stage_names = [s.strip() for s in args.stages.split(',')]
        stages = []
        for name in stage_names:
            try:
                stages.append(StageType(name))
            except ValueError:
                print(f"❌ Invalid stage name: {name}")
                print(f"Valid stages: {[s.value for s in StageType]}")
                return
    
    # Load configuration
    config = {}
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)
    
    # Process files
    successful = 0
    failed = 0
    
    for i, file_path in enumerate(files):
        print(f"\n📄 Processing ({i+1}/{len(files)}): {file_path.name}")
        
        success = await process_file_through_stages(
            file_path, stages, output_dir, config
        )
        
        if success:
            successful += 1
            processed_files.add(str(file_path.relative_to(input_dir)))
            
            # Update resume file
            import json
            with open(resume_file, 'w') as f:
                json.dump({
                    'completed': list(processed_files),
                    'total_files': len(all_files),
                    'successful': len(processed_files),
                    'failed': failed
                }, f, indent=2)
        else:
            failed += 1
    
    print(f"\n📊 Batch Processing Results:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   📋 Resume file: {resume_file}")


def main():
    parser = argparse.ArgumentParser(description="Batch process files using stages")
    parser.add_argument("input_dir", help="Input directory containing files")
    parser.add_argument("--output-dir", default="./batch_output", help="Output directory")
    parser.add_argument("--stages", default="all", help="Stages to execute (e.g., 'markdown-conversion,chunker' or 'all')")
    parser.add_argument("--optimize", action="store_true", help="Include markdown optimization stage (only with --stages=all)")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--resume", action="store_true", help="Enable resume capability")
    
    args = parser.parse_args()
    
    if args.resume:
        asyncio.run(batch_process_with_resume(args))
    else:
        asyncio.run(batch_process(args))


if __name__ == "__main__":
    main()
