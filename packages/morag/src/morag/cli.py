"""Command-line interface for MoRAG - Stage-based processing system."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import click
import structlog

try:
    from morag_stages import StageManager, StageType, StageStatus
    from morag_stages.models import StageContext
    STAGES_AVAILABLE = True
except ImportError:
    # Mock classes for when morag_stages is not available
    class StageManager:
        pass
    class StageType:
        pass
    class StageStatus:
        pass
    class StageContext:
        pass
    STAGES_AVAILABLE = False

logger = structlog.get_logger(__name__)


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file."""
    if not config_path:
        return {}

    config_path = Path(config_path)

    if config_path.suffix == '.json':
        with open(config_path) as f:
            return json.load(f)
    elif config_path.suffix in ['.yml', '.yaml']:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--llm-model', help='Default LLM model for all agents (e.g., gemini-1.5-flash, gpt-4)')
@click.option('--fact-extraction-agent-model', help='LLM model for fact extraction agent')
@click.option('--entity-extraction-agent-model', help='LLM model for entity extraction agent')
@click.option('--relation-extraction-agent-model', help='LLM model for relation extraction agent')
@click.option('--keyword-extraction-agent-model', help='LLM model for keyword extraction agent')
@click.option('--summarization-agent-model', help='LLM model for summarization agent')
@click.option('--content-analysis-agent-model', help='LLM model for content analysis agent')
@click.option('--markdown-optimizer-agent-model', help='LLM model for markdown optimizer agent')
@click.option('--chunking-agent-model', help='LLM model for chunking agent')
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool, llm_model: Optional[str],
        fact_extraction_agent_model: Optional[str], entity_extraction_agent_model: Optional[str],
        relation_extraction_agent_model: Optional[str], keyword_extraction_agent_model: Optional[str],
        summarization_agent_model: Optional[str], content_analysis_agent_model: Optional[str],
        markdown_optimizer_agent_model: Optional[str], chunking_agent_model: Optional[str]):
    """MoRAG - Stage-based processing system using canonical stage names."""
    # Check if stages are available
    if not STAGES_AVAILABLE:
        click.echo("❌ MoRAG Stages package not available. Please install morag-stages package.")
        ctx.exit(1)

    # Configure logging
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        )

    # Store in context
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    ctx.obj['verbose'] = verbose

    # Store model configurations
    ctx.obj['model_config'] = {
        'default_model': llm_model,
        'agent_models': {
            'fact_extraction': fact_extraction_agent_model,
            'entity_extraction': entity_extraction_agent_model,
            'relation_extraction': relation_extraction_agent_model,
            'keyword_extraction': keyword_extraction_agent_model,
            'summarization': summarization_agent_model,
            'content_analysis': content_analysis_agent_model,
            'markdown_optimizer': markdown_optimizer_agent_model,
            'chunking': chunking_agent_model,
        }
    }


@cli.command()
@click.argument('stage_name', type=click.Choice(['markdown-conversion', 'markdown-optimizer', 'chunker', 'fact-generator', 'ingestor']))
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', default='./output', help='Output directory')
@click.option('--webhook-url', help='Webhook URL for notifications')
@click.pass_context
def stage(ctx: click.Context, stage_name: str, input_file: str, output_dir: str, webhook_url: Optional[str]):
    """Execute a single stage using canonical stage names."""
    async def _execute_stage():
        stage_manager = StageManager()

        # Merge model configuration with stage config
        merged_config = ctx.obj['config'].copy()
        if 'model_config' in ctx.obj:
            merged_config['model_config'] = ctx.obj['model_config']

        # Create stage context
        context = StageContext(
            source_path=Path(input_file),
            output_dir=Path(output_dir),
            webhook_url=webhook_url,
            config=merged_config
        )

        # Execute stage
        try:
            stage_type = StageType(stage_name)
        except ValueError:
            click.echo(f"❌ Invalid stage name: {stage_name}")
            click.echo(f"Valid stages: {[s.value for s in StageType]}")
            sys.exit(1)

        try:
            result = await stage_manager.execute_stage(stage_type, [Path(input_file)], context)

            if result.status == StageStatus.COMPLETED:
                click.echo(f"✅ Stage {stage_name} completed successfully")
                click.echo(f"📁 Output files: {[str(f) for f in result.output_files]}")
                click.echo(f"⏱️  Execution time: {result.metadata.execution_time:.2f}s")
                if webhook_url:
                    click.echo(f"🔔 Webhook notification sent to: {webhook_url}")
            elif result.status == StageStatus.SKIPPED:
                click.echo(f"⏭️  Stage {stage_name} skipped (outputs already exist)")
            else:
                click.echo(f"❌ Stage {stage_name} failed: {result.error_message}")
                sys.exit(1)

        except Exception as e:
            click.echo(f"❌ Error executing stage {stage_name}: {str(e)}", err=True)
            sys.exit(1)

    asyncio.run(_execute_stage())


@cli.command()
@click.argument('stages')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', default='./output', help='Output directory')
@click.option('--webhook-url', help='Webhook URL for notifications')
@click.pass_context
def stages(ctx: click.Context, stages: str, input_file: str, output_dir: str, webhook_url: Optional[str]):
    """Execute a chain of stages using canonical stage names.

    STAGES: Comma-separated stage names (e.g., "markdown-conversion,chunker,fact-generator")
    INPUT_FILE: Path to the input file to process
    """
    async def _execute_stage_chain():
        stage_manager = StageManager()

        # Parse stage names
        stage_names = [s.strip() for s in stages.split(',')]

        # Convert to stage types
        stage_types = []
        for name in stage_names:
            try:
                stage_types.append(StageType(name))
            except ValueError:
                click.echo(f"❌ Invalid stage name: {name}")
                click.echo(f"Valid stages: {[s.value for s in StageType]}")
                sys.exit(1)

        # Merge model configuration with stage config
        merged_config = ctx.obj['config'].copy()
        if 'model_config' in ctx.obj:
            merged_config['model_config'] = ctx.obj['model_config']

        # Create context
        context = StageContext(
            source_path=Path(input_file),
            output_dir=Path(output_dir),
            webhook_url=webhook_url,
            config=merged_config
        )

        try:
            # Execute stage chain
            results = await stage_manager.execute_stage_chain(stage_types, [Path(input_file)], context)

            # Report results
            successful = 0
            failed = 0

            for result in results:
                if result.status == StageStatus.COMPLETED:
                    click.echo(f"✅ Stage {result.stage_type.value} completed")
                    successful += 1
                elif result.status == StageStatus.SKIPPED:
                    click.echo(f"⏭️  Stage {result.stage_type.value} skipped")
                    successful += 1
                else:
                    click.echo(f"❌ Stage {result.stage_type.value} failed: {result.error_message}")
                    failed += 1

            click.echo(f"\n📊 Stage Chain Results: {successful}/{len(results)} stages completed successfully")

            if failed > 0:
                sys.exit(1)

        except Exception as e:
            click.echo(f"❌ Error executing stage chain: {str(e)}", err=True)
            sys.exit(1)

    asyncio.run(_execute_stage_chain())


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', default='./output', help='Output directory')
@click.option('--optimize', is_flag=True, help='Include markdown optimization stage')
@click.option('--skip-stages', help='Comma-separated list of stages to skip')
@click.option('--webhook-url', help='Webhook URL for notifications')
@click.pass_context
def process(ctx: click.Context, input_file: str, output_dir: str, optimize: bool, skip_stages: Optional[str], webhook_url: Optional[str]):
    """Execute full pipeline (all stages)."""
    async def _execute_full_pipeline():
        stage_manager = StageManager()

        # Determine which stages to run
        stage_types = [StageType.MARKDOWN_CONVERSION, StageType.CHUNKER, StageType.FACT_GENERATOR, StageType.INGESTOR]

        if optimize:
            stage_types.insert(1, StageType.MARKDOWN_OPTIMIZER)

        if skip_stages:
            skip_stage_names = [s.strip() for s in skip_stages.split(',')]
            skip_stage_types = []
            for name in skip_stage_names:
                try:
                    skip_stage_types.append(StageType(name))
                except ValueError:
                    click.echo(f"❌ Invalid stage name to skip: {name}")
                    sys.exit(1)
            stage_types = [s for s in stage_types if s not in skip_stage_types]

        # Create context
        context = StageContext(
            source_path=Path(input_file),
            output_dir=Path(output_dir),
            webhook_url=webhook_url,
            config=ctx.obj['config']
        )

        try:
            # Execute pipeline
            results = await stage_manager.execute_stage_chain(stage_types, [Path(input_file)], context)

            # Report final results
            successful = sum(1 for r in results if r.status == StageStatus.COMPLETED)
            skipped = sum(1 for r in results if r.status == StageStatus.SKIPPED)
            total = len(results)

            click.echo(f"\n📊 Pipeline Results: {successful + skipped}/{total} stages completed successfully")

            if successful + skipped < total:
                sys.exit(1)

        except Exception as e:
            click.echo(f"❌ Error executing pipeline: {str(e)}", err=True)
            sys.exit(1)

    asyncio.run(_execute_full_pipeline())


@cli.command()
def list_stages():
    """List all available stages."""
    click.echo("Available stages:")
    for stage_type in StageType:
        click.echo(f"  - {stage_type.value}")

    click.echo("\nStage descriptions:")
    descriptions = {
        StageType.MARKDOWN_CONVERSION: "Convert input files to unified markdown format",
        StageType.MARKDOWN_OPTIMIZER: "LLM-based text improvement and error correction (optional)",
        StageType.CHUNKER: "Create summary, chunks, and contextual embeddings",
        StageType.FACT_GENERATOR: "Extract facts, entities, relations, and keywords",
        StageType.INGESTOR: "Database ingestion and storage"
    }

    for stage_type, description in descriptions.items():
        click.echo(f"  {stage_type.value}: {description}")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
