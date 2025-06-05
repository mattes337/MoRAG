"""Command-line interface for MoRAG."""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any

import click
import structlog

from morag.api import MoRAGAPI
from morag_services import ServiceConfig

logger = structlog.get_logger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool):
    """MoRAG - Modular Retrieval Augmented Generation System."""
    # Configure logging
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        )
    
    # Load configuration
    service_config = ServiceConfig()
    if config:
        # Load config from file if provided
        config_path = Path(config)
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
                # Update service config with loaded data
                for key, value in config_data.items():
                    if hasattr(service_config, key):
                        setattr(service_config, key, value)
    
    # Store in context
    ctx.ensure_object(dict)
    ctx.obj['config'] = service_config
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('url')
@click.option('--type', '-t', 'content_type', help='Content type (auto-detected if not provided)')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', 'output_format', default='json', 
              type=click.Choice(['json', 'text', 'markdown']), help='Output format')
@click.pass_context
def process_url(ctx: click.Context, url: str, content_type: Optional[str], 
                output: Optional[str], output_format: str):
    """Process content from a URL."""
    async def _process():
        api = MoRAGAPI(ctx.obj['config'])
        try:
            result = await api.process_url(url, content_type)
            
            # Format output
            if output_format == 'json':
                output_data = {
                    'success': result.success,
                    'content': result.content,
                    'metadata': result.metadata,
                    'processing_time': result.processing_time,
                    'error_message': result.error_message
                }
                formatted_output = json.dumps(output_data, indent=2)
            elif output_format == 'text':
                formatted_output = result.content
            elif output_format == 'markdown':
                formatted_output = result.content
            else:
                formatted_output = str(result)
            
            # Output to file or stdout
            if output:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
                click.echo(f"Output written to {output}")
            else:
                click.echo(formatted_output)
                
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            raise click.Abort()
        finally:
            await api.cleanup()
    
    asyncio.run(_process())


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--type', '-t', 'content_type', help='Content type (auto-detected if not provided)')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', 'output_format', default='json',
              type=click.Choice(['json', 'text', 'markdown']), help='Output format')
@click.pass_context
def process_file(ctx: click.Context, file_path: str, content_type: Optional[str],
                 output: Optional[str], output_format: str):
    """Process content from a file."""
    async def _process():
        api = MoRAGAPI(ctx.obj['config'])
        try:
            result = await api.process_file(file_path, content_type)
            
            # Format output
            if output_format == 'json':
                output_data = {
                    'success': result.success,
                    'content': result.content,
                    'metadata': result.metadata,
                    'processing_time': result.processing_time,
                    'error_message': result.error_message
                }
                formatted_output = json.dumps(output_data, indent=2)
            elif output_format == 'text':
                formatted_output = result.content
            elif output_format == 'markdown':
                formatted_output = result.content
            else:
                formatted_output = str(result)
            
            # Output to file or stdout
            if output:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
                click.echo(f"Output written to {output}")
            else:
                click.echo(formatted_output)
                
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            raise click.Abort()
        finally:
            await api.cleanup()
    
    asyncio.run(_process())


@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Maximum number of results')
@click.option('--format', '-f', 'output_format', default='json',
              type=click.Choice(['json', 'text']), help='Output format')
@click.pass_context
def search(ctx: click.Context, query: str, limit: int, output_format: str):
    """Search for similar content."""
    async def _search():
        api = MoRAGAPI(ctx.obj['config'])
        try:
            results = await api.search(query, limit)
            
            if output_format == 'json':
                formatted_output = json.dumps(results, indent=2)
            else:
                formatted_output = "\n".join([
                    f"Score: {r.get('score', 0):.3f} - {r.get('text', '')[:100]}..."
                    for r in results
                ])
            
            click.echo(formatted_output)
            
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            raise click.Abort()
        finally:
            await api.cleanup()
    
    asyncio.run(_search())


@cli.command()
@click.pass_context
def health(ctx: click.Context):
    """Check health status of all components."""
    async def _health():
        api = MoRAGAPI(ctx.obj['config'])
        try:
            status = await api.health_check()
            formatted_output = json.dumps(status, indent=2)
            click.echo(formatted_output)
            
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            raise click.Abort()
        finally:
            await api.cleanup()
    
    asyncio.run(_health())


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', 'output_format', default='json',
              type=click.Choice(['json', 'jsonl']), help='Output format')
@click.pass_context
def batch(ctx: click.Context, input_file: str, output: Optional[str], output_format: str):
    """Process multiple items from a batch file."""
    async def _batch():
        api = MoRAGAPI(ctx.obj['config'])
        try:
            # Load batch items
            with open(input_file) as f:
                if input_file.endswith('.jsonl'):
                    items = [json.loads(line) for line in f]
                else:
                    items = json.load(f)
            
            results = await api.process_batch(items)
            
            # Format output
            if output_format == 'jsonl':
                formatted_output = "\n".join([
                    json.dumps({
                        'success': r.success,
                        'content': r.content,
                        'metadata': r.metadata,
                        'processing_time': r.processing_time,
                        'error_message': r.error_message
                    }) for r in results
                ])
            else:
                formatted_output = json.dumps([{
                    'success': r.success,
                    'content': r.content,
                    'metadata': r.metadata,
                    'processing_time': r.processing_time,
                    'error_message': r.error_message
                } for r in results], indent=2)
            
            # Output to file or stdout
            if output:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
                click.echo(f"Output written to {output}")
            else:
                click.echo(formatted_output)
                
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            raise click.Abort()
        finally:
            await api.cleanup()
    
    asyncio.run(_batch())


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
