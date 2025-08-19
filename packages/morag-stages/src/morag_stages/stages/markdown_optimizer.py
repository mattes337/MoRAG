"""Markdown optimizer stage implementation."""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog

from ..models import Stage, StageType, StageStatus, StageResult, StageContext, StageMetadata
from ..exceptions import StageExecutionError, StageValidationError

# Import LLM services with graceful fallback
try:
    from morag_core.ai import create_agent_with_config, SummarizationAgent, AgentConfig
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_agent_with_config = None
    SummarizationAgent = None
    AgentConfig = None

logger = structlog.get_logger(__name__)


class MarkdownOptimizerStage(Stage):
    """Stage that optimizes markdown content using LLM."""
    
    def __init__(self, stage_type: StageType = StageType.MARKDOWN_OPTIMIZER):
        """Initialize markdown optimizer stage."""
        super().__init__(stage_type)
        
        if not LLM_AVAILABLE:
            logger.warning("LLM services not available for markdown optimization")
        
        self.agent = None
    
    async def execute(self, 
                     input_files: List[Path], 
                     context: StageContext) -> StageResult:
        """Execute markdown optimization on input files.
        
        Args:
            input_files: List of input markdown files
            context: Stage execution context
            
        Returns:
            Stage execution result
        """
        if len(input_files) != 1:
            raise StageValidationError(
                "Markdown optimizer stage requires exactly one input file",
                stage_type=self.stage_type.value,
                invalid_files=[str(f) for f in input_files]
            )
        
        input_file = input_files[0]
        config = context.get_stage_config(self.stage_type)
        
        logger.info("Starting markdown optimization", 
                   input_file=str(input_file),
                   config=config)
        
        try:
            # Generate output filename
            output_file = context.output_dir / f"{input_file.stem}.opt.md"
            context.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Read input markdown
            markdown_content = input_file.read_text(encoding='utf-8')
            
            # Extract metadata and content
            metadata, content = self._extract_metadata_and_content(markdown_content)
            
            # Optimize content if LLM is available and API key is configured
            api_key_available = self._check_api_key_available()
            if LLM_AVAILABLE and config.get('enabled', True) and api_key_available:
                try:
                    optimized_content = await self._optimize_with_llm(content, metadata, config)
                    optimization_applied = True
                except Exception as e:
                    logger.warning("LLM optimization failed, using basic cleanup", error=str(e))
                    optimized_content = self._basic_text_cleanup(content)
                    optimization_applied = False
            else:
                # Fallback: basic text cleanup
                optimized_content = self._basic_text_cleanup(content)
                optimization_applied = False
            
            # Reconstruct markdown with metadata
            final_markdown = self._reconstruct_markdown(metadata, optimized_content)
            
            # Write to file
            output_file.write_text(final_markdown, encoding='utf-8')
            
            # Create metadata
            stage_metadata = StageMetadata(
                execution_time=0.0,  # Will be set by manager
                start_time=datetime.now(),
                input_files=[str(input_file)],
                output_files=[str(output_file)],
                config_used=config,
                metrics={
                    "optimization_applied": optimization_applied,
                    "input_length": len(content),
                    "output_length": len(optimized_content),
                    "length_change": len(optimized_content) - len(content),
                    "has_metadata": bool(metadata)
                }
            )
            
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file],
                metadata=stage_metadata,
                data={
                    "optimization_applied": optimization_applied,
                    "original_length": len(content),
                    "optimized_length": len(optimized_content)
                }
            )
            
        except Exception as e:
            logger.error("Markdown optimization failed", 
                        input_file=str(input_file), 
                        error=str(e))
            raise StageExecutionError(
                f"Markdown optimization failed: {e}",
                stage_type=self.stage_type.value,
                original_error=e
            )
    
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for markdown optimization.
        
        Args:
            input_files: List of input file paths
            
        Returns:
            True if inputs are valid
        """
        if len(input_files) != 1:
            return False
        
        input_file = input_files[0]
        
        # Check if file exists and is markdown
        if not input_file.exists():
            return False
        
        if input_file.suffix.lower() not in ['.md', '.markdown']:
            return False
        
        return True
    
    def get_dependencies(self) -> List[StageType]:
        """Get stage dependencies.
        
        Returns:
            List containing markdown-conversion stage
        """
        return [StageType.MARKDOWN_CONVERSION]
    
    def get_expected_outputs(self, input_files: List[Path], context: StageContext) -> List[Path]:
        """Get expected output file paths.
        
        Args:
            input_files: List of input file paths
            context: Stage execution context
            
        Returns:
            List of expected output file paths
        """
        if len(input_files) != 1:
            return []
        
        input_file = input_files[0]
        output_file = context.output_dir / f"{input_file.stem}.opt.md"
        return [output_file]
    
    def is_optional(self) -> bool:
        """Check if this stage is optional.

        Returns:
            True - markdown optimizer is optional
        """
        return True

    def _check_api_key_available(self) -> bool:
        """Check if API key is available for LLM operations.

        Returns:
            True if API key is available
        """
        import os
        return bool(os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'))
    
    def _extract_metadata_and_content(self, markdown: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter and content from markdown.
        
        Args:
            markdown: Full markdown content
            
        Returns:
            Tuple of (metadata dict, content string)
        """
        # Check for YAML frontmatter
        if markdown.startswith('---\n'):
            parts = markdown.split('---\n', 2)
            if len(parts) >= 3:
                yaml_content = parts[1]
                content = parts[2]
                
                # Parse YAML metadata (simple parsing)
                metadata = {}
                for line in yaml_content.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        metadata[key] = value
                
                return metadata, content
        
        return {}, markdown
    
    def _reconstruct_markdown(self, metadata: Dict[str, Any], content: str) -> str:
        """Reconstruct markdown with metadata header.
        
        Args:
            metadata: Metadata dictionary
            content: Content string
            
        Returns:
            Complete markdown with metadata header
        """
        if not metadata:
            return content
        
        # Create YAML frontmatter
        yaml_lines = ["---"]
        for key, value in metadata.items():
            if isinstance(value, str):
                yaml_lines.append(f'{key}: "{value}"')
            else:
                yaml_lines.append(f'{key}: {value}')
        yaml_lines.append("---")
        yaml_lines.append("")  # Empty line after frontmatter
        
        return "\n".join(yaml_lines) + content
    
    async def _optimize_with_llm(self, content: str, metadata: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Optimize content using LLM.

        Args:
            content: Content to optimize
            metadata: Document metadata
            config: Stage configuration

        Returns:
            Optimized content
        """
        # For now, just return basic cleanup since LLM optimization needs more complex setup
        logger.info("LLM optimization not fully implemented, using basic cleanup")
        return self._basic_text_cleanup(content)
    
    def _basic_text_cleanup(self, content: str) -> str:
        """Basic text cleanup without LLM.
        
        Args:
            content: Content to clean up
            
        Returns:
            Cleaned up content
        """
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Fix common formatting issues
        content = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1 \2', content)
        
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        return content.strip()
    
    def _get_system_prompt(self, content_type: str, config: Dict[str, Any]) -> str:
        """Get system prompt for optimization.
        
        Args:
            content_type: Type of content being optimized
            config: Stage configuration
            
        Returns:
            System prompt string
        """
        base_prompt = """You are an expert content optimizer. Your task is to improve the readability and structure of the provided content while preserving all important information."""
        
        if config.get('fix_transcription_errors', True):
            base_prompt += " Fix any transcription errors you notice."
        
        if config.get('improve_structure', True):
            base_prompt += " Improve the document structure and formatting."
        
        if config.get('preserve_timestamps', True) and content_type in ['video', 'audio']:
            base_prompt += " IMPORTANT: Preserve all timestamp information exactly as provided."
        
        if config.get('preserve_metadata', True):
            base_prompt += " Preserve all metadata and structural elements."
        
        # Add content-type specific instructions
        if content_type == 'video':
            base_prompt += " This is a video transcript. Maintain speaker labels and timestamps."
        elif content_type == 'audio':
            base_prompt += " This is an audio transcript. Maintain speaker labels and timestamps."
        elif content_type == 'document':
            base_prompt += " This is a document. Maintain proper heading structure and formatting."
        
        return base_prompt
    
    def _get_user_prompt(self, content: str, metadata: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Get user prompt for optimization.
        
        Args:
            content: Content to optimize
            metadata: Document metadata
            config: Stage configuration
            
        Returns:
            User prompt string
        """
        prompt = f"Please optimize the following content:\n\n{content}"
        
        if metadata:
            prompt = f"Document metadata: {metadata}\n\n" + prompt
        
        return prompt
