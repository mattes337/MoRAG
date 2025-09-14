# Task 3: Implement markdown-optimizer Stage

## Overview
Implement the optional markdown-optimizer stage that uses LLM to improve and fix transcription errors in markdown text. This completely replaces any existing optimization logic.

## Objectives
- Design LLM prompt templates using **system/user message pattern**
- Implement text optimization pipeline with quality assessment
- Add configurable optimization strategies
- Preserve timestamps and metadata while improving content
- Create validation and quality metrics
- **Use system messages for instructions, user messages for content**
- **REMOVE ALL LEGACY OPTIMIZATION CODE**

## Deliverables

### 1. markdown-optimizer Stage Implementation (Complete Replacement)
```python
from morag_stages.models import Stage, StageType, StageResult, StageContext, StageStatus
from morag_services import GeminiLLMService
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import re
import yaml

class MarkdownOptimizerStage(Stage):
    def __init__(self, llm_service: GeminiLLMService):
        super().__init__(StageType.MARKDOWN_OPTIMIZER)
        self.llm_service = llm_service
        self.prompt_templates = PromptTemplateManager()
    
    async def execute(self, 
                     input_files: List[Path], 
                     context: StageContext) -> StageResult:
        """Optimize markdown content using LLM."""
        start_time = time.time()
        
        try:
            if len(input_files) != 1:
                raise ValueError("Stage 2 requires exactly one markdown input file")
            
            input_file = input_files[0]
            
            # Parse input markdown
            content, metadata = self._parse_markdown_file(input_file)
            
            # Determine optimization strategy
            config = context.config.get('stage2', {})
            optimization_type = self._determine_optimization_type(metadata, config)
            
            # Optimize content
            optimized_content = await self._optimize_content(
                content, metadata, optimization_type, config
            )
            
            # Validate optimization quality
            quality_metrics = self._assess_optimization_quality(
                content, optimized_content, metadata
            )
            
            # Generate optimized markdown output
            output_file = self._generate_optimized_markdown(
                optimized_content, metadata, input_file, context, quality_metrics
            )
            
            # Create optimization report
            report_file = self._create_optimization_report(
                quality_metrics, input_file, context
            )
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file, report_file],
                metadata={
                    "optimization_type": optimization_type,
                    "quality_improvement": quality_metrics.get('improvement_score', 0),
                    "original_length": len(content),
                    "optimized_length": len(optimized_content),
                    "processing_time": execution_time
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for Stage 2."""
        if len(input_files) != 1:
            return False
        
        input_file = input_files[0]
        
        # Check if file exists and is markdown
        return input_file.exists() and input_file.suffix == '.md'
    
    def get_dependencies(self) -> List[StageType]:
        """markdown-optimizer depends on markdown-conversion."""
        return [StageType.MARKDOWN_CONVERSION]
    
    def _parse_markdown_file(self, input_file: Path) -> tuple[str, Dict[str, Any]]:
        """Parse markdown file and extract metadata."""
        content = input_file.read_text(encoding='utf-8')
        
        # Extract YAML frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata = yaml.safe_load(parts[1])
                content = parts[2].strip()
            else:
                metadata = {}
        else:
            metadata = {}
        
        return content, metadata
    
    def _determine_optimization_type(self, 
                                   metadata: Dict[str, Any], 
                                   config: Dict[str, Any]) -> str:
        """Determine the type of optimization needed."""
        content_type = metadata.get('content_type', 'unknown')
        
        # Override from config
        if 'optimization_type' in config:
            return config['optimization_type']
        
        # Auto-detect based on content type
        if content_type in ['video', 'audio']:
            return 'transcription'
        elif content_type == 'document':
            return 'document'
        elif content_type == 'web':
            return 'web_content'
        else:
            return 'general'
    
    async def _optimize_content(self,
                               content: str,
                               metadata: Dict[str, Any],
                               optimization_type: str,
                               config: Dict[str, Any]) -> str:
        """Optimize content using LLM with system/user message pattern."""

        # Get appropriate prompt templates (system and user)
        system_message, user_message_template = self.prompt_templates.get_system_user_templates(optimization_type)

        # Prepare user message context
        user_context = {
            'content': content,
            'metadata': metadata,
            'preserve_timestamps': config.get('preserve_timestamps', True),
            'fix_transcription_errors': config.get('fix_transcription_errors', True),
            'improve_readability': config.get('improve_readability', True),
            'preserve_structure': config.get('preserve_structure', True)
        }

        # Generate user message
        user_message = user_message_template.format(**user_context)

        # Call LLM with system/user message pattern
        response = await self.llm_service.generate_text_with_system_message(
            system_message=system_message,
            user_message=user_message,
            model=config.get('llm_model', 'gemini-pro'),
            temperature=config.get('temperature', 0.3),
            max_tokens=config.get('max_tokens', 8000)
        )

        return response.strip()
    
    def _assess_optimization_quality(self, 
                                   original: str, 
                                   optimized: str, 
                                   metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of optimization."""
        
        metrics = {
            'original_word_count': len(original.split()),
            'optimized_word_count': len(optimized.split()),
            'length_change_ratio': len(optimized) / len(original) if original else 0,
            'timestamp_preservation': self._check_timestamp_preservation(original, optimized),
            'structure_preservation': self._check_structure_preservation(original, optimized),
            'readability_improvement': self._assess_readability_improvement(original, optimized)
        }
        
        # Calculate overall improvement score
        metrics['improvement_score'] = self._calculate_improvement_score(metrics)
        
        return metrics
    
    def _check_timestamp_preservation(self, original: str, optimized: str) -> float:
        """Check how well timestamps were preserved."""
        # Find timestamps in both versions
        timestamp_pattern = r'\[(\d{2}:\d{2}(?::\d{2})?(?:\s*-\s*\d{2}:\d{2}(?::\d{2})?)?)\]'
        
        original_timestamps = set(re.findall(timestamp_pattern, original))
        optimized_timestamps = set(re.findall(timestamp_pattern, optimized))
        
        if not original_timestamps:
            return 1.0  # No timestamps to preserve
        
        preserved = len(original_timestamps.intersection(optimized_timestamps))
        return preserved / len(original_timestamps)
    
    def _check_structure_preservation(self, original: str, optimized: str) -> float:
        """Check how well document structure was preserved."""
        # Count headers, lists, etc.
        def count_structure_elements(text: str) -> Dict[str, int]:
            return {
                'headers': len(re.findall(r'^#+\s', text, re.MULTILINE)),
                'lists': len(re.findall(r'^\s*[-*+]\s', text, re.MULTILINE)),
                'numbered_lists': len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE)),
                'code_blocks': len(re.findall(r'```', text)) // 2
            }
        
        original_structure = count_structure_elements(original)
        optimized_structure = count_structure_elements(optimized)
        
        # Calculate preservation ratio
        total_original = sum(original_structure.values())
        if total_original == 0:
            return 1.0
        
        preserved = sum(
            min(original_structure[key], optimized_structure[key])
            for key in original_structure
        )
        
        return preserved / total_original
    
    def _assess_readability_improvement(self, original: str, optimized: str) -> float:
        """Assess readability improvement (simplified metric)."""
        # Simple metrics: sentence length, word complexity
        def avg_sentence_length(text: str) -> float:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                return 0
            return sum(len(s.split()) for s in sentences) / len(sentences)
        
        original_avg = avg_sentence_length(original)
        optimized_avg = avg_sentence_length(optimized)
        
        # Prefer moderate sentence lengths (10-20 words)
        def score_sentence_length(avg_len: float) -> float:
            if 10 <= avg_len <= 20:
                return 1.0
            elif avg_len < 10:
                return avg_len / 10
            else:
                return 20 / avg_len
        
        original_score = score_sentence_length(original_avg)
        optimized_score = score_sentence_length(optimized_avg)
        
        return optimized_score - original_score
    
    def _calculate_improvement_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall improvement score."""
        weights = {
            'timestamp_preservation': 0.3,
            'structure_preservation': 0.3,
            'readability_improvement': 0.4
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
```

### 2. Prompt Template Manager (Complete Replacement with System/User Pattern)
```python
class PromptTemplateManager:
    """Manages LLM prompt templates using system/user message pattern."""

    def __init__(self):
        self.system_templates = {
            'transcription': self._get_transcription_system_template(),
            'document': self._get_document_system_template(),
            'web_content': self._get_web_content_system_template(),
            'general': self._get_general_system_template()
        }
        self.user_templates = {
            'transcription': self._get_transcription_user_template(),
            'document': self._get_document_user_template(),
            'web_content': self._get_web_content_user_template(),
            'general': self._get_general_user_template()
        }

    def get_system_user_templates(self, optimization_type: str) -> tuple[str, str]:
        """Get system and user message templates for optimization type."""
        system_template = self.system_templates.get(optimization_type, self.system_templates['general'])
        user_template = self.user_templates.get(optimization_type, self.user_templates['general'])
        return system_template, user_template
    
    def _get_transcription_system_template(self) -> str:
        return """You are an expert editor specializing in improving transcribed content. Your task is to optimize transcribed text while preserving its original meaning and structure.

REQUIREMENTS:
1. Fix transcription errors and improve readability
2. Preserve ALL timestamps in their original format [HH:MM:SS] or [HH:MM:SS - HH:MM:SS]
3. Maintain speaker labels and dialogue structure
4. Improve grammar and punctuation without changing meaning
5. Keep technical terms and proper nouns accurate
6. Preserve paragraph breaks and topic transitions

Provide the optimized version of the content, maintaining the exact same structure but with improved clarity and accuracy."""

    def _get_transcription_user_template(self) -> str:
        return """Please optimize the following transcribed content:

CONTENT TO OPTIMIZE:
{content}

METADATA:
- Content Type: {metadata.get('content_type', 'unknown')}
- Duration: {metadata.get('duration', 'unknown')}
- Language: {metadata.get('language', 'unknown')}

OPTIMIZATION SETTINGS:
- Preserve Timestamps: {preserve_timestamps}
- Fix Transcription Errors: {fix_transcription_errors}
- Improve Readability: {improve_readability}"""
    
    def _get_document_system_template(self) -> str:
        return """You are an expert editor specializing in document optimization. Your task is to improve document content while preserving its structure and meaning.

REQUIREMENTS:
1. Improve readability and flow
2. Fix grammar and punctuation errors
3. Preserve document structure (headers, lists, formatting)
4. Maintain technical accuracy
5. Keep all important information intact
6. Improve clarity without changing meaning

Provide the optimized version of the document content."""

    def _get_document_user_template(self) -> str:
        return """Please optimize the following document content:

CONTENT TO OPTIMIZE:
{content}

METADATA:
- Content Type: {metadata.get('content_type', 'unknown')}
- Source: {metadata.get('source_file', 'unknown')}

OPTIMIZATION SETTINGS:
- Preserve Structure: {preserve_structure}
- Improve Readability: {improve_readability}"""
    
    def _get_web_content_template(self) -> str:
        return """You are an expert editor specializing in web content optimization. Your task is to improve the following web content while preserving its key information.

REQUIREMENTS:
1. Remove navigation elements and boilerplate text
2. Improve content structure and readability
3. Preserve important links and references
4. Fix formatting issues from web scraping
5. Maintain factual accuracy
6. Create clear, coherent narrative flow

CONTENT TO OPTIMIZE:
{content}

METADATA:
- Source URL: {metadata.get('source_file', 'unknown')}
- Content Type: {metadata.get('content_type', 'unknown')}

Please provide the optimized version of the web content."""
    
    def _get_general_template(self) -> str:
        return """You are an expert editor. Your task is to improve the following content while preserving its original meaning and important structural elements.

REQUIREMENTS:
1. Improve readability and clarity
2. Fix grammar and punctuation errors
3. Preserve important structural elements
4. Maintain factual accuracy
5. Keep technical terms accurate
6. Improve overall flow and coherence

CONTENT TO OPTIMIZE:
{content}

Please provide the optimized version of the content."""
```

## Implementation Steps

1. **Create markdown-optimizer package structure**
2. **Implement MarkdownOptimizerStage class with system/user message pattern**
3. **Create prompt template system using system/user message separation**
4. **Add quality assessment metrics**
5. **Implement LLM integration with proper message structure**
6. **Add configuration support**
7. **Create optimization report generation**
8. **Add validation and error handling**
9. **REMOVE ALL LEGACY OPTIMIZATION CODE**
10. **Implement comprehensive testing and performance monitoring**

## Testing Requirements

- Unit tests for optimization logic
- LLM integration tests with mock responses
- Quality assessment validation
- Prompt template testing
- Error handling and edge cases
- Performance tests for large content

## Files to Create

- `packages/morag-stages/src/morag_stages/markdown_optimizer/__init__.py`
- `packages/morag-stages/src/morag_stages/markdown_optimizer/implementation.py`
- `packages/morag-stages/src/morag_stages/markdown_optimizer/prompts.py`
- `packages/morag-stages/src/morag_stages/markdown_optimizer/quality_assessment.py`
- `packages/morag-stages/tests/test_markdown_optimizer.py`
- `llm/MARKDOWN_OPTIMIZER.md` (LLM prompt documentation with system/user examples)

## Success Criteria

- Content optimization improves readability without losing information
- System/user message pattern is properly implemented
- Timestamps and structure are preserved correctly
- Quality metrics accurately assess optimization effectiveness
- LLM integration uses proper message structure and handles errors gracefully
- Performance is acceptable for typical content sizes
- **ALL LEGACY OPTIMIZATION CODE IS REMOVED**
- All tests pass with good coverage
