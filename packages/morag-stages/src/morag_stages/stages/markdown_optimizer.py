"""Markdown optimizer stage implementation."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from morag_core.config.unified import MarkdownOptimizerConfig

from ..error_handling import stage_error_handler, validation_error_handler
from ..exceptions import StageExecutionError, StageValidationError
from ..models import (
    Stage,
    StageContext,
    StageMetadata,
    StageResult,
    StageStatus,
    StageType,
)

# Import LLM services with graceful fallback
try:
    from morag_reasoning.llm import LLMClient
    from morag_reasoning.llm import LLMConfig as ReasoningLLMConfig

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

    # Create placeholder classes for runtime
    class LLMClient:  # type: ignore
        pass

    class ReasoningLLMConfig:  # type: ignore
        pass


logger = structlog.get_logger(__name__)


class MarkdownOptimizerStage(Stage):
    """Stage that optimizes markdown content using LLM."""

    def __init__(self, stage_type: StageType = StageType.MARKDOWN_OPTIMIZER):
        """Initialize markdown optimizer stage."""
        super().__init__(stage_type)

        if not LLM_AVAILABLE:
            logger.warning("LLM services not available for markdown optimization")

        self.agent = None

    @stage_error_handler("markdown_optimizer_execute")
    async def execute(
        self,
        input_files: List[Path],
        context: StageContext,
        output_dir: Optional[Path] = None,
    ) -> StageResult:
        """Execute markdown optimization on input files.

        Args:
            input_files: List of input markdown files
            context: Stage execution context
            output_dir: Optional output directory override

        Returns:
            Stage execution result
        """
        if len(input_files) != 1:
            raise StageValidationError(
                "Markdown optimizer stage requires exactly one input file",
                stage_type=self.stage_type.value,
                invalid_files=[str(f) for f in input_files],
            )

        input_file = input_files[0]

        # Load configuration from environment variables with context overrides
        context_config = context.get_stage_config(self.stage_type)
        config = MarkdownOptimizerConfig.from_env_and_overrides(context_config)

        logger.info(
            "Starting markdown optimization", input_file=str(input_file), config=config
        )

        try:
            # Get output directory from parameter or context
            effective_output_dir = output_dir or context.output_dir

            # Generate output filename
            output_file = effective_output_dir / f"{input_file.stem}.opt.md"
            effective_output_dir.mkdir(parents=True, exist_ok=True)

            # Read input markdown
            markdown_content = input_file.read_text(encoding="utf-8")

            # Extract metadata and content
            metadata, content = self._extract_metadata_and_content(markdown_content)

            # Optimize content if LLM is available and API key is configured
            api_key_available = self._check_api_key_available()
            if LLM_AVAILABLE and config.enabled and api_key_available:
                try:
                    optimized_content = await self._optimize_with_llm(
                        content, metadata, config, context
                    )
                    optimization_applied = True
                except Exception as e:
                    logger.warning(
                        "LLM optimization failed, using basic cleanup", error=str(e)
                    )
                    optimized_content = self._basic_text_cleanup(content)
                    optimization_applied = False
            else:
                # Fallback: basic text cleanup
                optimized_content = self._basic_text_cleanup(content)
                optimization_applied = False

            # Reconstruct markdown with metadata
            final_markdown = self._reconstruct_markdown(metadata, optimized_content)

            # Write to file
            output_file.write_text(final_markdown, encoding="utf-8")

            # Create metadata
            stage_metadata = StageMetadata(
                execution_time=0.0,  # Will be set by manager
                start_time=datetime.now(),
                input_files=[str(input_file)],
                output_files=[str(output_file)],
                config_used=config.model_dump()
                if hasattr(config, "model_dump")
                else config.__dict__,
                metrics={
                    "optimization_applied": optimization_applied,
                    "input_length": len(content),
                    "output_length": len(optimized_content),
                    "length_change": len(optimized_content) - len(content),
                    "has_metadata": bool(metadata),
                },
            )

            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file],
                metadata=stage_metadata,
                data={
                    "optimization_applied": optimization_applied,
                    "original_length": len(content),
                    "optimized_length": len(optimized_content),
                },
            )

        except Exception as e:
            logger.error(
                "Markdown optimization failed", input_file=str(input_file), error=str(e)
            )
            raise StageExecutionError(
                f"Markdown optimization failed: {e}",
                stage_type=self.stage_type.value,
                original_error=e,
            )

    @validation_error_handler("markdown_optimizer_validate_inputs")
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for markdown optimization.

        Args:
            input_files: List of input file paths

        Returns:
            True if inputs are valid
        """
        logger.debug(
            "Validating markdown optimizer inputs",
            input_count=len(input_files),
            input_files=[str(f) for f in input_files],
        )

        if len(input_files) != 1:
            logger.error(
                "Invalid input count for markdown optimizer",
                expected=1,
                actual=len(input_files),
                files=[str(f) for f in input_files],
            )
            return False

        input_file = input_files[0]

        # Check if file exists and is markdown
        if not input_file.exists():
            logger.error(
                "Input file does not exist for markdown optimizer",
                file_path=str(input_file),
            )
            return False

        if input_file.suffix.lower() not in [".md", ".markdown"]:
            logger.error(
                "Input file is not markdown for markdown optimizer",
                file_path=str(input_file),
                suffix=input_file.suffix,
            )
            return False

        logger.debug(
            "Input validation successful for markdown optimizer",
            file_path=str(input_file),
        )
        return True

    def get_dependencies(self) -> List[StageType]:
        """Get stage dependencies.

        Returns:
            List containing markdown-conversion stage
        """
        return [StageType.MARKDOWN_CONVERSION]

    def get_expected_outputs(
        self, input_files: List[Path], context: StageContext
    ) -> List[Path]:
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
        from ..file_manager import sanitize_filename

        sanitized_name = sanitize_filename(input_file.stem)
        output_file = context.output_dir / f"{sanitized_name}.opt.md"
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

        return bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )

    def _extract_metadata_and_content(
        self, markdown: str
    ) -> tuple[Dict[str, Any], str]:
        """Extract metadata and content from markdown (supports both YAML frontmatter and H1+H2 format).

        Args:
            markdown: Full markdown content

        Returns:
            Tuple of (metadata dict, content string)
        """
        # Check for YAML frontmatter (legacy format)
        if markdown.startswith("---\n"):
            parts = markdown.split("---\n", 2)
            if len(parts) >= 3:
                yaml_content = parts[1]
                content = parts[2]

                # Parse YAML metadata (simple parsing)
                metadata = {}
                for line in yaml_content.strip().split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        metadata[key] = value

                return metadata, content

        # Check for new H1+H2 format
        lines = markdown.split("\n")
        if lines and lines[0].startswith("# "):
            # Extract title from H1
            title_line = lines[0][2:].strip()
            metadata = {"title": title_line}

            # Look for information section
            info_section_start = -1
            for i, line in enumerate(lines):
                if line.startswith("## ") and "Information" in line:
                    info_section_start = i
                    break

            if info_section_start > 0:
                # Extract metadata from information section
                for i in range(info_section_start + 1, len(lines)):
                    line = lines[i].strip()
                    if line.startswith("- **") and "**:" in line:
                        # Parse metadata line: - **Key**: Value
                        key_end = line.find("**:", 4)
                        if key_end > 4:
                            key = line[4:key_end].strip()
                            value = line[key_end + 3 :].strip()
                            metadata[key.lower().replace(" ", "_")] = value
                    elif line.startswith("## ") or (line and not line.startswith("- ")):
                        # End of information section
                        break

            return metadata, markdown

        return {}, markdown

    def _reconstruct_markdown(self, metadata: Dict[str, Any], content: str) -> str:
        """Reconstruct markdown - return optimized content directly without adding structure.

        Args:
            metadata: Metadata dictionary (not used in current implementation)
            content: Optimized content string

        Returns:
            The optimized content as-is, without additional structure
        """
        # Clean up any unwanted markdown code block wrapping that the LLM might have added
        content = content.strip()

        # Remove markdown code block wrapping if present
        if content.startswith("```markdown\n") and content.endswith("\n```"):
            content = content[12:-4].strip()
        elif content.startswith("```\n") and content.endswith("\n```"):
            content = content[4:-4].strip()

        # Return the cleaned content directly
        return content

    async def _optimize_with_llm(
        self,
        content: str,
        metadata: Dict[str, Any],
        config: MarkdownOptimizerConfig,
        context: StageContext = None,
    ) -> str:
        """Optimize content using LLM with text splitting for large files.

        Args:
            content: Content to optimize
            metadata: Document metadata
            config: Stage configuration
            context: Stage execution context (for model overrides)

        Returns:
            Optimized content
        """
        try:
            # Get LLM configuration with stage-specific overrides
            unified_llm_config = config.get_llm_config()

            # Check for agent-specific model overrides from context
            model_override = None
            if context and hasattr(context, "config") and context.config:
                model_config = context.config.get("model_config", {})
                if model_config:
                    # Check for markdown optimizer specific model
                    markdown_optimizer_model = model_config.get("agent_models", {}).get(
                        "markdown_optimizer"
                    )
                    if markdown_optimizer_model:
                        model_override = markdown_optimizer_model
                        logger.info(
                            f"Using markdown optimizer agent model override: {model_override}"
                        )
                    # Check for default model override
                    elif model_config.get("default_model"):
                        model_override = model_config["default_model"]
                        logger.info(f"Using default model override: {model_override}")

            # Convert to reasoning LLMConfig format
            reasoning_config = ReasoningLLMConfig(
                provider=unified_llm_config.provider,
                model=model_override or unified_llm_config.model,
                api_key=unified_llm_config.api_key,
                temperature=unified_llm_config.temperature,
                max_tokens=unified_llm_config.max_tokens,
                max_retries=unified_llm_config.max_retries,
            )
            llm_client = LLMClient(reasoning_config)

            # Determine if content needs splitting
            if len(content) <= config.max_chunk_size:
                # Content is small enough to process in one request
                logger.info(
                    "Processing content in single LLM request",
                    content_length=len(content),
                )
                return await self._optimize_single_chunk(
                    llm_client, content, metadata, config
                )
            else:
                # Content is too large, split and process in chunks
                logger.info(
                    "Content too large, splitting for multiple LLM requests",
                    content_length=len(content),
                    max_chunk_size=config.max_chunk_size,
                )
                return await self._optimize_with_splitting(
                    llm_client, content, metadata, config
                )

        except ImportError:
            logger.warning("LLM reasoning module not available, using basic cleanup")
            return self._basic_text_cleanup(content)
        except Exception as e:
            logger.warning("LLM optimization failed, using basic cleanup", error=str(e))
            return self._basic_text_cleanup(content)

    def _basic_text_cleanup(self, content: str) -> str:
        """Basic text cleanup without LLM.

        Args:
            content: Content to clean up

        Returns:
            Cleaned up content
        """
        # Remove excessive whitespace
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

        # Fix common formatting issues
        content = re.sub(r"([.!?])\s*\n\s*([A-Z])", r"\1 \2", content)

        # Normalize line endings
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # Preserve formatting by only stripping leading/trailing whitespace, not internal formatting
        return content.strip(" \t")

    def _get_api_key(self) -> Optional[str]:
        """Get API key for LLM operations.

        Returns:
            API key string or None if not available
        """
        import os

        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    async def _optimize_single_chunk(
        self,
        llm_client,
        content: str,
        metadata: Dict[str, Any],
        config: MarkdownOptimizerConfig,
    ) -> str:
        """Optimize content in a single LLM request.

        Args:
            llm_client: LLM client instance
            content: Content to optimize
            metadata: Document metadata
            config: Stage configuration

        Returns:
            Optimized content
        """
        # Determine content type for appropriate prompting
        content_type = self._determine_content_type(metadata, content)

        # Create system and user prompts
        system_prompt = self._get_system_prompt(content_type, config)
        user_prompt = self._get_user_prompt(content, metadata, config)

        # Generate optimized content
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await llm_client.generate_from_messages(
            messages, max_tokens=config.max_tokens, temperature=config.temperature
        )

        # Preserve formatting by only stripping leading/trailing whitespace, not internal formatting
        if isinstance(response, str):
            return response.strip(" \t")
        else:
            return str(response).strip(" \t")

    async def _optimize_with_splitting(
        self,
        llm_client,
        content: str,
        metadata: Dict[str, Any],
        config: MarkdownOptimizerConfig,
    ) -> str:
        """Optimize content by splitting into chunks and processing separately.

        Args:
            llm_client: LLM client instance
            content: Content to optimize
            metadata: Document metadata
            config: Stage configuration

        Returns:
            Optimized content reassembled from chunks
        """
        # Split content into manageable chunks
        chunks = self._split_content_intelligently(content, config.max_chunk_size)

        logger.info(
            "Split content into chunks for optimization",
            num_chunks=len(chunks),
            avg_chunk_size=sum(len(c) for c in chunks) // len(chunks) if chunks else 0,
        )

        # Process each chunk
        optimized_chunks = []
        content_type = self._determine_content_type(metadata, content)

        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Processing chunk {i}/{len(chunks)}", chunk_size=len(chunk))

            try:
                # Create context-aware prompts for chunk processing
                system_prompt = self._get_chunk_system_prompt(
                    content_type, config, i, len(chunks)
                )
                user_prompt = self._get_chunk_user_prompt(
                    chunk, metadata, config, i, len(chunks)
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                optimized_chunk = await llm_client.generate_from_messages(
                    messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )

                # Preserve formatting by only stripping leading/trailing whitespace, not internal formatting
                optimized_chunks.append(optimized_chunk.strip(" \t"))

            except Exception as e:
                logger.warning(
                    f"Failed to optimize chunk {i}, using original", error=str(e)
                )
                optimized_chunks.append(chunk)

        # Reassemble optimized content
        return self._reassemble_chunks(optimized_chunks, content_type)

    def _determine_content_type(self, metadata: Dict[str, Any], content: str) -> str:
        """Determine the type of content for appropriate optimization.

        Args:
            metadata: Document metadata
            content: Content to analyze

        Returns:
            Content type string (video, audio, document, web_content, general)
        """
        # Check metadata for content type hints
        if metadata:
            source_type = metadata.get("source_type", "").lower()
            if "video" in source_type or "mp4" in source_type:
                return "video"
            elif "audio" in source_type or any(
                ext in source_type for ext in ["mp3", "wav", "flac"]
            ):
                return "audio"
            elif "pdf" in source_type or "document" in source_type:
                return "document"
            elif "web" in source_type or "html" in source_type:
                return "web_content"

        # Analyze content for type indicators
        if (
            "[" in content
            and "]" in content
            and any(time_pattern in content for time_pattern in [":", "min", "sec"])
        ):
            # Likely contains timestamps
            if "speaker" in content.lower() or any(
                speaker in content.lower()
                for speaker in ["person", "interviewer", "host"]
            ):
                return "video"  # Video with speakers
            else:
                return "audio"  # Audio transcript
        elif content.count("#") > 3:  # Multiple headings suggest document structure
            return "document"
        elif "http" in content or "www." in content:
            return "web_content"
        else:
            return "general"

    def _get_system_prompt(
        self, content_type: str, config: MarkdownOptimizerConfig
    ) -> str:
        """Get system prompt for optimization.

        Args:
            content_type: Type of content being optimized
            config: Stage configuration

        Returns:
            System prompt string
        """
        base_prompt = """You are a text correction specialist. Your ONLY task is to fix OCR errors, punctuation, and basic formatting issues while preserving ALL original content exactly as written.

CRITICAL CONTENT REQUIREMENT: You MUST preserve EVERY SINGLE WORD and ALL MEANING from the original content. DO NOT summarize, condense, rephrase, or change the meaning in any way. DO NOT add, remove, or generate any new content.

CRITICAL LANGUAGE REQUIREMENT: You MUST preserve the EXACT ORIGINAL LANGUAGE of the content. DO NOT translate or switch to any other language.

CRITICAL OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the corrected version of the provided content
- DO NOT wrap the content in code blocks (```markdown or ```)
- DO NOT add any explanatory text, metadata, or JSON wrapping
- Return the content exactly as it should appear in the final markdown file

YOUR ONLY ALLOWED CORRECTIONS:
1. Fix OCR errors where letters are missing spaces (e.g., "P anzliche" → "Pflanzliche")
2. Fix obvious character recognition errors (e.g., "fl" → "fl", "fi" → "fi")
3. Fix punctuation spacing and basic formatting
4. Fix broken words that are clearly OCR mistakes
5. Ensure proper markdown formatting structure

STRICTLY FORBIDDEN:
- DO NOT summarize or condense any content
- DO NOT rephrase or rewrite sentences
- DO NOT change the meaning or structure of paragraphs
- DO NOT remove any information, facts, or details
- DO NOT add new content or explanations
- DO NOT change the order of information
- DO NOT merge or split paragraphs unless fixing clear OCR line break errors

PRESERVE EXACTLY:
- All facts, data, and information
- All names, places, and specific details
- All lists, instructions, and recommendations
- All paragraph structure and organization
- All headings and subheadings
- All tables and structured content"""

        if config.fix_transcription_errors:
            base_prompt += " Fix any transcription errors you notice."

        if config.improve_structure:
            base_prompt += " Improve the document structure and formatting."

        if config.preserve_timestamps and content_type in ["video", "audio"]:
            base_prompt += (
                " IMPORTANT: Preserve all timestamp information exactly as provided."
            )

        if config.preserve_metadata:
            base_prompt += " Preserve all metadata and structural elements that contain substantive information."

        # Add content-type specific instructions
        if content_type == "video":
            base_prompt += " This is a video transcript. Maintain speaker labels and timestamps while removing any video-specific clutter like chapter markers or promotional segments."
        elif content_type == "audio":
            base_prompt += " This is an audio transcript. Maintain speaker labels and timestamps while removing any audio-specific clutter like intro/outro music references or promotional segments."
        elif content_type == "document":
            base_prompt += " This is a document. Maintain proper heading structure and formatting while removing document-specific clutter like page headers, footers, and navigation elements."

        return base_prompt

    def _get_user_prompt(
        self, content: str, metadata: Dict[str, Any], config: MarkdownOptimizerConfig
    ) -> str:
        """Get user prompt for optimization.

        Args:
            content: Content to optimize
            metadata: Document metadata
            config: Stage configuration

        Returns:
            User prompt string
        """
        # Extract language information for preservation
        language_instruction = "\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST preserve the EXACT ORIGINAL LANGUAGE of the content. DO NOT translate or switch to any other language, especially English. Analyze the content language and keep ALL content in that same language."

        if metadata:
            detected_language = metadata.get("language", "").lower()
            if detected_language:
                language_names = {
                    "en": "English",
                    "de": "German",
                    "fr": "French",
                    "es": "Spanish",
                    "it": "Italian",
                    "pt": "Portuguese",
                    "nl": "Dutch",
                    "ru": "Russian",
                    "zh": "Chinese",
                    "ja": "Japanese",
                    "ko": "Korean",
                }
                language_name = language_names.get(
                    detected_language, detected_language.title()
                )
                language_instruction = f"\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST preserve the original language ({language_name}) AT ALL COSTS. DO NOT translate or switch to any other language, especially English. Keep ALL content in {language_name}."

        # Check for existing document type prefixes in title to avoid duplication
        title_prefix_instruction = ""
        if content.strip().startswith("# "):
            first_line = content.split("\n")[0]
            if any(
                prefix in first_line
                for prefix in [
                    "# Video Analysis",
                    "# Document Analysis",
                    "# Audio Analysis",
                ]
            ):
                title_prefix_instruction = "\n\nIMPORTANT: The document already has a proper analysis prefix in the title. DO NOT add additional prefixes like 'Document Analysis:' to the existing title."

        # Add strong anti-hallucination instructions
        anti_hallucination_instruction = "\n\nCRITICAL: Work ONLY with the content provided below. DO NOT add any new content, examples, conversations, or text that is not in the original. DO NOT generate or hallucinate any additional content."

        prompt = f"Please optimize ONLY the following content:{language_instruction}{title_prefix_instruction}{anti_hallucination_instruction}\n\n{content}"

        if metadata:
            prompt = f"Document metadata: {metadata}\n\n" + prompt

        return prompt

    def _split_content_intelligently(
        self, content: str, max_chunk_size: int
    ) -> List[str]:
        """Split content into chunks while preserving structure.

        Args:
            content: Content to split
            max_chunk_size: Maximum size per chunk in characters

        Returns:
            List of content chunks
        """
        if len(content) <= max_chunk_size:
            return [content]

        chunks = []

        # Try to split at major section boundaries first (## headers)
        major_sections = re.split(r"\n(?=##\s)", content)

        current_chunk = ""

        for section in major_sections:
            section = section.strip()
            if not section:
                continue

            # If adding this section would exceed chunk size
            if current_chunk and len(current_chunk) + len(section) + 2 > max_chunk_size:
                # Try to split the current chunk at minor boundaries
                if current_chunk:
                    chunks.extend(
                        self._split_section_at_boundaries(current_chunk, max_chunk_size)
                    )
                current_chunk = section
            else:
                # Add section to current chunk
                if current_chunk:
                    current_chunk += f"\n\n{section}"
                else:
                    current_chunk = section

        # Handle remaining content
        if current_chunk:
            chunks.extend(
                self._split_section_at_boundaries(current_chunk, max_chunk_size)
            )

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split_section_at_boundaries(
        self, section: str, max_chunk_size: int
    ) -> List[str]:
        """Split a section at natural boundaries.

        Args:
            section: Section to split
            max_chunk_size: Maximum chunk size

        Returns:
            List of section chunks
        """
        if len(section) <= max_chunk_size:
            return [section]

        chunks = []

        # Try splitting at paragraph boundaries first
        paragraphs = section.split("\n\n")
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed chunk size
            if (
                current_chunk
                and len(current_chunk) + len(paragraph) + 2 > max_chunk_size
            ):
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If single paragraph is too large, split it further
                if len(paragraph) > max_chunk_size:
                    chunks.extend(
                        self._split_paragraph_at_sentences(paragraph, max_chunk_size)
                    )
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += f"\n\n{paragraph}"
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_paragraph_at_sentences(
        self, paragraph: str, max_chunk_size: int
    ) -> List[str]:
        """Split a large paragraph at sentence boundaries.

        Args:
            paragraph: Paragraph to split
            max_chunk_size: Maximum chunk size

        Returns:
            List of paragraph chunks
        """
        if len(paragraph) <= max_chunk_size:
            return [paragraph]

        # Split at sentence boundaries (simple approach)
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed chunk size
            if (
                current_chunk
                and len(current_chunk) + len(sentence) + 1 > max_chunk_size
            ):
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If single sentence is too large, split at word boundaries
                if len(sentence) > max_chunk_size:
                    chunks.extend(
                        self._split_at_word_boundaries(sentence, max_chunk_size)
                    )
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += f" {sentence}"
                else:
                    current_chunk = sentence

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_at_word_boundaries(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text at word boundaries as a last resort.

        Args:
            text: Text to split
            max_chunk_size: Maximum chunk size

        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        words = text.split()
        current_chunk = ""

        for word in words:
            # If adding this word would exceed chunk size
            if current_chunk and len(current_chunk) + len(word) + 1 > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                # Add word to current chunk
                if current_chunk:
                    current_chunk += f" {word}"
                else:
                    current_chunk = word

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_chunk_system_prompt(
        self,
        content_type: str,
        config: MarkdownOptimizerConfig,
        chunk_num: int,
        total_chunks: int,
    ) -> str:
        """Get system prompt for chunk processing.

        Args:
            content_type: Type of content being optimized
            config: Stage configuration
            chunk_num: Current chunk number (1-based)
            total_chunks: Total number of chunks

        Returns:
            System prompt string
        """
        base_prompt = f"""You are an expert content optimizer. You are processing chunk {chunk_num} of {total_chunks} from a larger document.

Your task is to improve the readability and structure of this content chunk while preserving all important information. This is part of a larger document, so maintain consistency and don't add introductory or concluding statements that assume this is a complete document.

CRITICAL CONTENT REQUIREMENT: You MUST work ONLY with the content provided in the user message. DO NOT add, generate, or hallucinate any new content. DO NOT add examples, conversations, or any content that is not in the original document.

CRITICAL LANGUAGE REQUIREMENT: You MUST preserve the EXACT ORIGINAL LANGUAGE of the content. DO NOT translate or switch to any other language, especially English or Spanish. Analyze the content language and keep ALL content in that same language.

CRITICAL OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the optimized version of the provided content
- DO NOT wrap the content in code blocks (```markdown or ```)
- DO NOT add any explanatory text, metadata, or JSON wrapping
- DO NOT add prefixes like "Document Analysis:" or "Content:" to titles
- DO NOT add document information sections or metadata sections
- DO NOT generate or add any new content, examples, or conversations
- Return the content exactly as it should appear in the final markdown file

CRITICAL FORMATTING REQUIREMENTS:
- PRESERVE ALL NEWLINES AND LINE BREAKS exactly as they appear in the original content
- PRESERVE ALL WHITESPACE FORMATTING including indentation and spacing
- Return content in valid markdown format
- DO NOT remove line breaks from transcriptions or structured content

REMOVE CLUTTER: Remove non-essential elements such as:
- Table of contents and index entries
- Acknowledgements and dedications
- Navigation elements and boilerplate text
- Repetitive headers and footers
- Page numbers and reference markers
- Copyright notices and disclaimers (unless specifically relevant to content)
- Advertisement sections and promotional content
- Redundant metadata and formatting artifacts

PRESERVE ESSENTIAL CONTENT: Keep all substantive information, facts, data, and meaningful structural elements."""

        if config.fix_transcription_errors:
            base_prompt += " Fix any transcription errors you notice."

        if config.improve_structure:
            base_prompt += " Improve the formatting and structure within this chunk."

        if config.preserve_timestamps and content_type in ["video", "audio"]:
            base_prompt += " CRITICAL: Preserve all timestamp information exactly as provided - do not modify timestamps."

        if config.preserve_metadata:
            base_prompt += " Preserve all metadata and structural elements that contain substantive information."

        # Add content-type specific instructions
        if content_type == "video":
            base_prompt += " This is part of a video transcript. Maintain speaker labels and timestamps exactly while removing any video-specific clutter like chapter markers or promotional segments."
        elif content_type == "audio":
            base_prompt += " This is part of an audio transcript. Maintain speaker labels and timestamps exactly while removing any audio-specific clutter like intro/outro music references or promotional segments."
        elif content_type == "document":
            base_prompt += " This is part of a document. Maintain proper heading structure and formatting while removing document-specific clutter like page headers, footers, and navigation elements."

        base_prompt += " Return only the optimized content without any explanatory text or metadata."

        return base_prompt

    def _get_chunk_user_prompt(
        self,
        chunk: str,
        metadata: Dict[str, Any],
        config: MarkdownOptimizerConfig,
        chunk_num: int,
        total_chunks: int,
    ) -> str:
        """Get user prompt for chunk processing.

        Args:
            chunk: Content chunk to optimize
            metadata: Document metadata
            config: Stage configuration
            chunk_num: Current chunk number
            total_chunks: Total number of chunks

        Returns:
            User prompt string
        """
        # Extract language information for preservation
        language_instruction = "\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST preserve the EXACT ORIGINAL LANGUAGE of the content. DO NOT translate or switch to any other language, especially English. Analyze the content language and keep ALL content in that same language."

        if metadata:
            detected_language = metadata.get("language", "").lower()
            if detected_language:
                language_names = {
                    "en": "English",
                    "de": "German",
                    "fr": "French",
                    "es": "Spanish",
                    "it": "Italian",
                    "pt": "Portuguese",
                    "nl": "Dutch",
                    "ru": "Russian",
                    "zh": "Chinese",
                    "ja": "Japanese",
                    "ko": "Korean",
                }
                language_name = language_names.get(
                    detected_language, detected_language.title()
                )
                language_instruction = f"\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST preserve the original language ({language_name}) AT ALL COSTS. DO NOT translate or switch to any other language, especially English. Keep ALL content in {language_name}."

        # Add strong anti-hallucination instructions for chunks
        anti_hallucination_instruction = "\n\nCRITICAL: Work ONLY with the content chunk provided below. DO NOT add any new content, examples, conversations, or text that is not in the original chunk. DO NOT generate or hallucinate any additional content."

        prompt = f"Optimize ONLY this content chunk ({chunk_num}/{total_chunks}):{language_instruction}{anti_hallucination_instruction}\n\n{chunk}"

        if chunk_num == 1 and metadata:
            prompt = f"Document context: {metadata}\n\n" + prompt

        return prompt

    def _reassemble_chunks(self, optimized_chunks: List[str], content_type: str) -> str:
        """Reassemble optimized chunks into final content.

        Args:
            optimized_chunks: List of optimized content chunks
            content_type: Type of content

        Returns:
            Reassembled optimized content
        """
        if not optimized_chunks:
            return ""

        # For most content types, simply join with double newlines
        if content_type in ["video", "audio"]:
            # For transcripts, preserve internal formatting but clean edges
            return "\n\n".join(
                chunk.strip(" \t") for chunk in optimized_chunks if chunk.strip(" \t")
            )
        else:
            # For documents and other content, preserve internal formatting but clean edges
            return "\n\n".join(
                chunk.strip(" \t") for chunk in optimized_chunks if chunk.strip(" \t")
            )
