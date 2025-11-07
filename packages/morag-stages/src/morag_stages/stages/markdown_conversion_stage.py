"""Main markdown conversion stage implementation."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog

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
from ..utils import detect_content_type, is_content_type

# Import sanitization function
try:
    from morag_core.exceptions import ValidationError
    from morag_core.utils.validation import sanitize_filepath
except ImportError:

    class ValidationError(Exception):
        """Fallback ValidationError class."""

        pass

    def sanitize_filepath(filepath: Union[str, Path], base_dir: Path = None) -> Path:
        """Fallback sanitization function with enhanced security."""
        import re
        from pathlib import Path

        if not filepath:
            raise ValidationError("Empty file path provided")

        path = Path(filepath)
        path_str = str(path)

        # Check for null bytes
        if "\x00" in path_str:
            raise ValidationError(f"Null byte detected in path: {filepath}")

        # Check for dangerous patterns
        dangerous_patterns = [
            r"[;&|`$()]",  # Shell metacharacters
            r"\$\(",  # Command substitution
            r"`.*`",  # Backtick command substitution
            r"\.\./",  # Directory traversal
            r"\.\.\\",  # Windows directory traversal
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, path_str):
                raise ValidationError(
                    f"Dangerous characters or patterns detected in path: {filepath}"
                )

        # Resolve and validate path
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValidationError(f"Failed to resolve path {filepath}: {str(e)}")

        # Basic path traversal protection
        if base_dir is None:
            base_dir = Path.cwd().resolve()
        else:
            base_dir = base_dir.resolve()

        try:
            resolved.relative_to(base_dir)
        except ValueError:
            raise ValidationError(
                f"Path traversal detected - path outside base directory: {filepath}"
            )

        # Additional filename validation
        filename = resolved.name
        if filename:
            # Check for filenames that start with multiple dots
            if filename.startswith(".."):
                raise ValidationError(
                    f"Filename cannot start with double dots: {filename}"
                )

        return resolved


from .conversion_processors import ConversionProcessors
from .converter_factory import ConverterFactory

logger = structlog.get_logger(__name__)

# Import core exceptions
try:
    from morag_core.exceptions import ProcessingError
except ImportError:

    class ProcessingError(Exception):  # type: ignore
        pass


# Import URL path utilities
try:
    from morag.utils.url_path import URLPath, get_url_string, is_url

    URL_PATH_AVAILABLE = True
except ImportError:
    URL_PATH_AVAILABLE = False

    # Fallback implementations
    def is_url(path_like) -> bool:
        return str(path_like).startswith(("http://", "https://"))

    def get_url_string(path_like) -> str:
        return str(path_like)


class MarkdownConversionStage(Stage):
    """Stage for converting various content types to markdown format."""

    def __init__(self, stage_type: StageType = StageType.MARKDOWN_CONVERSION):
        """Initialize the markdown conversion stage."""
        super().__init__(stage_type)

        # Store default configuration
        self.config = {
            "markitdown_enabled": True,
            "process_images": True,
            "process_audio": True,
            "process_video": True,
            "process_documents": True,
            "process_web": True,
            "output_format": "markdown",
            "quality_threshold": 0.3,
            "max_file_size_mb": 100,
        }

        # Initialize components
        self.converter_factory = ConverterFactory()
        self.processors = ConversionProcessors()

    @stage_error_handler("markdown_conversion_execute")
    async def execute(
        self,
        input_files: List[Path],
        context: StageContext,
        output_dir: Optional[Path] = None,
    ) -> StageResult:
        """Execute markdown conversion for input files.

        Args:
            input_files: List of input files or URLs
            context: Stage execution context
            output_dir: Optional output directory override

        Returns:
            StageResult with conversion results
        """
        # Get output directory from context if not provided
        if output_dir is None:
            output_dir = context.output_dir or Path.cwd()

        start_time = datetime.now()
        output_files_list = []

        try:
            # Validate inputs
            if not self.validate_inputs(input_files):
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()

                stage_metadata = StageMetadata(
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    input_files=[str(f) for f in input_files],
                    output_files=[],
                    config_used=self.config,
                )

                return StageResult(
                    stage_type=self.stage_type,
                    status=StageStatus.FAILED,
                    output_files=[],
                    metadata=stage_metadata,
                    error_message="Input validation failed",
                )

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process each input file
            results = []
            errors = []

            for input_file in input_files:
                try:
                    logger.info("Processing file", file=str(input_file))

                    # Detect content type
                    content_type = detect_content_type(input_file)

                    # Generate output filename
                    output_filename = self._generate_output_filename(
                        input_file, content_type
                    )
                    output_file = output_dir / output_filename

                    # Sanitize input file path for security
                    sanitized_input = input_file
                    if not is_url(str(input_file)):
                        # Only sanitize local file paths, not URLs
                        try:
                            # Use a safe base directory for validation (current working directory)
                            safe_base_dir = Path.cwd()
                            sanitized_input = sanitize_filepath(
                                input_file, base_dir=safe_base_dir
                            )
                        except (ValidationError, ValueError, Exception) as e:
                            logger.error(
                                "File path sanitization failed",
                                file=str(input_file),
                                error=str(e),
                            )
                            errors.append(
                                {
                                    "input_file": str(input_file),
                                    "error": f"File path sanitization failed: {str(e)}",
                                }
                            )
                            continue

                    # Process the file
                    result = await self.processors.process_file(
                        sanitized_input, output_file, content_type, self.config.copy()
                    )

                    if result["success"]:
                        output_files_list.append(output_file)
                        results.append(
                            {
                                "input_file": str(input_file),
                                "output_file": str(output_file),
                                "content_type": content_type,
                                "metadata": result.get("metadata", {}),
                                "quality_score": result.get("quality_score", 0.0),
                            }
                        )
                        logger.info(
                            "File processed successfully",
                            file=str(input_file),
                            output=str(output_file),
                        )
                    else:
                        errors.append(
                            {
                                "input_file": str(input_file),
                                "error": result.get("error", "Unknown error"),
                                "content_type": content_type,
                            }
                        )
                        logger.error(
                            "File processing failed",
                            file=str(input_file),
                            error=result.get("error"),
                        )

                except Exception as e:
                    error_msg = f"Failed to process {input_file}: {str(e)}"
                    errors.append({"input_file": str(input_file), "error": error_msg})
                    logger.error(
                        "Exception during file processing",
                        file=str(input_file),
                        error=str(e),
                    )

            # Determine overall status
            total_files = len(input_files)
            successful_files = len(results)

            if successful_files == total_files:
                status = StageStatus.COMPLETED
            elif successful_files > 0:
                status = StageStatus.PARTIAL
            else:
                status = StageStatus.FAILED

            # Create metadata
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            stage_metadata = StageMetadata(
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                input_files=[str(f) for f in input_files],
                output_files=[str(f) for f in output_files_list],
                config_used=self.config,
                metrics={
                    "total_files": total_files,
                    "successful_files": successful_files,
                    "failed_files": len(errors),
                    "success_rate": successful_files / total_files if total_files > 0 else 0.0,
                }
            )

            return StageResult(
                stage_type=self.stage_type,
                status=status,
                output_files=output_files_list,
                metadata=stage_metadata,
                data={"results": results, "errors": errors}
            )

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.error("Stage execution failed", error=str(e))

            stage_metadata = StageMetadata(
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                input_files=[str(f) for f in input_files],
                output_files=[str(f) for f in output_files_list],
                config_used=self.config,
            )

            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.FAILED,
                output_files=output_files_list,
                metadata=stage_metadata,
                error_message=str(e)
            )

    @validation_error_handler("markdown_conversion_validate_inputs")
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files/URLs."""
        if not input_files:
            logger.error("No input files provided")
            return False

        for input_file in input_files:
            try:
                if URL_PATH_AVAILABLE and hasattr(input_file, "is_url"):
                    if input_file.is_url():
                        # URL validation
                        url_str = get_url_string(input_file)
                        if not url_str.startswith(("http://", "https://")):
                            logger.error("Invalid URL format", url=url_str)
                            return False
                    else:
                        # File path validation
                        if not input_file.exists():
                            logger.error("File does not exist", file=str(input_file))
                            return False
                else:
                    # Fallback validation
                    if is_url(str(input_file)):
                        # Basic URL validation
                        if not str(input_file).startswith(("http://", "https://")):
                            logger.error("Invalid URL format", url=str(input_file))
                            return False
                    else:
                        # File path validation
                        path = Path(input_file)
                        if not path.exists():
                            logger.error("File does not exist", file=str(input_file))
                            return False

                        # Check file size
                        max_size_bytes = (
                            self.config.get("max_file_size_mb", 100) * 1024 * 1024
                        )
                        if path.stat().st_size > max_size_bytes:
                            logger.error(
                                "File too large",
                                file=str(input_file),
                                size_mb=path.stat().st_size / (1024 * 1024),
                            )
                            return False

            except Exception as e:
                logger.error(
                    "Error validating input", file=str(input_file), error=str(e)
                )
                return False

        return True

    def get_dependencies(self) -> List[StageType]:
        """Get list of stage dependencies."""
        return []  # Markdown conversion is usually the first stage

    def get_expected_outputs(
        self, input_files: List[Path], context: StageContext
    ) -> List[Path]:
        """Get expected output files."""
        output_dir = context.output_dir if context else Path.cwd()
        outputs = []

        for input_file in input_files:
            content_type = detect_content_type(input_file)
            output_filename = self._generate_output_filename(input_file, content_type)
            outputs.append(output_dir / output_filename)

        return outputs

    def _generate_output_filename(
        self, input_file: Path, content_type, metadata: Dict[str, Any] = None
    ) -> str:
        """Generate output filename for converted content."""
        if metadata is None:
            metadata = {}

        try:
            # Get base filename
            if (
                URL_PATH_AVAILABLE
                and hasattr(input_file, "is_url")
                and input_file.is_url()
            ):
                # Handle URL
                url_str = get_url_string(input_file)

                # Extract filename from URL path or use domain
                from urllib.parse import unquote, urlparse

                parsed = urlparse(url_str)

                if parsed.path and parsed.path != "/":
                    # Use path-based filename
                    path_parts = [p for p in parsed.path.split("/") if p]
                    if path_parts:
                        filename = unquote(path_parts[-1])
                        # Remove query parameters and fragments
                        filename = filename.split("?")[0].split("#")[0]
                        if filename and not filename.endswith(
                            (".html", ".htm", ".php", ".jsp", ".asp")
                        ):
                            base_name = Path(filename).stem
                        else:
                            base_name = (
                                path_parts[-1]
                                if path_parts
                                else parsed.netloc.replace(".", "_")
                            )
                    else:
                        base_name = parsed.netloc.replace(".", "_")
                else:
                    # Use domain name
                    base_name = parsed.netloc.replace(".", "_")

                # Clean the base name
                base_name = re.sub(r"[^\w\-_\.]", "_", base_name)

            else:
                # Handle local file
                path = Path(input_file)
                base_name = path.stem

            # Add content type suffix if needed
            type_suffix = ""
            if self._is_content_type(content_type, "video"):
                type_suffix = "_video"
            elif self._is_content_type(content_type, "audio"):
                type_suffix = "_audio"
            elif self._is_content_type(content_type, "image"):
                type_suffix = "_image"
            elif self._is_content_type(content_type, "web"):
                type_suffix = "_web"
            elif self._is_content_type(content_type, "youtube"):
                type_suffix = "_youtube"

            # Create final filename
            output_filename = f"{base_name}{type_suffix}.md"

            # Sanitize filename
            output_filename = re.sub(r'[<>:"/\\|?*]', "_", output_filename)
            output_filename = re.sub(
                r"__+", "_", output_filename
            )  # Replace multiple underscores

            return output_filename

        except Exception as e:
            logger.warning(
                "Error generating output filename", input=str(input_file), error=str(e)
            )
            # Fallback to simple naming
            return f"converted_{hash(str(input_file)) % 10000}.md"

    def _is_content_type(self, content_type, expected_type: str) -> bool:
        """Check if content type matches expected type."""
        if hasattr(content_type, "value"):
            content_type = content_type.value
        return is_content_type(content_type, expected_type)


__all__ = ["MarkdownConversionStage"]
