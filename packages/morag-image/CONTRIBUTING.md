# Contributing to morag-image

Thank you for your interest in contributing to the morag-image package! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read and respect these guidelines to maintain a positive and inclusive community.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- System dependencies (see [README.md](README.md))

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/morag.git
   cd morag/packages/morag-image
   ```
3. Add the original repository as an upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/morag.git
   ```

## Development Environment

### Setting Up

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode with all development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Environment Variables

For testing features that require API access:

```bash
# Linux/macOS
export GOOGLE_API_KEY=your-api-key

# Windows (PowerShell)
$env:GOOGLE_API_KEY = "your-api-key"
```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the [Coding Standards](#coding-standards)

3. Run tests to ensure your changes don't break existing functionality:
   ```bash
   pytest
   ```

4. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

5. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request from your fork to the original repository

## Pull Request Process

1. Ensure your PR includes a clear description of the changes and the purpose
2. Update documentation if necessary
3. Include tests for new functionality
4. Ensure all tests pass and code quality checks succeed
5. Address any review comments and make requested changes
6. Once approved, your PR will be merged

## Coding Standards

This project follows these coding standards:

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [flake8](https://flake8.pycqa.org/) for linting

### Type Annotations

- Use type annotations for function parameters and return values
- Use [mypy](https://mypy.readthedocs.io/) for type checking

### Docstrings

- Use Google-style docstrings
- Document all public classes, methods, and functions

Example:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of the function.
    
    Longer description explaining the function's purpose and behavior.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When and why this exception is raised
    """
    # Function implementation
```

## Testing

### Running Tests

Run the full test suite:

```bash
pytest
```

Run tests with coverage report:

```bash
pytest --cov=morag_image
```

Run specific tests:

```bash
pytest tests/test_processor.py
```

### Writing Tests

- Write tests for all new functionality
- Use pytest fixtures for common test setup
- Mock external dependencies (APIs, file system, etc.)
- Aim for high test coverage (>80%)

### Test Resources

Test resources (sample images, etc.) should be placed in the `tests/resources` directory.

## Documentation

### Updating Documentation

1. Update docstrings in the code
2. Update markdown files in the `docs/` directory
3. Add examples to the `examples/` directory for new features

### Building Documentation

If you've made significant changes to the documentation, please verify that it renders correctly.

## Questions?

If you have any questions or need help with the contribution process, please open an issue on GitHub.

Thank you for contributing to morag-image!