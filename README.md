# TrustyAI SDK

[![Ruff](https://github.com/trustyai-explainability/trustyai-sdk/actions/workflows/ruff.yml/badge.svg)](https://github.com/trustyai-explainability/trustyai-sdk/actions/workflows/ruff.yml)
[![Tests](https://github.com/trustyai-explainability/trustyai-sdk/actions/workflows/pytest.yml/badge.svg)](https://github.com/trustyai-explainability/trustyai-sdk/actions/workflows/pytest.yml)

A Python SDK for TrustyAI, providing tools for explaining, evaluating, and enhancing AI models.

## Installation

### Standard Installation

Install the package with core functionality and CLI:

```bash
pip install .
```

After installation, you can use both the Python API and CLI:

```bash
trustyai --help
trustyai info
trustyai model list
trustyai eval list-providers
```

### Full Installation

To install everything including evaluation support:

```bash
pip install .[all]
```

This includes all core, CLI, and evaluation dependencies.

## Additional Optional Dependencies

### Evaluation Support

For model evaluation capabilities:

```bash
pip install .[eval]
```

### Development Dependencies

For development, testing, and linting:

```bash
pip install .[dev]
```

## Usage

### Python API

```python
import numpy as np
from trustyai.core.model import TrustyModel
from trustyai.core.providers import ProviderRegistry

# Create a trusty model
model = TrustyModel(name="MyModel")

# Get explanations
X = np.random.rand(10, 5)
explanations = model.explain(X)
print(explanations)
```

### Command Line Interface

The CLI is available by default after installation:

```bash
# Display help
trustyai --help

# Show version information
trustyai --version

# Show general information
trustyai info

# List available models
trustyai model list

# List evaluation providers
trustyai eval list-providers

# List available validators
trustyai validators list

# Run a validator
trustyai validators run python-version
```

## Documentation

- [API Documentation](API.md) - Comprehensive API reference including validators

## Development

This project uses:
- [pytest](https://docs.pytest.org/) for testing
- [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- [mypy](https://mypy.readthedocs.io/) for type checking

```bash
# Install development dependencies
pip install .[dev]

# Run tests
make test

# Run linting
make lint

# Format code
make format
```

## License

Apache License 2.0