# TrustyAI SDK

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://trustyai-explainability.github.io/trustyai-sdk)
[![Ruff](https://github.com/trustyai-explainability/trustyai-sdk/actions/workflows/ruff.yml/badge.svg)](https://github.com/trustyai-explainability/trustyai-sdk/actions/workflows/ruff.yml)
[![Tests](https://github.com/trustyai-explainability/trustyai-sdk/actions/workflows/pytest.yml/badge.svg)](https://github.com/trustyai-explainability/trustyai-sdk/actions/workflows/pytest.yml)
[![PyPI version](https://badge.fury.io/py/trustyai.svg)](https://badge.fury.io/py/trustyai)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Python toolkit for testing machine learning models. You can run tests on your computer or scale them up using Kubernetes.

## ✨ Key Features

- **🔄 Multiple Testing Tools**: Works with LM Evaluation Harness, RAGAS, and other testing tools
- **☸️ Works with Kubernetes**: Scale up tests on Kubernetes with TrustyAI Operator
- **🖥️ Run Anywhere**: Test models on your computer or spread across cluster nodes
- **🛡️ Team Ready**: Built-in checks, monitoring, and works with OpenDataHub
- **🎯 Easy to Use**: Command line tools and Python code for different workflows

## 🚀 Quick Start

### Local Testing

```bash
trustyai eval execute \
  --provider lm-eval-harness \
  --execution-mode local \
  --model "microsoft/DialoGPT-medium" \
  --tasks "hellaswag,arc_easy" \
  --limit 10
```

### Kubernetes Testing

```bash
trustyai eval execute \
  --provider lm-eval-harness \
  --execution-mode kubernetes \
  --model "microsoft/DialoGPT-medium" \
  --tasks "hellaswag,arc_easy" \
  --namespace trustyai-eval \
  --cpu 4 \
  --memory 8Gi \
  --limit 50
```

## 📚 Documentation

**📖 [Complete Documentation](https://trustyai-explainability.github.io/trustyai-sdk)**

Quick links:
- [Getting Started Guide](https://trustyai-explainability.github.io/trustyai-sdk/getting-started.html)
- [Kubernetes Integration](https://trustyai-explainability.github.io/trustyai-sdk/kubernetes.html)
- [CLI Reference](https://trustyai-explainability.github.io/trustyai-sdk/cli-eval.html)
- [Examples](https://trustyai-explainability.github.io/trustyai-sdk/examples-local.html)

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