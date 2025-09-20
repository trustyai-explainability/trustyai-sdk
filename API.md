# TrustyAI SDK API Documentation

## Validators API

The TrustyAI SDK provides a comprehensive validation system to check system requirements, dependencies, and configurations for running TrustyAI operations.

### Overview

The validators system consists of:
- **BaseValidator**: Abstract base class for all validators
- **ValidatorRegistry**: Central registry for managing validators
- **ValidationResult**: Standardized result format for all validators

### Common Schema

All validators return a `ValidationResult` with the following structure:

```python
{
    "is_valid": bool,           # Whether validation passed
    "message": str,             # Human-readable status message
    "details": dict             # Additional details and metadata
}
```

### CLI Usage

#### List Available Validators

```bash
trustyai validators list
```

Shows all available validators with their descriptions and implementation classes.

#### Run a Validator

```bash
# Basic usage
trustyai validators run <validator_name>

# With configuration
trustyai validators run <validator_name> --config '{"key": "value"}'

# JSON output
trustyai validators run <validator_name> --json

# Custom implementation
trustyai validators run <validator_name> --implementation custom
```

### Available Validators

#### Local Environment Validators

##### `python-version`
Validates Python version requirements.

**Configuration:**
```json
{
    "min_version": "3.8"  // Minimum required Python version (default: "3.8")
}
```

**Example:**
```bash
trustyai validators run python-version --config '{"min_version": "3.9"}'
```

##### `package-dependencies`
Validates that required Python packages are installed.

**Configuration:**
```json
{
    "packages": ["numpy", "torch", "transformers"]  // List of required packages
}
```

**Example:**
```bash
trustyai validators run package-dependencies --config '{"packages": ["lm-eval", "torch"]}'
```

##### `environment-variables`
Validates that required environment variables are set.

**Configuration:**
```json
{
    "required_variables": ["HF_TOKEN", "OPENAI_API_KEY"],  // Must be set
    "optional_variables": ["CUDA_VISIBLE_DEVICES"]        // Nice to have
}
```

**Example:**
```bash
trustyai validators run environment-variables --config '{"required_variables": ["HF_TOKEN"]}'
```

##### `lm-eval-harness`
Validates lm-evaluation-harness installation and configuration.

**Configuration:** None required (uses defaults)

**Example:**
```bash
trustyai validators run lm-eval-harness
```

##### `trustyai-provider`
Validates TrustyAI provider availability.

**Configuration:** None required (auto-detects available providers)

**Example:**
```bash
trustyai validators run trustyai-provider
```

#### Kubernetes Environment Validators

**Note:** Kubernetes validators automatically initialize the Kubernetes client using the default context from your kubeconfig file. If you need to use a different context or kubeconfig file, you can configure this programmatically via the Python API.

##### `kubernetes-connectivity`
Validates basic connectivity to a Kubernetes cluster.

**Configuration:** None required (auto-detects cluster and lists namespaces)

**Example:**
```bash
trustyai validators run kubernetes-connectivity
```

##### `trustyai-operator`
Validates that the TrustyAI operator is deployed in the Kubernetes cluster.

**Configuration:**
```json
{
    "namespace": "trustyai-system"  // Expected operator namespace (default: "trustyai-system")
}
```

**Example:**
```bash
trustyai validators run trustyai-operator --config '{"namespace": "custom-namespace"}'
```

### Python API

#### Using the Validator Registry

```python
from trustyai.core.validators import ValidatorRegistry

# List available validators
validators = ValidatorRegistry.list_validators()
print(validators)

# Create and run a validator
validator = ValidatorRegistry.create_validator(
    name="python-version",
    implementation="default",
    config={"min_version": "3.9"}
)

if validator:
    result = validator.validate()
    print(f"Valid: {result.is_valid}")
    print(f"Message: {result.message}")
    print(f"Details: {result.details}")
```

#### Creating Custom Validators

```python
from trustyai.core.validators import BaseValidator, ValidationResult, validator

@validator("my-custom-validator")
class MyCustomValidator(BaseValidator):
    """Custom validator description."""

    def validate(self) -> ValidationResult:
        # Your validation logic here
        try:
            # Perform checks
            check_passed = True

            if check_passed:
                return ValidationResult(
                    is_valid=True,
                    message="Custom validation passed",
                    details={"custom_data": "value"}
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    message="Custom validation failed",
                    details={"error": "reason"}
                )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Custom validation error: {str(e)}",
                details={"exception": str(e)}
            )
```

### JSON Output Format

When using the `--json` flag, the output follows this structure:

```json
{
    "validator": "validator-name",
    "implementation": "default",
    "config": {
        "key": "value"
    },
    "result": {
        "is_valid": true,
        "message": "Validation message",
        "details": {
            "additional": "information"
        }
    }
}
```

### Error Handling

#### Validator Not Found
```json
{
    "validator": "non-existent",
    "implementation": "default",
    "config": {},
    "error": "Validator 'non-existent' not found."
}
```

#### Configuration Error
```json
{
    "validator": "package-dependencies",
    "implementation": "default",
    "config": {},
    "error": "Invalid JSON in --config: ..."
}
```

#### Runtime Error
```json
{
    "validator": "python-version",
    "implementation": "default",
    "config": {},
    "error": "Error running validator 'python-version': ..."
}
```

#### Common Kubernetes Validation Scenarios

##### Missing Kubernetes Package
```bash
$ trustyai validators run kubernetes-connectivity
❌ Validator: kubernetes-connectivity
Status: FAIL
Message: Failed to connect to Kubernetes cluster
Details:
  error: kubernetes_client_init_failed
  suggestion: Ensure you have a valid kubeconfig file and cluster access
```

To fix: Install the kubernetes package with `pip install trustyai[eval]`

##### No Cluster Access
```bash
$ trustyai validators run kubernetes-connectivity
❌ Validator: kubernetes-connectivity
Status: FAIL
Message: Failed to initialize Kubernetes client with default configuration
Details:
  error: kubernetes_client_init_failed
  suggestion: Ensure you have a valid kubeconfig file and cluster access
```

To fix: Ensure your kubeconfig is configured and you have cluster access

##### Successful Cluster Connection
```bash
$ trustyai validators run kubernetes-connectivity
✅ Validator: kubernetes-connectivity
Status: PASS
Message: Successfully connected to Kubernetes cluster with 8 namespaces
Details:
  namespace_count: 8
  sample_namespaces: ["default", "kube-system", "kube-public", "trustyai", "monitoring"]
  cluster_version: "1.28"
```

### Best Practices

1. **Use JSON output** for programmatic access and integration with other tools
2. **Specify configurations** when validators require specific settings
3. **Check all dependencies** before running complex operations by using multiple validators
4. **Create custom validators** for organization-specific requirements
5. **Handle validation failures** gracefully in automated workflows

### Integration Examples

#### Pre-flight Checks Script

```bash
#!/bin/bash
# Check all requirements before running evaluation

echo "Checking Python version..."
trustyai validators run python-version --json | jq -r '.result.is_valid'

echo "Checking dependencies..."
trustyai validators run package-dependencies --config '{"packages": ["lm-eval", "torch"]}' --json | jq -r '.result.is_valid'

echo "Checking environment..."
trustyai validators run environment-variables --config '{"required_variables": ["HF_TOKEN"]}' --json | jq -r '.result.is_valid'

echo "All checks complete!"
```

#### Python Integration

```python
import json
from trustyai.core.validators import ValidatorRegistry

def validate_environment():
    """Validate the complete environment setup."""

    validators_to_run = [
        ("python-version", {"min_version": "3.8"}),
        ("package-dependencies", {"packages": ["lm-eval", "torch"]}),
        ("trustyai-provider", {})
    ]

    results = []
    all_valid = True

    for validator_name, config in validators_to_run:
        validator = ValidatorRegistry.create_validator(validator_name, "default", config)
        if validator:
            result = validator.validate()
            results.append({
                "validator": validator_name,
                "result": result
            })
            if not result.is_valid:
                all_valid = False

    return all_valid, results

# Usage
is_valid, validation_results = validate_environment()
if is_valid:
    print("✅ Environment validation passed!")
else:
    print("❌ Environment validation failed!")
    for item in validation_results:
        if not item["result"].is_valid:
            print(f"  - {item['validator']}: {item['result'].message}")
```