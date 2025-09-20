"""Local environment validators for TrustyAI."""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Any, Dict

from . import BaseValidator, ValidationResult
from .registry import validator


@validator("python-version")
class PythonVersionValidator(BaseValidator):
    """Validates Python version requirements."""

    def validate(self) -> ValidationResult:
        """Check if Python version meets requirements."""
        min_version = self.config.get('min_version', '3.8')
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        try:
            min_major, min_minor = map(int, min_version.split('.'))
            current_major, current_minor = sys.version_info.major, sys.version_info.minor

            if (current_major, current_minor) >= (min_major, min_minor):
                return ValidationResult(
                    is_valid=True,
                    message=f"Python version {current_version} meets requirement >= {min_version}",
                    details={
                        "current_version": current_version,
                        "required_version": min_version,
                        "python_executable": sys.executable
                    }
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    message=f"Python version {current_version} does not meet requirement >= {min_version}",
                    details={
                        "current_version": current_version,
                        "required_version": min_version,
                        "python_executable": sys.executable
                    }
                )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Failed to validate Python version: {str(e)}",
                details={"error": str(e)}
            )


@validator("package-dependencies")
class PackageDependenciesValidator(BaseValidator):
    """Validates that required Python packages are installed."""

    def validate(self) -> ValidationResult:
        """Check if required packages are installed."""
        packages = self.config.get('packages', [])
        if not packages:
            return ValidationResult(
                is_valid=True,
                message="No packages specified to validate",
                details={"packages": []}
            )

        missing_packages = []
        installed_packages = {}

        for package in packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                installed_packages[package] = version
            except ImportError:
                missing_packages.append(package)

        if not missing_packages:
            return ValidationResult(
                is_valid=True,
                message=f"All required packages are installed: {', '.join(packages)}",
                details={
                    "packages": packages,
                    "installed_versions": installed_packages
                }
            )
        else:
            return ValidationResult(
                is_valid=False,
                message=f"Missing required packages: {', '.join(missing_packages)}",
                details={
                    "packages": packages,
                    "missing_packages": missing_packages,
                    "installed_packages": installed_packages
                }
            )


@validator("environment-variables")
class EnvironmentVariablesValidator(BaseValidator):
    """Validates that required environment variables are set."""

    def validate(self) -> ValidationResult:
        """Check if required environment variables are set."""
        import os

        required_vars = self.config.get('required_variables', [])
        optional_vars = self.config.get('optional_variables', [])

        missing_required = []
        present_vars = {}
        missing_optional = []

        # Check required variables
        for var in required_vars:
            value = os.getenv(var)
            if value is not None:
                present_vars[var] = "***" if "token" in var.lower() or "key" in var.lower() else value
            else:
                missing_required.append(var)

        # Check optional variables
        for var in optional_vars:
            value = os.getenv(var)
            if value is not None:
                present_vars[var] = "***" if "token" in var.lower() or "key" in var.lower() else value
            else:
                missing_optional.append(var)

        if not missing_required:
            message = f"All required environment variables are set"
            if missing_optional:
                message += f" (optional variables missing: {', '.join(missing_optional)})"

            return ValidationResult(
                is_valid=True,
                message=message,
                details={
                    "required_variables": required_vars,
                    "optional_variables": optional_vars,
                    "present_variables": list(present_vars.keys()),
                    "missing_optional": missing_optional
                }
            )
        else:
            return ValidationResult(
                is_valid=False,
                message=f"Missing required environment variables: {', '.join(missing_required)}",
                details={
                    "required_variables": required_vars,
                    "missing_required": missing_required,
                    "present_variables": list(present_vars.keys()),
                    "missing_optional": missing_optional
                }
            )


@validator("lm-eval-harness")
class LMEvalHarnessValidator(BaseValidator):
    """Validates lm-evaluation-harness installation and configuration."""

    def validate(self) -> ValidationResult:
        """Check if lm-evaluation-harness is properly installed."""
        try:
            # Try to import lm_eval
            import lm_eval
            version = getattr(lm_eval, '__version__', 'unknown')

            # Try to check if basic functionality works
            try:
                from lm_eval import tasks
                available_tasks = list(tasks.ALL_TASKS.keys())[:5]  # Just a few for testing

                return ValidationResult(
                    is_valid=True,
                    message=f"lm-evaluation-harness v{version} is properly installed",
                    details={
                        "version": version,
                        "sample_tasks": available_tasks,
                        "total_tasks": len(tasks.ALL_TASKS)
                    }
                )
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    message=f"lm-evaluation-harness is installed but not functioning properly: {str(e)}",
                    details={
                        "version": version,
                        "error": str(e)
                    }
                )

        except ImportError:
            return ValidationResult(
                is_valid=False,
                message="lm-evaluation-harness is not installed",
                details={
                    "suggestion": "Install with: pip install lm-eval[default]"
                }
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Failed to validate lm-evaluation-harness: {str(e)}",
                details={"error": str(e)}
            )


@validator("trustyai-provider")
class TrustyAIProviderValidator(BaseValidator):
    """Validates TrustyAI provider availability."""

    def validate(self) -> ValidationResult:
        """Check if TrustyAI providers are available."""
        try:
            from ...providers import ProviderRegistry

            # Import providers to trigger registration
            try:
                from ...providers.eval.lm_eval import LMEvalProvider
                from ...providers.eval.lm_eval_kubernetes import KubernetesLMEvalProvider
                from ...providers.eval.lm_eval_local import LocalLMEvalProvider
            except ImportError:
                pass  # Optional providers not available

            providers = ProviderRegistry.list_providers()
            eval_providers = providers.get('eval', [])

            if eval_providers:
                provider_names = [p['name'] for p in eval_providers]
                return ValidationResult(
                    is_valid=True,
                    message=f"TrustyAI providers are available: {', '.join(provider_names)}",
                    details={
                        "providers": eval_providers,
                        "provider_count": len(eval_providers)
                    }
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    message="No TrustyAI providers are available",
                    details={
                        "suggestion": "Install evaluation providers with: pip install trustyai[eval]"
                    }
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Failed to validate TrustyAI providers: {str(e)}",
                details={"error": str(e)}
            )