"""Simple validator system for TrustyAI providers."""
from __future__ import annotations

import abc
from typing import Any

from ..models import ExecutionMode


class ValidationResult:
    """Result of validation checks."""

    def __init__(self, is_valid: bool, message: str, details: dict[str, Any] | None = None):
        self.is_valid = is_valid
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return f"[{status}] {self.message}"

    def __repr__(self) -> str:
        return f"ValidationResult(is_valid={self.is_valid}, message='{self.message}')"


class BaseValidator(abc.ABC):
    """Base validator for all TrustyAI provider implementations."""

    def __init__(self, implementation: str, config: dict[str, Any]):
        self.implementation = implementation
        self.config = config

    @abc.abstractmethod
    def validate(self) -> ValidationResult:
        """Perform validation checks. Override this method for specific validation logic."""


class LocalValidator(BaseValidator):
    """Base validator for local execution environments."""

    def validate(self) -> ValidationResult:
        """Perform basic local environment validation."""
        # Check if implementation is available
        try:
            __import__(self.implementation)
            return ValidationResult(
                is_valid=True,
                message=f"Local validation passed for {self.implementation}",
                details={"implementation": self.implementation, "mode": "local"}
            )
        except ImportError:
            return ValidationResult(
                is_valid=False,
                message=f"Implementation '{self.implementation}' is not available",
                details={"implementation": self.implementation, "mode": "local"}
            )


class KubernetesValidator(BaseValidator):
    """Base validator for Kubernetes execution environments."""

    def __init__(self, implementation: str, config: dict[str, Any], k8s_client=None):
        """Initialise Kubernetes validator with optional Kubernetes client.

        Args:
            implementation: The implementation name
            config: Configuration dictionary
            k8s_client: Optional Kubernetes client instance
        """
        super().__init__(implementation, config)
        self.k8s_client = k8s_client

    def validate(self) -> ValidationResult:
        """Perform basic Kubernetes environment validation."""
        # TODO:Basic check
        namespace = self.config.get('namespace', 'trustyai')

        # If we have a k8s client, we can perform a better validation
        if self.k8s_client:
            try:
                # Test the client connection
                self.k8s_client.list_namespace()
                return ValidationResult(
                    is_valid=True,
                    message=f"Kubernetes validation passed for {self.implementation} using provided client",
                    details={
                        "implementation": self.implementation, 
                        "mode": "kubernetes",
                        "namespace": namespace,
                        "client_provided": True
                    }
                )
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    message=f"Kubernetes client validation failed: {str(e)}",
                    details={
                        "implementation": self.implementation,
                        "mode": "kubernetes",
                        "namespace": namespace,
                        "client_provided": True,
                        "error": str(e)
                    }
                )
        else:
            # Fallback to basic validation without client
            return ValidationResult(
                is_valid=True,
                message=f"Kubernetes validation passed for {self.implementation} (no client provided)",
                details={
                    "implementation": self.implementation, 
                    "mode": "kubernetes",
                    "namespace": namespace,
                    "client_provided": False
                }
            )


# Factory function for creating validators
def create_validator(implementation: str, execution_mode: ExecutionMode, config: dict[str, Any], k8s_client=None) -> BaseValidator:
    """Create the appropriate validator based on implementation and execution mode.

    Args:
        implementation: The implementation name
        execution_mode: The execution mode (local or kubernetes)
        config: Configuration dictionary
        k8s_client: Optional Kubernetes client for Kubernetes validators
    """

    if execution_mode == ExecutionMode.LOCAL:
        return LocalValidator(implementation, config)
    elif execution_mode == ExecutionMode.KUBERNETES:
        return KubernetesValidator(implementation, config, k8s_client)
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")


# Import Kubernetes-specific validators
from .kubernetes import TrustyAIOperatorValidator

__all__ = [
    "ValidationResult",
    "BaseValidator", 
    "LocalValidator",
    "KubernetesValidator",
    "TrustyAIOperatorValidator",
    "create_validator"
]
