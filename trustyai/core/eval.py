"""Evaluation provider infrastructure for TrustyAI."""

import abc
from typing import Any

from .models import ExecutionMode
from .providers import BaseProvider


class EvaluationProviderConfig:
    """Configuration for evaluation providers."""

    def __init__(
        self,
        evaluation_name: str,
        model: str,
        tasks: list[str],
        limit: int | None = None,
        metrics: list[str] | None = None,
        device: str = "cuda",
        deployment_mode: ExecutionMode = ExecutionMode.LOCAL,
        **kwargs: Any,
    ) -> None:
        """Initialize evaluation configuration.

        Args:
            evaluation_name: Name of the evaluation
            model: HuggingFace model identifier or path
            tasks: List of tasks to evaluate on
            limit: Optional limit for number of examples to evaluate
            metrics: Optional list of metrics to compute
            device: Device to run evaluation on ("cuda", "cpu", etc.)
            deployment_mode: Deployment mode (local or kubernetes)
            **kwargs: Additional provider-specific parameters
        """
        self.evaluation_name = evaluation_name
        self.model = model
        self.tasks = tasks
        self.limit = limit
        self.metrics = metrics or []
        self.device = device
        self.deployment_mode = deployment_mode
        self.additional_params = kwargs

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value from the configuration.

        Args:
            key: Parameter name
            default: Default value if parameter is not found

        Returns:
            Parameter value or default
        """
        return self.additional_params.get(key, default)


class EvalProvider(BaseProvider):
    """Base class for model evaluation providers."""

    def __init__(
        self,
        implementation: str = "default",
        execution_mode: str = "local",
        **config: Any,
    ) -> None:
        """Initialize the evaluation provider."""
        super().__init__(implementation, execution_mode, **config)

    @classmethod
    def get_provider_type(cls) -> str:
        """Return the type of provider."""
        return "eval"

    def _get_validator(self) -> Any:
        """Get validator - stub implementation."""
        from .providers import LocalEvaluationValidator
        return LocalEvaluationValidator(self.implementation, self.config)

    def execute(self, request: Any) -> Any:
        """Execute evaluation - compatibility layer."""
        # This is a compatibility layer for existing code
        # Real implementation should be in concrete subclasses
        return self.evaluate(request)

    @property
    def supported_deployment_modes(self) -> list[ExecutionMode]:
        """Return deployment modes supported by default."""
        return [ExecutionMode.LOCAL]

    @abc.abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Evaluate a model on tasks with specified metrics.

        Supports:
        - New style: evaluate(config: EvaluationProviderConfig, **kwargs)
        - Legacy style: evaluate(model_or_id, dataset, metrics=None, **kwargs)

        Args:
            *args: Either (config) or (model_or_id, dataset, [metrics])
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary of evaluation results
        """


    @abc.abstractmethod
    def list_available_datasets(self) -> list[str]:
        """List available evaluation datasets for this provider.

        Returns:
            List of dataset names supported by this provider
        """

    @abc.abstractmethod
    def list_available_metrics(self) -> list[str]:
        """List available evaluation metrics for this provider.

        Returns:
            List of metric names supported by this provider
        """
