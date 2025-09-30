"""Provider system for TrustyAI extensions following ADR-0010 specification."""

from __future__ import annotations

import abc
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .models import ExecutionMode, JobStatus, TrustyAIMetadata, TrustyAIRequest, TrustyAIResponse
from .validators import (
    BaseValidator,
    LocalValidator,
    KubernetesValidator,
    ValidationResult,
    create_validator,
)

if TYPE_CHECKING:
    pass


class BaseProvider(abc.ABC):
    """Base provider interface for all TrustyAI capabilities following ADR-0010."""

    def __init__(
        self, implementation: str, execution_mode: str = "local", k8s_client=None, **config: Any
    ):
        """Initialise provider with implementation and execution mode.

        Args:
            implementation: Specific implementation name (e.g., "lm-evaluation-harness")
            execution_mode: Execution mode ("local" or "kubernetes")
            k8s_client: Optional Kubernetes client for Kubernetes execution
            **config: Provider-specific configuration
        """
        self.implementation = implementation
        self.execution_mode = ExecutionMode(execution_mode)
        self.k8s_client = k8s_client
        self.config = config
        self.validator = self._get_validator()

    @abc.abstractmethod
    def _get_validator(self) -> BaseValidator:
        """Get the appropriate validator for this provider implementation."""
        return create_validator(
            self.implementation, self.execution_mode, self.config, self.k8s_client
        )

    @abc.abstractmethod
    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        """Execute the provider operation."""
        pass

    def validate(self) -> ValidationResult:
        """Validate system readiness for this provider."""
        return self.validator.validate()

    def _prepare_execution(self, request: TrustyAIRequest) -> Dict[str, Any]:
        """Prepare execution parameters from request."""
        return {
            "implementation": self.implementation,
            "execution_mode": self.execution_mode,
            "config": self.config,
            "request": request,
        }

    @classmethod
    @abc.abstractmethod
    def provider_type(cls) -> str:
        """Return the type of provider (e.g., 'evaluation', 'explainability')."""

    @classmethod
    def get_provider_name(cls) -> str:
        """Return the name of the provider implementation."""
        return getattr(cls, "_provider_name", cls.__name__)

    @classmethod
    def get_description(cls) -> str:
        """Return the description of the provider."""
        return cls.__doc__ or "No description available"

    @property
    def supported_deployment_modes(self) -> List[ExecutionMode]:
        """Return the deployment modes supported by this provider."""
        # Default implementation - subclasses should override
        return [ExecutionMode.LOCAL]

    def is_mode_supported(self, mode: ExecutionMode) -> bool:
        """Check if a specific deployment mode is supported."""
        return mode in self.supported_deployment_modes


class EvaluationProvider(BaseProvider):
    """Universal evaluation provider that delegates to platform-specific implementations."""

    def __init__(self, implementation: str, execution_mode: str = "local", **config: Any):
        """Initialise evaluation provider.

        Args:
            implementation: Evaluation implementation (e.g., "lm-evaluation-harness", "ragas")
            execution_mode: Execution mode ("local" or "kubernetes")
            **config: Provider-specific configuration
        """
        super().__init__(implementation, execution_mode, **config)

        # Initialise platform-specific executor
        if self.execution_mode == ExecutionMode.LOCAL:
            self.executor: Union[LocalEvaluationExecutor, KubernetesEvaluationExecutor] = (
                LocalEvaluationExecutor(implementation, config)
            )
        elif self.execution_mode == ExecutionMode.KUBERNETES:
            self.executor = KubernetesEvaluationExecutor(implementation, config)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")

    @classmethod
    def provider_type(cls) -> str:
        """Return the provider type."""
        return "evaluation"

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        """Execute evaluation using the configured executor."""
        return self.executor.execute_evaluation(request)

    def evaluate(
        self, model: str, tasks: List[str], parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Convenience method for evaluation."""
        # Create request from parameters
        from .models import EvaluationRequest, ModelReference

        model_ref = ModelReference(identifier=model, type="huggingface")
        request = EvaluationRequest(
            provider=self.implementation,
            model=model_ref,
            tasks=tasks,
            parameters=parameters or {},
            execution_mode=self.execution_mode,
        )

        response = self.execute(request)  # type: ignore[arg-type]
        return response.results if response.results else {}

    @property
    def supported_deployment_modes(self) -> List[ExecutionMode]:
        """Return supported deployment modes."""
        return [ExecutionMode.LOCAL, ExecutionMode.KUBERNETES]


class ExplainabilityProvider(BaseProvider):
    """Universal explainability provider that delegates to platform-specific implementations."""

    def __init__(self, implementation: str, execution_mode: str = "local", **config: Any):
        super().__init__(implementation, execution_mode, **config)

        if self.execution_mode == ExecutionMode.LOCAL:
            self.executor: Union[LocalExplainabilityExecutor, KubernetesExplainabilityExecutor] = (
                LocalExplainabilityExecutor(implementation, config)
            )
        elif self.execution_mode == ExecutionMode.KUBERNETES:
            self.executor = KubernetesExplainabilityExecutor(implementation, config)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")

    @classmethod
    def provider_type(cls) -> str:
        return "explainability"

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.executor.execute_explanation(request)

    @property
    def supported_deployment_modes(self) -> List[ExecutionMode]:
        return [ExecutionMode.LOCAL, ExecutionMode.KUBERNETES]


class BiasDetectionProvider(BaseProvider):
    """Universal bias detection provider that delegates to platform-specific implementations."""

    def __init__(self, implementation: str, execution_mode: str = "local", **config: Any):
        super().__init__(implementation, execution_mode, **config)

        if self.execution_mode == ExecutionMode.LOCAL:
            self.executor: Union[LocalBiasDetectionExecutor, KubernetesBiasDetectionExecutor] = (
                LocalBiasDetectionExecutor(implementation, config)
            )
        elif self.execution_mode == ExecutionMode.KUBERNETES:
            self.executor = KubernetesBiasDetectionExecutor(implementation, config)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")

    @classmethod
    def provider_type(cls) -> str:
        return "bias_detection"

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.executor.execute_bias_detection(request)

    @property
    def supported_deployment_modes(self) -> List[ExecutionMode]:
        return [ExecutionMode.LOCAL, ExecutionMode.KUBERNETES]


# Platform-specific executor interfaces
class BaseExecutor(abc.ABC):
    """Base class for platform-specific executors."""

    def __init__(self, implementation: str, config: Dict[str, Any]):
        self.implementation = implementation
        self.config = config

    @abc.abstractmethod
    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        """Execute the provider operation."""
        pass

    def _create_metadata(self, request: TrustyAIRequest) -> TrustyAIMetadata:
        """Create metadata for the operation."""
        return TrustyAIMetadata(
            job_id=str(uuid.uuid4()),
            provider_type=self.__class__.__name__.replace("Executor", "").lower(),
            implementation=self.implementation,
            execution_mode=ExecutionMode.LOCAL
            if "Local" in self.__class__.__name__
            else ExecutionMode.KUBERNETES,
            version="2.0.0a1",
        )


class LocalEvaluationExecutor(BaseExecutor):
    """Local machine execution for evaluation tasks."""

    def execute_evaluation(self, request: TrustyAIRequest) -> TrustyAIResponse:
        """Execute evaluation locally."""
        metadata = self._create_metadata(request)

        # TODO: Implement actual local evaluation logic
        # This would invoke lm-eval or other evaluation frameworks directly

        return TrustyAIResponse(
            metadata=metadata,
            status=JobStatus.COMPLETED,
            results={"status": "evaluation completed locally"},
            execution_info={"mode": "local", "implementation": self.implementation},
        )

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.execute_evaluation(request)


class KubernetesEvaluationExecutor(BaseExecutor):
    """Kubernetes cluster execution for evaluation tasks."""

    def execute_evaluation(self, request: TrustyAIRequest) -> TrustyAIResponse:
        """Execute evaluation on Kubernetes."""
        metadata = self._create_metadata(request)

        # TODO: Implement Kubernetes job deployment
        # This would create Custom Resources and deploy jobs to cluster

        return TrustyAIResponse(
            metadata=metadata,
            status=JobStatus.PENDING,
            results={"status": "evaluation job deployed to kubernetes"},
            execution_info={"mode": "kubernetes", "implementation": self.implementation},
        )

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.execute_evaluation(request)


# Placeholder executors for other providers
class LocalExplainabilityExecutor(BaseExecutor):
    def execute_explanation(self, request: TrustyAIRequest) -> TrustyAIResponse:
        metadata = self._create_metadata(request)
        return TrustyAIResponse(
            metadata=metadata,
            status=JobStatus.COMPLETED,
            results={"status": "explanation completed locally"},
        )

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.execute_explanation(request)


class KubernetesExplainabilityExecutor(BaseExecutor):
    def execute_explanation(self, request: TrustyAIRequest) -> TrustyAIResponse:
        metadata = self._create_metadata(request)
        return TrustyAIResponse(
            metadata=metadata,
            status=JobStatus.PENDING,
            results={"status": "explanation job deployed to kubernetes"},
        )

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.execute_explanation(request)


class LocalBiasDetectionExecutor(BaseExecutor):
    def execute_bias_detection(self, request: TrustyAIRequest) -> TrustyAIResponse:
        metadata = self._create_metadata(request)
        return TrustyAIResponse(
            metadata=metadata,
            status=JobStatus.COMPLETED,
            results={"status": "bias detection completed locally"},
        )

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.execute_bias_detection(request)


class KubernetesBiasDetectionExecutor(BaseExecutor):
    def execute_bias_detection(self, request: TrustyAIRequest) -> TrustyAIResponse:
        metadata = self._create_metadata(request)
        return TrustyAIResponse(
            metadata=metadata,
            status=JobStatus.PENDING,
            results={"status": "bias detection job deployed to kubernetes"},
        )

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.execute_bias_detection(request)


# Validator system is now implemented in validators.py


# Registry system for provider discovery
class ProviderRegistry:
    """Registry for provider implementations."""

    _providers: Dict[str, Dict[str, type[BaseProvider]]] = {}

    @classmethod
    def _get_deployment_modes(cls, provider_class: type[BaseProvider]) -> List[str]:
        """Safely get deployment modes from a provider class."""
        try:
            # Try to create instance without arguments first (for LMEvalProviderBase)
            instance = provider_class()
            return [mode.value for mode in instance.supported_deployment_modes]
        except (TypeError, Exception):
            # If that fails, try with default arguments (for BaseProvider)
            try:
                instance = provider_class("default", "local", k8s_client=None)
                return [mode.value for mode in instance.supported_deployment_modes]
            except Exception:
                # If both fail, return a default
                return ["local"]

    @classmethod
    def register_provider(cls, provider_class: type[BaseProvider]) -> None:
        """Register a provider implementation."""
        provider_type = provider_class.provider_type()
        provider_name = provider_class.get_provider_name()

        if provider_type not in cls._providers:
            cls._providers[provider_type] = {}

        cls._providers[provider_type][provider_name] = provider_class

    @classmethod
    def get_provider(
        cls,
        provider_type: str,
        provider_name: str | None = None,
    ) -> type[BaseProvider] | None:
        """Get a provider by type and optional name."""
        if provider_type not in cls._providers:
            return None

        # If no specific provider is requested, return the first one
        if provider_name is None and cls._providers[provider_type]:
            return next(iter(cls._providers[provider_type].values()))

        if provider_name is not None:
            return cls._providers[provider_type].get(provider_name)

        return None

    @classmethod
    def list_providers(cls, provider_type: str | None = None) -> Dict[str, List[Dict[str, Any]]]:
        """List available providers, filtered by type if specified."""
        result = {}

        if provider_type:
            if provider_type in cls._providers:
                result[provider_type] = [
                    {
                        "name": provider_name,
                        "description": provider_class.get_description(),
                        "deployment_modes": cls._get_deployment_modes(provider_class),
                    }
                    for provider_name, provider_class in cls._providers[provider_type].items()
                ]
        else:
            for type_name, providers in cls._providers.items():
                result[type_name] = [
                    {
                        "name": provider_name,
                        "description": provider_class.get_description(),
                        "deployment_modes": cls._get_deployment_modes(provider_class),
                    }
                    for provider_name, provider_class in providers.items()
                ]

        return result


# Note: Generic providers are not auto-registered to avoid cluttering the provider list
# Only concrete implementations should be registered


# Decorator for easy provider registration
def register_provider(provider_class: type[BaseProvider]) -> type[BaseProvider]:
    """Decorator to register a provider automatically."""
    ProviderRegistry.register_provider(provider_class)
    return provider_class


def register_eval_provider(provider_class: type[BaseProvider]) -> type[BaseProvider]:
    """Decorator specifically for evaluation providers."""
    if provider_class.provider_type() != "evaluation":
        raise ValueError("Provider must be of type 'evaluation' to use this decorator")
    return register_provider(provider_class)
