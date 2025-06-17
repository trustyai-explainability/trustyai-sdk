"""Provider system for TrustyAI extensions following ADR-0010 specification."""
from __future__ import annotations

import abc
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .models import ExecutionMode, JobStatus, TrustyAIMetadata, TrustyAIRequest, TrustyAIResponse

if TYPE_CHECKING:
    pass


class BaseProvider(abc.ABC):
    """Base provider interface for all TrustyAI capabilities following ADR-0010."""

    def __init__(self, implementation: str, execution_mode: str = "local", **config: Any):
        """Initialise provider with implementation and execution mode.

        Args:
            implementation: Specific implementation name (e.g., "lm-evaluation-harness")
            execution_mode: Execution mode ("local" or "kubernetes")
            **config: Provider-specific configuration
        """
        self.implementation = implementation
        self.execution_mode = ExecutionMode(execution_mode)
        self.config = config
        self.validator = self._get_validator()

    @abc.abstractmethod
    def _get_validator(self) -> BaseValidator:
        """Get the appropriate validator for this provider implementation."""
        pass

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
            'implementation': self.implementation,
            'execution_mode': self.execution_mode,
            'config': self.config,
            'request': request
        }

    @classmethod
    @abc.abstractmethod
    def get_provider_type(cls) -> str:
        """Return the type of provider (e.g., 'evaluation', 'explainability')."""

    @classmethod
    def get_provider_name(cls) -> str:
        """Return the name of the provider implementation."""
        return getattr(cls, '_provider_name', cls.__name__)

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
            self.executor: Union[LocalEvaluationExecutor, KubernetesEvaluationExecutor] = LocalEvaluationExecutor(implementation, config)
        elif self.execution_mode == ExecutionMode.KUBERNETES:
            self.executor = KubernetesEvaluationExecutor(implementation, config)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")

    @classmethod
    def get_provider_type(cls) -> str:
        """Return the provider type."""
        return "evaluation"

    def _get_validator(self) -> BaseValidator:
        """Get appropriate validator based on execution mode."""
        if self.execution_mode == ExecutionMode.LOCAL:
            return LocalEvaluationValidator(self.implementation, self.config)
        elif self.execution_mode == ExecutionMode.KUBERNETES:
            return KubernetesEvaluationValidator(self.implementation, self.config)
        else:
            raise ValueError(f"No validator for execution mode: {self.execution_mode}")

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        """Execute evaluation using the configured executor."""
        return self.executor.execute_evaluation(request)

    def evaluate(self, model: str, tasks: List[str], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convenience method for evaluation."""
        # Create request from parameters
        from .models import EvaluationRequest, ModelReference
        
        model_ref = ModelReference(identifier=model, type="huggingface")
        request = EvaluationRequest(
            provider=self.implementation,
            model=model_ref,
            tasks=tasks,
            parameters=parameters or {},
            execution_mode=self.execution_mode
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
            self.executor: Union[LocalExplainabilityExecutor, KubernetesExplainabilityExecutor] = LocalExplainabilityExecutor(implementation, config)
        elif self.execution_mode == ExecutionMode.KUBERNETES:
            self.executor = KubernetesExplainabilityExecutor(implementation, config)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")

    @classmethod
    def get_provider_type(cls) -> str:
        return "explainability"

    def _get_validator(self) -> BaseValidator:
        if self.execution_mode == ExecutionMode.LOCAL:
            return LocalExplainabilityValidator(self.implementation, self.config)
        else:
            return KubernetesExplainabilityValidator(self.implementation, self.config)

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
            self.executor: Union[LocalBiasDetectionExecutor, KubernetesBiasDetectionExecutor] = LocalBiasDetectionExecutor(implementation, config)
        elif self.execution_mode == ExecutionMode.KUBERNETES:
            self.executor = KubernetesBiasDetectionExecutor(implementation, config)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")

    @classmethod
    def get_provider_type(cls) -> str:
        return "bias_detection"

    def _get_validator(self) -> BaseValidator:
        if self.execution_mode == ExecutionMode.LOCAL:
            return LocalBiasDetectionValidator(self.implementation, self.config)
        else:
            return KubernetesBiasDetectionValidator(self.implementation, self.config)

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
            provider_type=self.__class__.__name__.replace('Executor', '').lower(),
            implementation=self.implementation,
            execution_mode=ExecutionMode.LOCAL if 'Local' in self.__class__.__name__ else ExecutionMode.KUBERNETES,
            version="2.0.0a1"
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
            execution_info={"mode": "local", "implementation": self.implementation}
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
            execution_info={"mode": "kubernetes", "implementation": self.implementation}
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
            results={"status": "explanation completed locally"}
        )

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.execute_explanation(request)


class KubernetesExplainabilityExecutor(BaseExecutor):
    def execute_explanation(self, request: TrustyAIRequest) -> TrustyAIResponse:
        metadata = self._create_metadata(request)
        return TrustyAIResponse(
            metadata=metadata,
            status=JobStatus.PENDING,
            results={"status": "explanation job deployed to kubernetes"}
        )

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.execute_explanation(request)


class LocalBiasDetectionExecutor(BaseExecutor):
    def execute_bias_detection(self, request: TrustyAIRequest) -> TrustyAIResponse:
        metadata = self._create_metadata(request)
        return TrustyAIResponse(
            metadata=metadata,
            status=JobStatus.COMPLETED,
            results={"status": "bias detection completed locally"}
        )

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.execute_bias_detection(request)


class KubernetesBiasDetectionExecutor(BaseExecutor):
    def execute_bias_detection(self, request: TrustyAIRequest) -> TrustyAIResponse:
        metadata = self._create_metadata(request)
        return TrustyAIResponse(
            metadata=metadata,
            status=JobStatus.PENDING,
            results={"status": "bias detection job deployed to kubernetes"}
        )

    def execute(self, request: TrustyAIRequest) -> TrustyAIResponse:
        return self.execute_bias_detection(request)


# Validator system as described in ADR
class ValidationResult:
    """Result of validation checks."""

    def __init__(self, checks: List[CheckResult]):
        self.checks = checks
        self.is_valid = all(check.passed for check in checks)
        self.issues = [check for check in checks if not check.passed]


class CheckResult:
    """Result of an individual validation check."""

    def __init__(self, name: str, passed: bool, message: str, suggestion: Optional[str] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.suggestion = suggestion
        self.category = "system"


class BaseValidator(abc.ABC):
    """Base validator for provider implementations."""

    def __init__(self, implementation: str, config: Dict[str, Any]):
        self.implementation = implementation
        self.config = config

    @abc.abstractmethod
    def validate(self) -> ValidationResult:
        """Perform validation checks."""
        pass


class LocalEvaluationValidator(BaseValidator):
    """Validator for local evaluation providers."""

    def validate(self) -> ValidationResult:
        """Validate local evaluation requirements."""
        checks = [
            self._check_implementation_available(),
            self._check_system_resources(),
            self._check_model_access()
        ]
        return ValidationResult(checks)

    def _check_implementation_available(self) -> CheckResult:
        """Check if the evaluation implementation is available."""
        # TODO: Check if lm-eval, ragas, etc. are installed
        return CheckResult(
            "implementation_available",
            True,
            f"{self.implementation} is available",
            None
        )

    def _check_system_resources(self) -> CheckResult:
        """Check system resources."""
        return CheckResult(
            "system_resources",
            True,
            "Sufficient system resources available",
            None
        )

    def _check_model_access(self) -> CheckResult:
        """Check model accessibility."""
        return CheckResult(
            "model_access",
            True,
            "Model access validated",
            None
        )


class KubernetesEvaluationValidator(BaseValidator):
    """Validator for Kubernetes evaluation providers."""

    def validate(self) -> ValidationResult:
        """Validate Kubernetes evaluation requirements."""
        checks = [
            self._check_cluster_connectivity(),
            self._check_trustyai_operator(),
            self._check_namespace_exists(),
            self._check_resource_quotas()
        ]
        return ValidationResult(checks)

    def _check_cluster_connectivity(self) -> CheckResult:
        """Verify cluster connectivity."""
        return CheckResult(
            "cluster_connectivity",
            True,  # TODO: Implement actual check
            "Kubernetes cluster is accessible",
            "Check your kubeconfig file and cluster status"
        )

    def _check_trustyai_operator(self) -> CheckResult:
        """Check TrustyAI operator presence."""
        return CheckResult(
            "trustyai_operator",
            True,  # TODO: Implement actual check
            "TrustyAI operator is available",
            "Install TrustyAI operator: kubectl apply -f trustyai-operator.yaml"
        )

    def _check_namespace_exists(self) -> CheckResult:
        """Check if target namespace exists."""
        namespace = self.config.get('namespace', 'trustyai')
        return CheckResult(
            "namespace_exists",
            True,  # TODO: Implement actual check
            f"Namespace '{namespace}' exists",
            f"Create namespace: kubectl create namespace {namespace}"
        )

    def _check_resource_quotas(self) -> CheckResult:
        """Check resource availability."""
        return CheckResult(
            "resource_quotas",
            True,  # TODO: Implement actual check
            "Sufficient cluster resources available",
            "Check resource quotas and limits in your namespace"
        )


# Placeholder validators for other providers
class LocalExplainabilityValidator(BaseValidator):
    def validate(self) -> ValidationResult:
        return ValidationResult([CheckResult("explainability_local", True, "Local explainability ready")])


class KubernetesExplainabilityValidator(BaseValidator):
    def validate(self) -> ValidationResult:
        return ValidationResult([CheckResult("explainability_k8s", True, "Kubernetes explainability ready")])


class LocalBiasDetectionValidator(BaseValidator):
    def validate(self) -> ValidationResult:
        return ValidationResult([CheckResult("bias_detection_local", True, "Local bias detection ready")])


class KubernetesBiasDetectionValidator(BaseValidator):
    def validate(self) -> ValidationResult:
        return ValidationResult([CheckResult("bias_detection_k8s", True, "Kubernetes bias detection ready")])


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
        except TypeError:
            # If that fails, try with a default implementation argument (for BaseProvider)
            try:
                instance = provider_class("default")
                return [mode.value for mode in instance.supported_deployment_modes]
            except Exception:
                # If both fail, return a default
                return ["local"]

    @classmethod
    def register_provider(cls, provider_class: type[BaseProvider]) -> None:
        """Register a provider implementation."""
        provider_type = provider_class.get_provider_type()
        provider_name = provider_class.get_provider_name()

        if provider_type not in cls._providers:
            cls._providers[provider_type] = {}

        cls._providers[provider_type][provider_name] = provider_class

    @classmethod
    def get_provider(
        cls, provider_type: str, provider_name: str | None = None,
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


# Auto-register the core providers
ProviderRegistry.register_provider(EvaluationProvider)
ProviderRegistry.register_provider(ExplainabilityProvider)
ProviderRegistry.register_provider(BiasDetectionProvider)


# Decorator for easy provider registration
def register_provider(provider_class: type[BaseProvider]) -> type[BaseProvider]:
    """Decorator to register a provider automatically."""
    ProviderRegistry.register_provider(provider_class)
    return provider_class


def register_eval_provider(provider_class: type[BaseProvider]) -> type[BaseProvider]:
    """Decorator specifically for evaluation providers."""
    if provider_class.get_provider_type() != "evaluation":
        raise ValueError("Provider must be of type 'evaluation' to use this decorator")
    return register_provider(provider_class)
