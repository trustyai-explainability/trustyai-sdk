"""LM Evaluation Harness provider for TrustyAI evaluation (legacy implementation)."""

import warnings
from typing import Any, Dict, List

from trustyai.core import DeploymentMode
from trustyai.core.eval import EvalProvider, EvaluationProviderConfig
from trustyai.core.kubernetes import KubernetesResource
from trustyai.core.registry import register_eval_provider

from .lm_eval_kubernetes import KubernetesLMEvalProvider
from .lm_eval_local import LocalLMEvalProvider


@register_eval_provider
class LMEvalProvider(EvalProvider):
    """LM Evaluation Harness provider for TrustyAI.
    
    This provider automatically delegates to the appropriate implementation
    (LocalLMEvalProvider or KubernetesLMEvalProvider) based on the deployment_mode
    specified in your configuration.
    
    Usage:
        # Local evaluation (default)
        config = EvaluationProviderConfig(
            deployment_mode=DeploymentMode.LOCAL,
            # ... other config
        )
        provider = Providers.eval.LMEvalProvider()
        results = provider.evaluate(config)
        
        # Kubernetes evaluation
        config = EvaluationProviderConfig(
            deployment_mode=DeploymentMode.KUBERNETES,
            # ... other config
        )
        provider = Providers.eval.LMEvalProvider()
        results = provider.evaluate(config)
    
    Note: When using Kubernetes deployment mode, only the LMEvalJob custom resource
    will be returned, without any supporting Deployment or Service resources.
    """

    def __init__(self) -> None:
        """Initialize the LM Eval provider wrapper."""
        self._local_provider = LocalLMEvalProvider()
        self._kubernetes_provider = KubernetesLMEvalProvider()
        
    @classmethod
    def get_provider_name(cls) -> str:
        """Return the name of this provider."""
        return "lm-eval-harness"

    @classmethod
    def provider_type(cls) -> str:
        """Return the type of provider."""
        return "eval"
        
    @classmethod
    def get_description(cls) -> str:
        """Return the description of the provider."""
        return (
            "LM Evaluation Harness for language model evaluation. "
            "Automatically delegates to local or Kubernetes implementation based on deployment mode."
        )

    def initialize(self, **kwargs: Any) -> None:
        """Initialize both the local and Kubernetes providers.
        
        Args:
            **kwargs: Additional configuration parameters
        """
        # Initialize both providers
        self._local_provider.initialize(**kwargs)
        self._kubernetes_provider.initialize(**kwargs)

    @property
    def supported_deployment_modes(self) -> List[DeploymentMode]:
        """Return the deployment modes supported by this provider."""
        return [DeploymentMode.LOCAL, DeploymentMode.KUBERNETES]

    def list_available_datasets(self) -> List[str]:
        """List available evaluation datasets for this provider.

        Returns:
            List of dataset names supported by this provider
        """
        # Delegate to local provider as both have the same datasets
        return self._local_provider.list_available_datasets()

    def list_available_metrics(self) -> List[str]:
        """List available evaluation metrics for this provider.

        Returns:
            List of metric names supported by this provider
        """
        # Delegate to local provider as both have the same metrics
        return self._local_provider.list_available_metrics()

    def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate a model using lm-evaluation-harness.

        Delegates to the appropriate specific provider based on deployment_mode.
        In Kubernetes mode, only the LMEvalJob custom resource will be returned,
        without any supporting Deployment or Service resources.

        Args:
            *args: Either (config) or (model_or_id, dataset, [metrics])
            **kwargs: Additional parameters

        Returns:
            Dictionary of evaluation results (for local mode) or deployment information (for Kubernetes mode)
        """
        # Determine which provider to use based on deployment mode
        # Extract deployment mode from kwargs or config
        if "deployment_mode" in kwargs:
            deployment_mode = kwargs["deployment_mode"]
            if isinstance(deployment_mode, str):
                deployment_mode = DeploymentMode(deployment_mode)
        elif args and isinstance(args[0], EvaluationProviderConfig):
            deployment_mode = args[0].deployment_mode
        else:
            # Default to local if not specified
            deployment_mode = DeploymentMode.LOCAL
            
        # Delegate to the appropriate provider
        if deployment_mode == DeploymentMode.KUBERNETES:
            return self._kubernetes_provider.evaluate(*args, **kwargs)
        else:
            return self._local_provider.evaluate(*args, **kwargs)
            
    def get_kubernetes_resources(self, config: Dict[str, Any]) -> List[KubernetesResource]:
        """Get Kubernetes resources needed by this provider.

        Delegates to the Kubernetes provider implementation.
        Note: Only returns the LMEvalJob custom resource without supporting
        Deployment or Service resources.

        Args:
            config: Provider configuration

        Returns:
            List of KubernetesResource objects (only the LMEvalJob)
        """
        return self._kubernetes_provider.get_kubernetes_resources(config)
