"""Core functionality for TrustyAI."""

from .eval import EvalProvider
from .kubernetes import (
    ConfigMapConverter,
    DeploymentConverter,
    KubernetesDeployer,
    KubernetesResource,
    KubernetesResourceConverter,
    ServiceConverter,
)
from .providers import BaseProvider, ProviderRegistry
from .models import ExecutionMode
# Backward compatibility aliases
Provider = BaseProvider
DeploymentMode = ExecutionMode
from .registry import provider as registry_provider, register_eval_provider, provider_registry
