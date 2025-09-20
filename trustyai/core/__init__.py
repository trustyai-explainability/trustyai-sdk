"""Core functionality for TrustyAI."""

from .eval import EvalProvider
from .kubernetes import (
    ConfigMapConverter,
    DeploymentConverter,
    KubernetesDeployer,
    KubernetesResource,
    KubernetesResourceConverter,
    ServiceConverter,
    kubernetes_client,
)
from .lmevaljob import LMEvalJob
from .models import ExecutionMode
from .providers import BaseProvider, ProviderRegistry

# Backward compatibility aliases
Provider = BaseProvider
DeploymentMode = ExecutionMode
from .registry import provider as registry_provider
from .registry import provider_registry, register_eval_provider
