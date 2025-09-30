"""TrustyAI Python SDK.

Example usage with the Providers class:
    from trustyai import Providers
    from trustyai.core import DeploymentMode
    from trustyai.core.eval import EvaluationProviderConfig
    
    # Create provider - deployment mode is handled by configuration
    provider = Providers.eval.LMEvalProvider()
    
    # Local evaluation (default)
    config = EvaluationProviderConfig(
        deployment_mode=DeploymentMode.LOCAL,
        model="gpt2",
        tasks=["hellaswag"]
    )
    results = provider.evaluate(config)
    
    # Kubernetes evaluation
    config = EvaluationProviderConfig(
        deployment_mode=DeploymentMode.KUBERNETES,
        model="gpt2", 
        tasks=["hellaswag"]
    )
    results = provider.evaluate(config)
    
    # Access other provider types
    # provider = Providers.explainability.SomeProvider()
    # provider = Providers.bias_detection.SomeProvider()
"""

__version__ = "1.0.0a1"

# Import provider system
from trustyai.core.providers import ProviderRegistry
from trustyai.providers import Providers

# Import core components
from trustyai.core.lmevaljob import LMEvalJob
from trustyai.providers.eval.utils import LMEvalJobBuilder

# Import Kubernetes client
try:
    from trustyai.core.trustyai_kubernetes_client import TrustyAIKubernetesClient
except ImportError:
    # Kubernetes dependencies not available
    pass

# Try to import and register providers
try:
    # Import LM Eval provider if available
    from trustyai.providers.eval.lm_eval import LMEvalProvider
except ImportError:
    # Optional dependency not available
    pass
