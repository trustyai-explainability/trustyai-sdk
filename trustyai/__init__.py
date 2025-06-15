"""TrustyAI Python SDK."""

__version__ = "2.0.0a1"

# Import provider system
from trustyai.core.providers import ProviderRegistry

# Try to import and register providers
try:
    # Import LM Eval provider if available
    from trustyai.providers.eval.lm_eval import LMEvalProvider
except ImportError:
    # Optional dependency not available
    pass
