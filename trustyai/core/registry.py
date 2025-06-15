"""Provider registry decorators for TrustyAI."""

from typing import Callable, Type, TypeVar

from trustyai.core.eval import EvalProvider
from trustyai.core.providers import BaseProvider, ProviderRegistry

T = TypeVar("T", bound=Type[BaseProvider])
E = TypeVar("E", bound=Type[EvalProvider])


def provider(cls: T) -> T:
    """Decorator to automatically register a provider class with the ProviderRegistry.

    Example:
        @registry.provider
        class MyProvider(Provider):
            ...

    Args:
        cls: The provider class to register

    Returns:
        The decorated class
    """
    ProviderRegistry.register_provider(cls)
    return cls


def register_eval_provider(cls: E) -> E:
    """Decorator to automatically register an evaluation provider class with the ProviderRegistry.

    Example:
        @registry.register_eval_provider
        class MyEvalProvider(EvalProvider):
            ...

    Args:
        cls: The evaluation provider class to register

    Returns:
        The decorated class
    """
    ProviderRegistry.register_provider(cls)
    return cls


# Create a namespace class for organized provider registration
class ProviderRegistration:
    """Namespace for provider registration functions."""
    
    class eval:
        """Namespace for evaluation provider registration functions."""
        
        @staticmethod
        def register_local(provider_name: str) -> Callable[[E], E]:
            """Register a local evaluation provider.
            
            Example:
                @provider.eval.register_local("lm-eval")
                class LocalLMEvalProvider(EvalProvider):
                    ...
                    
            Args:
                provider_name: The name for this provider implementation
                
            Returns:
                A decorator function for registering the provider
            """
            def decorator(cls: E) -> E:
                # Set the provider name
                cls._provider_name = provider_name
                # Register with the provider registry
                ProviderRegistry.register_provider(cls)
                return cls
            return decorator
            
        @staticmethod
        def register_kubernetes(provider_name: str) -> Callable[[E], E]:
            """Register a Kubernetes evaluation provider.
            
            Example:
                @provider.eval.register_kubernetes("lm-eval")
                class KubernetesLMEvalProvider(EvalProvider):
                    ...
                    
            Args:
                provider_name: The name for this provider implementation
                
            Returns:
                A decorator function for registering the provider
            """
            def decorator(cls: E) -> E:
                # Set the provider name
                cls._provider_name = provider_name
                # Register with the provider registry
                ProviderRegistry.register_provider(cls)
                return cls
            return decorator


# Create an instance for easy imports
provider_registry = ProviderRegistration() 