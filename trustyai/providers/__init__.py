"""Provider implementations for TrustyAI."""

# Import provider modules
from . import eval

from typing import Any, Dict, List, Type, Union, cast

from trustyai.core.providers import BaseProvider, ProviderRegistry


class ProviderNamespace:
    """Dynamic namespace for providers of a specific type."""
    
    def __init__(self, provider_type: str):
        """Initialise the provider namespace.
        
        Args:
            provider_type: The type of providers (e.g., 'eval', 'explainability')
        """
        self._provider_type = provider_type
        self._update_providers()
    
    def _update_providers(self) -> None:
        """Update the provider attributes based on the current registry."""
        # Get providers from registry
        providers = ProviderRegistry._providers.get(self._provider_type, {})
        
        # Add each provider as an attribute
        for provider_name, provider_class in providers.items():
            # Convert provider name to a valid Python attribute name
            attr_name = self._sanitise_name(provider_name)
            setattr(self, attr_name, provider_class)
    
    def _sanitise_name(self, name: str) -> str:
        """Convert provider name to a valid Python attribute name.
        
        Args:
            name: The original provider name
            
        Returns:
            A sanitised name suitable for use as a Python attribute
        """
        # Handle specific cases
        if name == "lm_eval_harness":
            return "LMEvalProvider"
        
        # Convert snake_case to PascalCase and add Provider suffix if needed
        parts = name.split('_')
        class_name = ''.join(word.capitalize() for word in parts)
        
        # Add Provider suffix if not already present
        if not class_name.endswith('Provider'):
            class_name += 'Provider'
            
        return class_name
    
    def __getattr__(self, name: str) -> Type[BaseProvider]:
        """Get a provider by name, with fallback to registry lookup.
        
        Args:
            name: The provider name to look up
            
        Returns:
            The provider class
            
        Raises:
            AttributeError: If the provider is not found
        """
        # Refresh providers in case new ones were registered
        self._update_providers()
        
        # Check if we have it as a direct attribute after refresh
        if name in self.__dict__:
            return cast(Type[BaseProvider], self.__dict__[name])
        
        # If still not found, look in registry by original name
        providers = ProviderRegistry._providers.get(self._provider_type, {})
        for provider_name, provider_class in providers.items():
            if self._sanitise_name(provider_name) == name:
                return provider_class
        
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Available providers: {list(providers.keys())}"
        )
    
    def __dir__(self) -> List[str]:
        """Return list of available provider names."""
        self._update_providers()
        return [attr for attr in self.__dict__ if not attr.startswith('_')]


class ProvidersClass:
    """Organised access to all TrustyAI providers by type.
    
    Usage:
        provider = Providers.eval.LMEvalProvider()
        provider = Providers.explainability.SomeExplainabilityProvider()
    """
    
    def __init__(self) -> None:
        """Initialise the Providers class."""
        self._namespaces: Dict[str, ProviderNamespace] = {}
        self._update_namespaces()
    
    def _update_namespaces(self) -> None:
        """Update provider namespaces based on the current registry."""
        # Get all provider types from the registry
        for provider_type in ProviderRegistry._providers.keys():
            if provider_type not in self._namespaces:
                self._namespaces[provider_type] = ProviderNamespace(provider_type)
            else:
                # Update existing namespace
                self._namespaces[provider_type]._update_providers()
    
    def __getattr__(self, name: str) -> ProviderNamespace:
        """Get a provider namespace by type.
        
        Args:
            name: The provider type name (e.g., 'eval', 'explainability')
            
        Returns:
            A ProviderNamespace for the requested type
            
        Raises:
            AttributeError: If the provider type is not found
        """
        # Refresh namespaces in case new provider types were registered
        self._update_namespaces()
        
        # Check if we have this namespace
        if name in self._namespaces:
            return self._namespaces[name]
        
        # Check if there are any providers registered for this type
        if name in ProviderRegistry._providers:
            self._namespaces[name] = ProviderNamespace(name)
            return self._namespaces[name]
        
        available_types = list(ProviderRegistry._providers.keys())
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Available provider types: {available_types}"
        )
    
    def __dir__(self) -> List[str]:
        """Return list of available provider types."""
        self._update_namespaces()
        return list(self._namespaces.keys())


# Create a singleton instance
Providers = ProvidersClass()

# Export for convenience
__all__ = ['Providers', 'ProviderNamespace', 'ProvidersClass']
