"""Validator registry for TrustyAI validators."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Type

from . import BaseValidator, ValidationResult


class ValidatorRegistry:
    """Registry for managing TrustyAI validators."""

    _validators: Dict[str, Type[BaseValidator]] = {}

    @classmethod
    def register(cls, name: str, validator_class: Type[BaseValidator]) -> None:
        """Register a validator class with a given name.

        Args:
            name: The name to register the validator under
            validator_class: The validator class to register
        """
        if not issubclass(validator_class, BaseValidator):
            raise ValueError(f"Validator class must inherit from BaseValidator")

        cls._validators[name] = validator_class

    @classmethod
    def get_validator(cls, name: str) -> Type[BaseValidator] | None:
        """Get a validator class by name.

        Args:
            name: The name of the validator

        Returns:
            The validator class or None if not found
        """
        return cls._validators.get(name)

    @classmethod
    def list_validators(cls) -> List[Dict[str, Any]]:
        """List all registered validators.

        Returns:
            A list of dictionaries with validator information
        """
        validators = []
        for name, validator_class in cls._validators.items():
            # Get docstring for description
            description = (validator_class.__doc__ or "").strip().split("\n")[0]

            validators.append(
                {
                    "name": name,
                    "class": validator_class.__name__,
                    "description": description,
                    "module": validator_class.__module__,
                }
            )

        return validators

    @classmethod
    def create_validator(
        cls, name: str, implementation: str, config: Dict[str, Any], **kwargs
    ) -> BaseValidator | None:
        """Create a validator instance by name.

        Args:
            name: The name of the validator
            implementation: The implementation string
            config: Configuration dictionary
            **kwargs: Additional keyword arguments for the validator constructor

        Returns:
            A validator instance or None if the validator is not found
        """
        validator_class = cls.get_validator(name)
        if not validator_class:
            return None

        # Check constructor signature to pass appropriate arguments
        sig = inspect.signature(validator_class.__init__)
        init_kwargs = {}

        for param_name in sig.parameters:
            if param_name == "self":
                continue
            elif param_name == "implementation":
                init_kwargs["implementation"] = implementation
            elif param_name == "config":
                init_kwargs["config"] = config
            elif param_name in kwargs:
                init_kwargs[param_name] = kwargs[param_name]

        return validator_class(**init_kwargs)


def validator(name: str):
    """Decorator to register a validator class.

    Args:
        name: The name to register the validator under
    """

    def decorator(validator_class: Type[BaseValidator]) -> Type[BaseValidator]:
        ValidatorRegistry.register(name, validator_class)
        return validator_class

    return decorator
