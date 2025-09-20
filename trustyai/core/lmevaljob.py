"""Python SDK for creating LMEvalJob Custom Resources."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ModelArg:
    """Model argument for LMEvalJob."""

    name: str
    value: str


@dataclass
class Loader:
    """Data loader configuration."""

    __type__: str = "load_hf"
    path: str = ""
    split: str = "train"


@dataclass
class PreprocessStep:
    """Preprocessing step configuration."""

    __type__: str
    mapper: dict[str, str] | None = None
    values: dict[str, Any] | None = None
    condition: str | None = None
    field_to_field: dict[str, str] | None = None
    field: str | None = None
    to_field: str | None = None


@dataclass
class TaskCard:
    """Task card configuration."""

    __type__: str = "task_card"
    loader: Loader = field(default_factory=Loader)
    preprocess_steps: list[PreprocessStep] = field(default_factory=list)
    task: str = ""
    templates: list[str] = field(default_factory=list)


@dataclass
class TemplateRef:
    """Template reference."""

    ref: str


@dataclass
class MetricRef:
    """Metric reference."""

    ref: str


@dataclass
class Card:
    """Card configuration for task recipe."""

    custom: str | None = None


@dataclass
class TaskRecipe:
    """Task recipe configuration."""

    card: Card | None = None
    template: TemplateRef | None = None
    format: str = ""
    metrics: list[MetricRef] = field(default_factory=list)


@dataclass
class TaskList:
    """Task list configuration."""

    taskRecipes: list[TaskRecipe] | None = None
    taskNames: list[str] | None = None


@dataclass
class Template:
    """Custom template definition."""

    name: str
    value: str


@dataclass
class Task:
    """Custom task definition."""

    name: str
    value: str


@dataclass
class Metric:
    """Custom metric definition."""

    name: str
    value: str


@dataclass
class CustomDefinitions:
    """Custom definitions for templates, tasks, and metrics."""

    templates: list[Template] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)


@dataclass
class EnvVar:
    """Environment variable for pod container."""

    name: str
    value: str


@dataclass
class Container:
    """Container configuration."""

    env: list[EnvVar] = field(default_factory=list)


@dataclass
class Pod:
    """Pod configuration."""

    container: Container = field(default_factory=Container)


@dataclass
class LMEvalJobSpec:
    """LMEvalJob specification."""

    model: str = "hf"
    modelArgs: list[ModelArg] = field(default_factory=list)
    taskList: TaskList | None = None
    custom: CustomDefinitions | None = None
    logSamples: bool = True
    allowOnline: bool = True
    allowCodeExecution: bool = True
    limit: str | None = None
    pod: Pod | None = None


@dataclass
class Metadata:
    """Kubernetes metadata."""

    name: str
    namespace: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)


@dataclass
class LMEvalJob:
    """LMEvalJob Custom Resource."""

    apiVersion: str = "trustyai.opendatahub.io/v1alpha1"
    kind: str = "LMEvalJob"
    metadata: Metadata = field(default_factory=lambda: Metadata(name=""))
    spec: LMEvalJobSpec = field(default_factory=LMEvalJobSpec)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return self._convert_to_dict(self)

    def to_yaml(self) -> str:
        """Convert to YAML representation."""
        data = self.to_dict()
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        """Convert to JSON representation."""
        data = self.to_dict()
        return json.dumps(data, indent=2)

    @staticmethod
    def _convert_to_dict(obj: Any) -> Any:
        """Recursively convert dataclass objects to dictionaries, omitting null/empty values."""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name, _field_def in obj.__dataclass_fields__.items():
                value = getattr(obj, field_name)

                # Skip None values
                if value is None:
                    continue

                # Handle lists
                if isinstance(value, list):
                    if value:  # Only include non-empty lists
                        converted_list = [LMEvalJob._convert_to_dict(item) for item in value]
                        # Filter out any None values from the converted list
                        converted_list = [item for item in converted_list if item is not None]
                        if converted_list:  # Only include if there are valid items
                            result[field_name] = converted_list

                # Handle dictionaries (including empty dicts from default_factory)
                elif isinstance(value, dict):
                    if value:  # Only include non-empty dicts
                        result[field_name] = value

                # Handle nested dataclass objects
                elif hasattr(value, '__dataclass_fields__'):
                    converted = LMEvalJob._convert_to_dict(value)
                    if converted:  # Only include non-empty converted objects
                        result[field_name] = converted

                # Handle all other values (strings, numbers, booleans, etc.)
                else:
                    # Include all non-None values, including empty strings, 0, False, etc.
                    result[field_name] = value

            return result
        return obj


