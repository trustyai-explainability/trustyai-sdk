"""Core data models for TrustyAI SDK."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ExecutionMode(str, Enum):
    """Execution modes for providers."""

    LOCAL = "local"
    KUBERNETES = "kubernetes"
    REMOTE = "remote"


class JobStatus(str, Enum):
    """Job status for operations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelReference:
    """Standardised model reference format."""

    identifier: str  # Model identifier (HF path, file path, API endpoint)
    type: str  # Model type (huggingface, local, api, etc.)
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetReference:
    """Standardised dataset reference format."""

    identifier: str  # Dataset identifier or path
    type: str  # Dataset type (file, huggingface, synthetic, etc.)
    format: Optional[str] = None  # Data format (json, csv, parquet, etc.)
    schema: Optional[Dict[str, Any]] = None  # Dataset schema information
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustyAIMetadata:
    """Common metadata across all TrustyAI operations."""

    job_id: str
    provider_type: str  # Type of provider (evaluation, explainability, etc.)
    implementation: str  # Specific implementation used
    execution_mode: ExecutionMode  # Execution environment
    version: str  # TrustyAI version used
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: JobStatus = JobStatus.PENDING


@dataclass
class TrustyAIRequest:
    """Base request model for all provider operations."""

    provider: str  # Provider implementation to use
    execution_mode: ExecutionMode = ExecutionMode.LOCAL
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class TrustyAIResponse:
    """Base response model for all provider operations."""

    metadata: TrustyAIMetadata
    status: JobStatus
    results: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_info: Dict[str, Any] = field(default_factory=dict)

    def is_successful(self) -> bool:
        """Check if the operation was successful."""
        return self.status == JobStatus.COMPLETED and not self.errors


@dataclass
class EvaluationRequest:
    """Request model for evaluation operations."""

    provider: str  # Provider implementation to use
    model: ModelReference
    tasks: List[str]  # Evaluation tasks to run
    execution_mode: ExecutionMode = ExecutionMode.LOCAL
    dataset: Optional[DatasetReference] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class EvaluationMetric:
    """Individual evaluation metric result."""

    name: str  # Metric name
    value: Union[float, int, str]  # Metric value
    unit: Optional[str] = None  # Metric unit if applicable
    confidence_interval: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Results for a single evaluation task."""

    task_name: str
    metrics: List[EvaluationMetric]
    samples_evaluated: int
    execution_time: float  # Execution time in seconds
    raw_results: Optional[Dict[str, Any]] = None  # Provider-specific raw results


@dataclass
class EvaluationResponse:
    """Response model for evaluation operations."""

    metadata: TrustyAIMetadata
    status: JobStatus
    model: ModelReference
    results: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_info: Dict[str, Any] = field(default_factory=dict)
    tasks: List[TaskResult] = field(default_factory=list)
    summary_metrics: List[EvaluationMetric] = field(default_factory=list)
    total_samples: int = 0
    total_execution_time: float = 0.0

    def is_successful(self) -> bool:
        """Check if the operation was successful."""
        return self.status == JobStatus.COMPLETED and not self.errors


# Configuration models
@dataclass
class ProviderConfig:
    """Base configuration for all providers."""

    implementation: str
    execution_mode: ExecutionMode = ExecutionMode.LOCAL
    timeout: int = 3600  # Operation timeout in seconds
    retry_policy: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_attempts": 3,
            "backoff_factor": 2.0,
            "exceptions": ["ConnectionError", "TimeoutError"],
        }
    )


@dataclass
class LocalExecutionConfig:
    """Configuration for local execution."""

    working_directory: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    resource_limits: Optional[Dict[str, str]] = None


@dataclass
class KubernetesExecutionConfig:
    """Configuration for Kubernetes execution."""

    namespace: str
    job_template: Optional[str] = None
    resource_limits: Dict[str, str] = field(default_factory=lambda: {"cpu": "2", "memory": "4Gi"})
    service_account: Optional[str] = None
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)


class TrustyAIEncoder(json.JSONEncoder):
    """Custom JSON encoder for TrustyAI data models."""

    def default(self, obj: Any) -> Any:
        """Handle serialisation of TrustyAI objects."""
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        return super().default(obj)
