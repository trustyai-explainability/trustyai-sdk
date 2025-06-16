"""Base class for LM Evaluation providers."""

from __future__ import annotations

from typing import Any, TypedDict

from trustyai.core import DeploymentMode
from trustyai.core.eval import EvalProvider, EvaluationProviderConfig


class LMEvalKwargs(TypedDict, total=False):
    """Typed kwargs for LMEval provider."""

    log_level: str
    no_gpu: bool
    batch_size: int
    max_length: int
    num_fewshot: int
    include_path: str | None
    deployment_mode: str | DeploymentMode
    config: EvaluationProviderConfig
    device: str
    limit: int | None
    metrics: list[str] | None


class LMEvalProviderBase(EvalProvider):
    """Base class for LM Evaluation Harness providers."""

    def __init__(self) -> None:
        """Initialize the base LM Eval provider."""
        self._lm_eval_module = None
        self._available_metrics = None

    @classmethod
    def get_description(cls) -> str:
        """Return the description of the provider."""
        return "LM Evaluation Harness for language model evaluation"

    def initialize(self, **kwargs) -> None:  # type: ignore
        """Initialize the LM Eval provider.

        Args:
            **kwargs: Additional configuration parameters

        Raises:
            ImportError: If lm-eval is not installed
        """
        try:
            import lm_eval
            from lm_eval.utils import setup_logging

            self._lm_eval_module = lm_eval
            # Setup logging for lm-eval
            setup_logging(kwargs.get("log_level", "INFO"))
        except ImportError:
            raise ImportError(  # noqa: B904, TRY003
                "lm-eval package is required for LMEvalProvider. "  # noqa: EM101
                "Install it with: pip install trustyai[eval]",
            )

        # Initialize available metrics
        self._load_available_metrics()

    def _load_available_metrics(self) -> None:
        """Load available metrics from lm-eval."""
        if self._lm_eval_module is None:
            return

        # Common metrics used in lm-evaluation-harness
        self._available_metrics = [
            "acc",
            "acc_norm",
            "perplexity",
            "bleu",
            "rouge",
            "exact_match",
            "f1",
            "precision",
            "recall",
            "matthews_correlation",
            "multiple_choice_grade",
            "wer",
            "ter",
        ]

    def list_available_datasets(self) -> list[str]:
        """List available evaluation datasets for this provider.

        Returns:
            List of dataset names supported by this provider
        """
        if self._lm_eval_module is None:
            self.initialize()

        try:
            # Import the task list
            from lm_eval.tasks import TaskManager

            task_manager = TaskManager()
            return sorted(task_manager.get_task_list())
        except (ImportError, AttributeError):
            return []

    def list_available_metrics(self) -> list[str]:
        """List available evaluation metrics for this provider.

        Returns:
            List of metric names supported by this provider
        """
        if self._available_metrics is None:
            self._load_available_metrics()

        return self._available_metrics or []
    
    def _parse_args_to_config(self, *args: Any, **kwargs) -> EvaluationProviderConfig:  # type: ignore
        """Parse arguments into a configuration object.

        Args:
            *args: Either (config) or (model_or_id, dataset, [metrics])
            **kwargs: Additional parameters

        Returns:
            EvaluationProviderConfig with parsed parameters

        Raises:
            ValueError: If arguments are invalid
        """
        # Debug logging for namespace
        if "namespace" in kwargs:
            print(f"[DEBUG - _parse_args_to_config] Namespace in kwargs: {kwargs['namespace']}")

        # Handle device parameter - respect config.device if present, otherwise use no_gpu flag
        use_gpu = not kwargs.get("no_gpu", False)
        default_device = "cuda" if use_gpu else "cpu"

        # Remove no_gpu from kwargs to avoid duplication
        kwargs_copy = kwargs.copy()
        if "no_gpu" in kwargs_copy:
            kwargs_copy.pop("no_gpu")

        # Get deployment mode
        deployment_mode = kwargs_copy.get("deployment_mode", DeploymentMode.LOCAL)
        if isinstance(deployment_mode, str):
            deployment_mode = DeploymentMode(deployment_mode)

        # Remove deployment_mode from kwargs to avoid duplication
        if "deployment_mode" in kwargs_copy:
            kwargs_copy.pop("deployment_mode")

        # Handle both new and legacy calling conventions
        if len(args) == 0:
            # Expecting config in kwargs
            if "config" not in kwargs:
                raise ValueError("Missing required argument: config")

            # Get the config from kwargs
            config = kwargs.pop("config")

            # Only override device if not explicitly set in config
            if not hasattr(config, 'device') or config.device is None:
                config.device = default_device
            config.deployment_mode = deployment_mode
            
            # Debug check for namespace
            print(f"[DEBUG - _parse_args_to_config] Args=0: has namespace? {'namespace' in config.additional_params}")
            if "namespace" in config.additional_params:
                print(f"[DEBUG - _parse_args_to_config] Namespace value: {config.additional_params['namespace']}")

            return config
        elif len(args) == 1:
            # Single argument should be config
            if isinstance(args[0], EvaluationProviderConfig):
                # Use existing config, only override device if not set or if no_gpu flag is present
                config = args[0]
                
                # Only override device if no_gpu flag is present or device is not set
                if kwargs.get("no_gpu", False) or not hasattr(config, 'device') or config.device is None:
                    config.device = default_device
                
                if "deployment_mode" not in kwargs:
                    config.deployment_mode = deployment_mode
                
                # Debug check for namespace
                print(f"[DEBUG - _parse_args_to_config] Args=1: has namespace? {'namespace' in config.additional_params}")
                if "namespace" in config.additional_params:
                    print(f"[DEBUG - _parse_args_to_config] Namespace value: {config.additional_params['namespace']}")

                return config
            else:
                raise ValueError(f"Expected EvaluationProviderConfig, got {type(args[0])}")
        else:
            # Legacy style: (model_or_id, dataset, [metrics])
            model_or_id = args[0]
            dataset = args[1]

            # Handle metrics properly
            metrics = None
            if len(args) > 2 and "metrics" not in kwargs:
                metrics = args[2]
            elif "metrics" in kwargs:
                metrics = kwargs.pop("metrics")

            # Create a new config
            config = EvaluationProviderConfig(
                evaluation_name=f"eval_{dataset}",
                model=model_or_id,
                tasks=[dataset],
                metrics=metrics,
                device=default_device,
                deployment_mode=deployment_mode,
                **kwargs_copy
            )
            
            # Debug check for namespace
            print(f"[DEBUG - _parse_args_to_config] Args={len(args)}: has namespace? {'namespace' in config.additional_params}")
            if "namespace" in config.additional_params:
                print(f"[DEBUG - _parse_args_to_config] Namespace value: {config.additional_params['namespace']}")

            return config 