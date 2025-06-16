"""Local LM Evaluation Harness provider for TrustyAI."""

from __future__ import annotations

from typing import Any, Dict

from trustyai.core import DeploymentMode
from trustyai.core.eval import EvaluationProviderConfig
from trustyai.core.registry import provider_registry
from trustyai.providers.eval.lm_eval_base import LMEvalProviderBase


@provider_registry.eval.register_local("lm-eval")
class LocalLMEvalProvider(LMEvalProviderBase):
    """Local implementation of LM Evaluation Harness for TrustyAI.
    
    This provider runs evaluation directly on the local machine.
    
    Note: Requires the 'eval' extra to be installed:
    pip install trustyai[eval]
    """
    
    def get_supported_deployment_modes(self) -> list[DeploymentMode]:
        """Return the deployment modes supported by this provider."""
        return [DeploymentMode.LOCAL]
    
    def evaluate(self, *args: Any, **kwargs) -> Dict[str, Any]:  # type: ignore
        """Evaluate a model using lm-evaluation-harness locally.

        Args:
            *args: Either (config) or (model_or_id, dataset, [metrics])
            **kwargs: Additional parameters

        Returns:
            Dictionary of evaluation results

        Raises:
            ValueError: If lm-eval is not initialized or configuration is invalid
            RuntimeError: If evaluation fails
        """
        # Parse arguments to get the configuration
        config = self._parse_args_to_config(*args, **kwargs)
        
        # Ensure we're using local mode
        if config.deployment_mode != DeploymentMode.LOCAL:
            config.deployment_mode = DeploymentMode.LOCAL
            
        return self._evaluate_local(config, **kwargs)
        
    def _evaluate_local(self, config: EvaluationProviderConfig, **kwargs) -> Dict[str, Any]:  # type: ignore
        """Run evaluation locally.

        Args:
            config: Evaluation configuration
            **kwargs: Additional parameters

        Returns:
            Dictionary of evaluation results

        Raises:
            ValueError: If lm-eval is not initialized
            RuntimeError: If evaluation fails
        """
        if self._lm_eval_module is None:
            self.initialize()

        if self._lm_eval_module is None:
            raise ValueError("LM Eval provider is not properly initialized")

        try:
            # Use device from config, or fallback to no_gpu flag if device not set
            if hasattr(config, 'device') and config.device:
                device = config.device
            else:
                # Fallback to no_gpu flag for backwards compatibility
                use_gpu = not kwargs.get("no_gpu", False)
                device = "cuda" if use_gpu else "cpu"
                config.device = device

            # Import necessary components from lm-eval
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM
            from lm_eval.tasks import TaskManager

            # Print diagnostics for device setting
            print(f"Using device: {device} for model evaluation")

            # Verify tasks are provided
            if not config.tasks:
                raise ValueError("At least one task must be specified in the config")

            # Get batch size
            batch_size = config.get_param("batch_size", 8)

            # Get max length (optional)
            max_length = config.get_param("max_length", 2048)

            # Create the model instance
            lm_obj = HFLM(
                pretrained=config.model,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )

            # Set up task manager
            include_path = config.get_param("include_path", None)
            task_manager = TaskManager(include_path=include_path)

            # Number of few-shot examples
            num_fewshot = config.get_param("num_fewshot", 0)

            # Run evaluation
            results = simple_evaluate(
                model=lm_obj,
                tasks=config.tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=config.limit,
                task_manager=task_manager,
            )

            # Filter metrics if specified in config
            if config.metrics:
                filtered_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
                for task_name, task_results in results["results"].items():
                    filtered_task_results = {}
                    for metric in config.metrics:
                        if metric in task_results:
                            filtered_task_results[metric] = task_results[metric]

                    if filtered_task_results:
                        if "results" not in filtered_results:
                            filtered_results["results"] = {}
                        filtered_results["results"][task_name] = filtered_task_results

                if filtered_results:
                    results = filtered_results

            return results
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}") from e 