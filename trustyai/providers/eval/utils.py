"""Utilities for LM Evaluation, including job builders and helper functions."""

from __future__ import annotations

import json
from typing import Any

from trustyai.core.lmevaljob import (
    Card,
    CustomDefinitions,
    EnvVar,
    LMEvalJob,
    Loader,
    Metric,
    MetricRef,
    ModelArg,
    Pod,
    PreprocessStep,
    TaskCard,
    TaskList,
    TaskRecipe,
    Template,
    Task,
    TemplateRef,
)


class LMEvalJobBuilder:
    """Builder class for creating LMEvalJob instances."""

    def __init__(self, name: str):
        """Initialize builder with job name."""
        self.job = LMEvalJob()
        self.job.metadata.name = name

    def namespace(self, namespace: str) -> LMEvalJobBuilder:
        """Set the namespace."""
        self.job.metadata.namespace = namespace
        return self

    def model(self, model_type: str = "hf") -> LMEvalJobBuilder:
        """Set the model type."""
        self.job.spec.model = model_type
        return self

    def add_model_arg(self, name: str, value: str) -> LMEvalJobBuilder:
        """Add a model argument."""
        self.job.spec.modelArgs.append(ModelArg(name=name, value=value))
        return self

    def pretrained_model(self, model_name: str) -> LMEvalJobBuilder:
        """Set the pretrained model name."""
        return self.add_model_arg("pretrained", model_name)

    def task_names(self, tasks: list[str]) -> LMEvalJobBuilder:
        """Set task names."""
        if not self.job.spec.taskList:
            self.job.spec.taskList = TaskList()
        self.job.spec.taskList.taskNames = tasks
        return self

    def task_recipes(self, recipes: list[TaskRecipe]) -> LMEvalJobBuilder:
        """Set task recipes."""
        if not self.job.spec.taskList:
            self.job.spec.taskList = TaskList()
        self.job.spec.taskList.taskRecipes = recipes
        return self

    def custom_card(self, card_json: str, template_ref: str, format_str: str, metrics: list[str]) -> LMEvalJobBuilder:
        """Add a custom card with template and metrics."""
        recipe = TaskRecipe(
            card=Card(custom=card_json),
            template=TemplateRef(ref=template_ref),
            format=format_str,
            metrics=[MetricRef(ref=metric) for metric in metrics]
        )
        return self.task_recipes([recipe])

    def custom_definitions(self, templates: list[Template] = None,
                          tasks: list[Task] = None,
                          metrics: list[Metric] = None) -> LMEvalJobBuilder:
        """Set custom definitions."""
        self.job.spec.custom = CustomDefinitions(
            templates=templates or [],
            tasks=tasks or [],
            metrics=metrics or []
        )
        return self

    def log_samples(self, log: bool = True) -> LMEvalJobBuilder:
        """Set log samples flag."""
        self.job.spec.logSamples = log
        return self

    def allow_online(self, allow: bool = True) -> LMEvalJobBuilder:
        """Set allow online flag."""
        self.job.spec.allowOnline = allow
        return self

    def allow_code_execution(self, allow: bool = True) -> LMEvalJobBuilder:
        """Set allow code execution flag."""
        self.job.spec.allowCodeExecution = allow
        return self

    def limit(self, limit_value: int | str) -> LMEvalJobBuilder:
        """Set evaluation limit."""
        self.job.spec.limit = str(limit_value)
        return self

    def env_var(self, name: str, value: str) -> LMEvalJobBuilder:
        """Add environment variable to pod container."""
        if not self.job.spec.pod:
            self.job.spec.pod = Pod()
        self.job.spec.pod.container.env.append(EnvVar(name=name, value=value))
        return self

    def hf_token(self, token: str) -> LMEvalJobBuilder:
        """Add HuggingFace token as environment variable."""
        return self.env_var("HF_TOKEN", token)

    def build(self) -> LMEvalJob:
        """Build and return the LMEvalJob instance."""
        return self.job

    @staticmethod
    def simple(name: str, model_name: str, tasks: list[str],
               namespace: str = "default", limit: int | str | None = None) -> LMEvalJob:
        """Create a simple LMEvalJob with basic configuration.

        Args:
            name: Name for the LMEvalJob
            model_name: Pretrained model name
            tasks: List of task names to evaluate
            namespace: Kubernetes namespace (default: "default")
            limit: Optional limit for evaluation samples

        Returns:
            Configured LMEvalJob instance
        """
        builder = (LMEvalJobBuilder(name)
                  .namespace(namespace)
                  .pretrained_model(model_name)
                  .task_names(tasks))

        if limit is not None:
            builder = builder.limit(limit)

        return builder.build()


# Helper functions for common use cases
def create_simple_lmeval_job(name: str, model_name: str, tasks: list[str],
                           namespace: str = "default", limit: int | str | None = None) -> LMEvalJob:
    """Create a simple LMEvalJob with basic configuration.

    .. deprecated:: 1.0.0
        Use :meth:`LMEvalJobBuilder.simple` instead.
    """
    import warnings
    warnings.warn(
        "create_simple_lmeval_job is deprecated. Use LMEvalJobBuilder.simple() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return LMEvalJobBuilder.simple(name, model_name, tasks, namespace, limit)


def create_task_card_json(task_card: TaskCard) -> str:
    """Create a JSON string from a TaskCard dataclass.

    Args:
        task_card: TaskCard instance with loader, preprocess_steps, task, and templates

    Returns:
        JSON string representation of the task card (with null values omitted)
    """
    def clean_dict(obj):
        """Convert object to dict and remove null/empty values."""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)

                if value is None:
                    continue

                if isinstance(value, list):
                    if value:  # Only include non-empty lists
                        cleaned_list = [clean_dict(item) for item in value]
                        cleaned_list = [item for item in cleaned_list if item]  # Remove empty items
                        if cleaned_list:
                            result[field_name] = cleaned_list
                elif isinstance(value, dict):
                    if value:  # Only include non-empty dicts
                        result[field_name] = value
                elif hasattr(value, '__dataclass_fields__'):
                    cleaned = clean_dict(value)
                    if cleaned:
                        result[field_name] = cleaned
                else:
                    result[field_name] = value
            return result
        return obj

    cleaned_data = clean_dict(task_card)
    return json.dumps(cleaned_data, indent=4)


# Helper functions for creating common preprocessing steps
def create_rename_splits_step(mapper: dict[str, str]) -> PreprocessStep:
    """Create a rename_splits preprocessing step."""
    return PreprocessStep(__type__="rename_splits", mapper=mapper)


def create_filter_by_condition_step(values: dict[str, Any], condition: str = "eq") -> PreprocessStep:
    """Create a filter_by_condition preprocessing step."""
    return PreprocessStep(__type__="filter_by_condition", values=values, condition=condition)


def create_rename_step(field_to_field: dict[str, str]) -> PreprocessStep:
    """Create a rename preprocessing step."""
    return PreprocessStep(__type__="rename", field_to_field=field_to_field)


def create_literal_eval_step(field: str) -> PreprocessStep:
    """Create a literal_eval preprocessing step."""
    return PreprocessStep(__type__="literal_eval", field=field)


def create_copy_step(field: str, to_field: str) -> PreprocessStep:
    """Create a copy preprocessing step."""
    return PreprocessStep(__type__="copy", field=field, to_field=to_field)


def create_llm_as_judge_metric(judge_model: str = "mistralai/Mistral-7B-Instruct-v0.2") -> Metric:
    """Create an LLM-as-a-Judge metric."""
    metric_config = {
        "__type__": "llm_as_judge",
        "inference_model": {
            "__type__": "hf_pipeline_based_inference_engine",
            "model_name": judge_model,
            "max_new_tokens": 256,
            "use_fp16": True
        },
        "template": "templates.response_assessment.rating.mt_bench_single_turn",
        "task": "rating.single_turn",
        "format": "formats.models.mistral.instruction",
        "main_score": f"{judge_model.replace('/', '_').replace('-', '_').lower()}_huggingface_template_mt_bench_single_turn"
    }

    return Metric(
        name="llmaaj_metric",
        value=json.dumps(metric_config, indent=4)
    )


def create_mt_bench_template() -> Template:
    """Create MT-Bench template."""
    template_config = {
        "__type__": "input_output_template",
        "instruction": "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n",
        "input_format": "[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]",
        "output_format": "[[{rating}]]",
        "postprocessors": ["processors.extract_mt_bench_rating_judgment"]
    }

    return Template(
        name="response_assessment.rating.mt_bench_single_turn",
        value=json.dumps(template_config, indent=4)
    )


def create_rating_task() -> Task:
    """Create rating task definition."""
    task_config = {
        "__type__": "task",
        "input_fields": {
            "question": "str",
            "answer": "str"
        },
        "outputs": {
            "rating": "float"
        },
        "metrics": ["metrics.spearman"]
    }

    return Task(
        name="response_assessment.rating.single_turn",
        value=json.dumps(task_config, indent=4)
    )