"""Unit tests for LMEvalJob SDK."""

import json

import pytest

from trustyai.core.lmevaljob import (
    LMEvalJob,
    Loader,
    TaskCard,
)
from trustyai.providers.eval.utils import (
    LMEvalJobBuilder,
    create_copy_step,
    create_filter_by_condition_step,
    create_literal_eval_step,
    create_llm_as_judge_metric,
    create_mt_bench_template,
    create_rating_task,
    create_rename_splits_step,
    create_rename_step,
    create_simple_lmeval_job,
    create_task_card_json,
)


# Helper function for MT-Bench specific configuration
def create_mt_bench_task_card(
    dataset_path: str = "OfirArviv/mt_bench_single_score_gpt4_judgement",
) -> str:
    """Create MT-Bench specific task card configuration for testing."""
    loader = Loader(__type__="load_hf", path=dataset_path, split="train")

    preprocess_steps = [
        create_rename_splits_step({"train": "test"}),
        create_filter_by_condition_step({"turn": 1}, "eq"),
        create_filter_by_condition_step({"reference": "[]"}, "eq"),
        create_rename_step(
            {
                "model_input": "question",
                "score": "rating",
                "category": "group",
                "model_output": "answer",
            }
        ),
        create_literal_eval_step("question"),
        create_copy_step("question/0", "question"),
        create_literal_eval_step("answer"),
        create_copy_step("answer/0", "answer"),
    ]

    task_card = TaskCard(
        __type__="task_card",
        loader=loader,
        preprocess_steps=preprocess_steps,
        task="tasks.response_assessment.rating.single_turn",
        templates=["templates.response_assessment.rating.mt_bench_single_turn"],
    )

    return create_task_card_json(task_card)


class TestLMEvalJob:
    """Test cases for LMEvalJob class."""

    def test_empty_lmevaljob_creation(self):
        """Test creating an empty LMEvalJob."""
        job = LMEvalJob()

        assert job.apiVersion == "trustyai.opendatahub.io/v1alpha1"
        assert job.kind == "LMEvalJob"
        assert job.metadata.name == ""
        assert job.spec.model == "hf"

    def test_lmevaljob_to_dict(self):
        """Test converting LMEvalJob to dictionary."""
        job = LMEvalJob()
        job.metadata.name = "test-job"
        job.metadata.namespace = "test-ns"

        result = job.to_dict()

        assert result["apiVersion"] == "trustyai.opendatahub.io/v1alpha1"
        assert result["kind"] == "LMEvalJob"
        assert result["metadata"]["name"] == "test-job"
        assert result["metadata"]["namespace"] == "test-ns"
        assert result["spec"]["model"] == "hf"

    def test_lmevaljob_to_yaml(self):
        """Test converting LMEvalJob to YAML."""
        job = LMEvalJob()
        job.metadata.name = "test-job"

        yaml_output = job.to_yaml()

        assert "apiVersion: trustyai.opendatahub.io/v1alpha1" in yaml_output
        assert "kind: LMEvalJob" in yaml_output
        assert "name: test-job" in yaml_output

    def test_lmevaljob_to_json(self):
        """Test converting LMEvalJob to JSON."""
        job = LMEvalJob()
        job.metadata.name = "test-job"

        json_output = job.to_json()
        parsed = json.loads(json_output)

        assert parsed["apiVersion"] == "trustyai.opendatahub.io/v1alpha1"
        assert parsed["kind"] == "LMEvalJob"
        assert parsed["metadata"]["name"] == "test-job"


class TestLMEvalJobBuilder:
    """Test cases for LMEvalJobBuilder class."""

    def test_builder_basic_creation(self):
        """Test basic job creation with builder."""
        job = (
            LMEvalJobBuilder("test-job")
            .namespace("test-ns")
            .pretrained_model("gpt2")
            .task_names(["hellaswag"])
            .build()
        )

        assert job.metadata.name == "test-job"
        assert job.metadata.namespace == "test-ns"
        assert len(job.spec.modelArgs) == 1
        assert job.spec.modelArgs[0].name == "pretrained"
        assert job.spec.modelArgs[0].value == "gpt2"
        assert job.spec.taskList.taskNames == ["hellaswag"]

    def test_builder_custom_card(self):
        """Test builder with custom card."""
        card_json = '{"__type__": "task_card", "task": "test"}'

        job = (
            LMEvalJobBuilder("test-job")
            .custom_card(
                card_json=card_json,
                template_ref="test-template",
                format_str="test-format",
                metrics=["test-metric"],
            )
            .build()
        )

        assert job.spec.taskList.taskRecipes is not None
        assert len(job.spec.taskList.taskRecipes) == 1

        recipe = job.spec.taskList.taskRecipes[0]
        assert recipe.card.custom == card_json
        assert recipe.template.ref == "test-template"
        assert recipe.format == "test-format"
        assert len(recipe.metrics) == 1
        assert recipe.metrics[0].ref == "test-metric"

    def test_builder_custom_definitions(self):
        """Test builder with custom definitions."""
        template = create_mt_bench_template()
        task = create_rating_task()
        metric = create_llm_as_judge_metric()

        job = (
            LMEvalJobBuilder("test-job")
            .custom_definitions(templates=[template], tasks=[task], metrics=[metric])
            .build()
        )

        assert job.spec.custom is not None
        assert len(job.spec.custom.templates) == 1
        assert len(job.spec.custom.tasks) == 1
        assert len(job.spec.custom.metrics) == 1

        assert job.spec.custom.templates[0].name == template.name
        assert job.spec.custom.tasks[0].name == task.name
        assert job.spec.custom.metrics[0].name == metric.name

    def test_builder_configuration_options(self):
        """Test builder configuration options."""
        job = (
            LMEvalJobBuilder("test-job")
            .model("custom")
            .limit(100)
            .log_samples(False)
            .allow_online(False)
            .allow_code_execution(False)
            .env_var("TEST_VAR", "test_value")
            .hf_token("test_token")
            .build()
        )

        assert job.spec.model == "custom"
        assert job.spec.limit == "100"
        assert job.spec.logSamples is False
        assert job.spec.allowOnline is False
        assert job.spec.allowCodeExecution is False

        assert job.spec.pod is not None
        assert len(job.spec.pod.container.env) == 2

        env_vars = {env.name: env.value for env in job.spec.pod.container.env}
        assert env_vars["TEST_VAR"] == "test_value"
        assert env_vars["HF_TOKEN"] == "test_token"


class TestHelperFunctions:
    """Test cases for helper functions."""

    def test_create_simple_lmeval_job(self):
        """Test simple job creation helper."""
        job = create_simple_lmeval_job(
            name="simple-test",
            model_name="gpt2",
            tasks=["hellaswag", "arc_easy"],
            namespace="test-ns",
        )

        assert job.metadata.name == "simple-test"
        assert job.metadata.namespace == "test-ns"
        assert job.spec.modelArgs[0].value == "gpt2"
        assert job.spec.taskList.taskNames == ["hellaswag", "arc_easy"]

    def test_create_task_card_json(self):
        """Test generic task card JSON creation."""
        loader = Loader(__type__="load_hf", path="test/dataset", split="train")
        preprocess_steps = [
            create_rename_splits_step({"train": "test"}),
            create_filter_by_condition_step({"field": "value"}, "eq"),
        ]
        task_card = TaskCard(
            __type__="task_card",
            loader=loader,
            preprocess_steps=preprocess_steps,
            task="test.task",
            templates=["test.template"],
        )

        card_json = create_task_card_json(task_card)
        card_data = json.loads(card_json)

        assert card_data["__type__"] == "task_card"
        assert card_data["loader"]["__type__"] == "load_hf"
        assert card_data["loader"]["path"] == "test/dataset"
        assert len(card_data["preprocess_steps"]) == 2
        assert card_data["task"] == "test.task"

    def test_create_mt_bench_task_card(self):
        """Test MT-Bench specific task card creation (test-specific configuration)."""
        card_json = create_mt_bench_task_card()
        card_data = json.loads(card_json)

        assert card_data["__type__"] == "task_card"
        assert card_data["loader"]["__type__"] == "load_hf"
        assert card_data["loader"]["path"] == "OfirArviv/mt_bench_single_score_gpt4_judgement"
        assert "preprocess_steps" in card_data
        assert len(card_data["preprocess_steps"]) == 8  # All the MT-Bench specific steps

    def test_preprocess_step_creators(self):
        """Test preprocessing step creation functions."""
        # Test rename_splits
        rename_step = create_rename_splits_step({"train": "test"})
        assert rename_step.__type__ == "rename_splits"
        assert rename_step.mapper == {"train": "test"}

        # Test filter_by_condition
        filter_step = create_filter_by_condition_step({"turn": 1}, "eq")
        assert filter_step.__type__ == "filter_by_condition"
        assert filter_step.values == {"turn": 1}
        assert filter_step.condition == "eq"

        # Test rename
        rename_fields_step = create_rename_step({"old": "new"})
        assert rename_fields_step.__type__ == "rename"
        assert rename_fields_step.field_to_field == {"old": "new"}

        # Test literal_eval
        literal_step = create_literal_eval_step("question")
        assert literal_step.__type__ == "literal_eval"
        assert literal_step.field == "question"

        # Test copy
        copy_step = create_copy_step("source", "dest")
        assert copy_step.__type__ == "copy"
        assert copy_step.field == "source"
        assert copy_step.to_field == "dest"

    def test_create_llm_as_judge_metric(self):
        """Test LLM-as-a-Judge metric creation."""
        metric = create_llm_as_judge_metric("test-model")
        metric_data = json.loads(metric.value)

        assert metric.name == "llmaaj_metric"
        assert metric_data["__type__"] == "llm_as_judge"
        assert metric_data["inference_model"]["model_name"] == "test-model"

    def test_create_mt_bench_template(self):
        """Test MT-Bench template creation."""
        template = create_mt_bench_template()
        template_data = json.loads(template.value)

        assert template.name == "response_assessment.rating.mt_bench_single_turn"
        assert template_data["__type__"] == "input_output_template"
        assert "instruction" in template_data

    def test_create_rating_task(self):
        """Test rating task creation."""
        task = create_rating_task()
        task_data = json.loads(task.value)

        assert task.name == "response_assessment.rating.single_turn"
        assert task_data["__type__"] == "task"
        assert "input_fields" in task_data
        assert "outputs" in task_data


class TestCompatibility:
    """Test cases to ensure compatibility"""

    def test_recreation(self):
        """Test that we can recreate."""

        task_card_json = create_mt_bench_task_card()

        # Create custom definitions
        mt_bench_template = create_mt_bench_template()
        rating_task = create_rating_task()
        llm_judge_metric = create_llm_as_judge_metric("mistralai/Mistral-7B-Instruct-v0.2")

        job = (
            LMEvalJobBuilder("custom-llmaaj-metric")
            .model("hf")
            .add_model_arg("pretrained", "google/flan-t5-small")
            .custom_card(
                card_json=task_card_json,
                template_ref="response_assessment.rating.mt_bench_single_turn",
                format_str="formats.models.mistral.instruction",
                metrics=["llmaaj_metric"],
            )
            .custom_definitions(
                templates=[mt_bench_template], tasks=[rating_task], metrics=[llm_judge_metric]
            )
            .log_samples(True)
            .allow_online(True)
            .allow_code_execution(True)
            .hf_token("<HF_TOKEN>")
            .build()
        )

        # Verify structure
        assert job.metadata.name == "custom-llmaaj-metric"
        assert job.spec.model == "hf"
        assert job.spec.logSamples is True
        assert job.spec.allowOnline is True
        assert job.spec.allowCodeExecution is True

        # Check model args
        assert len(job.spec.modelArgs) == 1
        assert job.spec.modelArgs[0].name == "pretrained"
        assert job.spec.modelArgs[0].value == "google/flan-t5-small"

        # Check task recipes
        assert job.spec.taskList.taskRecipes is not None
        assert len(job.spec.taskList.taskRecipes) == 1

        recipe = job.spec.taskList.taskRecipes[0]
        assert recipe.card.custom is not None
        assert '"__type__": "task_card"' in recipe.card.custom
        assert recipe.template.ref == "response_assessment.rating.mt_bench_single_turn"
        assert recipe.format == "formats.models.mistral.instruction"

        # Check custom definitions
        assert job.spec.custom is not None
        assert len(job.spec.custom.templates) == 1
        assert len(job.spec.custom.tasks) == 1
        assert len(job.spec.custom.metrics) == 1

        # Check environment variables
        assert job.spec.pod is not None
        env_vars = {env.name: env.value for env in job.spec.pod.container.env}
        assert env_vars["HF_TOKEN"] == "<HF_TOKEN>"

    def test_yaml_output_structure(self):
        """Test that YAML output has correct structure"""
        task_card_json = create_mt_bench_task_card()
        job = (
            LMEvalJobBuilder("test-job")
            .custom_card(
                card_json=task_card_json,
                template_ref="test-template",
                format_str="test-format",
                metrics=["test-metric"],
            )
            .build()
        )

        yaml_output = job.to_yaml()

        # Verify that card.custom is a string
        assert (
            'card:\n        custom: "' in yaml_output or 'card:\n          custom: "' in yaml_output
        )

        # Verify the JSON is properly escaped in YAML
        assert (
            '"__type__": "task_card"' in yaml_output
            or '\\"__type__\\": \\"task_card\\"' in yaml_output
        )

    def test_card_custom_field_format(self):
        """Test that card.custom field contains properly formatted JSON string."""
        card_json = '{"__type__": "task_card", "test": "value"}'

        job = (
            LMEvalJobBuilder("test-job")
            .custom_card(
                card_json=card_json,
                template_ref="test-template",
                format_str="test-format",
                metrics=["test-metric"],
            )
            .build()
        )

        # Verify the card.custom field contains the exact JSON string
        recipe = job.spec.taskList.taskRecipes[0]
        assert recipe.card.custom == card_json

        # Verify it's valid JSON
        parsed = json.loads(recipe.card.custom)
        assert parsed["__type__"] == "task_card"
        assert parsed["test"] == "value"

    def test_null_value_omission(self):
        """Test that null/None values are omitted from JSON/YAML output."""
        # Create a job with minimal configuration
        job = (
            LMEvalJobBuilder("clean-test")
            .pretrained_model("test-model")
            .task_names(["test-task"])
            .build()
        )

        # Convert to dict and JSON
        job_dict = job.to_dict()
        job_json = job.to_json()

        # Verify no None values in the dict
        def check_no_none_values(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    assert value is not None, f"Found None value at {path}.{key}"
                    check_no_none_values(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    assert item is not None, f"Found None value at {path}[{i}]"
                    check_no_none_values(item, f"{path}[{i}]")

        check_no_none_values(job_dict)

        # Verify no actual null values in JSON output (not just the string "null")
        import re

        # Look for JSON null values (": null" or "null,")
        null_pattern = r":\s*null[,\s}]"
        assert not re.search(null_pattern, job_json), f"Found JSON null values in: {job_json}"

        # Test TaskCard JSON creation also omits nulls
        loader = Loader(__type__="load_hf", path="test", split="train")
        step = create_rename_splits_step({"train": "test"})
        task_card = TaskCard(loader=loader, preprocess_steps=[step], task="test")

        card_json = create_task_card_json(task_card)
        # Check for actual JSON null values in TaskCard JSON
        assert not re.search(null_pattern, card_json), (
            f"Found JSON null values in TaskCard: {card_json}"
        )

        # Verify the created card JSON is clean
        card_data = json.loads(card_json)
        check_no_none_values(card_data)


if __name__ == "__main__":
    pytest.main([__file__])
