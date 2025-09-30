"""Tests for TrustyAI Kubernetes client functionality."""

import json
from unittest.mock import Mock, patch, MagicMock
import pytest

from trustyai.core.lmevaljob import LMEvalJob, Metadata, LMEvalJobSpec
from trustyai.core.trustyai_kubernetes_client import (
    TrustyAIKubernetesClient,
    TrustyAIResourceConverter,
    SubmittedResource,
)
from trustyai.providers.eval.utils import LMEvalJobBuilder


class TestTrustyAIResourceConverter:
    """Test the resource converter functionality."""

    def test_lmevaljob_to_kubernetes_resource(self):
        """Test converting LMEvalJob to KubernetesResource."""
        # Create a simple LMEvalJob
        job = LMEvalJobBuilder.simple(
            name="test-job", model_name="test-model", tasks=["task1", "task2"], namespace="test-ns"
        )

        # Convert to KubernetesResource
        resource = TrustyAIResourceConverter.lmevaljob_to_kubernetes_resource(job)

        # Verify the conversion
        assert resource.api_version == "trustyai.opendatahub.io/v1alpha1"
        assert resource.kind == "LMEvalJob"
        assert resource.metadata["name"] == "test-job"
        assert resource.metadata["namespace"] == "test-ns"
        assert resource.spec["model"] == "hf"


class TestSubmittedResource:
    """Test the SubmittedResource functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_deployer = Mock()
        self.mock_deployer._initialize_client.return_value = True
        self.mock_custom_objects_api = Mock()
        self.mock_deployer._custom_objects_api = self.mock_custom_objects_api

        self.resource = SubmittedResource(
            name="test-resource", namespace="test-ns", kind="LMEvalJob", deployer=self.mock_deployer
        )

    def test_init(self):
        """Test SubmittedResource initialization."""
        assert self.resource.name == "test-resource"
        assert self.resource.namespace == "test-ns"
        assert self.resource.kind == "LMEvalJob"
        assert self.resource._plural == "lmevaljobs"

    def test_get_status_success(self):
        """Test successful status retrieval."""
        mock_response = {"status": {"phase": "Running", "message": "Job is running"}}
        self.mock_custom_objects_api.get_namespaced_custom_object.return_value = mock_response

        status = self.resource.get_status()

        assert status == {"phase": "Running", "message": "Job is running"}
        self.mock_custom_objects_api.get_namespaced_custom_object.assert_called_once_with(
            group="trustyai.opendatahub.io",
            version="v1alpha1",
            namespace="test-ns",
            plural="lmevaljobs",
            name="test-resource",
        )

    def test_get_status_failure(self):
        """Test status retrieval failure."""
        self.mock_deployer._initialize_client.return_value = False

        status = self.resource.get_status()

        assert status is None

    def test_get_status_exception(self):
        """Test status retrieval with exception."""
        self.mock_custom_objects_api.get_namespaced_custom_object.side_effect = Exception(
            "API Error"
        )

        with patch("builtins.print") as mock_print:
            status = self.resource.get_status()

        assert status is None
        mock_print.assert_called_once_with("Error getting LMEvalJob status: API Error")

    def test_delete_success(self):
        """Test successful resource deletion."""
        success, message = self.resource.delete()

        assert success is True
        assert "Successfully deleted LMEvalJob 'test-resource'" in message
        self.mock_custom_objects_api.delete_namespaced_custom_object.assert_called_once()

    def test_delete_failure(self):
        """Test resource deletion failure."""
        self.mock_deployer._initialize_client.return_value = False

        success, message = self.resource.delete()

        assert success is False
        assert message == "Failed to initialize Kubernetes client"

    def test_is_running(self):
        """Test is_running status check."""
        self.mock_custom_objects_api.get_namespaced_custom_object.return_value = {
            "status": {"phase": "Running"}
        }

        assert self.resource.is_running() is True

    def test_is_completed(self):
        """Test is_completed status check."""
        self.mock_custom_objects_api.get_namespaced_custom_object.return_value = {
            "status": {"phase": "Succeeded"}
        }

        assert self.resource.is_completed() is True

    def test_is_failed(self):
        """Test is_failed status check."""
        self.mock_custom_objects_api.get_namespaced_custom_object.return_value = {
            "status": {"phase": "Failed"}
        }

        assert self.resource.is_failed() is True

    @patch("time.sleep")
    def test_wait_for_completion_success(self, mock_sleep):
        """Test waiting for completion - success case."""
        # First call returns running, second returns completed
        self.mock_custom_objects_api.get_namespaced_custom_object.side_effect = [
            {"status": {"phase": "Running"}},
            {"status": {"phase": "Succeeded"}},
        ]

        success, message = self.resource.wait_for_completion(timeout_seconds=30)

        assert success is True
        assert "completed successfully" in message

    @patch("time.sleep")
    def test_wait_for_completion_failure(self, mock_sleep):
        """Test waiting for completion - failure case."""
        self.mock_custom_objects_api.get_namespaced_custom_object.return_value = {
            "status": {"phase": "Failed"}
        }

        success, message = self.resource.wait_for_completion(timeout_seconds=30)

        assert success is False
        assert "failed with status: Failed" in message

    def test_str_representation(self):
        """Test string representation."""
        assert str(self.resource) == "LMEvalJob(name='test-resource', namespace='test-ns')"

    def test_repr_representation(self):
        """Test detailed string representation."""
        expected = "SubmittedResource(name='test-resource', namespace='test-ns', kind='LMEvalJob')"
        assert repr(self.resource) == expected


class TestTrustyAIKubernetesClient:
    """Test the TrustyAI Kubernetes client."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("trustyai.core.trustyai_kubernetes_client.KubernetesDeployer"):
            self.client = TrustyAIKubernetesClient()
            self.mock_deployer = self.client.deployer

    def test_init(self):
        """Test client initialization."""
        assert self.client.deployer is not None

    def test_init_with_params(self):
        """Test client initialization with parameters."""
        with patch(
            "trustyai.core.trustyai_kubernetes_client.KubernetesDeployer"
        ) as mock_deployer_class:
            client = TrustyAIKubernetesClient(kubeconfig="/path/to/config", context="test-context")

            mock_deployer_class.assert_called_once_with(
                kubeconfig="/path/to/config", context="test-context"
            )

    def test_submit_success(self):
        """Test successful resource submission."""
        # Create test job
        job = LMEvalJobBuilder.simple(
            name="test-job", model_name="test-model", tasks=["task1"], namespace="test-ns"
        )

        # Mock successful deployment
        self.mock_deployer.deploy_resource.return_value = (True, "Success")

        # Submit the job
        submitted = self.client.submit(job)

        # Verify the result
        assert submitted is not None
        assert submitted.name == "test-job"
        assert submitted.namespace == "test-ns"
        assert submitted.kind == "LMEvalJob"
        self.mock_deployer.deploy_resource.assert_called_once()

    def test_submit_failure(self):
        """Test failed resource submission."""
        job = LMEvalJobBuilder.simple(name="test-job", model_name="test-model", tasks=["task1"])

        # Mock failed deployment
        self.mock_deployer.deploy_resource.return_value = (False, "Deployment failed")

        with patch("builtins.print") as mock_print:
            submitted = self.client.submit(job)

        assert submitted is None
        mock_print.assert_called_once_with("Failed to submit LMEvalJob: Deployment failed")

    def test_list_resources_success(self):
        """Test successful resource listing."""
        # Mock the deployer initialization and API response
        self.mock_deployer._initialize_client.return_value = True
        self.mock_deployer._custom_objects_api = Mock()
        self.mock_deployer._custom_objects_api.list_namespaced_custom_object.return_value = {
            "items": [{"metadata": {"name": "job1"}}, {"metadata": {"name": "job2"}}]
        }

        resources = self.client.list_resources(namespace="test-ns")

        assert len(resources) == 2
        assert resources[0].name == "job1"
        assert resources[1].name == "job2"

    def test_list_resources_failure(self):
        """Test resource listing failure."""
        self.mock_deployer._initialize_client.return_value = False

        resources = self.client.list_resources()

        assert resources == []

    def test_get_resource_exists(self):
        """Test getting an existing resource."""
        with patch.object(SubmittedResource, "get_full_resource") as mock_get_full:
            mock_get_full.return_value = {"metadata": {"name": "test-job"}}

            resource = self.client.get_resource("test-job", "test-ns")

            assert resource is not None
            assert resource.name == "test-job"
            assert resource.namespace == "test-ns"

    def test_get_resource_not_exists(self):
        """Test getting a non-existent resource."""
        with patch.object(SubmittedResource, "get_full_resource") as mock_get_full:
            mock_get_full.return_value = None

            resource = self.client.get_resource("nonexistent", "test-ns")

            assert resource is None

    def test_generate_yaml(self):
        """Test YAML generation."""
        job = LMEvalJobBuilder.simple(name="test-job", model_name="test-model", tasks=["task1"])

        yaml_content = self.client.generate_yaml(job)

        assert "apiVersion: trustyai.opendatahub.io/v1alpha1" in yaml_content
        assert "kind: LMEvalJob" in yaml_content
        assert "name: test-job" in yaml_content

    def test_save_yaml_to_file_success(self):
        """Test successful YAML file saving."""
        job = LMEvalJobBuilder.simple(name="test-job", model_name="test-model", tasks=["task1"])

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            result = self.client.save_yaml_to_file(job, "/path/to/file.yaml")

            assert result is True
            mock_open.assert_called_once_with("/path/to/file.yaml", "w")
            mock_file.write.assert_called_once()

    def test_save_yaml_to_file_failure(self):
        """Test YAML file saving failure."""
        job = LMEvalJobBuilder.simple(name="test-job", model_name="test-model", tasks=["task1"])

        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with patch("builtins.print") as mock_print:
                result = self.client.save_yaml_to_file(job, "/invalid/path.yaml")

            assert result is False
            mock_print.assert_called_once_with("Error saving YAML file: Permission denied")


class TestLMEvalJobBuilderSimple:
    """Test the new static simple method."""

    def test_simple_method(self):
        """Test the simple static method."""
        job = LMEvalJobBuilder.simple(
            name="simple-test", model_name="gpt-2", tasks=["task1", "task2"], namespace="custom-ns"
        )

        assert job.metadata.name == "simple-test"
        assert job.metadata.namespace == "custom-ns"
        assert job.spec.model == "hf"
        assert len(job.spec.modelArgs) == 1
        assert job.spec.modelArgs[0].name == "pretrained"
        assert job.spec.modelArgs[0].value == "gpt-2"
        assert job.spec.taskList.taskNames == ["task1", "task2"]

    def test_simple_method_default_namespace(self):
        """Test the simple method with default namespace."""
        job = LMEvalJobBuilder.simple(name="simple-test", model_name="gpt-2", tasks=["task1"])

        assert job.metadata.namespace == "default"

    def test_simple_method_with_limit_int(self):
        """Test the simple method with integer limit."""
        job = LMEvalJobBuilder.simple(
            name="simple-test", model_name="gpt-2", tasks=["task1"], limit=100
        )

        assert job.spec.limit == "100"

    def test_simple_method_with_limit_str(self):
        """Test the simple method with string limit."""
        job = LMEvalJobBuilder.simple(
            name="simple-test", model_name="gpt-2", tasks=["task1"], limit="50"
        )

        assert job.spec.limit == "50"

    def test_simple_method_with_no_limit(self):
        """Test the simple method without limit."""
        job = LMEvalJobBuilder.simple(name="simple-test", model_name="gpt-2", tasks=["task1"])

        assert job.spec.limit is None

    def test_simple_method_with_all_params(self):
        """Test the simple method with all parameters."""
        job = LMEvalJobBuilder.simple(
            name="comprehensive-test",
            model_name="microsoft/DialoGPT-medium",
            tasks=["hellaswag", "arc_easy"],
            namespace="production",
            limit=200,
        )

        assert job.metadata.name == "comprehensive-test"
        assert job.metadata.namespace == "production"
        assert job.spec.modelArgs[0].value == "microsoft/DialoGPT-medium"
        assert job.spec.taskList.taskNames == ["hellaswag", "arc_easy"]
        assert job.spec.limit == "200"


class TestDeprecatedFunction:
    """Test the deprecated create_simple_lmeval_job function."""

    def test_deprecated_function_warning(self):
        """Test that the deprecated function issues a warning."""
        from trustyai.providers.eval.utils import create_simple_lmeval_job

        with pytest.warns(DeprecationWarning, match="create_simple_lmeval_job is deprecated"):
            job = create_simple_lmeval_job(
                name="deprecated-test", model_name="test-model", tasks=["task1"]
            )

        # Verify it still works
        assert job.metadata.name == "deprecated-test"
        assert job.spec.modelArgs[0].value == "test-model"

    def test_deprecated_function_with_limit(self):
        """Test that the deprecated function works with limit parameter."""
        from trustyai.providers.eval.utils import create_simple_lmeval_job

        with pytest.warns(DeprecationWarning):
            job = create_simple_lmeval_job(
                name="deprecated-test-limit",
                model_name="test-model",
                tasks=["task1"],
                namespace="test-ns",
                limit=42,
            )

        # Verify it works with all parameters
        assert job.metadata.name == "deprecated-test-limit"
        assert job.metadata.namespace == "test-ns"
        assert job.spec.modelArgs[0].value == "test-model"
        assert job.spec.limit == "42"
