"""Unit tests for Kubernetes resource handling."""

from typing import Any, dict
from unittest.mock import patch

import pytest

from trustyai.core.kubernetes import (
    ConfigMapConverter,
    DeploymentConverter,
    KubernetesResource,
    KubernetesResourceConverter,
    ServiceConverter,
)


class TestKubernetesResource:
    """Test cases for KubernetesResource class."""

    def test_to_yaml_basic(self):
        """Test basic YAML conversion of a Kubernetes resource."""
        resource = KubernetesResource(
            api_version="v1",
            kind="Pod",
            metadata={"name": "trustyai-service-pod", "namespace": "trustyai-ns"},
            spec={
                "containers": [
                    {
                        "name": "trustyai-service",
                        "image": "quay.io/trustyai/trustyai-service:latest",
                    }
                ]
            },
        )

        yaml_output = resource.to_yaml()

        assert "apiVersion: v1" in yaml_output
        assert "kind: Pod" in yaml_output
        assert "name: trustyai-service-pod" in yaml_output
        assert "namespace: trustyai-ns" in yaml_output
        assert "containers:" in yaml_output

    def test_to_yaml_with_status(self):
        """Test YAML conversion with status field."""
        resource = KubernetesResource(
            api_version="v1",
            kind="Pod",
            metadata={"name": "trustyai-explainer-pod"},
            spec={
                "containers": [
                    {
                        "name": "trustyai-explainer",
                        "image": "quay.io/trustyai/trustyai-explainer:latest",
                    }
                ]
            },
            status={"phase": "Running"},
        )

        yaml_output = resource.to_yaml()

        assert "status:" in yaml_output
        assert "phase: Running" in yaml_output


class MockConverter(KubernetesResourceConverter):
    """Mock converter for testing the abstract base class."""

    def convert(self, config: dict[str, Any]) -> KubernetesResource:
        """Mock implementation of convert method."""
        return KubernetesResource(
            api_version="v1",
            kind="MockResource",
            metadata={"name": config.get("name", "test")},
            spec={"data": config.get("data", {})},
        )


class TestKubernetesResourceConverter:
    """Test cases for KubernetesResourceConverter abstract class."""

    def test_to_yaml_calls_convert(self):
        """Test that to_yaml method calls convert and returns YAML."""
        converter = MockConverter()
        config = {
            "name": "trustyai-config",
            "data": {"model_endpoint": "http://trustyai-model-service:8080"},
        }

        yaml_output = converter.to_yaml(config)

        assert "apiVersion: v1" in yaml_output
        assert "kind: MockResource" in yaml_output
        assert "name: trustyai-config" in yaml_output
        assert "data:" in yaml_output
        assert "model_endpoint: http://trustyai-model-service:8080" in yaml_output

    def test_to_yaml_with_empty_config(self):
        """Test to_yaml with empty configuration."""
        converter = MockConverter()
        config = {}

        yaml_output = converter.to_yaml(config)

        assert "apiVersion: v1" in yaml_output
        assert "kind: MockResource" in yaml_output
        assert "name: test" in yaml_output  # Default from mock

    def test_convert_multiple(self):
        """Test converting multiple configurations."""
        converter = MockConverter()
        configs = [
            {"name": "trustyai-service-config", "data": {"service_port": "8080"}},
            {"name": "trustyai-explainer-config", "data": {"explainer_type": "lime"}},
        ]

        resources = converter.convert_multiple(configs)

        assert len(resources) == 2
        assert resources[0].metadata["name"] == "trustyai-service-config"
        assert resources[1].metadata["name"] == "trustyai-explainer-config"
        assert resources[0].spec["data"]["service_port"] == "8080"
        assert resources[1].spec["data"]["explainer_type"] == "lime"

    def test_convert_multiple_empty_list(self):
        """Test converting empty list of configurations."""
        converter = MockConverter()
        configs = []

        resources = converter.convert_multiple(configs)

        assert resources == []


class TestDeploymentConverter:
    """Test cases for DeploymentConverter class."""

    def test_to_yaml_basic_deployment(self):
        """Test basic deployment YAML conversion."""
        converter = DeploymentConverter()
        config = {
            "name": "trustyai-service",
            "image": "quay.io/trustyai/trustyai-service:latest",
            "replicas": 3,
            "container_port": 8080,
        }

        yaml_output = converter.to_yaml(config)

        assert "apiVersion: apps/v1" in yaml_output
        assert "kind: Deployment" in yaml_output
        assert "name: trustyai-service" in yaml_output
        assert "replicas: 3" in yaml_output
        assert "image: quay.io/trustyai/trustyai-service:latest" in yaml_output
        assert "containerPort: 8080" in yaml_output

    def test_to_yaml_deployment_with_namespace(self):
        """Test deployment YAML conversion with namespace."""
        converter = DeploymentConverter()
        config = {
            "name": "trustyai-explainer",
            "image": "quay.io/trustyai/trustyai-explainer:latest",
            "namespace": "trustyai-ns",
        }

        yaml_output = converter.to_yaml(config)

        assert "namespace: trustyai-ns" in yaml_output

    def test_to_yaml_deployment_with_custom_labels(self):
        """Test deployment YAML conversion with custom labels."""
        converter = DeploymentConverter()
        config = {
            "name": "trustyai-bias-detector",
            "image": "quay.io/trustyai/trustyai-bias-detector:latest",
            "labels": {"component": "bias-detection", "version": "v2.1.0"},
        }

        yaml_output = converter.to_yaml(config)

        assert "component: bias-detection" in yaml_output
        assert "version: v2.1.0" in yaml_output

    def test_to_yaml_deployment_defaults(self):
        """Test deployment YAML conversion with default values."""
        converter = DeploymentConverter()
        config = {}

        yaml_output = converter.to_yaml(config)

        assert "name: default-app" in yaml_output
        assert "image: nginx:latest" in yaml_output  # Keep original default for this test
        assert "replicas: 1" in yaml_output
        assert "containerPort: 80" in yaml_output


class TestServiceConverter:
    """Test cases for ServiceConverter class."""

    def test_to_yaml_basic_service(self):
        """Test basic service YAML conversion."""
        converter = ServiceConverter()
        config = {
            "name": "trustyai-service-svc",
            "port": 8080,
            "target_port": 8080,
            "service_type": "ClusterIP",
        }

        yaml_output = converter.to_yaml(config)

        assert "apiVersion: v1" in yaml_output
        assert "kind: Service" in yaml_output
        assert "name: trustyai-service-svc" in yaml_output
        assert "port: 8080" in yaml_output
        assert "targetPort: 8080" in yaml_output
        assert "type: ClusterIP" in yaml_output

    def test_to_yaml_service_with_nodeport(self):
        """Test service YAML conversion with NodePort."""
        converter = ServiceConverter()
        config = {
            "name": "trustyai-explainer-svc",
            "port": 8080,
            "service_type": "NodePort",
            "node_port": 30080,
        }

        yaml_output = converter.to_yaml(config)

        assert "type: NodePort" in yaml_output
        assert "nodePort: 30080" in yaml_output

    def test_to_yaml_service_with_namespace(self):
        """Test service YAML conversion with namespace."""
        converter = ServiceConverter()
        config = {
            "name": "trustyai-metrics-svc",
            "namespace": "trustyai-ns",
        }

        yaml_output = converter.to_yaml(config)

        assert "namespace: trustyai-ns" in yaml_output

    def test_to_yaml_service_defaults(self):
        """Test service YAML conversion with default values."""
        converter = ServiceConverter()
        config = {}

        yaml_output = converter.to_yaml(config)

        assert "name: default-service" in yaml_output
        assert "port: 80" in yaml_output
        assert "type: ClusterIP" in yaml_output


class TestConfigMapConverter:
    """Test cases for ConfigMapConverter class."""

    def test_to_yaml_basic_configmap(self):
        """Test basic ConfigMap YAML conversion."""
        converter = ConfigMapConverter()
        config = {
            "name": "trustyai-config",
            "data": {
                "model.endpoint": "http://trustyai-model-service:8080/v1/models",
                "explainer.enabled": "true",
                "bias.threshold": "0.8",
            },
        }

        yaml_output = converter.to_yaml(config)

        assert "apiVersion: v1" in yaml_output
        assert "kind: ConfigMap" in yaml_output
        assert "name: trustyai-config" in yaml_output
        assert "data:" in yaml_output
        assert "model.endpoint: http://trustyai-model-service:8080/v1/models" in yaml_output
        assert "explainer.enabled: 'true'" in yaml_output
        assert "bias.threshold: '0.8'" in yaml_output
        # ConfigMaps should not have spec field
        assert "spec:" not in yaml_output

    def test_to_yaml_configmap_with_namespace(self):
        """Test ConfigMap YAML conversion with namespace."""
        converter = ConfigMapConverter()
        config = {
            "name": "trustyai-metrics-config",
            "namespace": "trustyai-ns",
            "data": {"metrics.port": "9090"},
        }

        yaml_output = converter.to_yaml(config)

        assert "namespace: trustyai-ns" in yaml_output

    def test_to_yaml_configmap_empty_data(self):
        """Test ConfigMap YAML conversion with empty data."""
        converter = ConfigMapConverter()
        config = {
            "name": "trustyai-empty-config",
            "data": {},
        }

        yaml_output = converter.to_yaml(config)

        assert "name: trustyai-empty-config" in yaml_output
        assert "data: {}" in yaml_output

    def test_to_yaml_configmap_defaults(self):
        """Test ConfigMap YAML conversion with default values."""
        converter = ConfigMapConverter()
        config = {}

        yaml_output = converter.to_yaml(config)

        assert "name: default-config" in yaml_output
        assert "data: {}" in yaml_output


class TestYamlImportError:
    """Test cases for handling YAML import errors."""

    @patch("trustyai.core.kubernetes.yaml", None)
    def test_to_yaml_without_pyyaml(self):
        """Test that ImportError is raised when PyYAML is not available."""
        resource = KubernetesResource(
            api_version="v1",
            kind="Pod",
            metadata={"name": "trustyai-pod"},
            spec={
                "containers": [
                    {
                        "name": "trustyai-service",
                        "image": "quay.io/trustyai/trustyai-service:latest",
                    }
                ]
            },
        )

        with pytest.raises(ImportError, match="PyYAML is not installed"):
            resource.to_yaml()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_to_yaml_with_complex_nested_data(self):
        """Test YAML conversion with complex nested data structures."""
        converter = DeploymentConverter()
        config = {
            "name": "trustyai-operator",
            "image": "quay.io/trustyai/trustyai-operator:latest",
            "labels": {
                "app": "trustyai-operator",
                "component": "operator",
                "tier": "control-plane",
                "version": "v2.1.0",
            },
            "replicas": 1,
            "container_port": 8443,
            "namespace": "trustyai-system",
        }

        yaml_output = converter.to_yaml(config)

        # Verify all complex data is properly serialised
        assert "app: trustyai-operator" in yaml_output
        assert "component: operator" in yaml_output
        assert "tier: control-plane" in yaml_output
        assert "version: v2.1.0" in yaml_output
        assert "replicas: 1" in yaml_output
        assert "containerPort: 8443" in yaml_output
        assert "namespace: trustyai-system" in yaml_output

    def test_to_yaml_with_special_characters(self):
        """Test YAML conversion with special characters in data."""
        converter = ConfigMapConverter()
        config = {
            "name": "trustyai-special-config",
            "data": {
                "model.inference.url": "http://model-service:8080/v1/predict",
                "logging.config": '{"level": "INFO", "format": "json"}',
                "multiline.script": "#!/bin/bash\necho 'Starting TrustyAI'\nexec trustyai-service",
            },
        }

        yaml_output = converter.to_yaml(config)

        # Should handle special characters properly
        assert "model.inference.url:" in yaml_output
        assert "logging.config:" in yaml_output
        assert "multiline.script:" in yaml_output
