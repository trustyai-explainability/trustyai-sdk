"""Kubernetes-specific validators for TrustyAI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..kubernetes import kubernetes_client
from . import KubernetesValidator, ValidationResult

if TYPE_CHECKING:
    pass


class TrustyAIOperatorValidator(KubernetesValidator):
    """Validator for checking if the TrustyAI service operator is deployed in the cluster."""

    def __init__(self, implementation: str, config: dict, k8s_client=None, namespace: str = "trustyai-system"):
        """Initialize the TrustyAI operator validator.

        Args:
            implementation: The implementation name
            config: Configuration dictionary
            k8s_client: Optional Kubernetes client instance (deprecated, uses singleton now)
            namespace: Namespace where the TrustyAI operator is expected to be deployed
        """
        super().__init__(implementation, config, k8s_client)
        self.namespace = namespace

    def validate(self) -> ValidationResult:
        """Check if the TrustyAI service operator is deployed in the cluster.

        Returns:
            ValidationResult indicating whether the operator is deployed
        """
        if not kubernetes_client.is_initialized:
            return ValidationResult(
                is_valid=False,
                message="Kubernetes client not initialized - cannot validate TrustyAI operator",
                details={
                    "error": "no_kubernetes_client",
                    "suggestion": "Initialize Kubernetes client first: kubernetes_client.initialize()"
                }
            )
        return self._validate_with_client()

    def _validate_with_client(self) -> ValidationResult:
        """Validate using the provided Kubernetes client."""
        try:
            # Search for TrustyAI operator deployment across all namespaces
            apps_v1_api = kubernetes_client.client.AppsV1Api()

            # Search by label selector for TrustyAI operator
            label_selector = "app.kubernetes.io/part-of=trustyai"

            try:
                # List deployments with the TrustyAI label across all namespaces
                deployments = apps_v1_api.list_deployment_for_all_namespaces(
                    label_selector=label_selector
                )

                if not deployments.items:
                    # Fallback: search by deployment name pattern
                    deployments = apps_v1_api.list_deployment_for_all_namespaces()
                    trustyai_deployments = [
                        dep for dep in deployments.items
                        if "trustyai" in dep.metadata.name.lower() and "operator" in dep.metadata.name.lower()
                    ]

                    if not trustyai_deployments:
                        return ValidationResult(
                            is_valid=False,
                            message="TrustyAI operator deployment not found in any namespace",
                            details={
                                "error": "deployment_not_found",
                                "searched_namespaces": "all",
                                "label_selector": label_selector,
                                "client_provided": True,
                                "suggestion": "Install TrustyAI operator with documentation: https://trustyai.org/docs/main/trustyai-operator"
                            }
                        )

                    deployments.items = trustyai_deployments

                # Check the first found deployment
                deployment = deployments.items[0]
                namespace = deployment.metadata.namespace

                # Check if deployment is ready
                if deployment.status.ready_replicas and deployment.status.ready_replicas > 0:
                    return ValidationResult(
                        is_valid=True,
                        message=f"TrustyAI operator is deployed and ready in namespace '{namespace}'",
                        details={
                            "namespace": namespace,
                            "deployment": deployment.metadata.name,
                            "ready_replicas": deployment.status.ready_replicas,
                            "total_replicas": deployment.status.replicas,
                            "client_provided": True,
                            "found_by": "label_selector" if deployments.items else "name_pattern"
                        }
                    )
                return ValidationResult(
                    is_valid=False,
                    message=f"TrustyAI operator deployment exists but is not ready in namespace '{namespace}'",
                    details={
                        "namespace": namespace,
                        "deployment": deployment.metadata.name,
                        "ready_replicas": deployment.status.ready_replicas or 0,
                        "total_replicas": deployment.status.replicas or 0,
                        "client_provided": True,
                        "suggestion": f"Check operator logs: kubectl logs -n {namespace} deployment/{deployment.metadata.name}"
                    }
                )

            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    message=f"Error searching for TrustyAI operator deployment: {str(e)}",
                    details={
                        "error": "deployment_search_failed",
                        "client_provided": True,
                        "exception": str(e)
                    }
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Failed to validate TrustyAI operator with client: {str(e)}",
                details={
                    "error": "client_validation_failed",
                    "client_provided": True,
                    "exception": str(e)
                }
            )

    def check_custom_resource_definitions(self) -> ValidationResult:
        """Check if TrustyAI Custom Resource Definitions are installed.

        Returns:
            ValidationResult indicating whether the CRDs are installed
        """
        if not kubernetes_client.is_initialized:
            return ValidationResult(
                is_valid=False,
                message="Kubernetes client not initialized - cannot check TrustyAI CRDs",
                details={
                    "error": "no_kubernetes_client",
                    "suggestion": "Initialize Kubernetes client first: kubernetes_client.initialize()"
                }
            )
        return self._check_crds_with_client()

    def _check_crds_with_client(self) -> ValidationResult:
        """Check CRDs using the provided Kubernetes client."""
        try:
            # Check for TrustyAI CRDs
            apiextensions_v1_api = kubernetes_client.client.ApiextensionsV1Api()

            expected_crds = [
                "lmevaljobs.trustyai.redhat.com",
                "explanations.trustyai.redhat.com",
                "biasdetections.trustyai.redhat.com"
            ]

            missing_crds = []
            for crd_name in expected_crds:
                try:
                    apiextensions_v1_api.read_custom_resource_definition(name=crd_name)
                except Exception:
                    missing_crds.append(crd_name)

            if not missing_crds:
                return ValidationResult(
                    is_valid=True,
                    message="All TrustyAI Custom Resource Definitions are installed",
                    details={
                        "crds": expected_crds,
                        "client_provided": True
                    }
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    message=f"Missing TrustyAI Custom Resource Definitions: {missing_crds}",
                    details={
                        "expected_crds": expected_crds,
                        "missing_crds": missing_crds,
                        "client_provided": True,
                        "suggestion": "Install TrustyAI operator to get the required CRDs"
                    }
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Failed to check TrustyAI CRDs with client: {str(e)}",
                details={
                    "error": "crd_check_failed",
                    "client_provided": True,
                    "exception": str(e)
                }
            )
