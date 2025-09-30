"""Kubernetes client specifically for TrustyAI resources."""

from __future__ import annotations

import time
from typing import Any

from trustyai.core.kubernetes import KubernetesDeployer, KubernetesResource
from trustyai.core.lmevaljob import LMEvalJob


class TrustyAIResourceConverter:
    """Converter for TrustyAI specific resources."""

    @staticmethod
    def lmevaljob_to_kubernetes_resource(lmeval_job: LMEvalJob) -> KubernetesResource:
        """Convert an LMEvalJob to a KubernetesResource.

        Args:
            lmeval_job: LMEvalJob instance to convert

        Returns:
            KubernetesResource representation
        """
        resource_dict = lmeval_job.to_dict()

        return KubernetesResource(
            api_version=resource_dict["apiVersion"],
            kind=resource_dict["kind"],
            metadata=resource_dict["metadata"],
            spec=resource_dict["spec"],
        )


class SubmittedResource:
    """Represents a submitted TrustyAI resource with management methods."""

    def __init__(self, name: str, namespace: str, kind: str, deployer: KubernetesDeployer):
        """Initialize a submitted resource handle.

        Args:
            name: Name of the resource
            namespace: Namespace where the resource is located
            kind: Kind of the resource (e.g., "LMEvalJob")
            deployer: KubernetesDeployer instance for API calls
        """
        self.name = name
        self.namespace = namespace
        self.kind = kind
        self._deployer = deployer
        self._api_version = "trustyai.opendatahub.io/v1alpha1"
        self._group = "trustyai.opendatahub.io"
        self._version = "v1alpha1"
        self._plural = self._get_plural(kind)

    def _get_plural(self, kind: str) -> str:
        """Get the plural form of a resource kind."""
        if kind == "LMEvalJob":
            return "lmevaljobs"
        # Add other resource types as needed
        return kind.lower() + "s"

    def get_status(self) -> dict[str, Any] | None:
        """Get the status of the submitted resource.

        Returns:
            Resource status dictionary or None if not found/error
        """
        if not self._deployer._initialize_client():
            return None

        try:
            response = self._deployer._custom_objects_api.get_namespaced_custom_object(
                group=self._group,
                version=self._version,
                namespace=self.namespace,
                plural=self._plural,
                name=self.name,
            )
            return response.get("status", {})
        except Exception as e:
            print(f"Error getting {self.kind} status: {str(e)}")
            return None

    def get_full_resource(self) -> dict[str, Any] | None:
        """Get the full resource definition.

        Returns:
            Complete resource dictionary or None if not found/error
        """
        if not self._deployer._initialize_client():
            return None

        try:
            return self._deployer._custom_objects_api.get_namespaced_custom_object(
                group=self._group,
                version=self._version,
                namespace=self.namespace,
                plural=self._plural,
                name=self.name,
            )
        except Exception as e:
            print(f"Error getting {self.kind}: {str(e)}")
            return None

    def delete(self) -> tuple[bool, str | None]:
        """Delete the resource from the cluster.

        Returns:
            Tuple of (success, message)
        """
        if not self._deployer._initialize_client():
            return False, "Failed to initialize Kubernetes client"

        try:
            self._deployer._custom_objects_api.delete_namespaced_custom_object(
                group=self._group,
                version=self._version,
                namespace=self.namespace,
                plural=self._plural,
                name=self.name,
            )
            return (
                True,
                f"Successfully deleted {self.kind} '{self.name}' from namespace '{self.namespace}'",
            )
        except Exception as e:
            return False, f"Error deleting {self.kind}: {str(e)}"

    def get_logs(self) -> str | None:
        """Get logs from the pod associated with the resource.

        Returns:
            Log content as string or None if not found/error
        """
        if not self._deployer._initialize_client():
            return None

        try:
            core_v1_api = self._deployer._client.CoreV1Api(self._deployer._api_client)

            # List pods with label selector for the job
            pods = core_v1_api.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"job-name={self.name}"
            )

            if not pods.items:
                return None

            # Get logs from the first pod
            pod_name = pods.items[0].metadata.name
            logs = core_v1_api.read_namespaced_pod_log(name=pod_name, namespace=self.namespace)

            return logs
        except Exception as e:
            print(f"Error getting {self.kind} logs: {str(e)}")
            return None

    def wait_for_completion(self, timeout_seconds: int = 300) -> tuple[bool, str]:
        """Wait for the resource to complete.

        Args:
            timeout_seconds: Maximum time to wait in seconds

        Returns:
            Tuple of (completed_successfully, final_status)
        """
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            status = self.get_status()

            if status is None:
                return False, f"{self.kind} not found or error getting status"

            # Check for completion conditions
            phase = status.get("phase", "")
            if phase.lower() in ["succeeded", "completed"]:
                return True, f"{self.kind} completed successfully"
            elif phase.lower() in ["failed", "error"]:
                return False, f"{self.kind} failed with status: {phase}"

            # Wait before checking again
            time.sleep(10)

        return False, f"Timeout waiting for {self.kind} completion"

    def is_running(self) -> bool:
        """Check if the resource is currently running.

        Returns:
            True if running, False otherwise
        """
        status = self.get_status()
        if status is None:
            return False

        phase = status.get("phase", "").lower()
        return phase in ["running", "pending", "started"]

    def is_completed(self) -> bool:
        """Check if the resource has completed successfully.

        Returns:
            True if completed successfully, False otherwise
        """
        status = self.get_status()
        if status is None:
            return False

        phase = status.get("phase", "").lower()
        return phase in ["succeeded", "completed"]

    def is_failed(self) -> bool:
        """Check if the resource has failed.

        Returns:
            True if failed, False otherwise
        """
        status = self.get_status()
        if status is None:
            return False

        phase = status.get("phase", "").lower()
        return phase in ["failed", "error"]

    def __str__(self) -> str:
        """String representation of the submitted resource."""
        return f"{self.kind}(name='{self.name}', namespace='{self.namespace}')"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"SubmittedResource(name='{self.name}', namespace='{self.namespace}', kind='{self.kind}')"


class TrustyAIKubernetesClient:
    """Kubernetes client for TrustyAI resources."""

    def __init__(self, kubeconfig: str | None = None, context: str | None = None):
        """Initialize the TrustyAI Kubernetes client.

        Args:
            kubeconfig: Path to kubeconfig file
            context: Kubernetes context to use
        """
        self.deployer = KubernetesDeployer(kubeconfig=kubeconfig, context=context)

    def submit(self, resource: LMEvalJob) -> SubmittedResource | None:
        """Submit a TrustyAI resource to the Kubernetes cluster.

        Args:
            resource: TrustyAI resource to submit (e.g., LMEvalJob)

        Returns:
            SubmittedResource instance for managing the resource, or None if submission failed
        """
        # Convert to KubernetesResource
        k8s_resource = TrustyAIResourceConverter.lmevaljob_to_kubernetes_resource(resource)

        # Deploy the resource
        success, message = self.deployer.deploy_resource(k8s_resource)

        if success:
            return SubmittedResource(
                name=resource.metadata.name,
                namespace=resource.metadata.namespace or "default",
                kind=resource.kind,
                deployer=self.deployer,
            )
        else:
            print(f"Failed to submit {resource.kind}: {message}")
            return None

    def list_resources(
        self, namespace: str = "default", kind: str = "LMEvalJob"
    ) -> list[SubmittedResource]:
        """List all TrustyAI resources of a specific kind in a namespace.

        Args:
            namespace: Namespace to list resources from
            kind: Kind of resource to list

        Returns:
            List of SubmittedResource instances
        """
        if not self.deployer._initialize_client():
            return []

        try:
            plural = SubmittedResource._get_plural(None, kind)
            response = self.deployer._custom_objects_api.list_namespaced_custom_object(
                group="trustyai.opendatahub.io",
                version="v1alpha1",
                namespace=namespace,
                plural=plural,
            )

            resources = []
            for item in response.get("items", []):
                name = item["metadata"]["name"]
                resources.append(
                    SubmittedResource(
                        name=name, namespace=namespace, kind=kind, deployer=self.deployer
                    )
                )

            return resources
        except Exception as e:
            print(f"Error listing {kind} resources: {str(e)}")
            return []

    def get_resource(
        self, name: str, namespace: str = "default", kind: str = "LMEvalJob"
    ) -> SubmittedResource | None:
        """Get a handle to an existing TrustyAI resource.

        Args:
            name: Name of the resource
            namespace: Namespace where the resource is located
            kind: Kind of the resource

        Returns:
            SubmittedResource instance or None if not found
        """
        resource = SubmittedResource(
            name=name, namespace=namespace, kind=kind, deployer=self.deployer
        )

        # Check if the resource exists
        if resource.get_full_resource() is not None:
            return resource
        else:
            return None

    def generate_yaml(self, resource: LMEvalJob) -> str:
        """Generate YAML representation of a TrustyAI resource for kubectl apply.

        Args:
            resource: TrustyAI resource to convert to YAML

        Returns:
            YAML string representation
        """
        return resource.to_yaml()

    def save_yaml_to_file(self, resource: LMEvalJob, filepath: str) -> bool:
        """Save TrustyAI resource as YAML file for later kubectl apply.

        Args:
            resource: TrustyAI resource to save
            filepath: Path where to save the YAML file

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            yaml_content = self.generate_yaml(resource)
            with open(filepath, "w") as f:
                f.write(yaml_content)
            return True
        except Exception as e:
            print(f"Error saving YAML file: {str(e)}")
            return False
