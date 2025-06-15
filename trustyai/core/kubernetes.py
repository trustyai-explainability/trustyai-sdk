"""Kubernetes resource handling for TrustyAI."""

from __future__ import annotations

import abc
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, cast

import yaml

# Kubernetes client will be imported on demand to avoid dependency issues


class KubernetesResource:
    """Generic Kubernetes resource representation."""

    def __init__(
        self,
        api_version: str,
        kind: str,
        metadata: dict[str, Any],
        spec: dict[str, Any],
        status: dict[str, Any] | None = None,
    ):
        """Initialize a Kubernetes resource.

        Args:
            api_version: Kubernetes API version
            kind: Resource kind
            metadata: Resource metadata
            spec: Resource specification
            status: Optional resource status

        """
        self.api_version = api_version
        self.kind = kind
        self.metadata = metadata
        self.spec = spec
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        """Convert the resource to a dictionary representation.

        Returns:
            Dictionary representing the Kubernetes resource

        """
        resource_dict = {
            "apiVersion": self.api_version,
            "kind": self.kind,
            "metadata": self.metadata,
            "spec": self.spec,
        }

        if self.status:
            resource_dict["status"] = self.status

        return resource_dict

    def to_yaml(self) -> str:
        """Convert the resource to YAML format.

        Returns:
            YAML string representation of the resource

        """
        if yaml is None:
            raise ImportError("PyYAML is not installed. Install it with 'pip install pyyaml'.")  # noqa: EM101, TRY003
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KubernetesResource:
        """Create a resource from a dictionary.

        Args:
            data: Dictionary representing a Kubernetes resource

        Returns:
            KubernetesResource instance

        """
        return cls(
            api_version=cast(str, data.get("apiVersion", "")),
            kind=cast(str, data.get("kind", "")),
            metadata=data.get("metadata", {}),
            spec=data.get("spec", {}),
            status=data.get("status"),
        )


class KubernetesResourceConverter(abc.ABC):
    """Abstract base class for converting configs to Kubernetes resources."""

    @abc.abstractmethod
    def convert(self, config: Dict[str, Any]) -> KubernetesResource:
        """Convert a configuration dictionary to a Kubernetes resource.

        Args:
            config: Configuration dictionary

        Returns:
            KubernetesResource instance
        """
        pass

    def convert_to_yaml(self, config: Dict[str, Any]) -> str:
        """Convert a configuration to YAML representation.

        Args:
            config: Configuration dictionary

        Returns:
            YAML string representation
        """
        resource = self.convert(config)
        return resource.to_yaml()

    def convert_multiple(self, configs: List[Dict[str, Any]]) -> List[KubernetesResource]:
        """Convert multiple configurations to Kubernetes resources.

        Args:
            configs: List of configuration dictionaries

        Returns:
            List of KubernetesResource instances
        """
        return [self.convert(config) for config in configs]


class DeploymentConverter(KubernetesResourceConverter):
    """Example implementation for converting to Kubernetes Deployments."""

    def convert(self, config: Dict[str, Any]) -> KubernetesResource:
        """Convert a configuration to a Kubernetes Deployment.

        Args:
            config: Configuration dictionary with keys:
                - name: Deployment name
                - image: Container image
                - replicas: Number of replicas
                - container_port: Container port
                - labels: Labels to apply
                - namespace: Optional Kubernetes namespace

        Returns:
            KubernetesResource for a Deployment
        """
        name = config.get("name", "default-app")
        image = config.get("image", "nginx:latest")
        replicas = config.get("replicas", 1)
        container_port = config.get("container_port", 80)
        labels = config.get("labels", {"app": name})

        metadata = {"name": name, "labels": labels}

        # Add namespace if specified
        if "namespace" in config:
            metadata["namespace"] = config["namespace"]

        spec = {
            "replicas": replicas,
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [
                        {"name": name, "image": image, "ports": [{"containerPort": container_port}]}
                    ]
                },
            },
        }

        return KubernetesResource(
            api_version="apps/v1", kind="Deployment", metadata=metadata, spec=spec
        )


class ServiceConverter(KubernetesResourceConverter):
    """Example implementation for converting to Kubernetes Services."""

    def convert(self, config: Dict[str, Any]) -> KubernetesResource:
        """Convert a configuration to a Kubernetes Service.

        Args:
            config: Configuration dictionary with keys:
                - name: Service name
                - port: Service port
                - target_port: Target port
                - service_type: Service type (ClusterIP, NodePort, LoadBalancer)
                - selector: Pod selector labels
                - namespace: Optional Kubernetes namespace

        Returns:
            KubernetesResource for a Service
        """
        name = config.get("name", "default-service")
        port = config.get("port", 80)
        target_port = config.get("target_port", port)
        service_type = config.get("service_type", "ClusterIP")
        selector = config.get("selector", {"app": name})

        metadata = {"name": name, "labels": config.get("labels", {"service": name})}

        # Add namespace if specified
        if "namespace" in config:
            metadata["namespace"] = config["namespace"]

        spec = {
            "selector": selector,
            "ports": [{"port": port, "targetPort": target_port}],
            "type": service_type,
        }

        # Add optional nodePort for NodePort or LoadBalancer types
        if service_type in ["NodePort", "LoadBalancer"] and "node_port" in config:
            spec["ports"][0]["nodePort"] = config["node_port"]

        return KubernetesResource(api_version="v1", kind="Service", metadata=metadata, spec=spec)


class ConfigMapConverter(KubernetesResourceConverter):
    """Implementation for converting to Kubernetes ConfigMaps."""

    def convert(self, config: Dict[str, Any]) -> KubernetesResource:
        """Convert a configuration to a Kubernetes ConfigMap.

        Args:
            config: Configuration dictionary with keys:
                - name: ConfigMap name
                - data: Dictionary of key-value pairs for the ConfigMap
                - labels: Optional labels to apply
                - namespace: Optional Kubernetes namespace

        Returns:
            KubernetesResource for a ConfigMap
        """
        name = config.get("name", "default-config")
        data = config.get("data", {})

        metadata = {"name": name, "labels": config.get("labels", {"config": name})}

        # Add namespace if specified
        if "namespace" in config:
            metadata["namespace"] = config["namespace"]

        # Create a custom KubernetesResource
        resource = KubernetesResource(
            api_version="v1",
            kind="ConfigMap",
            metadata=metadata,
            spec={},  # ConfigMaps don't use the spec field
        )

        # Create a custom to_dict method
        original_to_dict = resource.to_dict

        def custom_to_dict() -> Dict[str, Any]:
            result = original_to_dict()
            result["data"] = data
            # Remove empty spec
            if not result["spec"]:
                del result["spec"]
            return result

        # Replace the method
        resource.to_dict = custom_to_dict  # type: ignore

        return resource


class KubernetesDeployer:
    """Class for deploying Kubernetes resources to clusters."""

    def __init__(self, kubeconfig: Optional[str] = None, context: Optional[str] = None):
        """Initialize the deployer.

        Args:
            kubeconfig: Path to kubeconfig file (defaults to KUBECONFIG env var or ~/.kube/config)
            context: Kubernetes context to use
        """
        self.kubeconfig = kubeconfig
        self.context = context
        self._client = None
        self._api_client = None
        self._custom_objects_api = None
        
    def _get_kubernetes_client(self) -> Tuple[Any, Any]:
        """Get kubernetes client library when needed, to avoid import-time dependency.
        
        Returns:
            Tuple of (kubernetes module, client module)
            
        Raises:
            ImportError: If kubernetes package is not installed
        """
        try:
            import kubernetes
            from kubernetes import client
            return kubernetes, client
        except ImportError:
            raise ImportError(
                "The kubernetes package is required for deployment. "
                "Install it with: pip install trustyai[eval]"
            )
    
    def _initialize_client(self) -> bool:
        """Initialize the kubernetes client.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            kubernetes, client = self._get_kubernetes_client()
            
            # Configure API client
            if self.kubeconfig:
                kubernetes.config.load_kube_config(
                    config_file=self.kubeconfig, context=self.context
                )
            else:
                try:
                    # Try to load from within cluster first
                    kubernetes.config.load_incluster_config()
                except kubernetes.config.config_exception.ConfigException:
                    # Fall back to user's kubeconfig
                    kubernetes.config.load_kube_config(context=self.context)
            
            self._client = client
            self._api_client = client.ApiClient()
            self._custom_objects_api = client.CustomObjectsApi(self._api_client)
            return True
        except Exception as e:
            print(f"Failed to initialize Kubernetes client: {str(e)}")
            return False

    def deploy_resource(self, resource: KubernetesResource) -> Tuple[bool, Optional[str]]:
        """Deploy a single Kubernetes resource to the cluster.

        Args:
            resource: KubernetesResource to deploy

        Returns:
            Tuple of (success, error_message) where success is True if deployment succeeded
        """
        # Check if it's a special resource type that needs direct API handling
        if resource.kind == "LMEvalJob":
            return self.deploy_lm_eval_job(resource)
        else:
            # Use kubectl for standard resources
            success = self.deploy_yaml(resource.to_yaml())
            return (success, None if success else "Failed to deploy using kubectl")
            
    def _namespace_exists(self, namespace: str) -> bool:
        """Check if a Kubernetes namespace exists.
        
        Args:
            namespace: Name of the namespace to check
            
        Returns:
            True if the namespace exists, False otherwise
        """
        if not self._initialize_client():
            return False
            
        try:
            # Ensure client is properly initialized
            if self._client is None:
                print("[DEBUG] Kubernetes client is not initialized")
                return False
                
            core_v1_api = self._client.CoreV1Api(self._api_client)
            namespaces = core_v1_api.list_namespace()
            for ns in namespaces.items:
                if ns.metadata.name == namespace:
                    return True
            return False
        except Exception as e:
            print(f"[DEBUG] Error checking namespace {namespace}: {str(e)}")
            return False
            
    def deploy_lm_eval_job(self, resource: KubernetesResource) -> Tuple[bool, Optional[str]]:
        """Deploy a LMEvalJob custom resource using the Kubernetes API directly.
        
        Args:
            resource: KubernetesResource for a LMEvalJob
            
        Returns:
            Tuple of (success, error_message)
        """
        # Initialize client if not already done
        if not self._initialize_client():
            print("[DEBUG] Failed to initialize Kubernetes client")
            return False, "Failed to initialize Kubernetes client"
            
        if self._custom_objects_api is None:
            print("[DEBUG] Kubernetes API client is not properly initialized")
            return False, "Kubernetes API client is not properly initialized"
            
        try:
            # Extract relevant info
            group, version = resource.api_version.split("/")
            namespace = resource.metadata.get("namespace", "default")
            plural = "lmevaljobs"  # Lowercase, plural form of the CRD
            
            print(f"[DEBUG] Deploying LMEvalJob to namespace: {namespace}")
            
            # Check if namespace exists
            if not self._namespace_exists(namespace):
                error_msg = f"Namespace '{namespace}' does not exist. Please create it first with: kubectl create namespace {namespace}"
                print(f"[DEBUG] {error_msg}")
                return False, error_msg
            
            print(f"[DEBUG] API Group: {group}, Version: {version}")
            print(f"[DEBUG] Resource metadata: {resource.metadata}")
            
            # Try to create the custom resource
            response = self._custom_objects_api.create_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                body=resource.to_dict()
            )
            
            # Check if creation was successful
            if response and "metadata" in response:
                name = response["metadata"].get("name", "unknown")
                print(f"[DEBUG] Successfully created LMEvalJob '{name}' in namespace '{namespace}'")
                return True, f"Successfully created LMEvalJob '{name}' in namespace '{namespace}'"
            else:
                print(f"[DEBUG] Received unexpected response: {response}")
                return False, "Received unexpected response from Kubernetes API"
                
        except Exception as e:
            error_message = str(e)
            print(f"[DEBUG] Exception when deploying LMEvalJob: {error_message}")
            import traceback
            traceback.print_exc()
            
            # Create a more detailed error message
            if "404" in error_message or "not found" in error_message.lower():
                detailed_msg = f"Failed to deploy LMEvalJob (CRD not found): {error_message}. Ensure the TrustyAI operator is installed."
            elif "403" in error_message or "forbidden" in error_message.lower():
                detailed_msg = f"Failed to deploy LMEvalJob (permission denied): {error_message}. Check RBAC permissions."
            elif "namespace" in error_message.lower() and "not found" in error_message.lower():
                detailed_msg = f"Failed to deploy LMEvalJob: {error_message}. The namespace may not exist."
            else:
                detailed_msg = f"Failed to deploy LMEvalJob: {error_message}"
                
            return False, detailed_msg

    def deploy_yaml(self, yaml_content: str) -> bool:
        """Deploy YAML content to the Kubernetes cluster using kubectl.

        Args:
            yaml_content: YAML content to deploy

        Returns:
            True if deployment succeeded, False otherwise
        """
        try:
            # Create a temporary file to hold the YAML
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                tmp.write(yaml_content)
                tmp_name = tmp.name

            # Build kubectl command
            cmd = ["kubectl"]

            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])

            if self.context:
                cmd.extend(["--context", self.context])

            cmd.extend(["apply", "-f", tmp_name])

            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            # Clean up the temporary file
            os.unlink(tmp_name)

            # Check result
            if result.returncode != 0:
                print(f"Error deploying resource: {result.stderr}")
                return False

            return True

        except Exception as e:
            print(f"Failed to deploy resource: {str(e)}")
            return False

    def deploy_resources(self, resources: List[KubernetesResource]) -> bool:
        """Deploy multiple Kubernetes resources to the cluster.

        Args:
            resources: List of KubernetesResource instances to deploy

        Returns:
            True if all deployments succeeded, False otherwise
        """
        if not resources:
            print("[DEBUG] No resources to deploy")
            return False
            
        print(f"[DEBUG] Attempting to deploy {len(resources)} resources")
        all_succeeded = True
        
        # LMEvalJob resources need special handling
        for i, resource in enumerate(resources):
            print(f"[DEBUG] Deploying resource {i+1}/{len(resources)}: {resource.kind}")
            success, error = self.deploy_resource(resource)
            if not success:
                print(f"[DEBUG] Failed to deploy {resource.kind}: {error}")
                all_succeeded = False
            else:
                print(f"[DEBUG] Successfully deployed {resource.kind}")
        
        return all_succeeded
