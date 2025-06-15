"""Kubernetes LM Evaluation Harness provider for TrustyAI."""

from __future__ import annotations

from typing import Any, Dict, List

from trustyai.core import DeploymentMode
from trustyai.core.eval import EvaluationProviderConfig
from trustyai.core.kubernetes import (
    KubernetesDeployer,
    KubernetesResource,
    KubernetesResourceConverter,
)
from trustyai.core.registry import provider_registry
from trustyai.providers.eval.lm_eval_base import LMEvalProviderBase


class LMEvalJobConverter(KubernetesResourceConverter):
    """Converter for TrustyAI LMEvalJob Custom Resources."""

    def convert(self, config: Dict[str, Any]) -> KubernetesResource:
        """Convert a configuration to a TrustyAI LMEvalJob custom resource.

        Args:
            config: Configuration dictionary with keys:
                - model: Model type (e.g. 'hf')
                - model_args: List of model arguments as name-value pairs
                - task_names: List of task names to run
                - log_samples: Whether to log samples
                - allow_online: Whether to allow online access
                - allow_code_execution: Whether to allow code execution
                - limit: Optional maximum number of samples to evaluate
                - namespace: Optional Kubernetes namespace

        Returns:
            KubernetesResource for a LMEvalJob custom resource
        """
        # Get required fields
        model = config.get("model", "hf")
        model_args = config.get("model_args", [])
        task_names = config.get("task_names", [])
        log_samples = config.get("log_samples", True)
        allow_online = config.get("allow_online", True)
        allow_code_execution = config.get("allow_code_execution", True)
        
        # Create metadata - don't specify a name to let Kubernetes generate it
        # The generateName field will create a unique name with this prefix
        metadata = {
            "generateName": "evaljob-"
        }
        
        # Add namespace if specified - IMPORTANT: This must be set for proper deployment
        if "namespace" in config:
            metadata["namespace"] = config["namespace"]
            print(f"[DEBUG] Setting namespace in LMEvalJob resource: {config['namespace']}")
        
        # Build the CR spec
        spec = {
            "model": model,
            "modelArgs": model_args,
            "taskList": {
                "taskNames": task_names
            },
            "logSamples": log_samples,
            "allowOnline": allow_online,
            "allowCodeExecution": allow_code_execution,
        }
        
        # Add limit if specified - MUST be a string according to the CRD
        if "limit" in config and config["limit"] is not None:
            # Convert the limit to a string as required by the CRD
            spec["limit"] = str(config["limit"])
            print(f"[DEBUG] Setting limit as string: {spec['limit']}")
        
        return KubernetesResource(
            api_version="trustyai.opendatahub.io/v1alpha1",
            kind="LMEvalJob",
            metadata=metadata,
            spec=spec
        )


@provider_registry.eval.register_kubernetes("lm-eval")
class KubernetesLMEvalProvider(LMEvalProviderBase):
    """Kubernetes implementation of LM Evaluation Harness for TrustyAI.
    
    This provider generates Kubernetes resources for running evaluations
    in a Kubernetes cluster using the TrustyAI operator.
    
    For Kubernetes deployment, the provider can generate LMEvalJob CustomResource
    definitions that match the TrustyAI operator's format:

    ```yaml
    apiVersion: trustyai.opendatahub.io/v1alpha1
    kind: LMEvalJob
    metadata:
      generateName: evaljob-  # Kubernetes will generate a unique name
    spec:
      model: hf
      modelArgs:
        - name: pretrained
          value: google/flan-t5-base
      taskList:
        taskNames:
          - "qnlieu"
      logSamples: true
      allowOnline: true
      allowCodeExecution: true
      limit: 10  # Optional, omitted if not specified
    ```

    Deployment parameters:
    - deploy: Set to True to deploy resources to the cluster (default: False)
    - kubeconfig: Path to kubeconfig file (optional)
    - context: Kubernetes context to use (optional)

    Note: Requires the 'eval' extra to be installed:
    pip install trustyai[eval]
    """
    
    def get_supported_deployment_modes(self) -> list[DeploymentMode]:
        """Return the deployment modes supported by this provider."""
        return [DeploymentMode.KUBERNETES]
    
    def evaluate(self, *args: Any, **kwargs) -> Dict[str, Any]:  # type: ignore
        """Prepare Kubernetes resources for evaluation and optionally deploy them.

        Args:
            *args: Either (config) or (model_or_id, dataset, [metrics])
            **kwargs: Additional parameters

        Returns:
            Dictionary with deployment information

        Raises:
            ValueError: If configuration is invalid
        """
        # Parse arguments to get the configuration
        config = self._parse_args_to_config(*args, **kwargs)
        
        # Ensure we're using Kubernetes mode
        if config.deployment_mode != DeploymentMode.KUBERNETES:
            config.deployment_mode = DeploymentMode.KUBERNETES
            
        return self._evaluate_kubernetes(config)
        
    def _evaluate_kubernetes(self, config: EvaluationProviderConfig) -> dict[str, Any]:
        """Prepare Kubernetes resources for evaluation and optionally deploy them.

        Args:
            config: Evaluation configuration

        Returns:
            Dictionary with deployment information

        Raises:
            ValueError: If configuration is invalid
        """
        # Debug logging
        print(f"[DEBUG - _evaluate_kubernetes] Config keys: {dir(config)}")
        print(f"[DEBUG - _evaluate_kubernetes] Config namespace: {config.get_param('namespace')}")
        
        # Generate Kubernetes resources
        resources = self.get_kubernetes_resources(config.__dict__)
        
        # Check if we should deploy the resources
        should_deploy = config.get_param("deploy", False)
        
        # If deployment is requested, deploy the resources
        deployment_status = "prepared"
        deployment_message = "Resources ready for deployment"
        error_info = None
        
        # Generate YAML for display
        yaml_resources = []
        for resource in resources:
            yaml_resources.append(resource.to_yaml())
        
        combined_yaml = "\n---\n".join(yaml_resources)
        
        if should_deploy:
            # Get optional kubeconfig path and context
            kubeconfig = config.get_param("kubeconfig", None)
            context = config.get_param("context", None)
            
            # Create deployer
            deployer = KubernetesDeployer(kubeconfig=kubeconfig, context=context)
            
            # Track detailed error messages from resource deployment
            deployment_errors = []
            
            # Deploy resources individually to capture detailed error messages
            success = True
            for i, resource in enumerate(resources):
                res_success, error_msg = deployer.deploy_resource(resource)
                if not res_success:
                    success = False
                    if error_msg:
                        deployment_errors.append(error_msg)
            
            if success:
                deployment_status = "deployed"
                deployment_message = "Resources successfully deployed to Kubernetes cluster"
                
                # Look for the name of the deployed LMEvalJob
                for resource in resources:
                    if resource.kind == "LMEvalJob":
                        namespace = resource.metadata.get("namespace", "default")
                        if "generateName" in resource.metadata:
                            deployment_message += f"\nCreated resource with prefix: {resource.metadata['generateName']}"
                        elif "name" in resource.metadata:
                            deployment_message += f"\nCreated resource with name: {resource.metadata['name']}"
                        deployment_message += f"\nNamespace: {namespace}"
                        break
            else:
                deployment_status = "failed"
                deployment_message = "Failed to deploy resources to Kubernetes cluster"
                
                # Provide detailed error message if available
                if deployment_errors:
                    error_info = deployment_errors[0]  # Use the first error message
                else:
                    error_info = "The deployment failed. Make sure the TrustyAI Operator is installed in your cluster."

        # Return deployment info
        result = {
            "status": deployment_status,
            "message": deployment_message,
            "provider": self.get_provider_name(),
            "deployment_mode": DeploymentMode.KUBERNETES.value,
            "resources_count": len(resources),
            "resources": [r.to_dict() for r in resources],
            "yaml": combined_yaml,  # Add combined YAML for easy display
        }
        
        # Add error info if present
        if error_info:
            result["error"] = error_info
            
        return result

    def get_kubernetes_resources(self, config: Dict[str, Any]) -> List[KubernetesResource]:
        """Get Kubernetes resources needed by this provider.

        Args:
            config: Provider configuration

        Returns:
            List of KubernetesResource objects containing just the LMEvalJob

        """
        resources = []

        # Extract relevant configuration
        model = config.get("model", "")
        tasks = config.get("tasks", [])
        limit = config.get("limit", None)

        # Get namespace directly from additional_params which contains CLI parameters
        namespace = None
        if "additional_params" in config and isinstance(config["additional_params"], dict):
            namespace = config["additional_params"].get("namespace")
        
        # If not in additional_params, try the top-level config
        if not namespace:
            namespace = config.get("namespace")
            
        # If still not found, use default
        if not namespace:
            namespace = "trustyai"
            
        print(f"[DEBUG] Using namespace for CR: {namespace}")

        # Create LMEvalJob custom resource
        lm_eval_job_config = {
            "model": "hf",
            "model_args": [
                {
                    "name": "pretrained",
                    "value": model
                }
            ],
            "task_names": tasks,
            "log_samples": True,
            "allow_online": True,
            "allow_code_execution": True,
            "namespace": namespace,  # Make sure to include the namespace here
        }
        
        # Add limit if specified - must be a string for the CRD
        if limit is not None:
            lm_eval_job_config["limit"] = str(limit)  # Convert to string for CRD compatibility
            print(f"[DEBUG] Setting limit in config as string: {lm_eval_job_config['limit']}")

        lm_eval_job_converter = LMEvalJobConverter()
        lm_eval_job = lm_eval_job_converter.convert(lm_eval_job_config)
        
        # Double check that namespace is set in the resource metadata
        if "namespace" not in lm_eval_job.metadata:
            print(f"[DEBUG] Namespace not found in resource metadata, setting it to: {namespace}")
            lm_eval_job.metadata["namespace"] = namespace

        # Only add the LMEvalJob resource
        resources.append(lm_eval_job)

        return resources 