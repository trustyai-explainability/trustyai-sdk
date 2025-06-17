"""Kubernetes LMEval provider for TrustyAI."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

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

    def convert(self, config: dict[str, Any]) -> KubernetesResource:
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

        # Create metadata with a UUID-based name that we control
        job_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
        job_name = f"evaljob-{job_id}"
        metadata = {
            "name": job_name
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
    - wait_for_completion: Set to True to wait for job completion and return results (default: False)
    - timeout: Maximum time to wait for job completion in seconds (default: 300)
    - kubeconfig: Path to kubeconfig file (optional)
    - context: Kubernetes context to use (optional)

    Note: Requires the 'eval' extra to be installed:
    pip install trustyai[eval]
    """

    @property
    def supported_deployment_modes(self) -> list[DeploymentMode]:
        """Return the deployment modes supported by this provider."""
        return [DeploymentMode.KUBERNETES]

    async def evaluate(self, *args: Any, **kwargs) -> dict[str, Any]:  # type: ignore
        """Evaluate using Kubernetes deployment with async/await support.

        This method can be used both synchronously and asynchronously:

        Sync usage:
            results = provider.evaluate(config)

        Async usage:
            results = await provider.evaluate(config)

        Args:
            *args: Either (config) or (model_or_id, dataset, [metrics])
            **kwargs: Additional parameters

        Returns:
            Dictionary with evaluation results or deployment information

        Raises:
            ValueError: If configuration is invalid
        """
        # Parse arguments to get the configuration
        config = self._parse_args_to_config(*args, **kwargs)

        # Ensure we're using Kubernetes mode
        if config.deployment_mode != DeploymentMode.KUBERNETES:
            config.deployment_mode = DeploymentMode.KUBERNETES

        return await self._evaluate_kubernetes_async(config)

    async def _evaluate_kubernetes_async(self, config: EvaluationProviderConfig) -> dict[str, Any]:
        """Asynchronous Kubernetes evaluation implementation.

        Args:
            config: Evaluation configuration

        Returns:
            Dictionary with evaluation results or deployment information
        """
        # Debug logging
        print(f"[DEBUG - _evaluate_kubernetes_async] Config keys: {dir(config)}")
        print(f"[DEBUG - _evaluate_kubernetes_async] Config namespace: {config.get_param('namespace')}")

        # Generate Kubernetes resources and track the job name
        resources = self.get_kubernetes_resources(config.__dict__)
        
        # Extract the job name from the generated resources
        job_name = None
        for resource in resources:
            if resource.kind == "LMEvalJob":
                job_name = resource.metadata.get("name")
                break

        # Check if we should deploy the resources
        should_deploy = config.get_param("deploy", False)
        wait_for_completion = config.get_param("wait_for_completion", False)
        timeout = config.get_param("timeout", 300)  # 5 minutes default

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
                else:
                    # Log successful deployment
                    if resource.kind == "LMEvalJob":
                        print(f"[DEBUG] Successfully deployed LMEvalJob: {job_name}")

            if success:
                deployment_status = "deployed"
                deployment_message = "Resources successfully deployed to Kubernetes cluster"

                # Show the deployed LMEvalJob details
                # Get namespace from config rather than CR metadata
                namespace = self._get_namespace_from_config(config)
                deployment_message += f"\nCreated LMEvalJob: {job_name}"
                deployment_message += f"\nNamespace: {namespace}"

                # If we should wait for completion, retrieve results
                if wait_for_completion and job_name:
                    print(f"[DEBUG] Waiting for completion of job: {job_name}")
                    try:
                        job_results = await self._wait_for_job_completion_async(
                            deployer, job_name, namespace, timeout
                        )
                        if job_results:
                            # Return results in the same format as local evaluation
                            return job_results
                        else:
                            deployment_message += "\nJob completed but no results found"
                    except Exception as e:
                        deployment_message += f"\nError retrieving results: {str(e)}"
                        error_info = str(e)

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



    async def _wait_for_job_completion_async(
        self, deployer: KubernetesDeployer, job_name: str, namespace: str, timeout: int
    ) -> dict[str, Any] | None:
        """Asynchronously wait for a LMEvalJob to complete and retrieve its results.

        Args:
            deployer: Kubernetes deployer instance
            job_name: Name of the LMEvalJob
            namespace: Kubernetes namespace
            timeout: Maximum time to wait in seconds

        Returns:
            Dictionary with evaluation results or None if not available
        """
        start_time = time.time()
        check_interval = 10  # Check every 10 seconds

        print(f"⏳ Waiting for job {job_name} to complete...")

        while time.time() - start_time < timeout:
            try:
                # Get the job status
                job_status = self._get_job_status(deployer, job_name, namespace)
                
                if job_status:
                    state = job_status.get("state", "").lower()
                    print(f"[DEBUG] Job {job_name} state: {state}")
                    
                    if state == "complete":
                        # Job completed successfully, retrieve results
                        results_json = job_status.get("results")
                        if results_json:
                            try:
                                results = json.loads(results_json)
                                print(f"✅ Job {job_name} completed successfully!")
                                return results
                            except json.JSONDecodeError as e:
                                print(f"[DEBUG] Failed to parse results JSON: {e}")
                                return None
                        else:
                            print(f"[DEBUG] Job completed but no results found")
                            return None
                    
                    elif state in ["failed", "error"]:
                        # Job failed
                        message = job_status.get("message", "Job failed")
                        print(f"❌ Job {job_name} failed: {message}")
                        return None
                    
                    else:
                        # Job still running
                        print(f"🔄 Job {job_name} still running (state: {state})...")
                
                # Use asyncio.sleep instead of time.sleep for async compatibility
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                print(f"[DEBUG] Error checking job status: {e}")
                await asyncio.sleep(check_interval)

        print(f"⏰ Timeout waiting for job {job_name} to complete")
        return None

    def _get_namespace_from_config(self, config: EvaluationProviderConfig) -> str:
        """Extract namespace from configuration with fallback to default.
        
        Args:
            config: Evaluation configuration
            
        Returns:
            Namespace string
        """
        # Get namespace directly from additional_params which contains CLI parameters
        namespace = None
        if hasattr(config, 'get_param'):
            namespace = config.get_param("namespace")
        
        # If not found, try the config dict directly
        if not namespace and hasattr(config, '__dict__'):
            config_dict = config.__dict__
            if "additional_params" in config_dict and isinstance(config_dict["additional_params"], dict):
                namespace = config_dict["additional_params"].get("namespace")
            
            # If not in additional_params, try the top-level config
            if not namespace:
                namespace = config_dict.get("namespace")
        
        # If still not found, use default
        if not namespace:
            namespace = "trustyai"
            
        return namespace

    def _get_job_status(
        self, deployer: KubernetesDeployer, job_name: str, namespace: str
    ) -> dict[str, Any] | None:
        """Get the status of a LMEvalJob.

        Args:
            deployer: Kubernetes deployer instance
            job_name: Name of the LMEvalJob
            namespace: Kubernetes namespace

        Returns:
            Dictionary with job status or None if not found
        """
        try:
            # Initialize the client if not already done
            if not deployer._initialize_client():
                print("[DEBUG] Failed to initialize Kubernetes client")
                return None
            
            # Access the custom objects API from the deployer
            custom_objects_api = deployer._custom_objects_api
            if not custom_objects_api:
                print("[DEBUG] CustomObjectsApi not available")
                return None

            # Get the LMEvalJob custom resource
            response = custom_objects_api.get_namespaced_custom_object(
                group="trustyai.opendatahub.io",
                version="v1alpha1",
                namespace=namespace,
                plural="lmevaljobs",
                name=job_name
            )

            # Extract status information
            status = response.get("status", {})
            return status

        except Exception as e:
            print(f"[DEBUG] Error getting job status: {e}")
            return None

    def get_kubernetes_resources(self, config: dict[str, Any]) -> list[KubernetesResource]:
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


def get_lmeval_job_results(job_name: str, namespace: str = "test") -> dict[str, Any] | None:
    """Retrieve results from a completed LMEvalJob.
    
    Args:
        job_name: Name of the LMEvalJob (e.g., "evaljob-qjhhz")
        namespace: Kubernetes namespace
        
    Returns:
        Dictionary with evaluation results or None if not available
    """
    try:
        from trustyai.core.kubernetes import KubernetesDeployer
        
        deployer = KubernetesDeployer()
        client, custom_objects_api = deployer._get_kubernetes_client()
        
        if not client or not custom_objects_api:
            print("❌ Failed to get Kubernetes client")
            return None
            
        # Get the LMEvalJob custom resource
        response = custom_objects_api.get_namespaced_custom_object(
            group="trustyai.opendatahub.io",
            version="v1alpha1",
            namespace=namespace,
            plural="lmevaljobs",
            name=job_name
        )
        
        # Extract status information
        status = response.get("status", {})
        state = status.get("state", "").lower()
        
        print(f"📊 Job {job_name} status: {state}")
        
        if state == "complete":
            results_json = status.get("results")
            if results_json:
                results = json.loads(results_json)
                print("✅ Successfully retrieved results!")
                return results
            else:
                print("❌ Job completed but no results found")
                return None
        else:
            print(f"⏳ Job not completed yet (state: {state})")
            return None
            
    except Exception as e:
        print(f"❌ Error retrieving job results: {e}")
        return None
