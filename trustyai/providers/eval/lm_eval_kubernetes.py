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
from trustyai.core.validators.kubernetes import TrustyAIOperatorValidator
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
        metadata = {"name": job_name}

        # Add namespace if specified - IMPORTANT: This must be set for proper deployment
        if "namespace" in config:
            metadata["namespace"] = config["namespace"]
            print(f"[DEBUG] Setting namespace in LMEvalJob resource: {config['namespace']}")

        # Build the CR spec
        spec = {
            "model": model,
            "modelArgs": model_args,
            "taskList": {"taskNames": task_names},
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
            spec=spec,
        )


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

    def __init__(self) -> None:
        """Initialize the Kubernetes LM Eval provider."""
        super().__init__()

        # Add TrustyAI operator validator
        # Get the k8s_client and config later when they're available
        self._operator_validator: TrustyAIOperatorValidator | None = None

    def _setup_validators(self, k8s_client=None, config: dict[str, Any] | None = None) -> None:
        """Setup validators with the provided k8s_client and config.

        Args:
            k8s_client: Optional Kubernetes client instance
            config: Configuration dictionary
        """
        # Clear existing validators
        self._validators.clear()

        # Add TrustyAI operator validator
        if config is None:
            config = {}

        # Get namespace from config or use default
        namespace = config.get("namespace", "trustyai-system")

        self._operator_validator = TrustyAIOperatorValidator(
            implementation="trustyai-operator",
            config=config,
            k8s_client=k8s_client,
            namespace=namespace,
        )
        self.add_validator(self._operator_validator)

    @classmethod
    def get_provider_name(cls) -> str:
        """Return the name of this provider."""
        return "lm-eval-kubernetes"

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
        """Asynchronous Kubernetes evaluation implementation using TrustyAI Kubernetes client.

        Args:
            config: Evaluation configuration

        Returns:
            Dictionary with evaluation results or deployment information
        """
        # Debug logging
        print(f"[DEBUG - _evaluate_kubernetes_async] Config keys: {dir(config)}")
        print(
            f"[DEBUG - _evaluate_kubernetes_async] Config namespace: {config.get_param('namespace')}"
        )

        # Setup validators with config
        config_dict = config.__dict__.copy()
        if hasattr(config, "additional_params"):
            config_dict.update(config.additional_params)

        # Get k8s_client if available for validation
        k8s_client = None
        try:
            from trustyai.core.kubernetes import kubernetes_client

            if kubernetes_client.is_initialized:
                k8s_client = kubernetes_client.client
            else:
                # Try to initialize the client
                if kubernetes_client.initialize():
                    k8s_client = kubernetes_client.client
        except Exception as e:
            print(f"[DEBUG] Failed to get Kubernetes client: {e}")
            pass

        self._setup_validators(k8s_client=k8s_client, config=config_dict)

        # Run validation before deployment
        print("🔍 Running Kubernetes environment validation...")
        validation_passed = self.validate_and_check()

        if not validation_passed:
            print("❌ Validation failed. Please fix the issues above before proceeding.")
            return {
                "status": "validation_failed",
                "message": "Kubernetes environment validation failed",
                "provider": self.get_provider_name(),
                "deployment_mode": DeploymentMode.KUBERNETES.value,
                "error": "Environment validation failed",
            }

        # Create LMEvalJob using the new dataclass structure
        from trustyai.core.lmevaljob import LMEvalJob, LMEvalJobSpec, Metadata, ModelArg, TaskList
        import uuid

        # Get configuration parameters
        model = config.get_param("model", "")
        tasks = config.get_param("tasks", [])
        limit = config.get_param("limit")
        namespace = self._get_namespace_from_config(config)

        # Create job metadata with unique name
        job_id = str(uuid.uuid4())[:8]
        job_name = f"evaljob-{job_id}"

        metadata = Metadata(name=job_name, namespace=namespace)

        # Create model arguments
        model_args = [ModelArg(name="pretrained", value=model)]

        # Create task list
        task_list = TaskList(taskNames=tasks)

        # Create job spec
        spec = LMEvalJobSpec(
            model="hf",
            modelArgs=model_args,
            taskList=task_list,
            logSamples=True,
            allowOnline=True,
            allowCodeExecution=True,
        )

        # Add limit if specified
        if limit is not None:
            spec.limit = str(limit)
            print(f"[DEBUG] Setting limit as string: {spec.limit}")

        # Create the LMEvalJob
        lm_eval_job = LMEvalJob(metadata=metadata, spec=spec)

        # Check if we should deploy the resources
        should_deploy = config.get_param("deploy", False)
        wait_for_completion = config.get_param("wait_for_completion", False)
        timeout = config.get_param("timeout", 300)  # 5 minutes default

        # Get optional kubeconfig path and context
        kubeconfig = config.get_param("kubeconfig", None)
        context = config.get_param("context", None)

        # Initialize TrustyAI Kubernetes client
        from trustyai.core.trustyai_kubernetes_client import TrustyAIKubernetesClient

        trustyai_client = TrustyAIKubernetesClient(kubeconfig=kubeconfig, context=context)

        # Generate YAML for display
        yaml_content = trustyai_client.generate_yaml(lm_eval_job)

        # If deployment is requested, submit the job
        deployment_status = "prepared"
        deployment_message = "LMEvalJob ready for deployment"
        error_info = None
        submitted_resource = None

        if should_deploy:
            print("🚀 Deploying LMEvalJob using TrustyAI Kubernetes client...")

            # Submit the LMEvalJob
            submitted_resource = trustyai_client.submit(lm_eval_job)

            if submitted_resource:
                deployment_status = "deployed"
                deployment_message = f"LMEvalJob successfully deployed to Kubernetes cluster"
                deployment_message += f"\nCreated LMEvalJob: {job_name}"
                deployment_message += f"\nNamespace: {namespace}"

                print(f"[DEBUG] Successfully deployed LMEvalJob: {job_name}")

                # If we should wait for completion, retrieve results
                if wait_for_completion and submitted_resource:
                    print(f"[DEBUG] Waiting for completion of job: {job_name}")
                    try:
                        job_results = await self._wait_for_job_completion_with_trustyai_client(
                            submitted_resource, timeout
                        )
                        if job_results:
                            # Return results in the same format as local evaluation
                            return job_results
                        deployment_message += "\nJob completed but no results found"
                    except Exception as e:
                        deployment_message += f"\nError retrieving results: {str(e)}"
                        error_info = str(e)
            else:
                deployment_status = "failed"
                deployment_message = "Failed to deploy LMEvalJob to Kubernetes cluster"
                error_info = "The deployment failed. Make sure the TrustyAI Operator is installed in your cluster."

        # Return deployment info
        result = {
            "status": deployment_status,
            "message": deployment_message,
            "provider": self.get_provider_name(),
            "deployment_mode": DeploymentMode.KUBERNETES.value,
            "job_name": job_name,
            "namespace": namespace,
            "yaml": yaml_content,
        }

        # Add error info if present
        if error_info:
            result["error"] = error_info

        # Add submitted resource info if available
        if submitted_resource:
            result["submitted_resource"] = {
                "name": submitted_resource.name,
                "namespace": submitted_resource.namespace,
                "kind": submitted_resource.kind,
            }

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
                            print("[DEBUG] Job completed but no results found")
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

    async def _wait_for_job_completion_with_trustyai_client(
        self, submitted_resource, timeout: int
    ) -> dict[str, Any] | None:
        """Wait for a LMEvalJob to complete using TrustyAI client and retrieve its results.

        Args:
            submitted_resource: SubmittedResource instance from TrustyAI client
            timeout: Maximum time to wait in seconds

        Returns:
            Dictionary with evaluation results or None if not available
        """
        import asyncio
        import time
        import json

        start_time = time.time()
        check_interval = 10  # Check every 10 seconds

        print(f"⏳ Waiting for job {submitted_resource.name} to complete...")

        while time.time() - start_time < timeout:
            try:
                # Get the job status using the submitted resource
                job_status = submitted_resource.get_status()

                if job_status:
                    state = job_status.get("state", "").lower()
                    print(f"[DEBUG] Job {submitted_resource.name} state: {state}")

                    if state == "complete":
                        # Job completed successfully, retrieve results
                        results_json = job_status.get("results")
                        if results_json:
                            try:
                                results = json.loads(results_json)
                                print(f"✅ Job {submitted_resource.name} completed successfully!")
                                return results
                            except json.JSONDecodeError as e:
                                print(f"[DEBUG] Failed to parse results JSON: {e}")
                                return None
                        else:
                            print("[DEBUG] Job completed but no results found")
                            return None

                    elif state in ["failed", "error"]:
                        # Job failed
                        message = job_status.get("message", "Job failed")
                        print(f"❌ Job {submitted_resource.name} failed: {message}")
                        return None

                    else:
                        # Job still running
                        print(f"🔄 Job {submitted_resource.name} still running (state: {state})...")

                # Use asyncio.sleep instead of time.sleep for async compatibility
                await asyncio.sleep(check_interval)

            except Exception as e:
                print(f"[DEBUG] Error checking job status: {e}")
                await asyncio.sleep(check_interval)

        print(f"⏰ Timeout waiting for job {submitted_resource.name} to complete")
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
        if hasattr(config, "get_param"):
            namespace = config.get_param("namespace")

        # If not found, try the config dict directly
        if not namespace and hasattr(config, "__dict__"):
            config_dict = config.__dict__
            if "additional_params" in config_dict and isinstance(
                config_dict["additional_params"], dict
            ):
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
                name=job_name,
            )

            # Extract status information
            return response.get("status", {})

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
        limit = config.get("limit")

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
            "model_args": [{"name": "pretrained", "value": model}],
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
        from trustyai.core.kubernetes import kubernetes_client

        # Initialize client if not already done
        if not kubernetes_client.is_initialized:
            if not kubernetes_client.initialize():
                print("❌ Failed to initialize Kubernetes client")
                return None

        # Get the LMEvalJob custom resource
        response = kubernetes_client.custom_objects_api.get_namespaced_custom_object(
            group="trustyai.opendatahub.io",
            version="v1alpha1",
            namespace=namespace,
            plural="lmevaljobs",
            name=job_name,
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
            print("❌ Job completed but no results found")
            return None
        print(f"⏳ Job not completed yet (state: {state})")
        return None

    except Exception as e:
        print(f"❌ Error retrieving job results: {e}")
        return None
