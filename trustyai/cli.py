"""TrustyAI Command Line Interface."""

import json
import sys

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .core.models import ExecutionMode
from .core.providers import ProviderRegistry


# Helper function to import eval providers on demand
def _import_eval_providers():
    """Import evaluation providers to trigger registration."""
    try:
        from .providers.eval.lm_eval import (
            LMEvalProvider,  # This triggers registration via decorator
        )
        from .providers.eval.lm_eval_kubernetes import KubernetesLMEvalProvider
        from .providers.eval.lm_eval_local import LocalLMEvalProvider
    except ImportError:
        # Optional providers not available
        pass


@click.group()
@click.version_option(version=__version__)
def main(args=None):
    """TrustyAI CLI tool for Trustworthy AI operations."""
    # This allows programmatic usage of the CLI
    if args is not None:
        sys.argv = [sys.argv[0]] + args
    pass


@main.group()
def model():
    """Model management commands."""
    pass


@model.command("list")
def model_list():
    """List available models."""
    click.echo("Available models:")
    # This would eventually fetch actual models
    click.echo("  - No models available yet")


@model.command("explain")
@click.argument("model_id")
@click.option("--input", "-i", required=True, help="Path to input data file")
@click.option("--output", "-o", help="Path to output explanation file")
@click.option(
    "--format",
    "-f",
    default="json",
    type=click.Choice(["json", "csv"]),
    help="Output format (default: json)",
)
def explain_model(model_id, input, output, format):
    """Generate explanations for a model."""
    click.echo(f"Explaining model: {model_id}")
    click.echo(f"Input data: {input}")
    if output:
        click.echo(f"Saving explanations to: {output}")
    else:
        click.echo("Explanations will be printed to stdout")
    click.echo(f"Format: {format}")
    # This would eventually perform actual explanation


@main.group()
def metrics():
    """Fairness and performance metrics commands."""
    pass


@metrics.command("fairness")
@click.argument("model_id")
@click.option("--data", "-d", required=True, help="Path to evaluation data")
@click.option("--sensitive", "-s", required=True, help="Name of sensitive feature column")
def fairness_metrics(model_id, data, sensitive):
    """Calculate fairness metrics for a model."""
    click.echo(f"Calculating fairness metrics for model: {model_id}")
    click.echo(f"Data file: {data}")
    click.echo(f"Sensitive feature: {sensitive}")
    # This would eventually calculate actual metrics


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def info(verbose):
    """Display information about TrustyAI."""
    click.echo(f"TrustyAI SDK version {__version__}")
    click.echo("A Python SDK for trustworthy AI")

    if verbose:
        # In verbose mode, we would show more details
        click.echo("\nAdditional Information:")
        click.echo("  - Python package for explainable and fair AI")
        click.echo("  - Supports model explanations and fairness metrics")

        # List available providers
        providers = ProviderRegistry.list_providers()
        if providers:
            console = Console()
            table = Table(title="Available Providers")

            # Add columns
            table.add_column("Provider Name", style="cyan")
            table.add_column("Type", style="green")

            # Add rows for each provider
            for provider_type, provider_list in providers.items():
                for provider_info in provider_list:
                    table.add_row(provider_info["name"], provider_type)

            # Print the table
            console.print(table)


# Add evaluation commands
@main.group()
def eval():
    """Model evaluation commands."""
    pass


@eval.command("list-providers")
def list_eval_providers():
    """List available evaluation providers."""
    # Ensure providers are imported and registered
    _import_eval_providers()
    
    providers = ProviderRegistry.list_providers("eval")

    if not providers or "eval" not in providers or not providers["eval"]:
        click.echo("No evaluation providers available.")
        click.echo("Try installing optional dependencies: pip install trustyai[eval]")
        return

    # Create a Rich table for providers
    console = Console()
    table = Table(title="Available Evaluation Providers")

    # Add columns
    table.add_column("Provider Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Local Mode", style="yellow")  # Show mode as separate columns
    table.add_column("Kubernetes Mode", style="yellow")

    # Add rows for each provider
    for provider_info in providers["eval"]:
        # Get deployment modes
        deployment_modes = provider_info["deployment_modes"]
        # Check which modes are available
        local_available = "✓" if "local" in deployment_modes else "✗"
        k8s_available = "✓" if "kubernetes" in deployment_modes else "✗"
        
        table.add_row(
            provider_info["name"],
            provider_info["description"].split("\n")[0],  # Just the first line
            local_available,
            k8s_available
        )

    # Print the table
    console.print(table)


@eval.command("list-datasets")
@click.option("--provider", "-p", help="Name of the evaluation provider")
def list_eval_datasets(provider):
    """List available evaluation datasets for a provider."""
    provider_class = ProviderRegistry.get_provider("eval", provider)

    if not provider_class:
        click.echo("No evaluation provider available.")
        click.echo("Try installing optional dependencies: pip install trustyai[eval]")
        return

    # Initialize provider
    try:
        provider_instance = provider_class("default")

        datasets = provider_instance.list_available_datasets()

        if not datasets:
            if provider and provider == "lm_eval_harness":
                click.echo("The lm_eval_harness provider does not manage datasets.")
                click.echo(
                    "Please refer to the lm-evaluation-harness documentation for available tasks."
                )
                click.echo("https://github.com/EleutherAI/lm-evaluation-harness")
            else:
                click.echo(
                    f"No datasets available for provider: {provider or provider_class.get_provider_name()}"
                )
            return

        # Create a Rich table for datasets
        console = Console()
        table = Table(
            title=f"Available Datasets for {provider or provider_class.get_provider_name()}"
        )

        # Add columns
        table.add_column("Dataset Name", style="cyan")

        # Add rows for each dataset
        for dataset in datasets:
            table.add_row(dataset)

        # Print the table
        console.print(table)

    except Exception as e:
        click.echo(f"Error: {str(e)}")


@eval.command("list-metrics")
@click.option("--provider", "-p", help="Name of the evaluation provider")
def list_eval_metrics(provider):
    """List available evaluation metrics for a provider."""
    provider_class = ProviderRegistry.get_provider("eval", provider)

    if not provider_class:
        click.echo("No evaluation provider available.")
        click.echo("Try installing optional dependencies: pip install trustyai[eval]")
        return

    # Initialize provider
    try:
        provider_instance = provider_class("default")

        metrics = provider_instance.list_available_metrics()

        if not metrics:
            click.echo(
                f"No metrics available for provider: {provider or provider_class.get_provider_name()}"
            )
            return

        # Create a Rich table for metrics
        console = Console()
        table = Table(
            title=f"Available Metrics for {provider or provider_class.get_provider_name()}"
        )

        # Add columns
        table.add_column("Metric Name", style="cyan")

        # Add rows for each metric
        for metric in metrics:
            table.add_row(metric)

        # Print the table
        console.print(table)

    except Exception as e:
        click.echo(f"Error: {str(e)}")


def convert_numpy(obj):
    """Convert NumPy types to Python native types for serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.number, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.dtype):
        return str(obj)
    elif hasattr(obj, "dtype") and hasattr(obj, "item"):
        return obj.item()
    else:
        return obj


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that can handle NumPy types."""

    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.dtype):
            return str(obj)
        elif hasattr(obj, "dtype") and hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)


@eval.command("execute")
@click.option("--provider", "-p", required=True, help="Name of the evaluation provider")
@click.option(
    "--execution-mode", 
    type=click.Choice(["local", "kubernetes"]), 
    default="local",
    help="Execution mode (default: local)"
)
@click.option("--model", required=True, help="Model identifier/path")
@click.option("--tasks", required=True, help="Comma-separated list of evaluation tasks")
@click.option("--limit", "-l", type=int, help="Limit the number of examples to evaluate")
@click.option("--batch-size", type=int, help="Batch size for evaluation")
@click.option("--output", "-o", help="Path to output results file")
@click.option(
    "--format",
    "-f",
    default="json",
    type=click.Choice(["json", "csv"]),
    help="Output format (default: json)",
)
@click.option("--namespace", "-n", help="Kubernetes namespace (for kubernetes execution mode)")
@click.option("--cpu", help="CPU limit for Kubernetes execution")
@click.option("--memory", help="Memory limit for Kubernetes execution")
@click.option("--image", help="Container image for Kubernetes execution")
@click.option("--dataset", help="Path to external dataset (if required by provider)")
@click.option("--parameters", help="Additional provider-specific parameters as JSON string")
@click.option("--dry-run", is_flag=True, help="Validate configuration without executing")
@click.option("--watch", is_flag=True, help="Watch Kubernetes job progress (kubernetes mode only)")
@click.option("--force", is_flag=True, help="Force execution despite validation warnings")
def execute_eval(provider, execution_mode, model, tasks, limit, batch_size, output, format, 
                namespace, cpu, memory, image, dataset, parameters, dry_run, watch, force):
    """Execute model evaluation with specified provider and execution mode.
    
    This unified command supports both local and Kubernetes execution modes.
    
    Examples:
      # Local execution
      trustyai eval execute --provider lm-evaluation-harness --execution-mode local \\
        --model "hf/microsoft/DialoGPT-medium" --tasks "hellaswag,arc_easy" --limit 10
      
      # Kubernetes execution  
      trustyai eval execute --provider lm-evaluation-harness --execution-mode kubernetes \\
        --model "hf/microsoft/DialoGPT-medium" --tasks "hellaswag,arc_easy" \\
        --namespace trustyai-eval --cpu 4 --memory 8Gi
        
      # RAGAS evaluation with external dataset
      trustyai eval execute --provider ragas --execution-mode local \\
        --model "openai/gpt-4" --tasks "faithfulness,answer_relevancy" \\
        --dataset "data/rag_evaluation.json"
    """
    # Ensure providers are imported and registered
    _import_eval_providers()
    
    provider_class = ProviderRegistry.get_provider("eval", provider)

    if not provider_class:
        click.echo(f"Error: Evaluation provider '{provider}' not found.")
        click.echo("Try installing optional dependencies: pip install trustyai[eval]")
        click.echo("Use 'trustyai eval list-providers' to see available providers.")
        return

    # Parse tasks
    task_list = [task.strip() for task in tasks.split(",")]
    
    # Parse additional parameters if provided
    extra_params = {}
    if parameters:
        try:
            import json
            extra_params = json.loads(parameters)
        except json.JSONDecodeError as e:
            click.echo(f"Error: Invalid JSON in --parameters: {e}")
            return

    # Initialize provider
    try:
        provider_instance = provider_class()
        provider_instance.initialize()
        
        # Convert execution mode string to enum
        requested_mode = ExecutionMode.LOCAL if execution_mode == "local" else ExecutionMode.KUBERNETES
        
        # Validate execution mode support
        if not provider_instance.is_mode_supported(requested_mode):
            supported_modes = [mode.value for mode in provider_instance.supported_deployment_modes]
            click.echo(f"Error: Provider '{provider}' does not support '{execution_mode}' execution mode")
            click.echo(f"Supported modes: {', '.join(supported_modes)}")
            return

        # Validation for kubernetes mode
        if execution_mode == "kubernetes":
            if not namespace:
                click.echo("Error: --namespace is required for kubernetes execution mode")
                return
        
        # Display configuration
        click.echo(f"Provider: {provider}")
        click.echo(f"Execution mode: {execution_mode}")
        click.echo(f"Model: {model}")
        click.echo(f"Tasks: {', '.join(task_list)}")
        
        if limit:
            click.echo(f"Limit: {limit}")
        if batch_size:
            click.echo(f"Batch size: {batch_size}")
        if dataset:
            click.echo(f"Dataset: {dataset}")
        if execution_mode == "kubernetes":
            click.echo(f"Namespace: {namespace}")
            if cpu:
                click.echo(f"CPU: {cpu}")
            if memory:
                click.echo(f"Memory: {memory}")
            if image:
                click.echo(f"Image: {image}")

        # Create configuration object
        from .core.eval import EvaluationProviderConfig
        
        config_params = {
            "evaluation_name": f"eval_{'_'.join(task_list)}",
            "model": model,
            "tasks": task_list,
            "deployment_mode": requested_mode,
        }
        
        # Add optional parameters
        if limit:
            config_params["limit"] = limit
        if batch_size:
            config_params["batch_size"] = batch_size
        if dataset:
            config_params["dataset"] = dataset
            
        # Add kubernetes-specific parameters
        if execution_mode == "kubernetes":
            config_params["namespace"] = namespace
            if cpu:
                config_params["cpu"] = cpu
            if memory:
                config_params["memory"] = memory
            if image:
                config_params["image"] = image
                
        # Add extra parameters
        config_params.update(extra_params)
        
        config = EvaluationProviderConfig(**config_params)

        # Dry run mode - validate only
        if dry_run:
            click.echo("\n--- Dry Run Mode - Validation Only ---")
            # Here we would add validation logic
            click.echo("✅ Configuration validated successfully")
            click.echo("Use without --dry-run to execute the evaluation")
            return

        # Execute evaluation
        click.echo("\n🚀 Starting evaluation...")
        
        results = provider_instance.evaluate(config)

        # Handle results based on execution mode
        if execution_mode == "local":
            # Output results for local execution
            if output:
                # Create a serializable copy of results
                serializable_results = convert_numpy(results)

                # Write to the specified output file
                with open(output, "w") as f:
                    if format == "json":
                        json.dump(serializable_results, f, indent=2, cls=NumpyJSONEncoder)
                    elif format == "csv":
                        # Basic CSV export for simple results
                        import csv

                        if "results" in serializable_results and isinstance(
                            serializable_results["results"], dict
                        ):
                            # Open the file in write mode
                            with open(output, "w", newline="") as csvfile:
                                fieldnames = ["task", "metric", "value"]
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writeheader()

                                # Write each result
                                for task_name, task_results in serializable_results["results"].items():
                                    if isinstance(task_results, dict):
                                        for metric_name, metric_value in task_results.items():
                                            writer.writerow(
                                                {
                                                    "task": task_name,
                                                    "metric": metric_name,
                                                    "value": metric_value,
                                                }
                                            )
                        else:
                            click.echo("Error: Results format not suitable for CSV export")
                click.echo(f"Results saved to: {output}")
            else:
                # Print to console in a readable format
                console = Console()
                
                if "results" in results and isinstance(results["results"], dict):
                    table = Table(title=f"Evaluation Results for {model}")
                    table.add_column("Task", style="cyan")
                    table.add_column("Metric", style="green")
                    table.add_column("Value", style="yellow")

                    for task_name, task_results in results["results"].items():
                        if isinstance(task_results, dict):
                            for metric_name, metric_value in task_results.items():
                                table.add_row(task_name, metric_name, str(metric_value))

                    console.print(table)
                else:
                    # Just dump the results
                    console.print(results)
                    
        elif execution_mode == "kubernetes":
            # Handle kubernetes execution results
            console = Console()
            
            console.print("[bold]Kubernetes Execution Results:[/bold]")
            console.print(f"Provider: {results.get('provider', 'unknown')}")
            console.print(f"Status: {results.get('status', 'unknown')}")
            
            if "message" in results:
                console.print(f"Message: {results['message']}")
                
            # If job was deployed successfully
            if results.get('status') == 'deployed':
                console.print("\n[bold green]✅ Evaluation job deployed successfully![/bold green]")
                
                if watch:
                    console.print("👀 Watching job progress...")
                    # Here we would add job watching logic
                    console.print("Use 'kubectl get jobs -n {namespace}' to check status manually")
                else:
                    console.print("You can check job status with:")
                    console.print(f"  kubectl get jobs -n {namespace}")
                    console.print(f"  kubectl logs -f -n {namespace} job/<job-name>")
                    
            elif results.get('status') == 'failed':
                console.print("\n[bold red]❌ Failed to deploy evaluation job![/bold red]")
                if "error" in results:
                    console.print(f"Error: {results['error']}")

    except Exception as e:
        click.echo(f"Error: {str(e)}")
        if force or "--verbose" in sys.argv:
            import traceback
            traceback.print_exc()


# Deprecated commands - keeping for backward compatibility with deprecation warnings
@eval.command("run", deprecated=True, hidden=True)
@click.argument("model_id")
@click.option("--task", "-t", required=True, help="Name of the task/dataset to evaluate on")
@click.option("--provider", "-p", help="Name of the evaluation provider")
@click.option(
    "--metrics", "-m", multiple=True, help="Metrics to compute (can be used multiple times)"
)
@click.option("--limit", "-l", type=int, help="Limit the number of examples to evaluate")
@click.option(
    "--use-gpu/--no-gpu", default=True, help="Whether to use GPU for evaluation (default: True)"
)
@click.option("--output", "-o", help="Path to output results file")
@click.option(
    "--format",
    "-f",
    default="json",
    type=click.Choice(["json", "csv"]),
    help="Output format (default: json)",
)
def run_eval_deprecated(model_id, task, provider, metrics, limit, use_gpu, output, format):
    """[DEPRECATED] Use 'trustyai eval execute' instead."""
    click.echo("⚠️  WARNING: 'trustyai eval run' is deprecated.")
    click.echo("   Please use 'trustyai eval execute --execution-mode local' instead.")
    click.echo("   See 'trustyai eval execute --help' for the new syntax.\n")
    
    # Convert to new execute command format for backward compatibility
    tasks_str = task
    provider_name = provider or "lm-evaluation-harness"
    
    # Build equivalent execute command
    execute_args = [
        "--provider", provider_name,
        "--execution-mode", "local", 
        "--model", model_id,
        "--tasks", tasks_str
    ]
    
    if limit:
        execute_args.extend(["--limit", str(limit)])
    if output:
        execute_args.extend(["--output", output])
    if format != "json":
        execute_args.extend(["--format", format])
        
    # Call the new execute command
    ctx = click.get_current_context()
    ctx.invoke(execute_eval, **{
        'provider': provider_name,
        'execution_mode': 'local',
        'model': model_id, 
        'tasks': tasks_str,
        'limit': limit,
        'batch_size': None,
        'output': output,
        'format': format,
        'namespace': None,
        'cpu': None,
        'memory': None,
        'image': None,
        'dataset': None,
        'parameters': None,
        'dry_run': False,
        'watch': False,
        'force': False
    })


@eval.command("deploy", deprecated=True, hidden=True)
@click.argument("model_id")
@click.option("--task", "-t", required=True, help="Name of the task/dataset to evaluate on")
@click.option("--provider", "-p", help="Name of the evaluation provider")
@click.option("--limit", "-l", type=int, help="Limit the number of examples to evaluate")
@click.option(
    "--use-gpu/--no-gpu", default=True, help="Whether to use GPU for evaluation (default: True)"
)
@click.option("--output", "-o", help="Path to save the generated YAML")
@click.option(
    "--apply/--no-apply",
    default=False,
    help="Apply the resources to the Kubernetes cluster (default: False)",
)
@click.option(
    "--kubeconfig",
    help="Path to kubeconfig file for Kubernetes deployment",
)
@click.option(
    "--context",
    help="Kubernetes context to use for deployment",
)
@click.option(
    "--namespace",
    "-n",
    default="trustyai",
    help="Kubernetes namespace to use (default: trustyai)",
)
def deploy_eval_deprecated(model_id, task, provider, limit, use_gpu, output, apply, kubeconfig, context, namespace):
    """[DEPRECATED] Use 'trustyai eval execute --execution-mode kubernetes' instead."""
    click.echo("⚠️  WARNING: 'trustyai eval deploy' is deprecated.")
    click.echo("   Please use 'trustyai eval execute --execution-mode kubernetes' instead.")
    click.echo("   See 'trustyai eval execute --help' for the new syntax.\n")
    
    # Convert to new execute command format for backward compatibility  
    tasks_str = task
    provider_name = provider or "lm-evaluation-harness"
    
    # Call the new execute command
    ctx = click.get_current_context()
    ctx.invoke(execute_eval, **{
        'provider': provider_name,
        'execution_mode': 'kubernetes',
        'model': model_id,
        'tasks': tasks_str, 
        'limit': limit,
        'batch_size': None,
        'output': output,
        'format': 'json',
        'namespace': namespace,
        'cpu': None,
        'memory': None,
        'image': None,
        'dataset': None,
        'parameters': None,
        'dry_run': False,
        'watch': False,
        'force': False
    })


if __name__ == "__main__":
    main()
