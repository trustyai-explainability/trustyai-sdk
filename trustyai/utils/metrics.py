"""Metrics for trustworthy AI evaluation."""

import numpy as np


def fairness_metric(y_true, y_pred, sensitive_feature):
    """Calculate a simple fairness metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        sensitive_feature: Array of sensitive feature values

    Returns:
        Dictionary with fairness metrics
    """
    # Placeholder implementation
    unique_groups = np.unique(sensitive_feature)
    results = {}

    for group in unique_groups:
        group_mask = sensitive_feature == group
        if np.sum(group_mask) > 0:
            group_accuracy = np.mean(y_true[group_mask] == y_pred[group_mask])
            results[f"group_{group}_accuracy"] = group_accuracy

    # Calculate max difference in accuracy as a basic fairness metric
    if len(results) > 1:
        accuracies = list(results.values())
        results["max_accuracy_disparity"] = max(accuracies) - min(accuracies)

    return results
