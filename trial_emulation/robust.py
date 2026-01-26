"""
Robust variance calculation.

Implements sandwich estimator for robust standard errors.
Maps R's robust.R function.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLSResults


def robust_calculation(model: Any, data_id: pd.Series) -> Dict[str, Any]:
    """
    Robust variance calculation.

    Performs calculation of robust standard errors based on
    cluster-robust variance estimation using sandwich estimator.

    Maps R's robust_calculation() function.

    Args:
        model: Fitted GLM model object (statsmodels)
        data_id: Series of ID values to identify clusters

    Returns:
        Dictionary with keys:
            - summary: DataFrame with robust standard errors and CIs
            - matrix: Robust covariance matrix
    """
    # Calculate cluster-robust variance-covariance matrix
    # Using a simplified sandwich estimator
    # For production, consider using statsmodels.stats.sandwich_covariance

    try:
        from statsmodels.stats.sandwich_covariance import cov_cluster

        # Get residuals and design matrix
        if hasattr(model, "resid_response"):
            residuals = model.resid_response
        else:
            residuals = model.resid_pearson

        # Cluster by ID
        var_matrix = cov_cluster(model, data_id.values)
    except ImportError:
        # Fallback: use basic sandwich estimator
        import warnings

        warnings.warn(
            "statsmodels.stats.sandwich_covariance not available, " "using basic sandwich estimator"
        )
        var_matrix = _basic_sandwich(model, data_id)

    se = np.sqrt(np.diag(var_matrix))

    # Create summary table
    coef_names = model.params.index
    output = pd.DataFrame(
        {
            "names": coef_names,
            "estimate": model.params.values,
            "robust_se": se,
        }
    )

    output["2.5%"] = output["estimate"] - (1.96 * output["robust_se"])
    output["97.5%"] = output["estimate"] + (1.96 * output["robust_se"])
    output["z"] = output["estimate"] / output["robust_se"]

    from scipy import stats

    output["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(output["z"])))

    result = {
        "summary": output,
        "matrix": var_matrix,
    }

    return result


def _basic_sandwich(model: Any, data_id: pd.Series) -> np.ndarray:
    """
    Basic sandwich estimator implementation.

    Simplified cluster-robust variance estimator.
    """
    # Get residuals
    if hasattr(model, "resid_response"):
        residuals = model.resid_response.values
    else:
        residuals = model.resid_pearson.values

    # Get design matrix
    if hasattr(model, "model"):
        X = model.model.exog
    else:
        raise ValueError("Cannot extract design matrix from model")

    # Calculate meat: sum over clusters of X' * e * e' * X
    meat = np.zeros((X.shape[1], X.shape[1]))

    for cluster_id in data_id.unique():
        cluster_mask = (data_id == cluster_id).values
        X_cluster = X[cluster_mask]
        e_cluster = residuals[cluster_mask]

        # Outer product of residuals
        ee = np.outer(e_cluster, e_cluster)
        meat += X_cluster.T @ ee @ X_cluster

    # Bread: (X'X)^-1
    bread = np.linalg.inv(X.T @ X)

    # Sandwich: bread * meat * bread
    var_matrix = bread @ meat @ bread

    return var_matrix
