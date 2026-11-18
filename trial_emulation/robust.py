"""Cluster-robust sandwich variance for marginal structural models."""

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats


def _cluster_meat(
    score_obs: np.ndarray, group_values: np.ndarray, positive_rows: np.ndarray
) -> np.ndarray:
    """Sum over positive-weight clusters of the outer product of cluster score sums.

    Each cluster's score sum runs over all of its rows (zero-weight rows have
    zero score), exactly as a per-cluster loop would, but in one pass over the
    rows: O(rows x parameters + clusters x parameters^2) instead of
    O(clusters x rows).
    """
    codes, _ = pd.factorize(group_values)
    n_groups = int(codes.max()) + 1
    cluster_scores = np.empty((n_groups, score_obs.shape[1]), dtype=float)
    for j in range(score_obs.shape[1]):
        cluster_scores[:, j] = np.bincount(codes, weights=score_obs[:, j], minlength=n_groups)
    has_positive = np.zeros(n_groups, dtype=bool)
    has_positive[codes[positive_rows]] = True
    cluster_scores = cluster_scores[has_positive]
    return cluster_scores.T @ cluster_scores


def robust_calculation(
    model: Any,
    data_id: pd.Series,
    *,
    small_sample_correction: bool = False,
) -> Dict[str, Any]:
    """Calculate a patient-clustered estimating-equation sandwich covariance.

    The fitted model must expose per-observation scores and a Hessian, as
    statsmodels GLM results do.  Frequency weights supplied to the GLM are part
    of both quantities, so causal/IP weights enter once in the estimating
    equation and are squared in the cluster-score meat.

    This is a fixed-weight covariance: uncertainty from estimating propensity,
    censoring, or measurement weights is not included.  Refit those nuisance
    models in a patient bootstrap (or use stacked estimating equations) for
    full-pipeline uncertainty.
    """
    groups = pd.Series(data_id).reset_index(drop=True)
    if groups.isna().any():
        raise ValueError("cluster IDs must not contain missing values")

    # A fixed working scale is required for estimating-equation inference.
    # Letting statsmodels infer Gaussian scale from frequency-weight residual
    # degrees of freedom can produce infinities when sum(weights) <= p, while
    # observed-information calculations can produce 0/0 for zero-weight rows.
    score_obs = np.asarray(model.model.score_obs(model.params, scale=1.0), dtype=float)
    hessian = np.asarray(model.model.hessian(model.params, scale=1.0, observed=False), dtype=float)
    if score_obs.ndim != 2:
        raise ValueError("model score_obs must be a two-dimensional matrix")
    if len(groups) != score_obs.shape[0]:
        raise ValueError(
            "cluster IDs and fitted model rows must align exactly: "
            f"got {len(groups)} IDs for {score_obs.shape[0]} rows"
        )
    if not np.isfinite(score_obs).all() or not np.isfinite(hessian).all():
        raise ValueError("non-finite score or Hessian in robust covariance")

    n_rows, n_parameters = score_obs.shape
    frequency_weights = np.asarray(
        getattr(model.model, "freq_weights", np.ones(n_rows)), dtype=float
    )
    positive_rows = frequency_weights > 0
    n_positive_rows = int(positive_rows.sum())
    group_values = groups.to_numpy()
    positive_groups = pd.unique(group_values[positive_rows])
    n_clusters = len(positive_groups)
    if n_clusters < 2:
        raise ValueError("cluster-robust covariance requires at least two positive-weight clusters")

    bread = np.linalg.pinv(-hessian)
    meat = _cluster_meat(score_obs, group_values, positive_rows)

    correction = 1.0
    if small_sample_correction:
        if n_positive_rows <= n_parameters:
            raise ValueError("small-sample correction requires positive-weight rows > parameters")
        correction = (n_clusters / (n_clusters - 1.0)) * (
            (n_positive_rows - 1.0) / (n_positive_rows - n_parameters)
        )
        meat *= correction

    var_matrix = bread @ meat @ bread
    var_matrix = (var_matrix + var_matrix.T) / 2.0
    diagonal = np.diag(var_matrix)
    if np.any(diagonal < -1e-12):
        raise ValueError("robust covariance has a materially negative variance")
    se = np.sqrt(np.maximum(diagonal, 0.0))

    coef_names = model.params.index
    estimates = np.asarray(model.params, dtype=float)
    output = pd.DataFrame(
        {
            "names": coef_names,
            "estimate": estimates,
            "robust_se": se,
        }
    )
    output["2.5%"] = output["estimate"] - (1.96 * output["robust_se"])
    output["97.5%"] = output["estimate"] + (1.96 * output["robust_se"])
    output["z"] = np.divide(
        output["estimate"],
        output["robust_se"],
        out=np.full(len(output), np.nan),
        where=output["robust_se"].to_numpy() > 0,
    )
    output["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(output["z"])))

    return {
        "summary": output,
        "matrix": var_matrix,
        "cov_params": var_matrix,
        "n_rows": int(n_rows),
        "n_positive_weight_rows": n_positive_rows,
        "n_clusters": int(n_clusters),
        "small_sample_correction": bool(small_sample_correction),
        "small_sample_factor": float(correction),
        "reference_distribution": "normal",
        "variance_scope": "patient-clustered sandwich conditional on fitted weights",
    }
