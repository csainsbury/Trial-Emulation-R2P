"""Marginal structural model fitting for binary and continuous outcomes."""

from typing import Any, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.formula.api import glm

from .formula_utils import add_rhs, parse_formula_like
from .robust import robust_calculation
from .types import TEDataPrep, TEMSM
from .utils import limit_weight, p99_weight, quiet_line, quiet_msg, quiet_print

_SUPPORTED_FAMILIES = {"binomial", "gaussian"}


def _family(name: str) -> Any:
    if name == "binomial":
        return sm.families.Binomial(link=sm.families.links.Logit())
    if name == "gaussian":
        return sm.families.Gaussian(link=sm.families.links.Identity())
    raise ValueError(f"outcome_family must be one of {sorted(_SUPPORTED_FAMILIES)}, got '{name}'")


def _prepare_formula_rows(
    data: pd.DataFrame,
    formula: str,
    weights: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build the design once and align rows, weights and clusters explicitly."""
    data = data.reset_index(drop=True)
    try:
        y_design, x_design = dmatrices(
            formula,
            data=data,
            return_type="dataframe",
            NA_action="drop",
        )
    except Exception as exc:
        raise ValueError(f"could not construct MSM design matrix: {exc}") from exc

    retained = x_design.index.to_numpy(dtype=int)
    if len(retained) < len(data):
        warnings.warn(
            "Data frame for outcome model contains missing data in model variables. "
            "Doing complete-case analysis on formula variables only.",
            UserWarning,
        )
    if len(retained) == 0:
        raise ValueError("No rows remain after complete-case filtering")

    model_data = data.iloc[retained].copy().reset_index(drop=True)
    model_weights = np.asarray(weights, dtype=float)[retained]
    # Rebuild after row removal so categorical levels found only in excluded
    # rows cannot leave all-zero design columns and a false rank deficiency.
    try:
        y_design, x_design = dmatrices(
            formula,
            data=model_data,
            return_type="dataframe",
            NA_action="raise",
        )
    except Exception as exc:
        raise ValueError(f"could not rebuild complete-case MSM design matrix: {exc}") from exc
    y_design = y_design.reset_index(drop=True)
    x_design = x_design.reset_index(drop=True)

    if not np.isfinite(y_design.to_numpy(dtype=float)).all():
        raise ValueError("outcome design contains non-finite values")
    if not np.isfinite(x_design.to_numpy(dtype=float)).all():
        raise ValueError("MSM design contains non-finite values")
    if not np.isfinite(model_weights).all() or np.any(model_weights < 0):
        raise ValueError("analysis weights must be finite and non-negative")
    if float(model_weights.sum()) <= 0:
        raise ValueError("analysis weights must have positive total weight")

    positive_design = x_design.to_numpy(dtype=float)[model_weights > 0]
    if positive_design.shape[0] < positive_design.shape[1]:
        raise ValueError("positive-weight rows are fewer than model parameters")
    rank = int(np.linalg.matrix_rank(positive_design))
    if rank < positive_design.shape[1]:
        raise ValueError(
            f"MSM design is rank deficient on positive-weight rows: "
            f"rank {rank} for {positive_design.shape[1]} columns"
        )

    return model_data, model_weights, y_design, x_design


def trial_msm(
    data: Union[pd.DataFrame, TEDataPrep],
    outcome_cov: Union[str, List[str]] = "~1",
    estimand_type: str = "ITT",
    model_var: Optional[Union[str, List[str]]] = None,
    first_followup: Optional[int] = None,
    last_followup: Optional[int] = None,
    analysis_weights: str = "asis",
    weight_limits: tuple = (0, np.inf),
    include_followup_time: Union[str, List[str]] = "followup_time + I(followup_time**2)",
    include_trial_period: Union[str, List[str]] = "trial_period + I(trial_period**2)",
    where_case: Optional[List[str]] = None,
    glm_function: str = "glm",
    use_sample_weights: bool = True,
    outcome_family: str = "binomial",
    small_sample_correction: bool = False,
    compute_robust_variance: bool = True,
    quiet: bool = False,
    **kwargs,
) -> TEMSM:
    """Fit a weighted marginal structural model.

    ``outcome_family='binomial'`` fits the package's traditional pooled-logistic
    time-to-event MSM. ``outcome_family='gaussian'`` fits an identity-link model
    for a continuous endpoint, including direct one-index target-trial frames.

    Analysis weights are estimating-equation weights.  The returned robust
    covariance is a patient-clustered sandwich conditional on those fitted
    weights; it does not include uncertainty from estimating nuisance models.

    Args:
        data: Expanded DataFrame, direct trial frame, or TEDataPrep object.
        outcome_cov: Covariates for the outcome model.
        estimand_type: ``ITT``, ``PP``, or ``As-Treated``.
        model_var: Variables representing the treatment strategy.
        first_followup: First follow-up time to include.
        last_followup: Last follow-up time to include.
        analysis_weights: ``asis``, ``unweighted``, ``p99``, or ``weight_limits``.
        weight_limits: Limits used when ``analysis_weights='weight_limits'``.
        include_followup_time: Follow-up-time terms in the formula.
        include_trial_period: Trial-period terms in the formula.
        where_case: Optional pandas-query conditions.
        glm_function: Retained for R-interface compatibility; only GLM is used.
        use_sample_weights: Multiply by ``sample_weight`` when present.
        outcome_family: ``binomial`` (logit) or ``gaussian`` (identity).
        small_sample_correction: Apply the conventional cluster sandwich factor.
        compute_robust_variance: Compute the patient-clustered sandwich. Set to
            ``False`` only when variance comes from an outer resampling procedure;
            the fitted GLM and predictions are unchanged and ``robust`` is empty.
        quiet: Suppress progress output.
        **kwargs: Additional arguments passed to statsmodels ``GLM.fit``.
    """
    if isinstance(data, TEDataPrep):
        data = data.data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a DataFrame or TEDataPrep object")
    data = data.copy()

    if estimand_type not in ["ITT", "PP", "As-Treated"]:
        raise ValueError(
            f"estimand_type must be one of 'ITT', 'PP', 'As-Treated', got '{estimand_type}'"
        )
    if analysis_weights not in ["asis", "unweighted", "p99", "weight_limits"]:
        raise ValueError(
            "analysis_weights must be one of 'asis', 'unweighted', 'p99', "
            f"'weight_limits', got '{analysis_weights}'"
        )
    if outcome_family not in _SUPPORTED_FAMILIES:
        _family(outcome_family)  # raise the canonical message
    if glm_function != "glm":
        raise ValueError("glm_function is retained for compatibility; only 'glm' is supported")
    forbidden_fit_kwargs = {"weights", "freq_weights", "var_weights"}.intersection(kwargs)
    if forbidden_fit_kwargs:
        raise ValueError(
            "weights are controlled by trial_msm; remove fit arguments: "
            + ", ".join(sorted(forbidden_fit_kwargs))
        )

    outcome_cov_str, _ = parse_formula_like(outcome_cov)
    include_followup_time_str, _ = parse_formula_like(include_followup_time)
    include_trial_period_str, _ = parse_formula_like(include_trial_period)

    if first_followup is not None or last_followup is not None:
        first_followup = max(0, first_followup) if first_followup is not None else 0
        last_followup = min(np.inf, last_followup) if last_followup is not None else np.inf
        data = data[
            (data["followup_time"] >= first_followup) & (data["followup_time"] <= last_followup)
        ].copy()
        if data.empty:
            raise ValueError("No rows remain after follow-up time filtering")

    quiet_msg(quiet, "Preparing for model fitting")
    if estimand_type in ["ITT", "PP"]:
        model_var_str = (
            "assigned_treatment" if model_var is None else parse_formula_like(model_var)[0]
        )
    else:
        model_var_str = (
            "dose + I(dose**2)" if model_var is None else parse_formula_like(model_var)[0]
        )

    model_formula = "outcome ~ 1"
    model_formula = add_rhs(model_formula, model_var_str)
    model_formula = add_rhs(model_formula, include_trial_period_str)
    model_formula = add_rhs(model_formula, include_followup_time_str)
    model_formula = add_rhs(model_formula, outcome_cov_str)

    if where_case:
        quiet_msg(quiet, f"Subsetting data with {', '.join(where_case)}")
        for condition in where_case:
            data = data.query(condition).copy()
        if data.empty:
            raise ValueError("No rows remain after where_case filtering")

    weights = (
        data["weight"].to_numpy(dtype=float).copy()
        if "weight" in data.columns
        else np.ones(len(data), dtype=float)
    )
    if use_sample_weights:
        if "sample_weight" not in data.columns:
            warnings.warn("'sample_weight' column not found. Using sample weights = 1.")
        else:
            weights *= data["sample_weight"].to_numpy(dtype=float)

    if analysis_weights == "p99":
        weights = p99_weight(weights)
    elif analysis_weights == "weight_limits":
        weights = limit_weight(weights, weight_limits[0], weight_limits[1])
    elif analysis_weights == "unweighted":
        weights = np.ones(len(data), dtype=float)

    model_data, model_weights, y_design, x_design = _prepare_formula_rows(
        data, model_formula, weights
    )
    if "id" not in model_data.columns:
        raise ValueError("data must contain an 'id' column for clustered variance")
    if model_data["id"].isna().any():
        raise ValueError("cluster IDs must not contain missing values")

    outcome = y_design.iloc[:, 0].to_numpy(dtype=float)
    if outcome_family == "binomial" and not np.isin(outcome, [0.0, 1.0]).all():
        raise ValueError("binomial outcome must contain only 0 and 1")

    if "assigned_treatment" in model_var_str and "assigned_treatment" in model_data.columns:
        arm = model_data["assigned_treatment"]
        for value in pd.unique(arm.dropna()):
            if float(model_weights[arm.to_numpy() == value].sum()) <= 0:
                raise ValueError(f"assigned_treatment={value!r} has no positive analysis weight")

    quiet_line(quiet)
    quiet_msg(quiet, "Fitting outcome model")
    fit_kwargs = {"disp": 0}
    fit_kwargs.update(kwargs)
    model = glm(
        model_formula,
        data=model_data,
        family=_family(outcome_family),
        freq_weights=model_weights,
        missing="raise",
    ).fit(**fit_kwargs)
    if hasattr(model, "converged") and not bool(model.converged):
        raise RuntimeError("MSM GLM did not converge")
    if not np.isfinite(np.asarray(model.params, dtype=float)).all():
        raise RuntimeError("MSM GLM returned non-finite parameters")

    quiet_print(quiet, model.summary())
    quiet_line(quiet)
    if compute_robust_variance:
        quiet_msg(quiet, "Calculating robust variance")
        robust_model = robust_calculation(
            model,
            model_data["id"],
            small_sample_correction=small_sample_correction,
        )
        quiet_msg(quiet, "Summary with robust standard error:")
        quiet_print(quiet, robust_model["summary"])
        quiet_line(quiet)
    else:
        robust_model = {}

    positive_weight_rows = int(np.count_nonzero(model_weights > 0))
    args = {
        "outcome_cov": outcome_cov_str,
        "estimand_type": estimand_type,
        "model_var": model_var_str,
        "first_followup": first_followup,
        "last_followup": last_followup,
        "analysis_weights": analysis_weights,
        "weight_limits": weight_limits,
        "include_followup_time": include_followup_time_str,
        "include_trial_period": include_trial_period_str,
        "where_case": where_case,
        "glm_function": glm_function,
        "use_sample_weights": use_sample_weights,
        "outcome_family": outcome_family,
        "small_sample_correction": small_sample_correction,
        "compute_robust_variance": compute_robust_variance,
        "model_formula": model_formula,
        "analysis_rows": int(len(model_data)),
        "analysis_clusters": int(model_data.loc[model_weights > 0, "id"].nunique()),
        "analysis_clusters_all_rows": int(model_data["id"].nunique()),
        "analysis_positive_weight_rows": positive_weight_rows,
        "analysis_weight_sum": float(model_weights.sum()),
        "working_model_scale": float(model.scale),
        "design_rank": int(
            np.linalg.matrix_rank(x_design.to_numpy(dtype=float)[model_weights > 0])
        ),
        "design_columns": int(x_design.shape[1]),
        "variance_scope": (
            robust_model["variance_scope"] if compute_robust_variance else "not_computed"
        ),
    }
    return TEMSM(model=model, robust=robust_model, args=args)


def fit_msm(*args, **kwargs):
    """Alias for :func:`trial_msm` for compatibility with the R interface."""
    return trial_msm(*args, **kwargs)
