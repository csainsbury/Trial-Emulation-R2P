"""
Marginal structural model fitting.

Fits MSMs for the sequence of emulated trials.
Maps R's modelling.R functions.
"""

from typing import Optional, Union, List, Dict, Any
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
from statsmodels.discrete.discrete_model import LogitResults

from .formula_utils import parse_formula_like, add_rhs
from .robust import robust_calculation
from .utils import quiet_msg, quiet_print, quiet_line, p99_weight, limit_weight
from .types import TEMSM, TEDataPrep


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
    quiet: bool = False,
    **kwargs,
) -> TEMSM:
    """
    Fit the marginal structural model for the sequence of emulated trials.

    Apply a weighted pooled logistic regression to fit the marginal structural model
    and calculates the robust covariance matrix using the sandwich estimator.

    Maps R's trial_msm() function.

    Args:
        data: Expanded DataFrame or TEDataPrep object
        outcome_cov: Covariates for outcome model
        estimand_type: Type of estimand ("ITT", "PP", or "As-Treated")
        model_var: Variables to include in outcome model
        first_followup: First follow-up time to include
        last_followup: Last follow-up time to include
        analysis_weights: Weight adjustment strategy ("asis", "unweighted", "p99", "weight_limits")
        weight_limits: Limits for weight truncation
        include_followup_time: Formula for follow-up time terms
        include_trial_period: Formula for trial period terms
        where_case: List of where conditions for subsetting
        glm_function: GLM function to use (for compatibility)
        use_sample_weights: Use case-control sampling weights?
        quiet: Suppress messages?
        **kwargs: Additional arguments passed to GLM

    Returns:
        TEMSM object containing:
            - model: Fitted GLM model
            - robust: Dictionary with robust variance results
            - args: Dictionary of parameters used
    """
    # Extract data if TEDataPrep object
    if isinstance(data, TEDataPrep):
        data = data.data

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a DataFrame or TEDataPrep object")

    # Basic validation
    if estimand_type not in ["ITT", "PP", "As-Treated"]:
        raise ValueError(
            f"estimand_type must be one of 'ITT', 'PP', 'As-Treated', got '{estimand_type}'"
        )

    if analysis_weights not in ["asis", "unweighted", "p99", "weight_limits"]:
        raise ValueError(
            f"analysis_weights must be one of 'asis', 'unweighted', 'p99', 'weight_limits', "
            f"got '{analysis_weights}'"
        )

    # Convert formulas
    outcome_cov_str, _ = parse_formula_like(outcome_cov)
    include_followup_time_str, _ = parse_formula_like(include_followup_time)
    include_trial_period_str, _ = parse_formula_like(include_trial_period)

    # Filter by follow-up time if specified
    if first_followup is not None or last_followup is not None:
        first_followup = max(0, first_followup) if first_followup is not None else 0
        last_followup = min(np.inf, last_followup) if last_followup is not None else np.inf
        data = data[
            (data["followup_time"] >= first_followup) & (data["followup_time"] <= last_followup)
        ].copy()

    quiet_msg(quiet, "Preparing for model fitting")

    # Set model_var based on estimand_type
    if estimand_type == "ITT":
        if model_var is None:
            model_var_str = "assigned_treatment"
        else:
            model_var_str, _ = parse_formula_like(model_var)
    elif estimand_type == "PP":
        if model_var is None:
            model_var_str = "assigned_treatment"
        else:
            model_var_str, _ = parse_formula_like(model_var)
    elif estimand_type == "As-Treated":
        if model_var is None:
            model_var_str = "dose + I(dose**2)"
        else:
            model_var_str, _ = parse_formula_like(model_var)

    # Build model formula
    model_formula = "outcome ~ 1"
    model_formula = add_rhs(model_formula, model_var_str)
    model_formula = add_rhs(model_formula, include_trial_period_str)
    model_formula = add_rhs(model_formula, include_followup_time_str)
    model_formula = add_rhs(model_formula, outcome_cov_str)

    # Apply where_case conditions
    if where_case:
        quiet_msg(quiet, f"Subsetting data with {', '.join(where_case)}")
        for condition in where_case:
            data = data.query(condition).copy()

    # Get weights
    if "weight" not in data.columns:
        w = np.ones(len(data))
    else:
        w = data["weight"].values.copy()

    # Apply sample weights if requested
    if use_sample_weights:
        if "sample_weight" not in data.columns:
            import warnings

            warnings.warn("'sample_weight' column not found. Using sample weights = 1.")
        else:
            w = w * data["sample_weight"].values

    # Adjust weights
    if analysis_weights == "asis":
        pass  # Nothing to do
    elif analysis_weights == "p99":
        w = p99_weight(w)
    elif analysis_weights == "weight_limits":
        w = limit_weight(w, weight_limits[0], weight_limits[1])
    elif analysis_weights == "unweighted":
        w = np.ones(len(data))

    # Check for missing data
    if data.isnull().any().any():
        import warnings

        warnings.warn(
            "Data frame for outcome model contains missing data. " "Doing complete-case analysis."
        )
        # Get boolean mask of rows without missing values before dropping
        # This ensures w (numpy array) and data are aligned
        complete_case_mask = ~data.isnull().any(axis=1)
        data = data[complete_case_mask].copy()
        w = w[complete_case_mask]

    # Assert alignment between data and weights
    assert len(data) == len(w), (
        f"Data and weights must have same length. " f"Got data: {len(data)}, weights: {len(w)}"
    )

    quiet_line(quiet)
    quiet_msg(quiet, "Fitting outcome model")

    # Fit model
    model = logit(model_formula, data=data).fit(weights=w, disp=0, **kwargs)

    quiet_print(quiet, model.summary())
    quiet_line(quiet)

    quiet_msg(quiet, "Calculating robust variance")
    robust_model = robust_calculation(model, data["id"])
    quiet_msg(quiet, "Summary with robust standard error:")
    quiet_print(quiet, robust_model["summary"])
    quiet_line(quiet)

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
        "model_formula": model_formula,
    }

    result = TEMSM(model=model, robust=robust_model, args=args)

    return result


def fit_msm(*args, **kwargs):
    """
    Alias for trial_msm for compatibility with R interface.
    """
    return trial_msm(*args, **kwargs)
