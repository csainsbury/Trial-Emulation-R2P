"""
Weight calculation functions.

Implements inverse probability weighting for treatment switching and censoring.
Maps R's weighting.R / data_utils.R weight functions (TrialEmulation 0.0.4.9).
"""

from typing import Dict, List, Optional, Any, Union
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import glm

from .formula_utils import parse_formula_like
from .utils import quiet_msg, quiet_print

# Internal response column for censoring models: R fits ``1 - cense ~ ...``,
# i.e. the probability of remaining uncensored.
_UNCENSORED = "_te_uncensored"


def weight_func(
    sw_data: pd.DataFrame,
    use_switch_weights: bool = False,
    use_censor_weights: bool = False,
    switch_n_cov: Union[str, List[str]] = "~1",
    switch_d_cov: Union[str, List[str]] = "~1",
    eligible_wts_0: Optional[str] = None,
    eligible_wts_1: Optional[str] = None,
    cense: Optional[str] = None,
    pool_cense_n: bool = False,
    pool_cense_d: bool = False,
    cense_d_cov: Union[str, List[str]] = "~1",
    cense_n_cov: Union[str, List[str]] = "~1",
    save_weight_models: bool = False,
    data_dir: Optional[str] = None,
    quiet: bool = False,
    glm_function: str = "glm",
    **kwargs,
) -> Dict[str, Any]:
    """
    Weight calculation function.

    Performs calculation for inverse probability weights of the data.
    Maps R's weight_func() function.

    Args:
        sw_data: DataFrame to calculate weights for
        use_switch_weights: Calculate switching weights?
        use_censor_weights: Calculate censoring weights?
        switch_n_cov: Numerator covariates for switching weights
        switch_d_cov: Denominator covariates for switching weights
        eligible_wts_0: Eligibility variable for weights when previous treatment = 0
        eligible_wts_1: Eligibility variable for weights when previous treatment = 1
        cense: Censoring variable name
        pool_cense_n: Pool numerator censoring models?
        pool_cense_d: Pool denominator censoring models?
        cense_d_cov: Denominator covariates for censoring weights
        cense_n_cov: Numerator covariates for censoring weights
        save_weight_models: Save model objects?
        data_dir: Directory to save models
        quiet: Suppress messages?
        glm_function: GLM function to use (for compatibility)
        **kwargs: Additional arguments passed to GLM

    Returns:
        Dictionary with keys:
            - data: DataFrame with weights added
            - switch_models: Dictionary of switching weight model summaries
            - censor_models: Dictionary of censoring weight model summaries
    """
    sw_data = sw_data.copy()

    switch_models: Dict[str, Any] = {}
    censor_models: Dict[str, Any] = {}

    elig_col_0 = _resolve_eligible_wts_col(sw_data, eligible_wts_0, "eligible_wts_0")
    elig_col_1 = _resolve_eligible_wts_col(sw_data, eligible_wts_1, "eligible_wts_1")

    # Switching weights
    if use_switch_weights:
        switch_d_formula = f"treatment ~ {parse_formula_like(switch_d_cov)[0]}"
        switch_n_formula = f"treatment ~ {parse_formula_like(switch_n_cov)[0]}"

        switch_results = fit_switch_weights(
            switch_d_cov=switch_d_formula,
            switch_n_cov=switch_n_formula,
            eligible_wts_0=elig_col_0,
            eligible_wts_1=elig_col_1,
            sw_data=sw_data,
            quiet=quiet,
            save_dir=data_dir,
            save_weight_models=save_weight_models,
            **kwargs,
        )
        sw_data = switch_results["sw_data"]
        switch_models = switch_results["switch_models"]

    # Censoring weights: models for P(cense = 0 | X).
    if use_censor_weights:
        if not cense:
            raise ValueError("cense must be specified when use_censor_weights=True")
        if cense not in sw_data.columns:
            raise ValueError(f"cense column '{cense}' not found in data")

        censor_results = fit_censor_weights(
            cense=cense,
            cense_d_cov=f"{_UNCENSORED} ~ {parse_formula_like(cense_d_cov)[0]}",
            cense_n_cov=f"{_UNCENSORED} ~ {parse_formula_like(cense_n_cov)[0]}",
            pool_cense_d=pool_cense_d,
            pool_cense_n=pool_cense_n,
            sw_data=sw_data,
            quiet=quiet,
            save_dir=data_dir,
            save_weight_models=save_weight_models,
            **kwargs,
        )
        sw_data = censor_results["sw_data"]
        censor_models = censor_results["censor_models"]

    # Combine switching-model probabilities into per-period weights.
    if not use_switch_weights:
        sw_data["wt"] = 1.0
    else:
        sw_data["wt"] = np.nan
        am_1 = sw_data["am_1"]
        treat = sw_data["treatment"]
        for lag, elig_col, p_n, p_d in (
            (0, elig_col_0, "p0_n", "p0_d"),
            (1, elig_col_1, "p1_n", "p1_d"),
        ):
            base = am_1 == lag
            if elig_col is not None:
                base = base & (sw_data[elig_col] == 1)
            has_p = sw_data[p_n].notna() & sw_data[p_d].notna()
            stay0 = base & (treat == 0) & has_p
            take1 = base & (treat == 1) & has_p
            sw_data.loc[stay0, "wt"] = (1.0 - sw_data.loc[stay0, p_n]) / (
                1.0 - sw_data.loc[stay0, p_d]
            )
            sw_data.loc[take1, "wt"] = sw_data.loc[take1, p_n] / sw_data.loc[take1, p_d]
            if elig_col is not None:
                sw_data.loc[(am_1 == lag) & (sw_data[elig_col] == 0), "wt"] = 1.0

    if not use_censor_weights:
        sw_data["wtC"] = 1.0
    else:
        sw_data["pC_d"] = sw_data["pC_d"].fillna(1.0)
        sw_data["pC_n"] = sw_data["pC_n"].fillna(1.0)
        sw_data["wtC"] = sw_data["pC_n"] / sw_data["pC_d"]

    sw_data["wt"] = sw_data["wt"] * sw_data["wtC"]

    ordered = ["cens_pool_d", "cens_d0", "cens_n0", "cens_d1", "cens_n1", "cens_pool_n"]
    censor_models = {k: censor_models[k] for k in ordered if k in censor_models}

    return {
        "data": sw_data,
        "switch_models": switch_models,
        "censor_models": censor_models,
    }


def _resolve_eligible_wts_col(
    sw_data: pd.DataFrame, arg: Optional[str], canonical: str
) -> Optional[str]:
    """Column holding the eligible_wts_* indicator, or None when not used.

    ``data_preparation`` renames the user's column to ``eligible_wts_0`` /
    ``eligible_wts_1`` (as R's ``select_data_cols`` does); direct callers of
    :func:`weight_func` may pass the original column name.
    """
    if arg is None or (isinstance(arg, str) and arg in ("", "NA")):
        return None
    if canonical in sw_data.columns:
        return canonical
    if arg in sw_data.columns:
        return arg
    raise ValueError(f"{canonical} column '{arg}' not found in data")


def _fit_weight_model(formula: str, data: pd.DataFrame, **kwargs) -> Any:
    """Fit a binomial-logit GLM, as R's ``glm(family = binomial)`` does."""
    if len(data) == 0:
        raise ValueError(f"No observations available to fit weight model '{formula}'")
    fit_kwargs = {k: v for k, v in kwargs.items() if k in ("maxiter", "tol")}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Pp]erfect separation.*")
        model = glm(
            formula,
            data=data,
            family=sm.families.Binomial(link=sm.families.links.Logit()),
            missing="drop",
        ).fit(**fit_kwargs)
    return model


def _fitted_probabilities(model: Any, column: str, data: pd.DataFrame) -> pd.DataFrame:
    """Fitted probabilities aligned to the rows used by the model, with id/period."""
    rows = model.model.data.row_labels
    used = data.loc[rows]
    return pd.DataFrame(
        {
            "id": used["id"].to_numpy(),
            "period": used["period"].to_numpy(),
            column: np.asarray(model.fittedvalues, dtype=float),
        }
    )


def fit_switch_weights(
    switch_d_cov: str,
    switch_n_cov: str,
    eligible_wts_0: Optional[str],
    eligible_wts_1: Optional[str],
    sw_data: pd.DataFrame,
    quiet: bool,
    save_dir: Optional[str],
    save_weight_models: bool,
    **kwargs,
) -> Dict[str, Any]:
    """
    Fit switching weight models.

    Maps R's fit_switch_weights(): denominator and numerator models for
    P(treatment = 1 | previous treatment = k), k = 0, 1, fitted on rows with
    ``eligible{k} == 1`` (and ``eligible_wts_{k} == 1`` when supplied).
    """
    switch_models: Dict[str, Any] = {}
    merged = []
    for k, elig_col in ((0, eligible_wts_0), (1, eligible_wts_1)):
        mask = sw_data[f"eligible{k}"] == 1
        if elig_col is not None:
            mask = mask & (sw_data[elig_col] == 1)
        subset = sw_data[mask]
        probs = None
        for part, formula, label in (
            ("d", switch_d_cov, "denominator"),
            ("n", switch_n_cov, "numerator"),
        ):
            model = _fit_weight_model(formula, subset, **kwargs)
            col = f"p{k}_{part}"
            fitted = _fitted_probabilities(model, col, subset)
            probs = (
                fitted if probs is None else probs.merge(fitted, on=["id", "period"], how="outer")
            )
            switch_models[f"switch_{part}{k}"] = process_weight_model(
                model,
                save_weight_models,
                save_dir,
                f"weight_model_switch_{part}{k}.pkl",
                f"P(treatment = 1 | previous treatment = {k}) for {label}",
                quiet,
            )
        merged.append(probs[["id", "period", f"p{k}_d", f"p{k}_n"]])

    for probs in merged:
        sw_data = sw_data.merge(probs, on=["id", "period"], how="left", validate="many_to_one")

    ordered = ["switch_d0", "switch_n0", "switch_d1", "switch_n1"]
    return {
        "sw_data": sw_data,
        "switch_models": {k: switch_models[k] for k in ordered},
    }


def fit_censor_weights(
    cense: str,
    cense_d_cov: str,
    cense_n_cov: str,
    pool_cense_d: bool,
    pool_cense_n: bool,
    sw_data: pd.DataFrame,
    quiet: bool,
    save_dir: Optional[str],
    save_weight_models: bool,
    **kwargs,
) -> Dict[str, Any]:
    """
    Fit informative censoring models for P(cense = 0 | X).

    Maps R's fit_censor_weights(). Models are pooled over previous treatment
    or fitted separately for ``eligible0`` / ``eligible1`` rows.
    """
    if pool_cense_d and not pool_cense_n:
        raise ValueError("Check the arguments for pooling censoring models!")

    censor_models: Dict[str, Any] = {}
    sw_data = sw_data.copy()
    sw_data[_UNCENSORED] = 1 - sw_data[cense]

    def fit(formula: str, data: pd.DataFrame, column: str, key: str, filename: str, desc: str):
        model = _fit_weight_model(formula, data, **kwargs)
        censor_models[key] = process_weight_model(
            model, save_weight_models, save_dir, filename, desc, quiet
        )
        return _fitted_probabilities(model, column, data)

    elig0 = sw_data[sw_data["eligible0"] == 1]
    elig1 = sw_data[sw_data["eligible1"] == 1]

    if pool_cense_d:
        cense_d = fit(
            cense_d_cov,
            sw_data,
            "pC_d",
            "cens_pool_d",
            "cense_model_pool_d.pkl",
            "Model for P(cense = 0 | X) for denominator",
        )
    else:
        cense_d0 = fit(
            cense_d_cov,
            elig0,
            "pC_d0",
            "cens_d0",
            "cense_model_d0.pkl",
            "Model for P(cense = 0 | X, previous treatment = 0) for denominator",
        )
        cense_d1 = fit(
            cense_d_cov,
            elig1,
            "pC_d1",
            "cens_d1",
            "cense_model_d1.pkl",
            "Model for P(cense = 0 | X, previous treatment = 1) for denominator",
        )

    if pool_cense_n:
        cense_n = fit(
            cense_n_cov,
            sw_data,
            "pC_n",
            "cens_pool_n",
            "cense_model_pool_n.pkl",
            "Model for P(cense = 0 | X) for numerator",
        )
    else:
        cense_n0 = fit(
            cense_n_cov,
            elig0,
            "pC_n0",
            "cens_n0",
            "cense_model_n0.pkl",
            "Model for P(cense = 0 | X, previous treatment = 0) for numerator",
        )
        cense_n1 = fit(
            cense_n_cov,
            elig1,
            "pC_n1",
            "cens_n1",
            "cense_model_n1.pkl",
            "Model for P(cense = 0 | X, previous treatment = 1) for numerator",
        )

    key = ["id", "period"]
    if pool_cense_d and pool_cense_n:
        sw_data = sw_data.merge(cense_d, on=key, how="left", validate="many_to_one")
        sw_data = sw_data.merge(cense_n, on=key, how="left", validate="many_to_one")
    elif not pool_cense_d and not pool_cense_n:
        for part in (cense_d0, cense_n0, cense_d1, cense_n1):
            sw_data = sw_data.merge(part, on=key, how="left", validate="many_to_one")
        lag0 = sw_data["am_1"] == 0
        lag1 = sw_data["am_1"] == 1
        sw_data["pC_n"] = np.nan
        sw_data["pC_d"] = np.nan
        sw_data.loc[lag0, "pC_n"] = sw_data.loc[lag0, "pC_n0"]
        sw_data.loc[lag0, "pC_d"] = sw_data.loc[lag0, "pC_d0"]
        sw_data.loc[lag1, "pC_n"] = sw_data.loc[lag1, "pC_n1"]
        sw_data.loc[lag1, "pC_d"] = sw_data.loc[lag1, "pC_d1"]
    else:  # pooled numerator, separate denominators (ITT default)
        # R: sw_data[cense_n, on = (id, period)] keeps the rows used by the
        # pooled numerator model.
        sw_data = sw_data.merge(cense_n, on=key, how="inner", validate="many_to_one")
        sw_data = sw_data.merge(cense_d0, on=key, how="left", validate="many_to_one")
        sw_data = sw_data.merge(cense_d1, on=key, how="left", validate="many_to_one")
        sw_data["pC_d"] = np.nan
        lag0 = sw_data["am_1"] == 0
        lag1 = sw_data["am_1"] == 1
        sw_data.loc[lag0, "pC_d"] = sw_data.loc[lag0, "pC_d0"]
        sw_data.loc[lag1, "pC_d"] = sw_data.loc[lag1, "pC_d1"]

    sw_data = sw_data.drop(columns=[_UNCENSORED])

    return {
        "sw_data": sw_data,
        "censor_models": censor_models,
    }


def _tidy(model: Any) -> pd.DataFrame:
    """Coefficient table in the layout of R's ``broom::tidy(glm)``."""
    names = ["(Intercept)" if n == "Intercept" else n for n in model.params.index]
    return pd.DataFrame(
        {
            "term": names,
            "estimate": np.asarray(model.params, dtype=float),
            "std.error": np.asarray(model.bse, dtype=float),
            "statistic": np.asarray(model.tvalues, dtype=float),
            "p.value": np.asarray(model.pvalues, dtype=float),
        }
    )


def process_weight_model(
    model: Any,
    save_weight_models: bool,
    save_dir: Optional[str],
    filename: str,
    description: str,
    quiet: bool,
) -> Dict[str, Any]:
    """
    Helper to process weight models.

    Maps R's process_weight_model(): ``summary`` is the coefficient table
    (as ``broom::tidy``) and ``fit_summary`` the model-level statistics (as
    ``broom::glance``).
    """
    quiet_msg(quiet, description)
    quiet_print(quiet, model.summary())

    result = {
        "description": description,
        "summary": _tidy(model),
        "fit_summary": {
            "null_deviance": float(model.null_deviance),
            "df_null": int(model.nobs - 1),
            "logLik": float(model.llf),
            "aic": float(model.aic),
            "deviance": float(model.deviance),
            "df_residual": int(model.df_resid),
            "nobs": int(model.nobs),
        },
    }

    if save_weight_models and save_dir:
        path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        result["path"] = path

    return result
