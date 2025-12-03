"""
Weight calculation functions.

Implements inverse probability weighting for treatment switching and censoring.
Maps R's weighting.R functions.
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
from statsmodels.discrete.discrete_model import LogitResults
import warnings

from .formula_utils import parse_formula_like
from .utils import quiet_msg, quiet_print


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
    **kwargs
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
    
    switch_models = {}
    censor_models = {}
    
    # Switching weights
    if use_switch_weights:
        switch_d_formula, _ = parse_formula_like(switch_d_cov)
        switch_n_formula, _ = parse_formula_like(switch_n_cov)
        
        # Add treatment to LHS
        switch_d_formula = f"treatment ~ {switch_d_formula}"
        switch_n_formula = f"treatment ~ {switch_n_formula}"
        
        switch_results = fit_switch_weights(
            switch_d_cov=switch_d_formula,
            switch_n_cov=switch_n_formula,
            eligible_wts_0=eligible_wts_0,
            eligible_wts_1=eligible_wts_1,
            sw_data=sw_data,
            quiet=quiet,
            save_dir=data_dir,
            save_weight_models=save_weight_models,
            **kwargs
        )
        sw_data = switch_results["sw_data"]
        switch_models = switch_results["switch_models"]
    
    # Censoring weights
    if use_censor_weights:
        if not cense:
            raise ValueError("cense must be specified when use_censor_weights=True")
        
        cense_d_formula, _ = parse_formula_like(cense_d_cov)
        cense_n_formula, _ = parse_formula_like(cense_n_cov)
        
        # Add (1 - cense) to LHS
        cense_d_formula = f"(1 - {cense}) ~ {cense_d_formula}"
        cense_n_formula = f"(1 - {cense}) ~ {cense_n_formula}"
        
        censor_results = fit_censor_weights(
            cense=cense,
            cense_d_cov=cense_d_formula,
            cense_n_cov=cense_n_formula,
            pool_cense_d=pool_cense_d,
            pool_cense_n=pool_cense_n,
            sw_data=sw_data,
            quiet=quiet,
            save_dir=data_dir,
            save_weight_models=save_weight_models,
            **kwargs
        )
        sw_data = censor_results["sw_data"]
        censor_models = censor_results["censor_models"]
    
    # Calculate switching weights
    if not use_switch_weights:
        sw_data["wt"] = 1.0
    else:
        sw_data["wt"] = np.nan
        
        # For eligible0 (previous treatment = 0)
        if eligible_wts_0 and eligible_wts_0 in sw_data.columns:
            mask_0_0 = (
                (sw_data["am_1"] == 0) &
                (sw_data[eligible_wts_0] == 1) &
                (sw_data["treatment"] == 0) &
                sw_data["p0_n"].notna() &
                sw_data["p0_d"].notna()
            )
            sw_data.loc[mask_0_0, "wt"] = (
                (1.0 - sw_data.loc[mask_0_0, "p0_n"]) / (1.0 - sw_data.loc[mask_0_0, "p0_d"])
            )
            
            mask_0_1 = (
                (sw_data["am_1"] == 0) &
                (sw_data[eligible_wts_0] == 1) &
                (sw_data["treatment"] == 1) &
                sw_data["p0_n"].notna() &
                sw_data["p0_d"].notna()
            )
            sw_data.loc[mask_0_1, "wt"] = (
                sw_data.loc[mask_0_1, "p0_n"] / sw_data.loc[mask_0_1, "p0_d"]
            )
            
            sw_data.loc[
                (sw_data["am_1"] == 0) & (sw_data[eligible_wts_0] == 0),
                "wt"
            ] = 1.0
        else:
            mask_0_0 = (
                (sw_data["am_1"] == 0) &
                (sw_data["treatment"] == 0) &
                sw_data["p0_n"].notna() &
                sw_data["p0_d"].notna()
            )
            sw_data.loc[mask_0_0, "wt"] = (
                (1.0 - sw_data.loc[mask_0_0, "p0_n"]) / (1.0 - sw_data.loc[mask_0_0, "p0_d"])
            )
            
            mask_0_1 = (
                (sw_data["am_1"] == 0) &
                (sw_data["treatment"] == 1) &
                sw_data["p0_n"].notna() &
                sw_data["p0_d"].notna()
            )
            sw_data.loc[mask_0_1, "wt"] = (
                sw_data.loc[mask_0_1, "p0_n"] / sw_data.loc[mask_0_1, "p0_d"]
            )
        
        # For eligible1 (previous treatment = 1)
        if eligible_wts_1 and eligible_wts_1 in sw_data.columns:
            mask_1_0 = (
                (sw_data["am_1"] == 1) &
                (sw_data[eligible_wts_1] == 1) &
                (sw_data["treatment"] == 0) &
                sw_data["p1_n"].notna() &
                sw_data["p1_d"].notna()
            )
            sw_data.loc[mask_1_0, "wt"] = (
                (1.0 - sw_data.loc[mask_1_0, "p1_n"]) / (1.0 - sw_data.loc[mask_1_0, "p1_d"])
            )
            
            mask_1_1 = (
                (sw_data["am_1"] == 1) &
                (sw_data[eligible_wts_1] == 1) &
                (sw_data["treatment"] == 1) &
                sw_data["p1_n"].notna() &
                sw_data["p1_d"].notna()
            )
            sw_data.loc[mask_1_1, "wt"] = (
                sw_data.loc[mask_1_1, "p1_n"] / sw_data.loc[mask_1_1, "p1_d"]
            )
            
            sw_data.loc[
                (sw_data["am_1"] == 1) & (sw_data[eligible_wts_1] == 0),
                "wt"
            ] = 1.0
        else:
            mask_1_0 = (
                (sw_data["am_1"] == 1) &
                (sw_data["treatment"] == 0) &
                sw_data["p1_n"].notna() &
                sw_data["p1_d"].notna()
            )
            sw_data.loc[mask_1_0, "wt"] = (
                (1.0 - sw_data.loc[mask_1_0, "p1_n"]) / (1.0 - sw_data.loc[mask_1_0, "p1_d"])
            )
            
            mask_1_1 = (
                (sw_data["am_1"] == 1) &
                (sw_data["treatment"] == 1) &
                sw_data["p1_n"].notna() &
                sw_data["p1_d"].notna()
            )
            sw_data.loc[mask_1_1, "wt"] = (
                sw_data.loc[mask_1_1, "p1_n"] / sw_data.loc[mask_1_1, "p1_d"]
            )
    
    # Calculate censoring weights
    if not use_censor_weights:
        sw_data["wtC"] = 1.0
    else:
        sw_data["pC_d"] = sw_data["pC_d"].fillna(1.0)
        sw_data["pC_n"] = sw_data["pC_n"].fillna(1.0)
        sw_data["wtC"] = sw_data["pC_n"] / sw_data["pC_d"]
    
    # Combine weights
    sw_data["wt"] = sw_data["wt"] * sw_data["wtC"]
    
    return {
        "data": sw_data,
        "switch_models": switch_models,
        "censor_models": censor_models,
    }


def fit_switch_weights(
    switch_d_cov: str,
    switch_n_cov: str,
    eligible_wts_0: Optional[str],
    eligible_wts_1: Optional[str],
    sw_data: pd.DataFrame,
    quiet: bool,
    save_dir: Optional[str],
    save_weight_models: bool,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit switching weight models.
    
    Maps R's fit_switch_weights() function.
    """
    switch_models = {}
    
    # Fit models for eligible0 == 1
    if eligible_wts_0 and eligible_wts_0 in sw_data.columns:
        eligible0_data = sw_data[(sw_data["eligible0"] == 1) & (sw_data[eligible_wts_0] == 1)].copy()
    else:
        eligible0_data = sw_data[sw_data["eligible0"] == 1].copy()
    
    if len(eligible0_data) > 0:
        # Denominator model
        model1 = logit(switch_d_cov, data=eligible0_data).fit(disp=0)
        switch_d0 = pd.DataFrame({
            "p0_d": model1.fittedvalues,
            "id": eligible0_data["id"].values,
            "period": eligible0_data["period"].values,
        })
        
        switch_models["switch_d0"] = process_weight_model(
            model1,
            save_weight_models,
            save_dir,
            "weight_model_switch_d0.pkl",
            "P(treatment = 1 | previous treatment = 0) for denominator",
            quiet
        )
        
        # Numerator model
        model2 = logit(switch_n_cov, data=eligible0_data).fit(disp=0)
        switch_n0 = pd.DataFrame({
            "p0_n": model2.fittedvalues,
            "id": eligible0_data["id"].values,
            "period": eligible0_data["period"].values,
        })
        
        switch_models["switch_n0"] = process_weight_model(
            model2,
            save_weight_models,
            save_dir,
            "weight_model_switch_n0.pkl",
            "P(treatment = 1 | previous treatment = 0) for numerator",
            quiet
        )
        
        switch_0 = switch_d0.merge(switch_n0, on=["id", "period"], how="outer")
    else:
        switch_0 = pd.DataFrame(columns=["id", "period", "p0_d", "p0_n"])
    
    # Fit models for eligible1 == 1
    if eligible_wts_1 and eligible_wts_1 in sw_data.columns:
        eligible1_data = sw_data[(sw_data["eligible1"] == 1) & (sw_data[eligible_wts_1] == 1)].copy()
    else:
        eligible1_data = sw_data[sw_data["eligible1"] == 1].copy()
    
    if len(eligible1_data) > 0:
        # Denominator model
        model3 = logit(switch_d_cov, data=eligible1_data).fit(disp=0)
        switch_d1 = pd.DataFrame({
            "p1_d": model3.fittedvalues,
            "id": eligible1_data["id"].values,
            "period": eligible1_data["period"].values,
        })
        
        switch_models["switch_d1"] = process_weight_model(
            model3,
            save_weight_models,
            save_dir,
            "weight_model_switch_d1.pkl",
            "P(treatment = 1 | previous treatment = 1) for denominator",
            quiet
        )
        
        # Numerator model
        model4 = logit(switch_n_cov, data=eligible1_data).fit(disp=0)
        switch_n1 = pd.DataFrame({
            "p1_n": model4.fittedvalues,
            "id": eligible1_data["id"].values,
            "period": eligible1_data["period"].values,
        })
        
        switch_models["switch_n1"] = process_weight_model(
            model4,
            save_weight_models,
            save_dir,
            "weight_model_switch_n1.pkl",
            "P(treatment = 1 | previous treatment = 1) for numerator",
            quiet
        )
        
        switch_1 = switch_d1.merge(switch_n1, on=["id", "period"], how="outer")
    else:
        switch_1 = pd.DataFrame(columns=["id", "period", "p1_d", "p1_n"])
    
    # Merge back to sw_data
    sw_data = sw_data.merge(switch_0[["id", "period", "p0_d", "p0_n"]], on=["id", "period"], how="left")
    sw_data = sw_data.merge(switch_1[["id", "period", "p1_d", "p1_n"]], on=["id", "period"], how="left")
    
    return {
        "sw_data": sw_data,
        "switch_models": switch_models,
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
    **kwargs
) -> Dict[str, Any]:
    """
    Fit informative censoring models.
    
    Maps R's fit_censor_weights() function.
    """
    censor_models = {}
    
    # Create (1 - cense) variable
    sw_data["_cense_1minus"] = 1 - sw_data[cense]
    
    if pool_cense_d:
        # Fit pooled denominator model
        model1 = logit(cense_d_cov.replace(cense, "_cense_1minus"), data=sw_data).fit(disp=0)
        cense_d = pd.DataFrame({
            "pC_d": model1.fittedvalues,
            "id": sw_data["id"].values,
            "period": sw_data["period"].values,
        })
        
        censor_models["cens_pool_d"] = process_weight_model(
            model1,
            save_weight_models,
            save_dir,
            "cense_model_pool_d.pkl",
            "Model for P(cense = 0 | X) for denominator",
            quiet
        )
    else:
        # Fit separate denominator models
        # eligible0
        eligible0_data = sw_data[sw_data["eligible0"] == 1].copy()
        if len(eligible0_data) > 0:
            model1 = logit(cense_d_cov.replace(cense, "_cense_1minus"), data=eligible0_data).fit(disp=0)
            cense_d0 = pd.DataFrame({
                "pC_d0": model1.fittedvalues,
                "id": eligible0_data["id"].values,
                "period": eligible0_data["period"].values,
            })
            
            censor_models["cens_d0"] = process_weight_model(
                model1,
                save_weight_models,
                save_dir,
                "cense_model_d0.pkl",
                "Model for P(cense = 0 | X, previous treatment = 0) for denominator",
                quiet
            )
        else:
            cense_d0 = pd.DataFrame(columns=["id", "period", "pC_d0"])
        
        # eligible1
        eligible1_data = sw_data[sw_data["eligible1"] == 1].copy()
        if len(eligible1_data) > 0:
            model3 = logit(cense_d_cov.replace(cense, "_cense_1minus"), data=eligible1_data).fit(disp=0)
            cense_d1 = pd.DataFrame({
                "pC_d1": model3.fittedvalues,
                "id": eligible1_data["id"].values,
                "period": eligible1_data["period"].values,
            })
            
            censor_models["cens_d1"] = process_weight_model(
                model3,
                save_weight_models,
                save_dir,
                "cense_model_d1.pkl",
                "Model for P(cense = 0 | X, previous treatment = 1) for denominator",
                quiet
            )
        else:
            cense_d1 = pd.DataFrame(columns=["id", "period", "pC_d1"])
    
    if pool_cense_n:
        # Fit pooled numerator model
        model2 = logit(cense_n_cov.replace(cense, "_cense_1minus"), data=sw_data).fit(disp=0)
        cense_n = pd.DataFrame({
            "pC_n": model2.fittedvalues,
            "id": sw_data["id"].values,
            "period": sw_data["period"].values,
        })
        
        censor_models["cens_pool_n"] = process_weight_model(
            model2,
            save_weight_models,
            save_dir,
            "cense_model_pool_n.pkl",
            "Model for P(cense = 0 | X) for numerator",
            quiet
        )
    else:
        # Fit separate numerator models
        # eligible0
        eligible0_data = sw_data[sw_data["eligible0"] == 1].copy()
        if len(eligible0_data) > 0:
            model2 = logit(cense_n_cov.replace(cense, "_cense_1minus"), data=eligible0_data).fit(disp=0)
            cense_n0 = pd.DataFrame({
                "pC_n0": model2.fittedvalues,
                "id": eligible0_data["id"].values,
                "period": eligible0_data["period"].values,
            })
            
            censor_models["cens_n0"] = process_weight_model(
                model2,
                save_weight_models,
                save_dir,
                "cense_model_n0.pkl",
                "Model for P(cense = 0 | X, previous treatment = 0) for numerator",
                quiet
            )
        else:
            cense_n0 = pd.DataFrame(columns=["id", "period", "pC_n0"])
        
        # eligible1
        eligible1_data = sw_data[sw_data["eligible1"] == 1].copy()
        if len(eligible1_data) > 0:
            model4 = logit(cense_n_cov.replace(cense, "_cense_1minus"), data=eligible1_data).fit(disp=0)
            cense_n1 = pd.DataFrame({
                "pC_n1": model4.fittedvalues,
                "id": eligible1_data["id"].values,
                "period": eligible1_data["period"].values,
            })
            
            censor_models["cens_n1"] = process_weight_model(
                model4,
                save_weight_models,
                save_dir,
                "cense_model_n1.pkl",
                "Model for P(cense = 0 | X, previous treatment = 1) for numerator",
                quiet
            )
        else:
            cense_n1 = pd.DataFrame(columns=["id", "period", "pC_n1"])
    
    # Combine results
    if pool_cense_d and pool_cense_n:
        sw_data = sw_data.merge(cense_d[["id", "period", "pC_d"]], on=["id", "period"], how="left")
        sw_data = sw_data.merge(cense_n[["id", "period", "pC_n"]], on=["id", "period"], how="left")
    elif not pool_cense_d and not pool_cense_n:
        cense_0 = cense_d0.merge(cense_n0, on=["id", "period"], how="outer")
        cense_1 = cense_d1.merge(cense_n1, on=["id", "period"], how="outer")
        
        sw_data = sw_data.merge(cense_0[["id", "period", "pC_d0", "pC_n0"]], on=["id", "period"], how="left")
        sw_data = sw_data.merge(cense_1[["id", "period", "pC_d1", "pC_n1"]], on=["id", "period"], how="left")
        
        sw_data.loc[sw_data["am_1"] == 0, "pC_n"] = sw_data.loc[sw_data["am_1"] == 0, "pC_n0"]
        sw_data.loc[sw_data["am_1"] == 0, "pC_d"] = sw_data.loc[sw_data["am_1"] == 0, "pC_d0"]
        sw_data.loc[sw_data["am_1"] == 1, "pC_n"] = sw_data.loc[sw_data["am_1"] == 1, "pC_n1"]
        sw_data.loc[sw_data["am_1"] == 1, "pC_d"] = sw_data.loc[sw_data["am_1"] == 1, "pC_d1"]
    elif not pool_cense_d and pool_cense_n:
        sw_data = sw_data.merge(cense_n[["id", "period", "pC_n"]], on=["id", "period"], how="left")
        sw_data = sw_data.merge(cense_d0[["id", "period", "pC_d0"]], on=["id", "period"], how="left")
        sw_data = sw_data.merge(cense_d1[["id", "period", "pC_d1"]], on=["id", "period"], how="left")
        
        sw_data.loc[sw_data["am_1"] == 0, "pC_d"] = sw_data.loc[sw_data["am_1"] == 0, "pC_d0"]
        sw_data.loc[sw_data["am_1"] == 1, "pC_d"] = sw_data.loc[sw_data["am_1"] == 1, "pC_d1"]
    elif pool_cense_d and not pool_cense_n:
        raise ValueError("Check the arguments for pooling censoring models!")
    
    # Clean up temporary column
    if "_cense_1minus" in sw_data.columns:
        sw_data = sw_data.drop(columns=["_cense_1minus"])
    
    return {
        "sw_data": sw_data,
        "censor_models": censor_models,
    }


def process_weight_model(
    model: LogitResults,
    save_weight_models: bool,
    save_dir: Optional[str],
    filename: str,
    description: str,
    quiet: bool
) -> Dict[str, Any]:
    """
    Helper to process weight models.
    
    Maps R's process_weight_model() function.
    """
    quiet_msg(quiet, description)
    quiet_print(quiet, model.summary())
    
    result = {
        "description": description,
        "summary": model.summary().tables[1],  # Coefficients table
        "fit_summary": {
            "aic": model.aic,
            "bic": model.bic,
            "llf": model.llf,
            "nobs": model.nobs,
        },
    }
    
    if save_weight_models and save_dir:
        import os
        import pickle
        path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        result["path"] = path
    
    return result

