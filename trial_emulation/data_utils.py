"""
Data utilities for column selection and data preparation checks.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
from .formula_utils import parse_formula_like


def select_data_cols(data: pd.DataFrame, args: Dict[str, Any]) -> pd.DataFrame:
    """
    Select the required columns from the data and rename.
    
    Maps R's select_data_cols() function.
    
    Args:
        data: DataFrame containing all required columns
        args: Dictionary of arguments from data_preparation
        
    Returns:
        DataFrame with required columns renamed
    """
    # Get column mappings
    id_col = args.get("id", "id")
    period_col = args.get("period", "period")
    treatment_col = args.get("treatment", "treatment")
    outcome_col = args.get("outcome", "outcome")
    eligible_col = args.get("eligible", "eligible")
    
    # Rename columns to canonical names
    rename_map = {
        id_col: "id",
        period_col: "period",
        outcome_col: "outcome",
        eligible_col: "eligible",
        treatment_col: "treatment",
    }
    
    data = data.rename(columns=rename_map)
    
    # Get additional columns needed
    eligible_wts_0 = args.get("eligible_wts_0")
    eligible_wts_1 = args.get("eligible_wts_1")
    cense = args.get("cense")
    where_var = args.get("where_var")
    
    # Extract variables from formulas
    formula_keys = ["outcome_cov", "switch_n_cov", "switch_d_cov", "cense_n_cov", "cense_d_cov"]
    formula_vars = []
    for key in formula_keys:
        if key in args:
            _, vars_list = parse_formula_like(args[key])
            formula_vars.extend(vars_list)
    
    # Collect all additional columns
    additional_cols = []
    if eligible_wts_0 and eligible_wts_0 != "NA":
        additional_cols.append(eligible_wts_0)
    if eligible_wts_1 and eligible_wts_1 != "NA":
        additional_cols.append(eligible_wts_1)
    if cense and cense != "NA":
        additional_cols.append(cense)
    if where_var:
        if isinstance(where_var, list):
            additional_cols.extend(where_var)
        else:
            additional_cols.append(where_var)
    
    additional_cols.extend(formula_vars)
    additional_cols = list(set(additional_cols))  # Remove duplicates
    
    # Rename eligible_wts columns if specified
    if eligible_wts_0 and eligible_wts_0 != "NA" and eligible_wts_0 in data.columns:
        data = data.rename(columns={eligible_wts_0: "eligible_wts_0"})
    if eligible_wts_1 and eligible_wts_1 != "NA" and eligible_wts_1 in data.columns:
        data = data.rename(columns={eligible_wts_1: "eligible_wts_1"})
    
    # Select required columns
    required_cols = ["id", "period", "outcome", "eligible", "treatment"]
    keep_cols = required_cols + [col for col in additional_cols if col in data.columns]
    
    # Remove duplicates while preserving order
    seen = set()
    keep_cols_unique = []
    for col in keep_cols:
        if col not in seen:
            seen.add(col)
            keep_cols_unique.append(col)
    
    data = data[keep_cols_unique].copy()
    
    # Sort by id and period
    data = data.sort_values(["id", "period"]).reset_index(drop=True)
    
    return data


def check_args_data_preparation(
    data: pd.DataFrame,
    id: str = "id",
    period: str = "period",
    treatment: str = "treatment",
    outcome: str = "outcome",
    eligible: str = "eligible",
    model_var: Optional[Union[str, List[str]]] = None,
    outcome_cov: Union[str, List[str]] = "~1",
    estimand_type: str = "ITT",
    switch_n_cov: Union[str, List[str]] = "~1",
    switch_d_cov: Union[str, List[str]] = "~1",
    first_period: Optional[int] = None,
    last_period: Optional[int] = None,
    use_censor_weights: bool = False,
    cense: Optional[str] = None,
    pool_cense: str = "none",
    cense_d_cov: Union[str, List[str]] = "~1",
    cense_n_cov: Union[str, List[str]] = "~1",
    eligible_wts_0: Optional[str] = None,
    eligible_wts_1: Optional[str] = None,
    where_var: Optional[Union[str, List[str]]] = None,
    data_dir: Optional[str] = None,
    save_weight_models: bool = False,
    glm_function: str = "glm",
    chunk_size: int = 500,
    separate_files: bool = False,
    quiet: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Check arguments to data_preparation and convert types.
    
    Maps R's check_args_data_preparation() function.
    
    Returns:
        Dictionary of checked and converted arguments
    """
    # Basic validation
    required_cols = [id, period, treatment, outcome, eligible]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if estimand_type not in ["ITT", "PP", "As-Treated"]:
        raise ValueError(f"estimand_type must be one of 'ITT', 'PP', 'As-Treated', got '{estimand_type}'")
    
    if use_censor_weights and not cense:
        raise ValueError("cense must be specified when use_censor_weights=True")
    
    if pool_cense not in ["none", "both", "numerator"]:
        raise ValueError(f"pool_cense must be one of 'none', 'both', 'numerator', got '{pool_cense}'")
    
    # Check for deprecated arguments
    if "use_weight" in kwargs:
        raise ValueError(
            "Argument 'use_weight' is no longer supported. "
            "Use 'estimand_type' to control weighting behaviour."
        )
    
    # Handle data_dir
    if separate_files or save_weight_models:
        if not data_dir:
            raise ValueError("data_dir must be specified when separate_files=True or save_weight_models=True")
        import os
        if os.path.exists(data_dir):
            # Check for existing trial files
            trial_files = [f for f in os.listdir(data_dir) if f.startswith("trial_") and f.endswith(".csv")]
            if trial_files:
                raise ValueError(
                    f"trial_*.csv files already exist in {data_dir}. "
                    "Remove them or specify a different data_dir."
                )
        else:
            os.makedirs(data_dir, exist_ok=True)
    else:
        data_dir = None
    
    # Convert formulas
    outcome_cov_str, _ = parse_formula_like(outcome_cov)
    switch_n_cov_str, _ = parse_formula_like(switch_n_cov)
    switch_d_cov_str, _ = parse_formula_like(switch_d_cov)
    cense_d_cov_str, _ = parse_formula_like(cense_d_cov)
    cense_n_cov_str, _ = parse_formula_like(cense_n_cov)
    
    # Build keeplist
    _, outcome_vars = parse_formula_like(outcome_cov)
    keeplist = ["id", "trial_period", "followup_time", "outcome", "weight", "treatment"]
    if where_var:
        if isinstance(where_var, list):
            keeplist.extend(where_var)
        else:
            keeplist.append(where_var)
    keeplist.extend(outcome_vars)
    
    args = {
        "id": id,
        "period": period,
        "treatment": treatment,
        "outcome": outcome,
        "eligible": eligible,
        "model_var": model_var,
        "outcome_cov": outcome_cov_str,
        "outcome_cov_vars": outcome_vars,
        "estimand_type": estimand_type,
        "switch_n_cov": switch_n_cov_str,
        "switch_d_cov": switch_d_cov_str,
        "first_period": first_period,
        "last_period": last_period,
        "use_censor_weights": use_censor_weights,
        "cense": cense,
        "pool_cense": pool_cense,
        "cense_d_cov": cense_d_cov_str,
        "cense_n_cov": cense_n_cov_str,
        "eligible_wts_0": eligible_wts_0,
        "eligible_wts_1": eligible_wts_1,
        "where_var": where_var,
        "data_dir": data_dir,
        "save_weight_models": save_weight_models,
        "glm_function": glm_function,
        "chunk_size": chunk_size,
        "separate_files": separate_files,
        "quiet": quiet,
        "keeplist": keeplist,
    }
    
    return args


def check_estimand_data_prep(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check and set defaults based on estimand_type.
    
    Maps R's check_estimand_data_prep() function.
    
    Args:
        args: Dictionary of arguments
        
    Returns:
        Updated arguments dictionary
    """
    estimand_type = args["estimand_type"]
    model_var = args.get("model_var")
    use_censor_weights = args["use_censor_weights"]
    pool_cense = args["pool_cense"]
    
    from .formula_utils import parse_formula_like
    
    if estimand_type == "ITT":
        model_var_str = "assigned_treatment"
        censor_at_switch = False
        use_switch_weights = False
        if use_censor_weights:
            if pool_cense == "none":  # Default for ITT
                pool_cense = "numerator"
            if pool_cense not in ["both", "numerator"]:
                raise ValueError(
                    f"pool_cense must be 'both' or 'numerator' for estimand_type='ITT', got '{pool_cense}'"
                )
    elif estimand_type == "PP":
        model_var_str = "assigned_treatment"
        censor_at_switch = True
        use_switch_weights = True
        if use_censor_weights:
            if pool_cense == "none":  # Default for PP
                pass  # Keep as "none"
            if pool_cense not in ["none", "both", "numerator"]:
                raise ValueError(
                    f"pool_cense must be one of 'none', 'both', 'numerator' for estimand_type='PP', got '{pool_cense}'"
                )
    elif estimand_type == "As-Treated":
        if model_var is None:
            model_var_str = "dose"
        else:
            model_var_str, _ = parse_formula_like(model_var)
        censor_at_switch = False
        use_switch_weights = True
        if use_censor_weights:
            if pool_cense == "none":  # Default for As-Treated
                pass  # Keep as "none"
            if pool_cense not in ["none", "both", "numerator"]:
                raise ValueError(
                    f"pool_cense must be one of 'none', 'both', 'numerator' for estimand_type='As-Treated', got '{pool_cense}'"
                )
    else:
        raise ValueError(f"Unknown estimand_type: {estimand_type}")
    
    args["pool_cense_n"] = pool_cense in ["both", "numerator"]
    args["pool_cense_d"] = pool_cense == "both"
    args["model_var"] = model_var_str
    args["censor_at_switch"] = censor_at_switch
    args["use_switch_weights"] = use_switch_weights
    
    # Add model_var variables to keeplist
    _, model_vars = parse_formula_like(model_var_str)
    keeplist = args.get("keeplist", [])
    keeplist.extend(model_vars)
    args["keeplist"] = list(set(keeplist))  # Remove duplicates
    
    return args

