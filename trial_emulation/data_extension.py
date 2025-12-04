"""
Data extension functions.

Expands the longitudinal data into a sequence of trials.
Maps R's data_extension.R functions.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import os

from .utils import quiet_msg


def data_extension(
    data: pd.DataFrame,
    keeplist: List[str],
    outcome_cov: Optional[str] = None,
    first_period: Optional[int] = None,
    last_period: Optional[int] = None,
    censor_at_switch: bool = False,
    where_var: Optional[Union[str, List[str]]] = None,
    data_dir: Optional[str] = None,
    separate_files: bool = False,
    chunk_size: int = 200
) -> Dict[str, Any]:
    """
    Data extension function.
    
    Expands the longitudinal data into a sequence of trials.
    Maps R's data_extension() function.
    
    Args:
        data: DataFrame to expand
        keeplist: List of variable names to keep in final output
        outcome_cov: Formula for covariate adjustment (for extracting variables)
        first_period: First period value to start expanding
        last_period: Last period value to expand
        censor_at_switch: Use censoring for per-protocol analysis
        where_var: Variables used in where conditions
        data_dir: Directory to save data if separate_files=True
        separate_files: Save expanded data in separate CSV files for each trial
        chunk_size: Number of IDs to expand in each chunk
        
    Returns:
        Dictionary with keys:
            - data: Expanded DataFrame or list of file paths
            - min_period: First trial period
            - max_period: Last trial period
            - N: Total number of observations
            - data_template: Zero-row DataFrame with columns
    """
    # Input validation
    if data.empty:
        raise ValueError("Input data is empty")
    
    required_cols = ["id", "period", "treatment", "outcome", "eligible"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not keeplist:
        raise ValueError("keeplist cannot be empty")
    
    if separate_files:
        if not data_dir:
            raise ValueError("data_dir must be specified when separate_files=True")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
    
    if first_period is None:
        first_period = int(data["period"].min())
    if last_period is None:
        last_period = int(data["period"].max())
    
    # Extract outcome covariate variables
    if outcome_cov:
        from .formula_utils import parse_formula_like
        _, outcomeCov_var = parse_formula_like(outcome_cov)
    else:
        outcomeCov_var = []
    
    if separate_files:
        all_ids = data["id"].unique()
        n_chunks = int(np.ceil(len(all_ids) / chunk_size))
        ids_split = np.array_split(all_ids, n_chunks)
        
        N = 0
        switch_data_list = []
        
        for ids_chunk in ids_split:
            chunk_data = data[data["id"].isin(ids_chunk)].copy()
            switch_data = expand(
                sw_data=chunk_data,
                outcomeCov_var=outcomeCov_var,
                where_var=where_var,
                use_censor=censor_at_switch,
                minperiod=first_period,
                maxperiod=last_period,
                keeplist=keeplist
            )
            
            N += len(switch_data)
            
            # Append to trial files
            for p in switch_data["trial_period"].unique():
                file_p = os.path.join(data_dir, f"trial_{p}.csv")
                trial_data = switch_data[switch_data["trial_period"] == p]
                trial_data.to_csv(file_p, mode="a", header=not os.path.exists(file_p), index=False)
            
            switch_data_list.append(switch_data)
        
        # Get list of created files
        files = [
            os.path.join(data_dir, f"trial_{p}.csv")
            for p in range(first_period, last_period + 1)
            if os.path.exists(os.path.join(data_dir, f"trial_{p}.csv"))
        ]
        
        # Create template from first chunk
        if switch_data_list:
            data_template = switch_data_list[0].iloc[:0].copy()
        else:
            data_template = pd.DataFrame(columns=keeplist)
        
        return {
            "data": files,
            "min_period": first_period,
            "max_period": last_period,
            "N": N,
            "data_template": data_template,
        }
    else:
        switch_data = expand(
            sw_data=data,
            outcomeCov_var=outcomeCov_var,
            where_var=where_var,
            use_censor=censor_at_switch,
            minperiod=first_period,
            maxperiod=last_period,
            keeplist=keeplist
        )
        
        return {
            "data": switch_data,
            "min_period": first_period,
            "max_period": last_period,
            "N": len(switch_data),
            "data_template": switch_data.iloc[:0].copy(),
        }


def expand(
    sw_data: pd.DataFrame,
    outcomeCov_var: List[str],
    where_var: Optional[Union[str, List[str]]],
    use_censor: bool,
    maxperiod: int,
    minperiod: int,
    keeplist: List[str]
) -> pd.DataFrame:
    """
    Expand function.
    
    Performs the data expansion for a given dataset.
    Maps R's expand() function.
    
    Args:
        sw_data: DataFrame to expand
        outcomeCov_var: List of baseline variables used in final model
        where_var: Variables used in where conditions
        use_censor: Use censoring for per-protocol analysis
        maxperiod: Maximum period
        minperiod: Minimum period
        keeplist: List of variable names to keep
        
    Returns:
        Expanded DataFrame
    """
    # Create temporary data structure
    temp_data = pd.DataFrame({
        "id": sw_data["id"].values,
        "period": sw_data["period"].values,
        "switch": sw_data.get("switch", 0).values,
    })
    
    # Initialize columns
    temp_data["wtprod"] = 1.0
    temp_data["elgcount"] = 0.0
    temp_data["expand"] = 0.0
    temp_data["treat"] = 0.0
    temp_data["dosesum"] = 0.0
    
    # Mark rows to expand
    expand_mask = (
        (sw_data["eligible"] == 1) &
        sw_data["treatment"].notna() &
        (sw_data["period"] >= minperiod) &
        (sw_data["period"] <= maxperiod)
    )
    temp_data.loc[expand_mask, "expand"] = 1
    
    # Calculate weight0 (cumulative product of weights)
    sw_data["weight0"] = 1.0
    sw_data.loc[sw_data["first"], "weight0"] = 1.0
    sw_data["weight0"] = sw_data.groupby("id")["wt"].cumprod()
    
    temp_data["wtprod"] = sw_data["weight0"].values
    temp_data["treat"] = sw_data["treatment"].values
    temp_data["dosesum"] = sw_data.get("cumA", 0).values
    temp_data["elgcount"] = sw_data["eligible"].values
    
    # Calculate init (initial treatment at eligibility)
    temp_data["init"] = np.nan
    eligible_mask = sw_data["eligible"] == 1
    temp_data.loc[eligible_mask, "init"] = sw_data.loc[eligible_mask, "treatment"].values
    
    # Forward fill init within each ID
    temp_data["init"] = temp_data.groupby("id")["init"].ffill()
    
    # Copy outcome covariates
    if outcomeCov_var:
        for var in outcomeCov_var:
            if var in sw_data.columns:
                temp_data[var] = sw_data[var].values
    
    # Copy where variables
    if where_var:
        if isinstance(where_var, list):
            for var in where_var:
                if var in sw_data.columns:
                    temp_data[var] = sw_data[var].values
        else:
            if where_var in sw_data.columns:
                temp_data[where_var] = sw_data[where_var].values
    
    # Create expansion index
    # For each row, create period+1 rows (one for each trial_period from 0 to period)
    expand_index_list = []
    trial_period_list = []
    for idx, row in sw_data.iterrows():
        period_val = int(row["period"])
        for trial_period in range(period_val + 1):
            expand_index_list.append(idx)
            trial_period_list.append(trial_period)
    
    expand_index = pd.Series(expand_index_list)
    
    # Create expanded data using iloc for proper indexing
    expand_idx_array = np.array(expand_index_list)
    switch_data = pd.DataFrame({
        "id": sw_data.iloc[expand_idx_array]["id"].values,
        "period_new": sw_data.iloc[expand_idx_array]["period"].values,
        "cumA_new": sw_data.iloc[expand_idx_array]["cumA"].values if "cumA" in sw_data.columns else np.zeros(len(expand_index_list)),
        "treatment_new": sw_data.iloc[expand_idx_array]["treatment"].values,
        "outcome_new": sw_data.iloc[expand_idx_array]["outcome"].values,
        "time_of_event": sw_data.iloc[expand_idx_array]["time_of_event"].values,
        "weight0": sw_data.iloc[expand_idx_array]["weight0"].values,
        "trial_period": trial_period_list,
    })
    
    if use_censor:
        switch_data["switch_new"] = sw_data.iloc[expand_idx_array]["switch"].values
    else:
        switch_data["switch_new"] = 0
    
    switch_data["index"] = range(len(switch_data))
    
    # Merge with temp_data
    merge_cols = ["id", "period", "wtprod", "expand", "treat", "dosesum", "elgcount", "init"]
    if outcomeCov_var:
        merge_cols.extend([var for var in outcomeCov_var if var in temp_data.columns])
    if where_var:
        if isinstance(where_var, list):
            merge_cols.extend([var for var in where_var if var in temp_data.columns])
        elif where_var in temp_data.columns:
            merge_cols.append(where_var)
    
    switch_data = switch_data.merge(
        temp_data[merge_cols],
        left_on=["id", "trial_period"],
        right_on=["id", "period"],
        how="left"
    )
    
    switch_data = switch_data.sort_values("index").reset_index(drop=True)
    
    # Calculate followup_time
    switch_data["followup_time"] = switch_data["period_new"] - switch_data["trial_period"]
    
    # Calculate dose if in keeplist
    if "dose" in keeplist:
        switch_data["dose"] = switch_data["cumA_new"] - switch_data["dosesum"] + switch_data["treat"]
    
    # Set switch_new to 0 at followup_time == 0
    switch_data.loc[switch_data["followup_time"] == 0, "switch_new"] = 0
    
    # Expand until switch - update expand flag based on switch_new
    # Only update for groups where expand was originally 1
    expand_results = []
    for (id_val, trial_period_val), group in switch_data.groupby(["id", "trial_period"]):
        if group["expand"].iloc[0] == 1:  # Only process if originally marked for expansion
            expand_flag = expand_until_switch(group["switch_new"].values, len(group))
            expand_results.extend(expand_flag)
        else:
            expand_results.extend([0] * len(group))
    
    # Update expand column
    switch_data["expand"] = expand_results
    
    # Calculate weight
    switch_data["weight"] = switch_data["weight0"] / switch_data["wtprod"]
    
    # Calculate case/outcome
    switch_data["case"] = 0
    if not use_censor:
        switch_data.loc[
            (switch_data["time_of_event"] == switch_data["period_new"]) &
            (switch_data["outcome_new"] == 1),
            "case"
        ] = 1
    else:
        switch_data.loc[switch_data["switch_new"] == 1, "case"] = np.nan
        switch_data.loc[
            (switch_data["switch_new"] == 0) &
            (switch_data["time_of_event"] == switch_data["period_new"]) &
            (switch_data["outcome_new"] == 1),
            "case"
        ] = 1
    
    # Rename columns
    switch_data = switch_data.rename(columns={"case": "outcome", "init": "assigned_treatment", "treatment_new": "treatment"})
    
    # Filter to expanded rows
    switch_data = switch_data[switch_data["expand"] == 1].copy()
    
    # Select keeplist columns
    available_cols = [col for col in keeplist if col in switch_data.columns]
    switch_data = switch_data[available_cols].copy()
    
    return switch_data


def expand_until_switch(s: np.ndarray, n: int) -> np.ndarray:
    """
    Check expand flag after treatment switch.
    
    Check if patients have switched treatment in eligible trials
    and set expand = 0 after first switch.
    
    Args:
        s: Numeric vector where 1 indicates a treatment switch
        n: Length of s
        
    Returns:
        Vector of indicator values up until first switch
    """
    result = np.ones(n, dtype=int)
    first_switch_idx = np.where(s == 1)[0]
    
    if len(first_switch_idx) > 0:
        first_switch = first_switch_idx[0]
        result[first_switch:] = 0
    
    return result

