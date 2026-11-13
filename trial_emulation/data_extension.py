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
    chunk_size: int = 200,
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
                keeplist=keeplist,
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
            keeplist=keeplist,
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
    keeplist: List[str],
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
    sw_data = sw_data.reset_index(drop=True)
    n_rows = len(sw_data)
    periods = sw_data["period"].to_numpy()
    eligible = sw_data["eligible"].to_numpy()
    treatment = sw_data["treatment"].to_numpy(dtype=float)

    # Per-row trial-start attributes (R: temp_data). A row can start a trial
    # when the patient is eligible, treatment is observed and the period lies
    # within [minperiod, maxperiod].
    temp_data = pd.DataFrame(
        {
            "id": sw_data["id"].to_numpy(),
            "period": periods,
            "expand": (
                (eligible == 1)
                & ~np.isnan(treatment)
                & (periods >= minperiod)
                & (periods <= maxperiod)
            ).astype(int),
        }
    )

    # Cumulative product of the per-period weights within patient (weight0).
    sw_data["weight0"] = sw_data.groupby("id", sort=False)["wt"].cumprod()
    temp_data["wtprod"] = sw_data["weight0"].to_numpy()
    temp_data["treat"] = treatment
    temp_data["dosesum"] = sw_data["cumA"].to_numpy(dtype=float)
    temp_data["elgcount"] = eligible

    # Assigned treatment: treatment at an eligible period; for ineligible
    # periods R carries the previous row's treatment (such rows never start a
    # trial, so this only matters for internal consistency).
    previous_treatment = np.concatenate([[np.nan], treatment[:-1]])
    temp_data["init"] = np.where(eligible == 1, treatment, previous_treatment)

    extra_vars: List[str] = []
    for var in list(outcomeCov_var or []) + _as_list(where_var):
        if var in sw_data.columns and var not in extra_vars and var not in temp_data.columns:
            extra_vars.append(var)
            temp_data[var] = sw_data[var].to_numpy()

    # Expand every observed row into trials starting at periods 0..period
    # (R: rep(seq_len(nrow), period + 1) and sequence(period + 1, from = 0)).
    n_copies = periods.astype(np.int64) + 1
    expand_index = np.repeat(np.arange(n_rows), n_copies)
    offsets = np.repeat(np.cumsum(n_copies) - n_copies, n_copies)
    trial_period = np.arange(len(expand_index)) - offsets

    switch_data = pd.DataFrame(
        {
            "id": sw_data["id"].to_numpy()[expand_index],
            "period_new": periods[expand_index],
            "cumA_new": sw_data["cumA"].to_numpy(dtype=float)[expand_index],
            "treatment_new": treatment[expand_index],
            "switch_new": (sw_data["switch"].to_numpy()[expand_index] if use_censor else 0),
            "outcome_new": sw_data["outcome"].to_numpy(dtype=float)[expand_index],
            "time_of_event": sw_data["time_of_event"].to_numpy(dtype=float)[expand_index],
            "weight0": sw_data["weight0"].to_numpy(dtype=float)[expand_index],
            "trial_period": trial_period,
            "index": np.arange(len(expand_index)),
        }
    )

    # Attach trial-start attributes. R joins switch_data[temp_data, on =
    # (id, trial_period = period)], which keeps only (patient, trial) pairs
    # whose trial period is an observed period of that patient.
    switch_data = switch_data.merge(
        temp_data.rename(columns={"period": "trial_period"}),
        on=["id", "trial_period"],
        how="inner",
        validate="many_to_one",
    )
    switch_data = switch_data.sort_values("index", kind="stable").reset_index(drop=True)

    switch_data["followup_time"] = switch_data["period_new"] - switch_data["trial_period"]

    if "dose" in keeplist:
        switch_data["dose"] = (
            switch_data["cumA_new"] - switch_data["dosesum"] + switch_data["treat"]
        )

    switch_data.loc[switch_data["followup_time"] == 0, "switch_new"] = 0

    # Within each trial that may expand, keep follow-up up to (not including)
    # the first treatment switch (R: expand_until_switch by id, trial_period).
    # Rows are in index order, i.e. increasing period within each trial.
    switched_so_far = switch_data.groupby(["id", "trial_period"], sort=False)["switch_new"].cumsum()
    switch_data["expand"] = np.where((switch_data["expand"] == 1) & (switched_so_far == 0), 1, 0)

    switch_data["weight"] = switch_data["weight0"] / switch_data["wtprod"]

    event_now = (switch_data["time_of_event"] == switch_data["period_new"]) & (
        switch_data["outcome_new"] == 1
    )
    if not use_censor:
        case = np.where(event_now, 1.0, 0.0)
    else:
        case = np.where(
            switch_data["switch_new"] == 1,
            np.nan,
            np.where(event_now, 1.0, 0.0),
        )
    switch_data["case"] = case

    switch_data = switch_data.rename(
        columns={"case": "outcome", "init": "assigned_treatment", "treatment_new": "treatment"}
    )

    switch_data = switch_data[switch_data["expand"] == 1]
    missing = [col for col in keeplist if col not in switch_data.columns]
    if missing:
        raise ValueError(f"Variables not available in expanded data: {missing}")
    return switch_data[list(keeplist)].reset_index(drop=True)


def _as_list(x: Optional[Union[str, List[str]]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [v for v in x if v is not None]


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
