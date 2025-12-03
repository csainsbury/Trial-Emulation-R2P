"""
Case-control sampling functions.

Performs case-control sampling of expanded data.
Maps R's sampling.R functions.
"""

from typing import Optional, List, Union
import pandas as pd
import numpy as np

from .types import TEDataPrep


def case_control_sampling_trials(
    data_prep: TEDataPrep,
    p_control: Union[float, List[float]],
    subset_condition: Optional[str] = None,
    sort: bool = False
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Case-control sampling of expanded data for the sequence of emulated trials.
    
    Perform case-control sampling of expanded data to create a data set of reduced size
    and calculate sampling weights to be used in trial_msm().
    
    Maps R's case_control_sampling_trials() function.
    
    Args:
        data_prep: Result from data_preparation()
        p_control: Control sampling probability (or list of probabilities)
        subset_condition: Expression used to subset the trial data before sampling
        sort: Sort data before sampling for reproducibility
        
    Returns:
        DataFrame or list of DataFrames (if multiple p_control values)
        with additional column 'sample_weight' containing sampling weights
    """
    if isinstance(p_control, (int, float)):
        p_control = [p_control]
    
    if isinstance(data_prep.data, list):
        # Separate files
        trial_samples = _sample_data_prep_sep(
            data_prep, p_control, subset_condition, sort
        )
    else:
        # Single DataFrame
        trial_samples = _sample_data_prep_dt(
            data_prep, p_control, subset_condition, sort
        )
    
    if len(p_control) > 1:
        # Return list of DataFrames, one per p_control value
        result_list = []
        for sample_id, sample_data in enumerate(trial_samples):
            sample_data["sample_id"] = sample_id
            result_list.append(sample_data)
        return result_list
    else:
        # Return single DataFrame
        result = pd.concat(trial_samples, ignore_index=True)
        return result


def _sample_data_prep_sep(
    data_prep: TEDataPrep,
    p_control: List[float],
    subset_condition: Optional[str],
    sort: bool
) -> List[pd.DataFrame]:
    """
    Sample utility for data_preparation result with separate_files = True.
    """
    import os
    
    trial_files = data_prep.data
    existing_files = [f for f in trial_files if os.path.exists(f)]
    
    if not existing_files:
        raise ValueError("No trial files could be found.")
    
    template = data_prep.data_template
    
    all_samples = []
    for trial_file in existing_files:
        period_data = pd.read_csv(trial_file)
        samples = _sample_from_period(
            period_data, p_control, subset_condition, sort
        )
        all_samples.append(samples)
    
    return all_samples


def _sample_data_prep_dt(
    data_prep: TEDataPrep,
    p_control: List[float],
    subset_condition: Optional[str],
    sort: bool
) -> List[pd.DataFrame]:
    """
    Sample utility for data_preparation result with separate_files = False.
    """
    periods = sorted(data_prep.data["trial_period"].unique())
    
    all_samples = []
    for period in periods:
        period_data = data_prep.data[data_prep.data["trial_period"] == period].copy()
        samples = _sample_from_period(
            period_data, p_control, subset_condition, sort
        )
        all_samples.append(samples)
    
    return all_samples


def _sample_from_period(
    period_data: pd.DataFrame,
    p_control: List[float],
    subset_condition: Optional[str],
    sort: bool
) -> pd.DataFrame:
    """
    Sample from period.
    
    Break period data into follow-up periods and sample.
    """
    if subset_condition:
        period_data = period_data.query(subset_condition).copy()
    
    if sort:
        period_data = period_data.sort_values(["followup_time", "id"]).reset_index(drop=True)
    
    # Split by follow-up time
    followup_groups = period_data.groupby("followup_time")
    
    all_samples = []
    for p in p_control:
        sample_list = []
        for followup_time, group in followup_groups:
            sampled = _do_sampling(group, p_control=p)
            sample_list.append(sampled)
        
        if sample_list:
            combined = pd.concat(sample_list, ignore_index=True)
            if len(p_control) > 1:
                combined["sample_id"] = p_control.index(p)
            all_samples.append(combined)
    
    if len(p_control) > 1:
        return pd.concat(all_samples, ignore_index=True)
    else:
        return all_samples[0] if all_samples else pd.DataFrame()


def _do_sampling(data: pd.DataFrame, p_control: float = 0.01) -> pd.DataFrame:
    """
    Sample controls and cases from data subset by trial and follow-up time.
    
    For the given data, find cases and sample a number of controls.
    """
    # Cases occurred at each period and follow-up visit
    cases_mask = data["outcome"] == 1
    cases = data[cases_mask].copy()
    ncase = len(cases)
    
    # Controls (still under follow-up and events haven't occurred)
    controls_mask = data["outcome"] == 0
    controls = data[controls_mask].copy()
    n_controls = len(controls)
    
    # Sample controls
    n_sample = np.random.binomial(1, p_control, n_controls).sum()
    if n_sample > 0:
        controlselect = controls.sample(n=n_sample, random_state=None)
    else:
        controlselect = pd.DataFrame(columns=controls.columns)
    
    # Combine cases and sampled controls
    dataall = pd.concat([cases, controlselect], ignore_index=True)
    
    # Add sample weights
    dataall["sample_weight"] = np.concatenate([
        np.ones(ncase),  # Cases get weight 1
        np.full(len(controlselect), 1.0 / p_control)  # Controls get weight 1/p_control
    ])
    
    return dataall

