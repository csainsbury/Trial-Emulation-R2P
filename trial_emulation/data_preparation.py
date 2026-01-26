"""
Data preparation function.

Main entry point for preparing data for the sequence of emulated target trials.
Maps R's data_preparation.R function.
"""

from typing import Optional, Union, List, Dict, Any
import pandas as pd

from .data_utils import (
    check_args_data_preparation,
    check_estimand_data_prep,
    select_data_cols,
)
from .data_manipulation import data_manipulation
from .weighting import weight_func
from .data_extension import data_extension
from .types import TEDataPrep
from .utils import quiet_msg, quiet_line


def data_preparation(
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
    **kwargs,
) -> TEDataPrep:
    """
    Prepare data for the sequence of emulated target trials.

    This function expands observational data in the person-time format (i.e., the 'long' format)
    to emulate a sequence of target trials and also estimates the inverse probability of
    treatment and censoring weights as required.

    Maps R's data_preparation() function.

    Args:
        data: Input DataFrame in long format
        id: Column name for patient identifier
        period: Column name for time period
        treatment: Column name for treatment indicator
        outcome: Column name for outcome indicator
        eligible: Column name for eligibility indicator
        model_var: Variables to include in outcome model (formula or list)
        outcome_cov: Covariates for outcome model (formula or list)
        estimand_type: Type of estimand ("ITT", "PP", or "As-Treated")
        switch_n_cov: Numerator covariates for switching weights
        switch_d_cov: Denominator covariates for switching weights
        first_period: First period to include (None = use min)
        last_period: Last period to include (None = use max)
        use_censor_weights: Use censoring weights?
        cense: Column name for censoring indicator
        pool_cense: Pooling strategy for censoring ("none", "both", "numerator")
        cense_d_cov: Denominator covariates for censoring weights
        cense_n_cov: Numerator covariates for censoring weights
        eligible_wts_0: Eligibility variable for weights when previous treatment = 0
        eligible_wts_1: Eligibility variable for weights when previous treatment = 1
        where_var: Variables used in where conditions
        data_dir: Directory to save models/data
        save_weight_models: Save weight model objects?
        glm_function: GLM function to use (for compatibility)
        chunk_size: Number of individuals per chunk when separate_files=True
        separate_files: Save expanded data in separate CSV files?
        quiet: Suppress progress messages?
        **kwargs: Additional arguments passed to weight functions

    Returns:
        TEDataPrep object containing:
            - data: Expanded DataFrame or list of file paths
            - min_period: First trial period
            - max_period: Last trial period
            - N: Total number of observations
            - data_template: Zero-row DataFrame template
            - switch_models: Dictionary of switching weight model summaries
            - censor_models: Dictionary of censoring weight model summaries
            - parameters: Dictionary of parameters used
    """
    # Check and convert arguments
    args = check_args_data_preparation(
        data=data,
        id=id,
        period=period,
        treatment=treatment,
        outcome=outcome,
        eligible=eligible,
        model_var=model_var,
        outcome_cov=outcome_cov,
        estimand_type=estimand_type,
        switch_n_cov=switch_n_cov,
        switch_d_cov=switch_d_cov,
        first_period=first_period,
        last_period=last_period,
        use_censor_weights=use_censor_weights,
        cense=cense,
        pool_cense=pool_cense,
        cense_d_cov=cense_d_cov,
        cense_n_cov=cense_n_cov,
        eligible_wts_0=eligible_wts_0,
        eligible_wts_1=eligible_wts_1,
        where_var=where_var,
        data_dir=data_dir,
        save_weight_models=save_weight_models,
        glm_function=glm_function,
        chunk_size=chunk_size,
        separate_files=separate_files,
        quiet=quiet,
        **kwargs,
    )

    args = check_estimand_data_prep(args)

    # Select and rename columns
    data = select_data_cols(data, args)

    quiet_msg(quiet, "Starting data manipulation")
    data = data_manipulation(data, use_censor=args["censor_at_switch"])

    # Calculate weights if needed
    if args["use_censor_weights"] or args["use_switch_weights"]:
        weight_result = weight_func(
            sw_data=data,
            use_switch_weights=args["use_switch_weights"],
            use_censor_weights=args["use_censor_weights"],
            switch_n_cov=args["switch_n_cov"],
            switch_d_cov=args["switch_d_cov"],
            eligible_wts_0=args.get("eligible_wts_0"),
            eligible_wts_1=args.get("eligible_wts_1"),
            cense=args.get("cense"),
            pool_cense_n=args["pool_cense_n"],
            pool_cense_d=args["pool_cense_d"],
            cense_d_cov=args["cense_d_cov"],
            cense_n_cov=args["cense_n_cov"],
            save_weight_models=args["save_weight_models"],
            data_dir=args.get("data_dir"),
            quiet=args["quiet"],
            glm_function=args["glm_function"],
            **kwargs,
        )
        data = weight_result["data"]
        switch_models = weight_result.get("switch_models")
        censor_models = weight_result.get("censor_models")
    else:
        data["wt"] = 1.0
        switch_models = None
        censor_models = None

    quiet_msg(quiet, "Starting data extension")
    result = data_extension(
        data=data,
        keeplist=args["keeplist"],
        outcome_cov=args["outcome_cov"],
        first_period=args["first_period"],
        last_period=args["last_period"],
        censor_at_switch=args["censor_at_switch"],
        where_var=args.get("where_var"),
        data_dir=args.get("data_dir"),
        separate_files=args["separate_files"],
        chunk_size=args["chunk_size"],
    )

    quiet_msg(quiet, "Summary of extended data:")
    quiet_msg(quiet, f"Number of observations: {result['N']}")
    quiet_line(quiet)

    # Create result object
    prep_result = TEDataPrep(
        data=result["data"],
        min_period=result["min_period"],
        max_period=result["max_period"],
        N=result["N"],
        data_template=result["data_template"],
        switch_models=switch_models,
        censor_models=censor_models,
        parameters=args,
    )

    return prep_result
