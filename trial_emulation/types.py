"""
Type definitions and result classes for TrialEmulation.
"""

from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class TEDataPrep:
    """
    Result object from data_preparation().
    
    Mirrors the R TE_data_prep class structure.
    """
    data: Union[pd.DataFrame, List[str]]  # DataFrame or list of file paths
    min_period: int
    max_period: int
    N: int
    data_template: pd.DataFrame
    switch_models: Optional[Dict[str, Any]] = None
    censor_models: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set class name for compatibility."""
        self.__class__.__name__ = "TE_data_prep"


@dataclass
class TEMSM:
    """
    Result object from trial_msm().
    
    Mirrors the R TE_msm class structure.
    """
    model: Any  # statsmodels GLM model
    robust: Dict[str, Any]  # Robust variance results
    args: Dict[str, Any]  # Parameters used
    
    def __post_init__(self):
        """Set class name for compatibility."""
        self.__class__.__name__ = "TE_msm"

