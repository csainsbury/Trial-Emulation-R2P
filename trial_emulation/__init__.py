"""
TrialEmulation: Python port of the R TrialEmulation package.

This package implements target trial emulation methods to apply randomized
clinical trial design and analysis in an observational setting. Using
marginal structural models, it can estimate intention-to-treat and
per-protocol effects in emulated trials using electronic health records.

The main workflow consists of:
1. Data preparation (including handling of eligibility, treatment/outcome/censoring, covariates)
2. Inverse-probability weights for treatment switching and censoring
3. Expansion of the longitudinal data into a stack of emulated trials
4. Fitting marginal structural models (MSMs) for time-to-event outcomes
5. Estimating ITT, PP, and As-Treated estimands
"""

__version__ = "0.0.5.2"

from .data_preparation import data_preparation
from .msm import trial_msm, fit_msm
from .sampling import case_control_sampling_trials
from .types import TEDataPrep

__all__ = [
    "data_preparation",
    "trial_msm",
    "fit_msm",
    "case_control_sampling_trials",
    "TEDataPrep",
]
