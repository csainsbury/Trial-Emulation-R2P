"""
Example usage of trial_emulation package.

This script demonstrates basic usage of the trial emulation workflow.
"""

import pandas as pd
import numpy as np
import trial_emulation as te

# Create example data
np.random.seed(123)
n_ids = 100
n_periods = 10

data_list = []
for id_val in range(1, n_ids + 1):
    for period in range(1, n_periods + 1):
        # Simulate treatment, eligibility, outcome
        eligible = np.random.binomial(1, 0.8) if period >= 2 else 0
        treatment = np.random.binomial(1, 0.5) if eligible else 0
        outcome = np.random.binomial(1, 0.1 + 0.05 * treatment)
        
        data_list.append({
            "id": id_val,
            "period": period,
            "treatment": treatment,
            "outcome": outcome,
            "eligible": eligible,
            "age": np.random.normal(50, 10),
            "sex": np.random.binomial(1, 0.5),
        })

data = pd.DataFrame(data_list)

print("Example data:")
print(data.head())
print(f"\nTotal rows: {len(data)}")

# Prepare data for trial emulation
print("\n=== Data Preparation ===")
prep = te.data_preparation(
    data=data,
    id="id",
    period="period",
    treatment="treatment",
    outcome="outcome",
    eligible="eligible",
    estimand_type="ITT",
    outcome_cov=["age", "sex"],
    use_censor_weights=False,
    quiet=False,
)

print(f"\nExpanded data summary:")
print(f"  Number of observations: {prep.N}")
print(f"  Trial periods: {prep.min_period} to {prep.max_period}")

# Fit marginal structural model
print("\n=== Fitting MSM ===")
msm = te.trial_msm(
    data=prep,
    outcome_cov=["age", "sex"],
    estimand_type="ITT",
    quiet=False,
)

print("\n=== Results ===")
print("\nRobust summary:")
print(msm.robust["summary"])

print("\nModel coefficients:")
print(msm.model.params)

