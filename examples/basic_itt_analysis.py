"""
Basic Intention-to-Treat (ITT) Analysis Example

This example demonstrates a simple ITT analysis using TrialEmulation.
It generates synthetic data and walks through the complete workflow.
"""

import numpy as np
import pandas as pd
import trial_emulation as te

print("=" * 60)
print("TrialEmulation: Basic ITT Analysis Example")
print("=" * 60)

# Step 1: Generate synthetic data
print("\n1. Generating synthetic data...")
np.random.seed(42)

n_patients = 100
n_periods = 8

data_list = []
for patient_id in range(1, n_patients + 1):
    age = np.random.randint(45, 75)
    baseline_risk = np.random.random()

    treated = False
    has_outcome = False

    for period in range(n_periods):
        if has_outcome:
            continue

        # Treatment initiation (simplified)
        if not treated and period > 0:
            treatment_prob = 0.15
            treated = np.random.random() < treatment_prob

        treatment = 1 if treated else 0

        # Outcome probability (treatment reduces risk)
        outcome_prob = 0.05 + 0.01 * period
        if treatment:
            outcome_prob *= 0.6  # 40% risk reduction
        outcome_prob *= 1 + baseline_risk

        outcome = 1 if np.random.random() < outcome_prob else 0
        if outcome:
            has_outcome = True

        data_list.append(
            {
                "id": patient_id,
                "period": period,
                "treatment": treatment,
                "outcome": outcome,
                "censored": outcome,
                "eligible": 1,
                "age": age,
                "baseline_risk": baseline_risk,
            }
        )

data = pd.DataFrame(data_list)

print(f"   Created dataset with {len(data)} person-periods")
print(f"   {data['id'].nunique()} patients, {data['period'].nunique()} periods")
print(f"   Treatment initiations: {(data.groupby('id')['treatment'].max() == 1).sum()}")
print(f"   Outcomes observed: {data['outcome'].sum()}")

# Step 2: Prepare data for trial emulation
print("\n2. Preparing data for trial emulation...")
print("   Expanding data into sequence of emulated trials...")

prep = te.data_preparation(
    data=data,
    id="id",
    period="period",
    treatment="treatment",
    outcome="outcome",
    eligible="eligible",
    estimand_type="ITT",
    outcome_cov=["age", "baseline_risk"],
    use_censor_weights=False,
    quiet=True,
)

print(f"   Expanded to {prep.N} observations")
print(f"   Trial periods: {prep.min_period} to {prep.max_period}")

# Step 3: Fit marginal structural model
print("\n3. Fitting marginal structural model...")
print("   Using weighted pooled logistic regression...")

msm = te.trial_msm(data=prep, outcome_cov=["age"], estimand_type="ITT", quiet=True)

print("   Model fitted successfully")

# Step 4: Extract and display results
print("\n4. Results:")
print("-" * 60)

# Get treatment effect
treatment_coef = msm.model.params["assigned_treatment"]
robust_se = msm.robust["bse"]["assigned_treatment"]
z_score = treatment_coef / robust_se
p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if "stats" in dir() else None

# Convert log OR to OR
odds_ratio = np.exp(treatment_coef)
or_lower = np.exp(treatment_coef - 1.96 * robust_se)
or_upper = np.exp(treatment_coef + 1.96 * robust_se)

print(f"\nTreatment Effect (Intention-to-Treat):")
print(f"  Log Odds Ratio: {treatment_coef:.4f} (SE: {robust_se:.4f})")
print(f"  Odds Ratio: {odds_ratio:.4f} (95% CI: [{or_lower:.4f}, {or_upper:.4f}])")

if treatment_coef < 0:
    risk_reduction = (1 - odds_ratio) * 100
    print(f"  Interpretation: {risk_reduction:.1f}% relative risk reduction")
else:
    risk_increase = (odds_ratio - 1) * 100
    print(f"  Interpretation: {risk_increase:.1f}% relative risk increase")

print("\nFull Model Summary (with robust standard errors):")
print("-" * 60)
print(msm.robust["summary"])

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)

print("\nNotes:")
print("- This analysis uses an intention-to-treat (ITT) estimand")
print("- ITT estimates the effect of treatment assignment at baseline")
print("- Robust standard errors account for clustering by patient ID")
print("- For real data, consider adding censoring weights if needed")
