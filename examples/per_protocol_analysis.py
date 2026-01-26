"""
Per-Protocol Analysis with Inverse Probability Weighting

This example demonstrates a per-protocol analysis that accounts for
treatment switching and censoring using inverse probability weights.
"""

import numpy as np
import pandas as pd
import trial_emulation as te
from scipy import stats

print("=" * 60)
print("TrialEmulation: Per-Protocol Analysis Example")
print("=" * 60)

# Step 1: Generate synthetic data with treatment switching
print("\n1. Generating synthetic data with treatment switching...")
np.random.seed(123)

n_patients = 150
n_periods = 10

data_list = []
for patient_id in range(1, n_patients + 1):
    age = np.random.randint(40, 80)
    comorbidity_score = np.random.randint(0, 5)

    treatment_status = 0  # Start untreated
    has_outcome = False

    for period in range(n_periods):
        if has_outcome:
            continue

        # Treatment initiation or discontinuation
        if treatment_status == 0 and period > 0:
            # Probability of starting treatment
            start_prob = 0.10 + 0.01 * comorbidity_score
            if np.random.random() < start_prob:
                treatment_status = 1

        elif treatment_status == 1:
            # Probability of discontinuing treatment
            stop_prob = 0.05
            if np.random.random() < stop_prob:
                treatment_status = 0

        treatment = treatment_status

        # Outcome probability
        hazard = 0.03 + 0.005 * period + 0.01 * comorbidity_score / 5
        if treatment:
            hazard *= 0.5  # 50% hazard reduction with treatment

        outcome = 1 if np.random.random() < hazard else 0
        if outcome:
            has_outcome = True

        data_list.append({
            'id': patient_id,
            'period': period,
            'treatment': treatment,
            'outcome': outcome,
            'censored': outcome,  # Censored at outcome
            'eligible': 1,
            'age': age,
            'comorbidity_score': comorbidity_score
        })

data = pd.DataFrame(data_list)

print(f"   Created dataset with {len(data)} person-periods")
print(f"   {data['id'].nunique()} patients across {data['period'].nunique()} periods")

# Count treatment switches
switches = data.groupby('id')['treatment'].apply(lambda x: (x.diff() != 0).sum()).sum()
print(f"   Treatment switches observed: {switches}")
print(f"   Outcomes observed: {data['outcome'].sum()}")

# Step 2: Prepare data with per-protocol estimand
print("\n2. Preparing data for per-protocol analysis...")
print("   Using per-protocol estimand with censoring weights...")

prep = te.data_preparation(
    data=data,
    id="id",
    period="period",
    treatment="treatment",
    outcome="outcome",
    eligible="eligible",
    estimand_type="PP",  # Per-protocol
    outcome_cov=["age", "comorbidity_score"],
    use_censor_weights=True,
    cense="censored",
    cense_d_cov=["age", "comorbidity_score"],
    cense_n_cov="~1",
    quiet=True
)

print(f"   Expanded to {prep.N} observations")
print(f"   Censoring weights calculated")
print(f"   Trial periods: {prep.min_period} to {prep.max_period}")

# Step 3: Fit marginal structural model
print("\n3. Fitting marginal structural model with weights...")

msm = te.trial_msm(
    data=prep,
    outcome_cov=["age", "comorbidity_score"],
    estimand_type="PP",
    analysis_weights="p99",  # Truncate extreme weights at 99th percentile
    quiet=True
)

print("   Model fitted with truncated weights")

# Step 4: Display results
print("\n4. Results:")
print("-" * 60)

# Extract treatment effect
treatment_coef = msm.model.params["assigned_treatment"]
robust_se = msm.robust["bse"]["assigned_treatment"]
z_score = treatment_coef / robust_se
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

# Convert to OR and CI
odds_ratio = np.exp(treatment_coef)
or_lower = np.exp(treatment_coef - 1.96 * robust_se)
or_upper = np.exp(treatment_coef + 1.96 * robust_se)

print(f"\nPer-Protocol Treatment Effect:")
print(f"  Log Odds Ratio: {treatment_coef:.4f} (Robust SE: {robust_se:.4f})")
print(f"  Odds Ratio: {odds_ratio:.4f}")
print(f"  95% CI: [{or_lower:.4f}, {or_upper:.4f}]")
print(f"  P-value: {p_value:.4f}")

if treatment_coef < 0:
    risk_reduction = (1 - odds_ratio) * 100
    print(f"  Interpretation: {risk_reduction:.1f}% relative risk reduction")
    print(f"  among those who adhered to treatment protocol")

print("\nWeight Summary:")
if isinstance(prep.data, pd.DataFrame) and 'wt' in prep.data.columns:
    weights = prep.data['wt'].dropna()
    print(f"  Mean weight: {weights.mean():.3f}")
    print(f"  Median weight: {weights.median():.3f}")
    print(f"  Min weight: {weights.min():.3f}")
    print(f"  Max weight: {weights.max():.3f}")
    print(f"  % of weights > 10: {(weights > 10).mean() * 100:.1f}%")

print("\nFull Model Summary:")
print("-" * 60)
print(msm.robust["summary"])

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)

print("\nInterpretation:")
print("- Per-protocol analysis estimates effect among adherers")
print("- Censoring weights account for informative censoring at treatment switches")
print("- Weight truncation (p99) reduces influence of extreme weights")
print("- Compare with ITT analysis to assess impact of non-adherence")

print("\nNext steps:")
print("- Compare with ITT estimate to quantify adherence effect")
print("- Perform sensitivity analyses with different weight truncation")
print("- Check for residual confounding with additional covariates")
