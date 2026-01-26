"""
Comparing Different Estimands: ITT vs PP vs As-Treated

This example demonstrates how to compare different causal estimands
(ITT, Per-Protocol, As-Treated) using the same dataset.
"""

import numpy as np
import pandas as pd
import trial_emulation as te
from scipy import stats

print("=" * 70)
print("TrialEmulation: Comparing Different Estimands")
print("=" * 70)

# Step 1: Generate realistic synthetic data
print("\n1. Generating synthetic data...")
np.random.seed(456)

n_patients = 200
n_periods = 8

data_list = []
for patient_id in range(1, n_patients + 1):
    age = np.random.randint(40, 80)
    female = np.random.randint(0, 2)

    treatment_status = 0
    cumulative_dose = 0
    has_outcome = False

    for period in range(n_periods):
        if has_outcome:
            continue

        # Treatment dynamics
        if treatment_status == 0:
            # Treatment initiation (more likely for older patients)
            start_prob = 0.08 + 0.002 * age
            if np.random.random() < start_prob:
                treatment_status = 1
        else:
            # Treatment discontinuation (20% per period)
            if np.random.random() < 0.20:
                treatment_status = 0

        treatment = treatment_status
        cumulative_dose += treatment  # Cumulative periods on treatment

        # Outcome model: treatment has protective effect
        log_hazard = -4.0 + 0.02 * age + 0.3 * female + 0.05 * period
        if treatment:
            log_hazard -= 0.6  # Strong protective effect (OR ~ 0.55)

        hazard = np.exp(log_hazard) / (1 + np.exp(log_hazard))
        outcome = 1 if np.random.random() < hazard else 0

        if outcome:
            has_outcome = True

        data_list.append({
            'id': patient_id,
            'period': period,
            'treatment': treatment,
            'outcome': outcome,
            'censored': outcome,
            'eligible': 1,
            'age': age,
            'female': female,
            'cumulative_dose': cumulative_dose
        })

data = pd.DataFrame(data_list)

print(f"   Dataset: {len(data)} observations, {data['id'].nunique()} patients")
print(f"   Ever treated: {(data.groupby('id')['treatment'].max() == 1).sum()}")
print(f"   Total outcomes: {data['outcome'].sum()}")

# Analyze treatment patterns
switches = data.groupby('id')['treatment'].apply(lambda x: (x.diff() != 0).sum()).sum()
print(f"   Treatment switches: {switches}")

# Step 2: Analyze with different estimands
print("\n2. Analyzing with three different estimands...")
print("   This may take a minute...")

results = {}

# 2a. Intention-to-Treat (ITT)
print("\n   a) Intention-to-Treat (ITT)...")
prep_itt = te.data_preparation(
    data=data,
    id="id",
    period="period",
    treatment="treatment",
    outcome="outcome",
    eligible="eligible",
    estimand_type="ITT",
    outcome_cov=["age", "female"],
    use_censor_weights=False,
    quiet=True
)

msm_itt = te.trial_msm(
    data=prep_itt,
    outcome_cov=["age", "female"],
    estimand_type="ITT",
    quiet=True
)

coef_itt = msm_itt.model.params["assigned_treatment"]
se_itt = msm_itt.robust["bse"]["assigned_treatment"]
or_itt = np.exp(coef_itt)
or_itt_lower = np.exp(coef_itt - 1.96 * se_itt)
or_itt_upper = np.exp(coef_itt + 1.96 * se_itt)

results['ITT'] = {
    'coef': coef_itt,
    'se': se_itt,
    'or': or_itt,
    'ci_lower': or_itt_lower,
    'ci_upper': or_itt_upper
}

print(f"      OR: {or_itt:.3f} (95% CI: [{or_itt_lower:.3f}, {or_itt_upper:.3f}])")

# 2b. Per-Protocol (PP)
print("\n   b) Per-Protocol (PP)...")
prep_pp = te.data_preparation(
    data=data,
    id="id",
    period="period",
    treatment="treatment",
    outcome="outcome",
    eligible="eligible",
    estimand_type="PP",
    outcome_cov=["age", "female"],
    use_censor_weights=True,
    cense="censored",
    quiet=True
)

msm_pp = te.trial_msm(
    data=prep_pp,
    outcome_cov=["age", "female"],
    estimand_type="PP",
    analysis_weights="p99",
    quiet=True
)

coef_pp = msm_pp.model.params["assigned_treatment"]
se_pp = msm_pp.robust["bse"]["assigned_treatment"]
or_pp = np.exp(coef_pp)
or_pp_lower = np.exp(coef_pp - 1.96 * se_pp)
or_pp_upper = np.exp(coef_pp + 1.96 * se_pp)

results['PP'] = {
    'coef': coef_pp,
    'se': se_pp,
    'or': or_pp,
    'ci_lower': or_pp_lower,
    'ci_upper': or_pp_upper
}

print(f"      OR: {or_pp:.3f} (95% CI: [{or_pp_lower:.3f}, {or_pp_upper:.3f}])")

# 2c. As-Treated
print("\n   c) As-Treated...")
prep_at = te.data_preparation(
    data=data,
    id="id",
    period="period",
    treatment="treatment",
    outcome="outcome",
    eligible="eligible",
    estimand_type="As-Treated",
    outcome_cov=["age", "female"],
    use_censor_weights=True,
    cense="censored",
    quiet=True
)

msm_at = te.trial_msm(
    data=prep_at,
    outcome_cov=["age", "female"],
    estimand_type="As-Treated",
    analysis_weights="p99",
    quiet=True
)

# For As-Treated, look at dose coefficient
dose_term = "dose"  # May need to check actual column name in model
if dose_term in msm_at.model.params:
    coef_at = msm_at.model.params[dose_term]
    se_at = msm_at.robust["bse"][dose_term]
    or_at = np.exp(coef_at)
    or_at_lower = np.exp(coef_at - 1.96 * se_at)
    or_at_upper = np.exp(coef_at + 1.96 * se_at)

    results['As-Treated'] = {
        'coef': coef_at,
        'se': se_at,
        'or': or_at,
        'ci_lower': or_at_lower,
        'ci_upper': or_at_upper
    }

    print(f"      OR per dose unit: {or_at:.3f} (95% CI: [{or_at_lower:.3f}, {or_at_upper:.3f}])")

# Step 3: Compare results
print("\n" + "=" * 70)
print("3. Comparison of Estimands")
print("=" * 70)

print("\n{:<15} {:<10} {:<10} {:<25}".format("Estimand", "Odds Ratio", "Std Error", "95% CI"))
print("-" * 70)

for estimand, vals in results.items():
    ci_str = f"[{vals['ci_lower']:.3f}, {vals['ci_upper']:.3f}]"
    print("{:<15} {:<10.3f} {:<10.3f} {:<25}".format(
        estimand, vals['or'], vals['se'], ci_str
    ))

# Interpretation
print("\n" + "=" * 70)
print("Interpretation")
print("=" * 70)

print("\nITT (Intention-to-Treat):")
print("  - Effect of treatment assignment, regardless of adherence")
print("  - Most conservative estimate, includes non-adherers")
print("  - Relevant for policy decisions about offering treatment")

print("\nPP (Per-Protocol):")
print("  - Effect among those who adhere to assigned treatment")
print("  - Uses censoring weights to handle treatment switching")
print("  - Typically larger effect size than ITT")
print("  - Relevant for biological/mechanistic questions")

if 'As-Treated' in results:
    print("\nAs-Treated:")
    print("  - Effect of actually receiving treatment")
    print("  - Accounts for cumulative exposure (dose)")
    print("  - May differ from ITT/PP if treatment effect is cumulative")

print("\nKey Insights:")
print("  - ITT is usually most conservative (smallest effect)")
print("  - PP typically shows larger effects (among adherers)")
print("  - Differences indicate impact of non-adherence")
print("  - All three estimands answer different causal questions")

print("\n" + "=" * 70)
print("Analysis complete!")
print("=" * 70)
