"""
Simulation checks for continuous-outcome (Gaussian) marginal structural models.

These tests run the full package pipeline (data_preparation -> expanded trial
data -> weighted Gaussian MSM) on simulated data with a known treatment effect
on a continuous outcome measured at every follow-up period:

* ITT: baseline-confounded treatment that never changes. The effect is
  recovered by inverse-probability-of-treatment weighting of the expanded
  ITT data and, alternatively, by baseline covariate adjustment.
* PP: randomised-like baseline arms with treatment switching driven by a
  time-varying prognostic factor. The per-protocol effect is recovered only
  when the switching weights fitted by ``data_preparation`` are applied.
* Bootstrap: ``compute_robust_variance=False`` gives the same point estimate,
  and a patient bootstrap of the whole pipeline gives a standard error of the
  same size as the conditional sandwich.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.formula.api import logit

from trial_emulation import data_preparation, trial_msm

TRUE_EFFECT = 2.0


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _long_frame(treatment, covariate, outcome, k):
    n = treatment.shape[0]
    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(1, n + 1), k),
            "period": np.tile(np.arange(k), n),
            "treatment": treatment.ravel(),
            "L": covariate.ravel(),
            "y": outcome.ravel(),
            "outcome": 0,  # no terminating event; y is the continuous endpoint
            "eligible": np.tile((np.arange(k) == 0).astype(int), n),
        }
    )


def simulate_itt(n=3000, k=3, seed=11):
    """Baseline confounding by L0; treatment fixed after baseline."""
    rng = np.random.default_rng(seed)
    l0 = rng.normal(size=n)
    a = (rng.random(n) < _expit(0.8 * l0)).astype(int)
    treatment = np.repeat(a[:, None], k, axis=1)
    covariate = np.repeat(l0[:, None], k, axis=1)
    y = 1.0 + TRUE_EFFECT * treatment + 1.5 * covariate + rng.normal(size=(n, k))
    return _long_frame(treatment, covariate, y, k)


def simulate_pp(n=4000, k=4, seed=1):
    """Switching depends on the current value of a prognostic factor L_t."""
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=(n, k))
    treatment = np.zeros((n, k), dtype=int)
    treatment[:, 0] = rng.random(n) < _expit(-1.0 + 0.3 * covariate[:, 0])
    for t in range(1, k):
        p_treat = np.where(
            treatment[:, t - 1] == 0,
            _expit(-1.0 + 0.3 * covariate[:, t]),  # start treatment
            _expit(1.0 - 1.5 * covariate[:, t]),  # stay on treatment
        )
        treatment[:, t] = rng.random(n) < p_treat
    y = 1.0 + TRUE_EFFECT * treatment + 2.0 * covariate + rng.normal(size=(n, k))
    return _long_frame(treatment, covariate, y, k)


def _attach_continuous_outcome(prep, raw):
    expanded = prep.data.copy()
    expanded["period"] = expanded["trial_period"] + expanded["followup_time"]
    outcome = raw[["id", "period", "y"]].rename(columns={"y": "outcome"})
    merged = expanded.drop(columns="outcome").merge(outcome, on=["id", "period"], how="left")
    assert len(merged) == len(expanded)
    assert merged["outcome"].notna().all()
    return merged


def _effect(result):
    row = result.robust["summary"].set_index("names").loc["assigned_treatment"]
    return float(row["estimate"]), float(row["robust_se"])


def test_itt_continuous_effect_recovered_with_baseline_weights():
    raw = simulate_itt()
    prep = data_preparation(raw, estimand_type="ITT", outcome_cov="~ L", quiet=True)
    data = _attach_continuous_outcome(prep, raw)
    assert (data["weight"] == 1.0).all()

    common = dict(
        estimand_type="ITT",
        outcome_family="gaussian",
        include_trial_period="~1",
        include_followup_time="followup_time",
        use_sample_weights=False,
        quiet=True,
    )
    naive, naive_se = _effect(trial_msm(data, **common))

    baseline = data[data["followup_time"] == 0]
    ps_model = logit("assigned_treatment ~ L", data=baseline).fit(disp=0)
    ps = pd.Series(np.asarray(ps_model.predict(baseline)), index=baseline["id"].to_numpy())
    p = data["id"].map(ps).to_numpy()
    a = data["assigned_treatment"].to_numpy()
    weighted_data = data.assign(weight=data["weight"] * np.where(a == 1, 1.0 / p, 1.0 / (1.0 - p)))
    ipw, ipw_se = _effect(trial_msm(weighted_data, **common))
    adjusted, adjusted_se = _effect(trial_msm(data, outcome_cov="~ L", **common))

    assert abs(naive - TRUE_EFFECT) > 0.5 + 4 * naive_se
    assert abs(ipw - TRUE_EFFECT) < 4 * ipw_se
    assert abs(adjusted - TRUE_EFFECT) < 4 * adjusted_se


@pytest.fixture(scope="module")
def pp_data():
    raw = simulate_pp()
    prep = data_preparation(
        raw, estimand_type="PP", switch_d_cov="~ L", switch_n_cov="~ 1", quiet=True
    )
    return raw, prep, _attach_continuous_outcome(prep, raw)


PP_MSM = dict(
    estimand_type="PP",
    outcome_family="gaussian",
    include_trial_period="~1",
    include_followup_time="followup_time",
    first_followup=1,  # L_0 also drives baseline treatment; follow-up rows only
    use_sample_weights=False,
    quiet=True,
)


def test_pp_continuous_effect_recovered_only_with_switch_weights(pp_data):
    _, prep, data = pp_data
    # PP censoring at switch removes rows; weights are not all one.
    assert prep.N < len(prep.data["id"].unique()) * 4
    assert data["weight"].std() > 0.05

    weighted, weighted_se = _effect(trial_msm(data, **PP_MSM))
    naive, naive_se = _effect(trial_msm(data, analysis_weights="unweighted", **PP_MSM))

    assert abs(weighted - TRUE_EFFECT) < 4 * weighted_se
    assert TRUE_EFFECT - naive > 0.3 + 4 * naive_se


def test_bootstrap_mode_matches_point_estimate_and_sandwich_scale(pp_data):
    raw, _, data = pp_data
    full = trial_msm(data, **PP_MSM)
    no_sandwich = trial_msm(data, compute_robust_variance=False, **PP_MSM)
    np.testing.assert_allclose(
        np.asarray(no_sandwich.model.params), np.asarray(full.model.params), rtol=1e-12
    )
    assert no_sandwich.robust == {}
    assert no_sandwich.args["variance_scope"] == "not_computed"

    # Patient bootstrap of the whole pipeline (weights refitted per resample).
    rng = np.random.default_rng(2026)
    n_ids = raw["id"].nunique()
    k = len(raw) // n_ids  # every simulated patient has k consecutive rows
    estimates = []
    for _ in range(40):
        draw = rng.integers(0, n_ids, size=n_ids)
        rows = (draw[:, None] * k + np.arange(k)).ravel()
        resample = raw.iloc[rows].reset_index(drop=True)
        resample["id"] = np.repeat(np.arange(1, n_ids + 1), k)
        prep = data_preparation(
            resample, estimand_type="PP", switch_d_cov="~ L", switch_n_cov="~ 1", quiet=True
        )
        fit = trial_msm(
            _attach_continuous_outcome(prep, resample),
            compute_robust_variance=False,
            **PP_MSM,
        )
        estimates.append(float(fit.model.params["assigned_treatment"]))

    bootstrap_se = float(np.std(estimates, ddof=1))
    _, sandwich_se = _effect(full)
    assert 0.6 < bootstrap_se / sandwich_se < 1.6
    assert abs(np.mean(estimates) - TRUE_EFFECT) < 4 * sandwich_se
