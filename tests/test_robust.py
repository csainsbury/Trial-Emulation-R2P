"""The vectorised cluster sandwich matches the original per-cluster loop."""

import os

import numpy as np
import pandas as pd
import pytest

from trial_emulation import data_preparation, trial_msm
from trial_emulation.robust import _cluster_meat, robust_calculation

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _loop_meat(score_obs, group_values, positive_rows):
    """The 0.0.5.2 meat: one boolean scan of all rows per positive-weight cluster."""
    meat = np.zeros((score_obs.shape[1], score_obs.shape[1]), dtype=float)
    for group in pd.unique(group_values[positive_rows]):
        cluster_score = score_obs[group_values == group].sum(axis=0)
        meat += np.outer(cluster_score, cluster_score)
    return meat


def _loop_reference(model, data_id, small_sample_correction=False):
    """The 0.0.5.2 implementation: one boolean scan of all rows per cluster."""
    groups = pd.Series(data_id).reset_index(drop=True)
    score_obs = np.asarray(model.model.score_obs(model.params, scale=1.0), dtype=float)
    hessian = np.asarray(model.model.hessian(model.params, scale=1.0, observed=False), dtype=float)
    n_rows, n_parameters = score_obs.shape
    frequency_weights = np.asarray(
        getattr(model.model, "freq_weights", np.ones(n_rows)), dtype=float
    )
    positive_rows = frequency_weights > 0
    n_positive_rows = int(positive_rows.sum())
    group_values = groups.to_numpy()
    positive_groups = pd.unique(group_values[positive_rows])
    n_clusters = len(positive_groups)
    bread = np.linalg.pinv(-hessian)
    meat = np.zeros((n_parameters, n_parameters), dtype=float)
    for group in positive_groups:
        cluster_score = score_obs[group_values == group].sum(axis=0)
        meat += np.outer(cluster_score, cluster_score)
    if small_sample_correction:
        meat *= (n_clusters / (n_clusters - 1.0)) * (
            (n_positive_rows - 1.0) / (n_positive_rows - n_parameters)
        )
    var_matrix = bread @ meat @ bread
    return (var_matrix + var_matrix.T) / 2.0


def _fits():
    trial_example = pd.read_csv(os.path.join(DATA_DIR, "trial_example.csv"))
    censored = pd.read_csv(os.path.join(DATA_DIR, "data_censored.csv"))
    fits = {}
    prep = data_preparation(
        trial_example, estimand_type="ITT", outcome_cov="~ nvarA + nvarB", quiet=True
    )
    fits["te_itt"] = (prep.data, dict(estimand_type="ITT", outcome_cov="~ nvarA + nvarB"))
    prep = data_preparation(
        trial_example,
        estimand_type="PP",
        outcome_cov="~ nvarA + nvarB",
        switch_n_cov="~ nvarA + nvarB",
        switch_d_cov="~ nvarA + nvarB + nvarC + catvarA",
        quiet=True,
    )
    fits["te_pp"] = (prep.data, dict(estimand_type="PP", outcome_cov="~ nvarA + nvarB"))
    prep = data_preparation(
        censored,
        estimand_type="As-Treated",
        outcome_cov="~ x1 + x2 + x3 + x4 + age_s",
        switch_n_cov="~ x1",
        switch_d_cov="~ x1 + x2 + x3 + age_s",
        use_censor_weights=True,
        cense="censored",
        cense_d_cov="~ x1 + x2 + x3 + x4 + age_s",
        cense_n_cov="~ x3 + x4",
        pool_cense="both",
        quiet=True,
    )
    fits["dc_at"] = (
        prep.data,
        dict(
            estimand_type="As-Treated",
            outcome_cov="~ x1 + x2",
            include_followup_time="followup_time",
            include_trial_period="trial_period",
        ),
    )
    # Continuous outcome with some zero-weight rows and whole zero-weight clusters.
    rng = np.random.default_rng(5)
    n = 600
    frame = pd.DataFrame(
        {
            "id": rng.integers(0, 80, n),
            "assigned_treatment": rng.integers(0, 2, n),
            "trial_period": rng.integers(0, 4, n),
            "followup_time": rng.integers(0, 6, n),
            "x": rng.normal(size=n),
        }
    )
    frame["outcome"] = 1 + 0.5 * frame["assigned_treatment"] + frame["x"] + rng.normal(size=n)
    frame["weight"] = rng.uniform(0.2, 3, n)
    frame.loc[rng.random(n) < 0.1, "weight"] = 0.0
    frame.loc[frame["id"] < 3, "weight"] = 0.0
    fits["gaussian_zero_weights"] = (
        frame,
        dict(outcome_family="gaussian", outcome_cov="~ x", use_sample_weights=False),
    )
    return fits


FITS = _fits()


@pytest.mark.parametrize("name", list(FITS))
@pytest.mark.parametrize("small_sample_correction", [False, True])
def test_vectorised_sandwich_matches_cluster_loop(name, small_sample_correction):
    data, kwargs = FITS[name]
    fit = trial_msm(data, quiet=True, small_sample_correction=small_sample_correction, **kwargs)
    ids = fit.model.model.data.frame["id"]
    # The changed piece, the cluster meat, matches the loop to 1e-10.
    model = fit.model
    score_obs = np.asarray(model.model.score_obs(model.params, scale=1.0), dtype=float)
    group_values = pd.Series(ids).reset_index(drop=True).to_numpy()
    positive_rows = np.asarray(model.model.freq_weights, dtype=float) > 0
    loop_meat = _loop_meat(score_obs, group_values, positive_rows)
    np.testing.assert_allclose(
        _cluster_meat(score_obs, group_values, positive_rows),
        loop_meat,
        rtol=1e-10,
        atol=1e-10 * np.abs(loop_meat).max(),
    )
    # The full covariance goes through the pseudo-inverse of an ill-conditioned
    # Hessian (unscaled squared time terms), which amplifies summation-order
    # rounding in the meat (observed on the macOS runners: SEs ~5e-8 relative, near-zero
    # off-diagonal covariances up to 3e-5 relative).
    reference = _loop_reference(model, ids, small_sample_correction)
    result = robust_calculation(model, ids, small_sample_correction=small_sample_correction)
    # Compare on the correlation scale: off-diagonal covariances near zero
    # cancel and carry the amplified rounding (~1e-7 of sqrt(var_i var_j)).
    sd = np.sqrt(np.diag(reference))
    np.testing.assert_allclose(
        result["matrix"] / np.outer(sd, sd), reference / np.outer(sd, sd), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(fit.robust["matrix"], result["matrix"], rtol=0, atol=0)
    np.testing.assert_allclose(
        result["summary"]["robust_se"], np.sqrt(np.diag(reference)), rtol=1e-6
    )
