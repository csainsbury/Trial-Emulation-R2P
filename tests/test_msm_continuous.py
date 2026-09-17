"""Continuous-outcome and causal-weight tests for trial_msm."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.formula.api import logit

from trial_emulation.msm import trial_msm


def continuous_frame() -> pd.DataFrame:
    rows = []
    for patient_id in range(24):
        treatment = patient_id % 2
        baseline = (patient_id % 5) - 2.0
        outcome = 3.0 + 2.5 * treatment + 0.4 * baseline + ((patient_id % 3) - 1) * 0.2
        rows.append(
            {
                "id": patient_id,
                "outcome": outcome,
                "assigned_treatment": treatment,
                "followup_time": 0,
                "trial_period": 0,
                "baseline": baseline,
                "weight": 0.5 + (patient_id % 7) / 3.0,
            }
        )
    return pd.DataFrame(rows)


def binary_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260917)
    data = continuous_frame()
    probability = 1.0 / (
        1.0 + np.exp(-(-0.8 + 0.7 * data["assigned_treatment"] + 0.2 * data["baseline"]))
    )
    data["outcome"] = rng.binomial(1, probability)
    data["weight"] = 1 + (np.arange(len(data)) % 3)
    return data


def fit_continuous(data: pd.DataFrame):
    return trial_msm(
        data=data,
        outcome_family="gaussian",
        outcome_cov="~1",
        include_followup_time="~1",
        include_trial_period="~1",
        use_sample_weights=False,
        quiet=True,
    )


def manual_gaussian_cluster_covariance(result, data: pd.DataFrame) -> np.ndarray:
    model = result.model
    x = np.asarray(model.model.exog, dtype=float)
    y = np.asarray(model.model.endog, dtype=float)
    fitted = np.asarray(model.fittedvalues, dtype=float)
    weights = data["weight"].to_numpy(dtype=float)
    groups = data["id"].to_numpy()
    bread = np.linalg.pinv(x.T @ (weights[:, None] * x))
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        residual = (y[mask] - fitted[mask])[:, None]
        score = (weights[mask, None] * x[mask] * residual).sum(axis=0)
        meat += np.outer(score, score)
    return bread @ meat @ bread


def test_gaussian_treatment_coefficient_matches_weighted_arm_means():
    data = continuous_frame()[
        ["id", "outcome", "assigned_treatment", "followup_time", "trial_period", "weight"]
    ]
    result = fit_continuous(data)
    treated = data["assigned_treatment"] == 1
    expected = np.average(data.loc[treated, "outcome"], weights=data.loc[treated, "weight"])
    expected -= np.average(data.loc[~treated, "outcome"], weights=data.loc[~treated, "weight"])
    assert result.model.params["assigned_treatment"] == pytest.approx(expected, abs=1e-12)
    assert result.args["outcome_family"] == "gaussian"


def test_gaussian_cluster_covariance_matches_manual_estimating_equation():
    data = continuous_frame()[
        ["id", "outcome", "assigned_treatment", "followup_time", "trial_period", "weight"]
    ]
    result = fit_continuous(data)
    expected = manual_gaussian_cluster_covariance(result, data)
    np.testing.assert_allclose(result.robust["matrix"], expected, rtol=1e-10, atol=1e-12)
    assert result.robust["variance_scope"].startswith("patient-clustered sandwich")


def test_weight_scale_does_not_change_coefficient_or_robust_covariance():
    data = continuous_frame()[
        ["id", "outcome", "assigned_treatment", "followup_time", "trial_period", "weight"]
    ]
    result = fit_continuous(data)
    scaled = data.copy()
    scaled["weight"] *= 17.0
    scaled_result = fit_continuous(scaled)
    np.testing.assert_allclose(result.model.params, scaled_result.model.params, atol=1e-11)
    np.testing.assert_allclose(
        result.robust["matrix"], scaled_result.robust["matrix"], rtol=1e-10, atol=1e-12
    )


def test_splitting_identical_rows_within_cluster_is_invariant():
    data = continuous_frame()[
        ["id", "outcome", "assigned_treatment", "followup_time", "trial_period", "weight"]
    ]
    result = fit_continuous(data)
    split = pd.concat([data.assign(weight=data["weight"] / 2.0)] * 2, ignore_index=True)
    split_result = fit_continuous(split)
    np.testing.assert_allclose(result.model.params, split_result.model.params, atol=1e-11)
    np.testing.assert_allclose(
        result.robust["matrix"], split_result.robust["matrix"], rtol=1e-10, atol=1e-12
    )


def test_unweighted_binomial_glm_matches_logit_coefficients():
    data = binary_frame()
    result = trial_msm(
        data=data,
        outcome_family="binomial",
        outcome_cov="baseline",
        include_followup_time="~1",
        include_trial_period="~1",
        analysis_weights="unweighted",
        use_sample_weights=False,
        quiet=True,
    )
    reference = logit("outcome ~ assigned_treatment + baseline", data=data).fit(disp=0)
    np.testing.assert_allclose(result.model.params, reference.params, rtol=1e-8, atol=1e-10)


def test_zero_weights_are_supported_for_both_families():
    continuous = continuous_frame()
    continuous.loc[0, "weight"] = 0.0
    continuous_result = fit_continuous(continuous)
    assert np.isfinite(continuous_result.robust["matrix"]).all()
    assert continuous_result.robust["n_positive_weight_rows"] == len(continuous) - 1

    binary = binary_frame()
    binary.loc[0, "weight"] = 0.0
    binary_result = trial_msm(
        data=binary,
        outcome_family="binomial",
        outcome_cov="baseline",
        include_followup_time="~1",
        include_trial_period="~1",
        use_sample_weights=False,
        quiet=True,
    )
    assert np.isfinite(binary_result.robust["matrix"]).all()
    assert binary_result.robust["n_positive_weight_rows"] == len(binary) - 1


def test_gaussian_covariance_is_invariant_when_weight_sum_is_at_most_parameters():
    data = continuous_frame()
    result = fit_continuous(data)
    for target_sum in [1.0, 2.0]:
        scaled = data.copy()
        scaled["weight"] *= target_sum / scaled["weight"].sum()
        scaled_result = fit_continuous(scaled)
        np.testing.assert_allclose(result.model.params, scaled_result.model.params, atol=1e-11)
        np.testing.assert_allclose(
            result.robust["matrix"],
            scaled_result.robust["matrix"],
            rtol=1e-10,
            atol=1e-12,
        )


def test_weighted_binomial_matches_integer_row_duplication_and_manual_sandwich():
    data = binary_frame()
    weighted = trial_msm(
        data=data,
        outcome_family="binomial",
        outcome_cov="baseline",
        include_followup_time="~1",
        include_trial_period="~1",
        use_sample_weights=False,
        quiet=True,
    )
    repeated = data.loc[data.index.repeat(data["weight"].astype(int))].copy()
    repeated["weight"] = 1.0
    duplicated = trial_msm(
        data=repeated,
        outcome_family="binomial",
        outcome_cov="baseline",
        include_followup_time="~1",
        include_trial_period="~1",
        use_sample_weights=False,
        quiet=True,
    )
    np.testing.assert_allclose(weighted.model.params, duplicated.model.params, atol=1e-10)

    x = np.asarray(weighted.model.model.exog, dtype=float)
    y = np.asarray(weighted.model.model.endog, dtype=float)
    mu = np.asarray(weighted.model.fittedvalues, dtype=float)
    weights = data["weight"].to_numpy(dtype=float)
    groups = data["id"].to_numpy()
    bread = np.linalg.pinv(x.T @ ((weights * mu * (1.0 - mu))[:, None] * x))
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        score = (weights[mask, None] * x[mask] * (y[mask] - mu[mask])[:, None]).sum(axis=0)
        meat += np.outer(score, score)
    expected = bread @ meat @ bread
    np.testing.assert_allclose(weighted.robust["matrix"], expected, rtol=1e-10, atol=1e-12)


def test_formula_row_drop_keeps_weights_and_clusters_aligned():
    data = continuous_frame()
    data.index = np.arange(100, 100 + len(data))
    data.loc[107, "baseline"] = np.nan
    shuffled = data.sample(frac=1.0, random_state=9)
    with pytest.warns(UserWarning, match="complete-case analysis"):
        result = trial_msm(
            data=shuffled,
            outcome_family="gaussian",
            outcome_cov="baseline",
            include_followup_time="~1",
            include_trial_period="~1",
            use_sample_weights=False,
            quiet=True,
        )
    assert result.args["analysis_rows"] == len(data) - 1
    assert result.robust["n_clusters"] == len(data) - 1


def test_categorical_level_only_in_dropped_row_does_not_cause_false_rank_failure():
    data = continuous_frame()
    data["site"] = np.where((data.index // 2) % 2 == 0, "A", "B")
    data.loc[0, "site"] = "C"
    data.loc[0, "baseline"] = np.nan
    with pytest.warns(UserWarning, match="complete-case analysis"):
        result = trial_msm(
            data=data,
            outcome_family="gaussian",
            outcome_cov="baseline + C(site)",
            include_followup_time="~1",
            include_trial_period="~1",
            use_sample_weights=False,
            quiet=True,
        )
    assert "C(site)[T.C]" not in result.model.params.index


def test_can_skip_conditional_robust_variance_for_outer_resampling():
    data = binary_frame()
    result = trial_msm(
        data=data,
        outcome_family="binomial",
        outcome_cov="baseline",
        include_followup_time="~1",
        include_trial_period="~1",
        use_sample_weights=False,
        compute_robust_variance=False,
        quiet=True,
    )

    assert result.robust == {}
    assert result.args["compute_robust_variance"] is False
    assert result.args["variance_scope"] == "not_computed"
    assert result.args["analysis_clusters"] == data["id"].nunique()
    prediction = np.asarray(result.model.predict(data), dtype=float)
    assert np.isfinite(prediction).all()
    assert ((prediction > 0) & (prediction < 1)).all()


def test_invalid_weights_and_outcome_family_fail_closed():
    data = continuous_frame()
    bad = data.copy()
    bad.loc[0, "weight"] = -1
    with pytest.raises(ValueError, match="finite and non-negative"):
        fit_continuous(bad)
    with pytest.raises(ValueError, match="outcome_family"):
        trial_msm(data=data, outcome_family="poisson", quiet=True)
