"""The Newton/IRLS solver reproduces statsmodels GLM fits.

Coefficients must agree with ``GLM.fit`` to 1e-6 relative (in practice they
agree to ~1e-10), downstream weights and robust standard errors likewise, and
the returned object must behave as a statsmodels GLM results object.
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm

from trial_emulation import data_preparation, trial_msm
from trial_emulation._glm import fit_formula_glm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COEF_RTOL = 1e-6
KEYS = ["id", "trial_period", "followup_time"]


def _read(name):
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))


PIPELINES = {
    "te_pp": (
        "trial_example",
        dict(
            estimand_type="PP",
            outcome_cov="~ nvarA + nvarB",
            switch_n_cov="~ nvarA + nvarB",
            switch_d_cov="~ nvarA + nvarB + nvarC + catvarA",
        ),
        dict(estimand_type="PP", outcome_cov="~ nvarA + nvarB"),
    ),
    "dc_itt_cens": (
        "data_censored",
        dict(
            estimand_type="ITT",
            outcome_cov="~ x1 + x2 + x3 + x4 + age_s",
            use_censor_weights=True,
            cense="censored",
            cense_d_cov="~ x1 + x2 + x3 + x4 + age_s",
            cense_n_cov="~ x3 + x4",
            pool_cense="numerator",
        ),
        dict(
            estimand_type="ITT",
            outcome_cov="~ x1 + x2",
            include_followup_time="followup_time",
            include_trial_period="trial_period",
        ),
    ),
    "dc_at_cens": (
        "data_censored",
        dict(
            estimand_type="As-Treated",
            outcome_cov="~ x1 + x2 + x3 + x4 + age_s",
            switch_n_cov="~ x1",
            switch_d_cov="~ x1 + x2 + x3 + age_s",
            use_censor_weights=True,
            cense="censored",
            cense_d_cov="~ x1 + x2 + x3 + x4 + age_s",
            cense_n_cov="~ x3 + x4",
            pool_cense="none",
        ),
        dict(
            estimand_type="As-Treated",
            outcome_cov="~ x1 + x2",
            include_followup_time="followup_time",
            include_trial_period="trial_period",
        ),
    ),
}


def _run(name, solver):
    dataset, prep_kwargs, msm_kwargs = PIPELINES[name]
    prep = data_preparation(_read(dataset), quiet=True, solver=solver, **prep_kwargs)
    data = prep.data.sort_values(KEYS).reset_index(drop=True)
    fit = trial_msm(data, quiet=True, solver=solver, **msm_kwargs)
    return prep, data, fit


@pytest.fixture(scope="module", params=list(PIPELINES))
def both_solvers(request):
    return request.param, _run(request.param, "numpy"), _run(request.param, "statsmodels")


def _weight_models(prep):
    return {**(prep.switch_models or {}), **(prep.censor_models or {})}


def test_weight_models_match_statsmodels(both_solvers):
    _, (prep_np, _, _), (prep_sm, _, _) = both_solvers
    models_np, models_sm = _weight_models(prep_np), _weight_models(prep_sm)
    assert list(models_np) == list(models_sm) and models_np
    for key in models_np:
        assert models_np[key]["solver"] == "numpy"
        assert models_sm[key]["solver"] == "statsmodels"
        left, right = models_np[key]["summary"], models_sm[key]["summary"]
        assert list(left["term"]) == list(right["term"])
        for column in ("estimate", "std.error", "statistic", "p.value"):
            np.testing.assert_allclose(left[column], right[column], rtol=COEF_RTOL, err_msg=key)
        for stat, value in models_sm[key]["fit_summary"].items():
            assert models_np[key]["fit_summary"][stat] == pytest.approx(value, rel=1e-9)


def test_weights_match_statsmodels(both_solvers):
    _, (_, data_np, _), (_, data_sm, _) = both_solvers
    pd.testing.assert_frame_equal(
        data_np.drop(columns="weight"), data_sm.drop(columns="weight"), check_exact=True
    )
    np.testing.assert_allclose(data_np["weight"], data_sm["weight"], rtol=1e-9)


def test_msm_matches_statsmodels(both_solvers):
    _, (_, _, fit_np), (_, _, fit_sm) = both_solvers
    assert fit_np.args["solver_used"] == "numpy"
    assert fit_sm.args["solver_used"] == "statsmodels"
    np.testing.assert_allclose(fit_np.model.params, fit_sm.model.params, rtol=COEF_RTOL)
    np.testing.assert_allclose(fit_np.model.bse, fit_sm.model.bse, rtol=COEF_RTOL)
    np.testing.assert_allclose(
        fit_np.robust["summary"]["robust_se"], fit_sm.robust["summary"]["robust_se"], rtol=1e-6
    )
    assert fit_np.model.llf == pytest.approx(fit_sm.model.llf, rel=1e-10)
    assert fit_np.model.deviance == pytest.approx(fit_sm.model.deviance, rel=1e-9)
    assert fit_np.model.fit_history["iteration"] == fit_sm.model.fit_history["iteration"]


def _synthetic(seed=3, n=4000):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "a": rng.integers(0, 2, n),
            "t": rng.integers(0, 60, n),
            "x": rng.normal(size=n),
            "g": rng.choice(["u", "v", "w"], n),
        }
    )
    eta = -2 + 0.5 * frame["a"] + 0.02 * frame["t"] + 0.6 * frame["x"]
    frame["yb"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    frame["yc"] = 50 + 3 * frame["a"] + 0.1 * frame["t"] + 4 * frame["x"] + rng.normal(0, 7, n)
    weights = rng.uniform(0.1, 4.0, n)
    weights[rng.random(n) < 0.05] = 0.0
    return frame, weights


@pytest.mark.parametrize(
    "family,response", [("binomial", "yb"), ("gaussian", "yc")], ids=["logit", "identity"]
)
def test_weighted_fit_matches_statsmodels(family, response):
    frame, weights = _synthetic()
    formula = f"{response} ~ a + t + I(t**2) + x + C(g)"
    fam = (
        sm.families.Binomial()
        if family == "binomial"
        else sm.families.Gaussian(sm.families.links.Identity())
    )
    reference = glm(formula, data=frame, family=fam, freq_weights=weights).fit()
    fitted = fit_formula_glm(formula, frame, family, freq_weights=weights, missing="raise")
    assert fitted.mle_settings["solver"] == "numpy"
    assert fitted.converged
    np.testing.assert_allclose(fitted.params, reference.params, rtol=COEF_RTOL)
    np.testing.assert_allclose(fitted.bse, reference.bse, rtol=COEF_RTOL)
    np.testing.assert_allclose(fitted.fittedvalues, reference.fittedvalues, rtol=1e-9)
    assert fitted.scale == pytest.approx(reference.scale, rel=1e-9)
    assert fitted.llf == pytest.approx(reference.llf, rel=1e-10)
    assert fitted.null_deviance == pytest.approx(reference.null_deviance, rel=1e-10)
    np.testing.assert_allclose(
        fitted.model.score_obs(fitted.params, scale=1.0),
        reference.model.score_obs(reference.params, scale=1.0),
        rtol=1e-6,
        atol=1e-8,
    )
    # The results object keeps the statsmodels interface.
    assert "Generalized Linear Model Regression Results" in str(fitted.summary())
    newdata = frame.head(7)
    np.testing.assert_allclose(fitted.predict(newdata), reference.predict(newdata), rtol=1e-8)
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_allclose(restored.params, fitted.params, rtol=0)


def test_rank_deficient_design_falls_back_to_statsmodels():
    frame, weights = _synthetic(n=500)
    frame["x_copy"] = 2.0 * frame["x"]
    formula = "yb ~ a + x + x_copy"
    reference = glm(formula, data=frame, family=sm.families.Binomial(), freq_weights=weights).fit()
    fitted = fit_formula_glm(formula, frame, "binomial", freq_weights=weights)
    assert fitted.mle_settings["solver"] == "statsmodels"
    assert "conditioned" in fitted.mle_settings["solver_fallback_reason"]
    np.testing.assert_array_equal(fitted.params, reference.params)


def test_other_fit_options_use_statsmodels():
    frame, weights = _synthetic(n=500)
    fitted = fit_formula_glm(
        "yb ~ a + x", frame, "binomial", freq_weights=weights, fit_kwargs={"scale": "X2"}
    )
    assert fitted.mle_settings["solver"] == "statsmodels"
    assert "scale" in fitted.mle_settings["solver_fallback_reason"]


def test_unknown_solver_is_rejected():
    frame, _ = _synthetic(n=50)
    with pytest.raises(ValueError, match="solver must be one of"):
        fit_formula_glm("yb ~ a", frame, "binomial", solver="fast")
    with pytest.raises(ValueError, match="solver must be one of"):
        data_preparation(_read("trial_example"), quiet=True, solver="fast")


def test_torch_solver_matches_numpy():
    pytest.importorskip("torch")
    frame, weights = _synthetic()
    for family, response in (("binomial", "yb"), ("gaussian", "yc")):
        formula = f"{response} ~ a + t + I(t**2) + x + C(g)"
        reference = fit_formula_glm(formula, frame, family, freq_weights=weights)
        fitted = fit_formula_glm(formula, frame, family, freq_weights=weights, solver="torch")
        assert fitted.mle_settings["solver"] == "torch"
        np.testing.assert_allclose(fitted.params, reference.params, rtol=1e-10)
        np.testing.assert_allclose(fitted.bse, reference.bse, rtol=1e-10)
