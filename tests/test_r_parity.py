"""
Parity tests against the R TrialEmulation package (0.0.4.9, git f432103).

Reference outputs in ``tests/data/r_reference/`` were produced by
``tests/data/r_reference/make_r_reference.R`` from the package's public
example datasets ``trial_example`` and ``data_censored``. They cover the
expanded trial data (row counts, per-column missingness and sums, per-trial
checksums, weight quantiles, a deterministic sample of rows), the weight-model
coefficient tables and the MSM coefficients with robust standard errors, for
ITT, PP and As-Treated preparation (with and without censoring weights).

Tolerances:

* Expansion structure (rows, keys, outcome, treatment, assigned treatment,
  dose, covariates): exact.
* Weights: relative 1e-5. Weight models are fitted by IRLS in both packages;
  the two convergence rules differ slightly.
* Weight-model and MSM coefficients: within 1e-4 of their standard error.
* Robust standard errors: relative 1e-3. R's ``sandwich::vcovCL`` multiplies
  the HC0 cluster sandwich by G/(G - 1); the Python default omits that
  finite-cluster factor (see CHANGELOG 0.0.5.0), so Python is compared with
  both R's HC0 value and, after applying G/(G - 1), with R's reported value.
"""

import os

import numpy as np
import pandas as pd
import pytest

from trial_emulation import data_preparation, trial_msm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REF_DIR = os.path.join(DATA_DIR, "r_reference")

WEIGHT_RTOL = 1e-5
COEF_SE_FRACTION = 1e-4
SE_RTOL = 1e-3

_TE_SWITCH = dict(
    switch_n_cov="~ nvarA + nvarB",
    switch_d_cov="~ nvarA + nvarB + nvarC + catvarA",
)
_DC_SWITCH = dict(switch_n_cov="~ x1", switch_d_cov="~ x1 + x2 + x3 + age_s")
_DC_CENSOR = dict(
    use_censor_weights=True,
    cense="censored",
    cense_d_cov="~ x1 + x2 + x3 + x4 + age_s",
    cense_n_cov="~ x3 + x4",
)
_DC_COV = "~ x1 + x2 + x3 + x4 + age_s"
_LINEAR_TIME = dict(include_followup_time="followup_time", include_trial_period="trial_period")

CONFIGS = {
    "te_itt": (
        "trial_example",
        dict(estimand_type="ITT", outcome_cov="~ nvarA + nvarB"),
        dict(estimand_type="ITT", outcome_cov="~ nvarA + nvarB"),
    ),
    "te_pp": (
        "trial_example",
        dict(estimand_type="PP", outcome_cov="~ nvarA + nvarB", **_TE_SWITCH),
        dict(estimand_type="PP", outcome_cov="~ nvarA + nvarB"),
    ),
    "te_at": (
        "trial_example",
        dict(estimand_type="As-Treated", outcome_cov="~ nvarA + nvarB", **_TE_SWITCH),
        dict(estimand_type="As-Treated", outcome_cov="~ nvarA + nvarB"),
    ),
    "dc_itt_cens": (
        "data_censored",
        dict(estimand_type="ITT", outcome_cov=_DC_COV, pool_cense="numerator", **_DC_CENSOR),
        dict(estimand_type="ITT", outcome_cov="~ x1 + x2", **_LINEAR_TIME),
    ),
    "dc_pp_cens": (
        "data_censored",
        dict(
            estimand_type="PP", outcome_cov=_DC_COV, pool_cense="none", **_DC_SWITCH, **_DC_CENSOR
        ),
        None,  # two events after PP censoring: the MSM is quasi-separated in R too
    ),
    "dc_at_cens": (
        "data_censored",
        dict(
            estimand_type="As-Treated",
            outcome_cov=_DC_COV,
            pool_cense="both",
            **_DC_SWITCH,
            **_DC_CENSOR,
        ),
        dict(estimand_type="As-Treated", outcome_cov="~ x1 + x2", **_LINEAR_TIME),
    ),
}

KEYS = ["id", "trial_period", "followup_time"]


def _load_input(name):
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))


def _ref(name):
    return pd.read_csv(os.path.join(REF_DIR, name))


@pytest.fixture(scope="module", params=list(CONFIGS))
def prepared(request):
    name = request.param
    dataset, prep_kwargs, msm_kwargs = CONFIGS[name]
    prep = data_preparation(_load_input(dataset), quiet=True, **prep_kwargs)
    data = prep.data.sort_values(KEYS).reset_index(drop=True)
    return name, prep, data, msm_kwargs


def test_expansion_shape_and_columns(prepared):
    name, prep, data, _ = prepared
    meta = _ref("expanded_meta.csv").set_index("config").loc[name]
    columns = _ref("expanded_column_summary.csv")
    columns = columns[columns["config"] == name]

    assert prep.N == len(data) == int(meta["N"])
    assert prep.min_period == int(meta["min_period"])
    assert prep.max_period == int(meta["max_period"])
    assert data["id"].nunique() == int(meta["n_ids"])
    assert data["trial_period"].nunique() == int(meta["n_trials"])
    assert list(data.columns) == list(columns["column"])


def test_expansion_column_missingness_and_sums(prepared):
    name, _, data, _ = prepared
    columns = _ref("expanded_column_summary.csv")
    columns = columns[columns["config"] == name]
    for row in columns.itertuples(index=False):
        values = data[row.column].astype(float)
        assert int(values.isna().sum()) == int(row.n_na), row.column
        rtol = WEIGHT_RTOL if row.column == "weight" else 1e-12
        np.testing.assert_allclose(values.sum(), row.sum, rtol=rtol, err_msg=row.column)
        np.testing.assert_allclose(values.min(), row.min, rtol=rtol, err_msg=row.column)
        np.testing.assert_allclose(values.max(), row.max, rtol=rtol, err_msg=row.column)


def test_expansion_per_trial_checksums(prepared):
    name, _, data, _ = prepared
    ref = _ref(f"{name}_by_trial.csv")
    agg = {
        "n": ("id", "size"),
        "n_ids": ("id", "nunique"),
        "sum_weight": ("weight", "sum"),
        "sum_outcome": ("outcome", "sum"),
        "sum_followup": ("followup_time", "sum"),
        "sum_treatment": ("treatment", "sum"),
    }
    if "assigned_treatment" in data:
        agg["sum_assigned"] = ("assigned_treatment", "sum")
    if "dose" in data:
        agg["sum_dose"] = ("dose", "sum")
    got = (
        data.assign(weight_sq=data["weight"] ** 2)
        .groupby("trial_period")
        .agg(sum_weight_sq=("weight_sq", "sum"), **agg)
    )
    ref = ref.set_index("trial_period")
    assert list(got.index) == list(ref.index)
    for col in ref.columns:
        rtol = WEIGHT_RTOL if col.startswith("sum_weight") else 0
        np.testing.assert_allclose(
            got[col].to_numpy(float), ref[col].to_numpy(float), rtol=rtol, err_msg=col
        )


def test_expansion_weight_quantiles(prepared):
    name, _, data, _ = prepared
    meta = _ref("expanded_meta.csv").set_index("config").loc[name]
    probs = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
    got = np.quantile(data["weight"].to_numpy(), probs)  # numpy "linear" = R type 7
    want = meta[["w_q000", "w_q001", "w_q025", "w_q050", "w_q075", "w_q099", "w_q100"]]
    np.testing.assert_allclose(got, want.to_numpy(float), rtol=WEIGHT_RTOL)
    np.testing.assert_allclose(data["weight"].mean(), meta["w_mean"], rtol=WEIGHT_RTOL)


def test_expansion_sample_rows(prepared):
    name, _, data, _ = prepared
    ref = _ref(f"{name}_sample_rows.csv")
    merged = ref.merge(data, on=KEYS, how="left", suffixes=("_r", "_py"), indicator=True)
    assert (merged["_merge"] == "both").all()
    for col in ref.columns:
        if col in KEYS:
            continue
        rtol = WEIGHT_RTOL if col == "weight" else 1e-12
        np.testing.assert_allclose(
            merged[f"{col}_py"].to_numpy(float),
            merged[f"{col}_r"].to_numpy(float),
            rtol=rtol,
            err_msg=col,
        )


def test_weight_models_match_r(prepared):
    name, prep, _, _ = prepared
    path = os.path.join(REF_DIR, f"{name}_weight_models.csv")
    if not os.path.exists(path):
        assert not prep.switch_models and not prep.censor_models
        return
    ref = pd.read_csv(path)
    models = {**(prep.switch_models or {}), **(prep.censor_models or {})}
    assert sorted(models) == sorted(ref["model"].unique())
    for model_name, ref_model in ref.groupby("model", sort=False):
        got = models[model_name]["summary"].set_index("term")
        ref_model = ref_model.set_index("term")
        assert list(got.index) == list(ref_model.index), model_name
        assert models[model_name]["fit_summary"]["nobs"] == int(ref_model["nobs"].iloc[0])
        diff = np.abs(got["estimate"] - ref_model["estimate"]) / ref_model["std.error"]
        assert (diff < COEF_SE_FRACTION).all(), (model_name, diff.max())
        np.testing.assert_allclose(got["std.error"], ref_model["std.error"], rtol=SE_RTOL)


def _r_name(name):
    return (
        name.replace(" ", "")
        .replace("**", "^")
        .replace("Intercept", "(Intercept)")
        .replace("((Intercept))", "(Intercept)")
    )


def test_msm_matches_r(prepared):
    name, prep, _, msm_kwargs = prepared
    if msm_kwargs is None:
        pytest.skip("no MSM reference for this configuration")
    fit = trial_msm(prep, use_sample_weights=False, quiet=True, **msm_kwargs)
    got = fit.robust["summary"].assign(term=lambda d: d["names"].map(_r_name)).set_index("term")
    ref = _ref(f"{name}_msm.csv").set_index("names")

    assert sorted(got.index) == sorted(ref.index)
    got = got.loc[ref.index]
    assert fit.args["analysis_rows"] == int(ref["n_obs"].iloc[0])
    n_clusters = int(ref["n_clusters"].iloc[0])
    assert fit.robust["n_clusters"] == n_clusters

    diff = np.abs(got["estimate"] - ref["estimate"]) / ref["robust_se"]
    assert (diff < COEF_SE_FRACTION).all(), diff.max()
    np.testing.assert_allclose(got["robust_se"], ref["robust_se_no_cadjust"], rtol=SE_RTOL)
    cadjust = np.sqrt(n_clusters / (n_clusters - 1.0))
    np.testing.assert_allclose(got["robust_se"] * cadjust, ref["robust_se"], rtol=SE_RTOL)


def test_itt_expansion_has_complete_weights_and_assignment():
    """Regression: ITT expansion must give every row a weight and an arm."""
    prep = data_preparation(
        _load_input("trial_example"),
        estimand_type="ITT",
        outcome_cov=["nvarA", "nvarB"],
        quiet=True,
    )
    assert prep.data["weight"].notna().all()
    assert (prep.data["weight"] == 1.0).all()
    assert prep.data["assigned_treatment"].notna().all()
    assert "dose" not in prep.data.columns


def test_python_censor_fallback_matches_r(monkeypatch):
    """The pure-Python censoring state machine gives the R PP expansion."""
    import trial_emulation.data_manipulation as dm

    monkeypatch.setattr(dm, "HAS_CPP", False)
    _, prep_kwargs, _ = CONFIGS["dc_pp_cens"]
    prep = data_preparation(_load_input("data_censored"), quiet=True, **prep_kwargs)
    data = prep.data.sort_values(KEYS).reset_index(drop=True)
    ref = _ref("dc_pp_cens_by_trial.csv")
    got = data.groupby("trial_period").agg(n=("id", "size"), sum_weight=("weight", "sum"))
    np.testing.assert_array_equal(got["n"].to_numpy(), ref["n"].to_numpy())
    np.testing.assert_allclose(
        got["sum_weight"].to_numpy(), ref["sum_weight"].to_numpy(), rtol=WEIGHT_RTOL
    )
