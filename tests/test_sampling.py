"""Tests for case-control sampling of expanded trial data and weight truncation."""

import numpy as np
import pandas as pd

from trial_emulation import case_control_sampling_trials
from trial_emulation.types import TEDataPrep
from trial_emulation.utils import p99_weight


def _prep():
    rng = np.random.default_rng(5)
    rows = []
    for trial_period in range(3):
        for followup_time in range(4):
            for pid in range(1, 201):
                rows.append(
                    {
                        "id": pid,
                        "trial_period": trial_period,
                        "followup_time": followup_time,
                        "outcome": int(rng.random() < 0.05),
                        "weight": 1.0,
                        "assigned_treatment": pid % 2,
                    }
                )
    data = pd.DataFrame(rows)
    return TEDataPrep(
        data=data,
        min_period=0,
        max_period=2,
        N=len(data),
        data_template=data.iloc[:0],
    )


def test_sampling_keeps_all_cases_and_weights_controls():
    prep = _prep()
    np.random.seed(1)
    sample = case_control_sampling_trials(prep, p_control=0.2)
    cases = prep.data["outcome"].sum()
    assert int(sample["outcome"].sum()) == cases
    assert (sample.loc[sample["outcome"] == 1, "sample_weight"] == 1.0).all()
    assert np.allclose(sample.loc[sample["outcome"] == 0, "sample_weight"], 5.0)
    n_controls = int((prep.data["outcome"] == 0).sum())
    assert 0.15 * n_controls < (sample["outcome"] == 0).sum() < 0.25 * n_controls
    assert "sample_id" not in sample.columns


def test_sampling_with_several_probabilities_returns_one_frame_per_probability():
    prep = _prep()
    np.random.seed(2)
    samples = case_control_sampling_trials(prep, p_control=[0.1, 1.0])
    assert isinstance(samples, list) and len(samples) == 2
    for sample, p in zip(samples, [0.1, 1.0]):
        assert "sample_id" not in sample.columns
        assert set(sample["trial_period"]) == {0, 1, 2}
        assert np.allclose(sample.loc[sample["outcome"] == 0, "sample_weight"], 1.0 / p)
    # p_control = 1 keeps every row.
    assert len(samples[1]) == prep.N


def test_p99_weight_uses_r_type1_quantiles():
    # R: quantile(1:100, c(0.01, 0.99), type = 1) == c(1, 99)
    truncated = p99_weight(np.arange(1, 101, dtype=float))
    assert truncated.min() == 1.0
    assert truncated.max() == 99.0
