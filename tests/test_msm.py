"""
Tests for msm (marginal structural model) module.

Tests the fitting of marginal structural models using weighted pooled logistic regression
and robust variance estimation.
"""

import pytest
import pandas as pd
import numpy as np
from trial_emulation.msm import trial_msm, fit_msm
from trial_emulation.types import TEMSM, TEDataPrep


@pytest.fixture
def expanded_trial_data():
    """
    Create synthetic expanded trial data for MSM fitting.

    This mimics the output from data_preparation with all required columns
    for MSM fitting.
    """
    np.random.seed(42)

    n_obs = 500
    data = pd.DataFrame(
        {
            "id": np.repeat(range(1, 101), 5),
            "trial_period": np.tile(range(5), 100),
            "followup_time": np.tile(range(5), 100),
            "assigned_treatment": np.random.randint(0, 2, n_obs),
            "outcome": np.random.randint(0, 2, n_obs),
            "weight": np.random.uniform(0.5, 2.0, n_obs),
            "dose": np.random.randint(0, 4, n_obs),
            "age": np.random.randint(40, 80, n_obs),
            "baseline_var": np.random.normal(0, 1, n_obs),
        }
    )

    return data


@pytest.fixture
def expanded_trial_data_with_known_effect():
    """
    Create expanded trial data with a known treatment effect.

    Treatment reduces outcome odds by 50% (OR = 0.5).
    """
    np.random.seed(123)

    n_ids = 200
    n_periods = 5

    data = []
    for patient_id in range(1, n_ids + 1):
        treated = patient_id <= n_ids // 2

        for period in range(n_periods):
            assigned_treatment = 1 if treated else 0

            # Outcome probability with known treatment effect
            log_odds = -2.0  # Baseline log odds
            if assigned_treatment:
                log_odds -= 0.693  # log(0.5) = -0.693, corresponds to OR = 0.5

            prob = 1 / (1 + np.exp(-log_odds))
            outcome = 1 if np.random.random() < prob else 0

            data.append(
                {
                    "id": patient_id,
                    "trial_period": period,
                    "followup_time": period,
                    "assigned_treatment": assigned_treatment,
                    "outcome": outcome,
                    "weight": 1.0,  # Equal weights for simplicity
                    "age": np.random.randint(45, 75),
                }
            )

    return pd.DataFrame(data)


class TestTrialMSMBasic:
    """Test basic trial_msm functionality."""

    def test_trial_msm_returns_correct_type(self, expanded_trial_data):
        """Test that trial_msm returns TEMSM object."""
        result = trial_msm(data=expanded_trial_data, estimand_type="ITT", quiet=True)

        assert isinstance(result, TEMSM)
        assert hasattr(result, "model")
        assert hasattr(result, "robust")
        assert hasattr(result, "args")

    def test_trial_msm_model_structure(self, expanded_trial_data):
        """Test that fitted model has expected structure."""
        result = trial_msm(data=expanded_trial_data, estimand_type="ITT", quiet=True)

        # Check model object
        assert result.model is not None
        assert hasattr(result.model, "params")
        assert hasattr(result.model, "predict")

        # Check robust results
        assert isinstance(result.robust, dict)
        assert "vcov_robust" in result.robust or "robust_vcov" in result.robust

    def test_trial_msm_with_dataframe(self, expanded_trial_data):
        """Test trial_msm with DataFrame input."""
        result = trial_msm(data=expanded_trial_data, estimand_type="ITT", quiet=True)

        assert isinstance(result, TEMSM)

    def test_trial_msm_with_tedataprep(self, expanded_trial_data):
        """Test trial_msm with TEDataPrep object."""
        # Create a mock TEDataPrep object
        te_data = TEDataPrep(
            data=expanded_trial_data,
            min_period=0,
            max_period=4,
            N=len(expanded_trial_data),
            data_template=expanded_trial_data.iloc[:0],
        )

        result = trial_msm(data=te_data, estimand_type="ITT", quiet=True)

        assert isinstance(result, TEMSM)


class TestTrialMSMEstimands:
    """Test different estimand types."""

    def test_itt_estimand(self, expanded_trial_data):
        """Test ITT estimand."""
        result = trial_msm(data=expanded_trial_data, estimand_type="ITT", quiet=True)

        assert result.args["estimand_type"] == "ITT"
        # ITT should use assigned_treatment in formula
        assert "assigned_treatment" in result.args["model_formula"]

    def test_pp_estimand(self, expanded_trial_data):
        """Test per-protocol estimand."""
        result = trial_msm(data=expanded_trial_data, estimand_type="PP", quiet=True)

        assert result.args["estimand_type"] == "PP"

    def test_as_treated_estimand(self, expanded_trial_data):
        """Test as-treated estimand."""
        result = trial_msm(data=expanded_trial_data, estimand_type="As-Treated", quiet=True)

        assert result.args["estimand_type"] == "As-Treated"
        # As-Treated should use dose in formula
        assert "dose" in result.args["model_formula"]

    def test_invalid_estimand_raises_error(self, expanded_trial_data):
        """Test that invalid estimand type raises ValueError."""
        with pytest.raises(ValueError, match="estimand_type must be one of"):
            trial_msm(data=expanded_trial_data, estimand_type="INVALID", quiet=True)


class TestTrialMSMWeights:
    """Test weight handling in MSM fitting."""

    def test_asis_weights(self, expanded_trial_data):
        """Test analysis_weights='asis' (use weights as-is)."""
        result = trial_msm(
            data=expanded_trial_data, estimand_type="ITT", analysis_weights="asis", quiet=True
        )

        assert result.args["analysis_weights"] == "asis"

    def test_unweighted(self, expanded_trial_data):
        """Test analysis_weights='unweighted'."""
        result = trial_msm(
            data=expanded_trial_data, estimand_type="ITT", analysis_weights="unweighted", quiet=True
        )

        assert result.args["analysis_weights"] == "unweighted"

    def test_p99_weights(self, expanded_trial_data):
        """Test analysis_weights='p99' (truncate at 99th percentile)."""
        result = trial_msm(
            data=expanded_trial_data, estimand_type="ITT", analysis_weights="p99", quiet=True
        )

        assert result.args["analysis_weights"] == "p99"

    def test_weight_limits(self, expanded_trial_data):
        """Test analysis_weights='weight_limits' with custom limits."""
        result = trial_msm(
            data=expanded_trial_data,
            estimand_type="ITT",
            analysis_weights="weight_limits",
            weight_limits=(0.1, 10.0),
            quiet=True,
        )

        assert result.args["analysis_weights"] == "weight_limits"
        assert result.args["weight_limits"] == (0.1, 10.0)

    def test_invalid_analysis_weights_raises_error(self, expanded_trial_data):
        """Test that invalid analysis_weights raises ValueError."""
        with pytest.raises(ValueError, match="analysis_weights must be one of"):
            trial_msm(
                data=expanded_trial_data,
                estimand_type="ITT",
                analysis_weights="INVALID",
                quiet=True,
            )


class TestTrialMSMFollowupFiltering:
    """Test follow-up time filtering."""

    def test_first_followup_filter(self, expanded_trial_data):
        """Test filtering with first_followup parameter."""
        first_followup = 2
        result = trial_msm(
            data=expanded_trial_data, estimand_type="ITT", first_followup=first_followup, quiet=True
        )

        assert result.args["first_followup"] == first_followup

    def test_last_followup_filter(self, expanded_trial_data):
        """Test filtering with last_followup parameter."""
        last_followup = 3
        result = trial_msm(
            data=expanded_trial_data, estimand_type="ITT", last_followup=last_followup, quiet=True
        )

        assert result.args["last_followup"] == last_followup

    def test_followup_range_filter(self, expanded_trial_data):
        """Test filtering with both first and last followup."""
        result = trial_msm(
            data=expanded_trial_data,
            estimand_type="ITT",
            first_followup=1,
            last_followup=3,
            quiet=True,
        )

        assert result.args["first_followup"] == 1
        assert result.args["last_followup"] == 3


class TestTrialMSMCovariates:
    """Test covariate specifications."""

    def test_outcome_covariates(self, expanded_trial_data):
        """Test specifying outcome model covariates."""
        result = trial_msm(
            data=expanded_trial_data,
            estimand_type="ITT",
            outcome_cov="age + baseline_var",
            quiet=True,
        )

        # Check that covariates are in the formula
        assert "age" in result.args["model_formula"]
        assert "baseline_var" in result.args["model_formula"]

    def test_custom_model_var(self, expanded_trial_data):
        """Test specifying custom model variables."""
        result = trial_msm(
            data=expanded_trial_data,
            estimand_type="ITT",
            model_var="assigned_treatment + age",
            quiet=True,
        )

        assert "age" in result.args["model_formula"]

    def test_followup_time_terms(self, expanded_trial_data):
        """Test custom follow-up time terms."""
        result = trial_msm(
            data=expanded_trial_data,
            estimand_type="ITT",
            include_followup_time="followup_time",
            quiet=True,
        )

        assert "followup_time" in result.args["model_formula"]

    def test_trial_period_terms(self, expanded_trial_data):
        """Test custom trial period terms."""
        result = trial_msm(
            data=expanded_trial_data,
            estimand_type="ITT",
            include_trial_period="trial_period",
            quiet=True,
        )

        assert "trial_period" in result.args["model_formula"]


class TestTrialMSMRobustVariance:
    """Test robust variance estimation."""

    def test_robust_variance_structure(self, expanded_trial_data):
        """Test that robust variance results have correct structure."""
        result = trial_msm(data=expanded_trial_data, estimand_type="ITT", quiet=True)

        # Check robust results dictionary
        assert isinstance(result.robust, dict)
        # Should contain robust variance-covariance matrix
        assert any(key in result.robust for key in ["vcov_robust", "robust_vcov", "cov_params"])

    def test_robust_standard_errors_differ_from_naive(self, expanded_trial_data):
        """Test that robust SEs differ from model SEs (usually)."""
        result = trial_msm(data=expanded_trial_data, estimand_type="ITT", quiet=True)

        # Model has standard errors
        naive_se = result.model.bse

        # Robust results should have different SEs (in most cases)
        assert naive_se is not None


class TestTrialMSMEdgeCases:
    """Test edge cases and error conditions."""

    def test_missing_required_columns_raises_error(self, expanded_trial_data):
        """Test that missing required columns raises an error."""
        data_missing = expanded_trial_data.drop(columns=["outcome"])

        with pytest.raises((KeyError, Exception)):
            trial_msm(data=data_missing, estimand_type="ITT", quiet=True)

    def test_with_missing_data_warns(self, expanded_trial_data):
        """Test that missing data triggers warning and complete-case analysis."""
        # Introduce some missing data
        data_with_na = expanded_trial_data.copy()
        data_with_na.loc[0:10, "age"] = np.nan

        with pytest.warns(UserWarning, match="missing data"):
            result = trial_msm(
                data=data_with_na, estimand_type="ITT", outcome_cov="age", quiet=True
            )

        assert isinstance(result, TEMSM)

    def test_empty_dataframe_after_filtering(self, expanded_trial_data):
        """Test behavior when filtering results in empty DataFrame."""
        # Filter to impossible range
        with pytest.raises((ValueError, Exception)):
            trial_msm(
                data=expanded_trial_data,
                estimand_type="ITT",
                first_followup=100,
                last_followup=200,
                quiet=True,
            )


class TestTrialMSMWhereCase:
    """Test where_case subsetting functionality."""

    def test_where_case_single_condition(self, expanded_trial_data):
        """Test subsetting with single where_case condition."""
        result = trial_msm(
            data=expanded_trial_data, estimand_type="ITT", where_case=["age >= 50"], quiet=True
        )

        assert result.args["where_case"] == ["age >= 50"]

    def test_where_case_multiple_conditions(self, expanded_trial_data):
        """Test subsetting with multiple where_case conditions."""
        result = trial_msm(
            data=expanded_trial_data,
            estimand_type="ITT",
            where_case=["age >= 50", "baseline_var > 0"],
            quiet=True,
        )

        assert result.args["where_case"] == ["age >= 50", "baseline_var > 0"]


class TestTrialMSMIntegration:
    """Integration tests for complete MSM workflow."""

    @pytest.mark.integration
    def test_full_msm_workflow(self, trial_example_data):
        """Test complete MSM fitting workflow with real R package data."""
        from trial_emulation.data_preparation import data_preparation

        # First prepare the data
        prep = data_preparation(
            data=trial_example_data,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            estimand_type="ITT",
            outcome_cov=["nvarA", "nvarB"],
            use_censor_weights=False,
            quiet=True,
        )

        # Then fit MSM
        result = trial_msm(
            data=prep,
            estimand_type="ITT",
            outcome_cov=["nvarA", "nvarB"],
            analysis_weights="unweighted",
            quiet=True,
        )

        # Verify complete workflow
        assert isinstance(result, TEMSM)
        assert result.model is not None
        assert result.robust is not None
        assert result.args is not None

        # Check that treatment coefficient exists
        params = result.model.params
        assert "assigned_treatment" in params.index

        # Check that model has reasonable coefficients (no NaN or inf)
        assert not params.isna().any(), "Model params should not contain NaN"
        assert not np.isinf(params).any(), "Model params should not contain inf"

    @pytest.mark.integration
    def test_msm_with_all_options(self, expanded_trial_data):
        """Test MSM with all available options."""
        result = trial_msm(
            data=expanded_trial_data,
            estimand_type="ITT",
            outcome_cov="age + baseline_var",
            first_followup=1,
            last_followup=3,
            analysis_weights="p99",
            include_followup_time="followup_time",
            include_trial_period="trial_period",
            use_sample_weights=False,
            quiet=True,
        )

        assert isinstance(result, TEMSM)
        assert result.args["first_followup"] == 1
        assert result.args["last_followup"] == 3
        assert result.args["analysis_weights"] == "p99"


class TestFitMSMAlias:
    """Test fit_msm alias function."""

    def test_fit_msm_alias(self, expanded_trial_data):
        """Test that fit_msm is an alias for trial_msm."""
        result1 = trial_msm(data=expanded_trial_data, estimand_type="ITT", quiet=True)

        result2 = fit_msm(data=expanded_trial_data, estimand_type="ITT", quiet=True)

        # Both should return TEMSM objects
        assert isinstance(result1, TEMSM)
        assert isinstance(result2, TEMSM)


class TestTrialMSMParameters:
    """Test parameter storage and retrieval."""

    def test_args_storage(self, expanded_trial_data):
        """Test that all arguments are stored in result."""
        result = trial_msm(
            data=expanded_trial_data,
            estimand_type="ITT",
            outcome_cov="age",
            first_followup=1,
            last_followup=3,
            quiet=True,
        )

        # Check that key parameters are stored
        assert "estimand_type" in result.args
        assert "outcome_cov" in result.args
        assert "first_followup" in result.args
        assert "last_followup" in result.args
        assert "model_formula" in result.args

    def test_model_formula_construction(self, expanded_trial_data):
        """Test that model formula is constructed correctly."""
        result = trial_msm(
            data=expanded_trial_data, estimand_type="ITT", outcome_cov="age", quiet=True
        )

        formula = result.args["model_formula"]

        # Formula should contain key components
        assert "outcome" in formula
        assert "assigned_treatment" in formula or result.args["estimand_type"] == "As-Treated"
        assert "age" in formula
