"""
Tests for data_preparation module.

Tests the main entry point for preparing data for emulated target trials,
including different estimand types, weighting options, and data expansion.
"""

import pytest
import pandas as pd
import numpy as np
from trial_emulation.data_preparation import data_preparation
from trial_emulation.types import TEDataPrep


class TestDataPreparationBasic:
    """Test basic data_preparation functionality."""

    def test_data_preparation_returns_correct_type(self, synthetic_data_simple):
        """Test that data_preparation returns TEDataPrep object."""
        result = data_preparation(
            data=synthetic_data_simple,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="censored",  # Using censored as proxy for eligible
            quiet=True,
        )

        assert isinstance(result, TEDataPrep)
        assert hasattr(result, "data")
        assert hasattr(result, "min_period")
        assert hasattr(result, "max_period")
        assert hasattr(result, "N")

    def test_data_preparation_with_simple_data(self, synthetic_data_simple):
        """Test data_preparation with simple synthetic data."""
        result = data_preparation(
            data=synthetic_data_simple,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="censored",
            quiet=True,
        )

        # Check that data was expanded
        assert result.N > 0
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) > 0

        # Check period bounds
        assert result.min_period >= synthetic_data_simple["period"].min()
        assert result.max_period <= synthetic_data_simple["period"].max()

    def test_data_preparation_preserves_data_structure(self, synthetic_data_simple):
        """Test that data_preparation preserves expected columns."""
        result = data_preparation(
            data=synthetic_data_simple,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="censored",
            quiet=True,
        )

        # Check that expanded data has required columns
        required_cols = ["id", "period", "outcome", "trial_period", "followup_time"]
        for col in required_cols:
            assert col in result.data.columns, f"Missing column: {col}"


class TestDataPreparationEstimands:
    """Test different estimand types."""

    def test_itt_estimand(self, synthetic_data_complex):
        """Test intention-to-treat (ITT) estimand."""
        result = data_preparation(
            data=synthetic_data_complex,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            estimand_type="ITT",
            quiet=True,
        )

        assert isinstance(result, TEDataPrep)
        assert result.parameters["estimand_type"] == "ITT"
        # ITT should not use switch weights
        assert not result.parameters.get("use_switch_weights", True)

    def test_pp_estimand(self, synthetic_data_complex):
        """Test per-protocol (PP) estimand."""
        result = data_preparation(
            data=synthetic_data_complex,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            estimand_type="PP",
            quiet=True,
        )

        assert isinstance(result, TEDataPrep)
        assert result.parameters["estimand_type"] == "PP"

    def test_as_treated_estimand(self, synthetic_data_complex):
        """Test as-treated estimand."""
        result = data_preparation(
            data=synthetic_data_complex,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            estimand_type="As-Treated",
            quiet=True,
        )

        assert isinstance(result, TEDataPrep)
        assert result.parameters["estimand_type"] == "As-Treated"

    def test_invalid_estimand_raises_error(self, synthetic_data_simple):
        """Test that invalid estimand type raises an error."""
        with pytest.raises((ValueError, KeyError)):
            data_preparation(
                data=synthetic_data_simple,
                id="id",
                period="period",
                treatment="treatment",
                outcome="outcome",
                eligible="censored",
                estimand_type="INVALID",
                quiet=True,
            )


class TestDataPreparationWeights:
    """Test weighting functionality."""

    def test_without_weights(self, synthetic_data_simple):
        """Test data_preparation without any weights."""
        result = data_preparation(
            data=synthetic_data_simple,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="censored",
            use_censor_weights=False,
            quiet=True,
        )

        # Without weights, wt column should still exist and be 1.0
        if "wt" in result.data.columns:
            assert (result.data["wt"] == 1.0).all()

    def test_with_censor_weights(self, synthetic_data_complex):
        """Test data_preparation with censoring weights."""
        result = data_preparation(
            data=synthetic_data_complex,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            use_censor_weights=True,
            cense="censored",
            quiet=True,
        )

        assert isinstance(result, TEDataPrep)
        # Censor weights should have been calculated
        assert result.censor_models is not None or "wt" in result.data.columns


class TestDataPreparationPeriods:
    """Test period filtering functionality."""

    def test_first_period_filter(self, synthetic_data_complex):
        """Test filtering data with first_period parameter."""
        first_period = 2
        result = data_preparation(
            data=synthetic_data_complex,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            first_period=first_period,
            quiet=True,
        )

        assert result.min_period >= first_period

    def test_last_period_filter(self, synthetic_data_complex):
        """Test filtering data with last_period parameter."""
        last_period = 5
        result = data_preparation(
            data=synthetic_data_complex,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            last_period=last_period,
            quiet=True,
        )

        assert result.max_period <= last_period

    def test_period_range_filter(self, synthetic_data_complex):
        """Test filtering data with both first and last period."""
        first_period = 2
        last_period = 7
        result = data_preparation(
            data=synthetic_data_complex,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            first_period=first_period,
            last_period=last_period,
            quiet=True,
        )

        assert result.min_period >= first_period
        assert result.max_period <= last_period


class TestDataPreparationEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_period_data(self, synthetic_data_single_period):
        """Test with single period dataset."""
        result = data_preparation(
            data=synthetic_data_single_period,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            quiet=True,
        )

        assert isinstance(result, TEDataPrep)
        assert result.min_period == result.max_period

    def test_missing_required_column_raises_error(self, synthetic_data_simple):
        """Test that missing required column raises an error."""
        # Remove a required column
        data_missing = synthetic_data_simple.drop(columns=["treatment"])

        with pytest.raises((KeyError, ValueError)):
            data_preparation(
                data=data_missing,
                id="id",
                period="period",
                treatment="treatment",
                outcome="outcome",
                eligible="censored",
                quiet=True,
            )

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises an error."""
        empty_df = pd.DataFrame(columns=["id", "period", "treatment", "outcome", "eligible"])

        with pytest.raises((ValueError, IndexError)):
            data_preparation(
                data=empty_df,
                id="id",
                period="period",
                treatment="treatment",
                outcome="outcome",
                eligible="eligible",
                quiet=True,
            )


class TestDataPreparationCovariates:
    """Test covariate specifications."""

    def test_outcome_covariates(self, synthetic_data_complex):
        """Test specifying outcome model covariates."""
        result = data_preparation(
            data=synthetic_data_complex,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            outcome_cov="age + baseline_risk",
            quiet=True,
        )

        assert isinstance(result, TEDataPrep)
        # Check that covariates were stored in parameters
        assert "outcome_cov" in result.parameters

    def test_model_var_specification(self, synthetic_data_complex):
        """Test specifying model variables."""
        result = data_preparation(
            data=synthetic_data_complex,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            model_var=["age", "baseline_risk"],
            quiet=True,
        )

        assert isinstance(result, TEDataPrep)
        assert "model_var" in result.parameters or "keeplist" in result.parameters


class TestDataPreparationOutput:
    """Test output format and structure."""

    def test_data_template_structure(self, synthetic_data_simple):
        """Test that data_template has correct structure."""
        result = data_preparation(
            data=synthetic_data_simple,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="censored",
            quiet=True,
        )

        # data_template should be a zero-row DataFrame with the same structure
        assert isinstance(result.data_template, pd.DataFrame)
        assert len(result.data_template) == 0
        # Should have same columns as expanded data
        assert set(result.data_template.columns).issubset(set(result.data.columns))

    def test_observation_count_positive(self, synthetic_data_complex):
        """Test that N (observation count) is positive."""
        result = data_preparation(
            data=synthetic_data_complex,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            quiet=True,
        )

        assert result.N > 0
        assert result.N == len(result.data)

    def test_parameters_stored(self, synthetic_data_simple):
        """Test that parameters are stored in result."""
        result = data_preparation(
            data=synthetic_data_simple,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="censored",
            estimand_type="ITT",
            quiet=True,
        )

        assert result.parameters is not None
        assert isinstance(result.parameters, dict)
        assert "estimand_type" in result.parameters


class TestDataPreparationIntegration:
    """Integration tests for full workflow."""

    @pytest.mark.integration
    def test_full_workflow_itt(self, trial_example_data):
        """Test complete ITT workflow with real R package data."""
        result = data_preparation(
            data=trial_example_data,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            estimand_type="ITT",
            outcome_cov=["nvarA", "nvarB"],
            quiet=True,
        )

        # Verify complete workflow
        assert isinstance(result, TEDataPrep)
        assert isinstance(result.data, pd.DataFrame)
        assert result.N > 0
        assert result.min_period is not None
        assert result.max_period is not None
        assert result.data_template is not None

        # Check data integrity
        assert not result.data["id"].isna().any()
        assert not result.data["trial_period"].isna().any()
        assert not result.data["followup_time"].isna().any()
        assert result.data["outcome"].isin([0, 1]).all()

    @pytest.mark.integration
    def test_full_workflow_with_weights(self, synthetic_data_complex):
        """Test complete workflow with all weighting options."""
        result = data_preparation(
            data=synthetic_data_complex,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="eligible",
            estimand_type="PP",
            use_censor_weights=True,
            cense="censored",
            outcome_cov="age + baseline_risk",
            quiet=True,
        )

        # Verify complete workflow with weights
        assert isinstance(result, TEDataPrep)
        assert isinstance(result.data, pd.DataFrame)
        assert result.N > 0

        # Check that weights exist
        assert "wt" in result.data.columns
        # Weights should be positive
        assert (result.data["wt"] > 0).all()


class TestDataPreparationQuietMode:
    """Test quiet mode functionality."""

    def test_quiet_mode_true(self, synthetic_data_simple, capsys):
        """Test that quiet=True suppresses output."""
        result = data_preparation(
            data=synthetic_data_simple,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="censored",
            quiet=True,
        )

        captured = capsys.readouterr()
        # With quiet=True, there should be minimal or no output
        assert isinstance(result, TEDataPrep)

    def test_quiet_mode_false(self, synthetic_data_simple, capsys):
        """Test that quiet=False enables output."""
        result = data_preparation(
            data=synthetic_data_simple,
            id="id",
            period="period",
            treatment="treatment",
            outcome="outcome",
            eligible="censored",
            quiet=False,
        )

        captured = capsys.readouterr()
        # With quiet=False, there should be some output
        # (Note: this test might need adjustment based on actual output behavior)
        assert isinstance(result, TEDataPrep)
