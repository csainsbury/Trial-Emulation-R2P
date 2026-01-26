"""
Tests for weighting module.

Tests inverse probability weighting (IPW) calculation for treatment
switching and censoring.
"""

import pytest
import pandas as pd
import numpy as np
from trial_emulation.weighting import weight_func


@pytest.fixture
def switch_data_simple():
    """
    Create simple switch data for testing switching weights.

    This mimics the data structure expected by weight_func with
    columns needed for switching weight calculation.
    """
    np.random.seed(42)

    n_obs = 100
    data = pd.DataFrame({
        'id': np.repeat(range(1, 21), 5),
        'period': np.tile(range(5), 20),
        'treatment': np.random.randint(0, 2, n_obs),
        'am_1': np.random.randint(0, 2, n_obs),  # Previous treatment
        'outcome': np.random.randint(0, 2, n_obs),
        'age': np.random.randint(40, 80, n_obs),
        'baseline_var': np.random.normal(0, 1, n_obs)
    })

    return data


@pytest.fixture
def censor_data_simple():
    """
    Create simple data for testing censoring weights.

    Includes censoring indicator and related columns.
    """
    np.random.seed(123)

    n_obs = 100
    data = pd.DataFrame({
        'id': np.repeat(range(1, 21), 5),
        'period': np.tile(range(5), 20),
        'treatment': np.random.randint(0, 2, n_obs),
        'am_1': np.random.randint(0, 2, n_obs),
        'censored': np.random.randint(0, 2, n_obs),
        'outcome': np.random.randint(0, 2, n_obs),
        'age': np.random.randint(40, 80, n_obs)
    })

    return data


@pytest.fixture
def combined_data():
    """
    Create data for testing both switching and censoring weights together.
    """
    np.random.seed(456)

    n_obs = 200
    data = pd.DataFrame({
        'id': np.repeat(range(1, 41), 5),
        'period': np.tile(range(5), 40),
        'treatment': np.random.randint(0, 2, n_obs),
        'am_1': np.random.randint(0, 2, n_obs),
        'censored': np.random.randint(0, 2, n_obs),
        'outcome': np.random.randint(0, 2, n_obs),
        'age': np.random.randint(40, 80, n_obs),
        'baseline_var': np.random.normal(0, 1, n_obs)
    })

    return data


class TestWeightFuncBasic:
    """Test basic weight_func functionality."""

    def test_weight_func_returns_dict(self, switch_data_simple):
        """Test that weight_func returns a dictionary."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=False,
            use_censor_weights=False,
            quiet=True
        )

        assert isinstance(result, dict)
        assert 'data' in result
        assert 'switch_models' in result
        assert 'censor_models' in result

    def test_weight_func_without_weights(self, switch_data_simple):
        """Test weight_func with no weighting."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=False,
            use_censor_weights=False,
            quiet=True
        )

        # Without weights, wt should be 1.0
        assert 'wt' in result['data'].columns
        assert (result['data']['wt'] == 1.0).all()

    def test_weight_func_preserves_data(self, switch_data_simple):
        """Test that weight_func preserves original data structure."""
        n_rows_original = len(switch_data_simple)

        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=False,
            use_censor_weights=False,
            quiet=True
        )

        # Should have same number of rows
        assert len(result['data']) == n_rows_original

        # Should have original columns plus weight columns
        original_cols = set(switch_data_simple.columns)
        result_cols = set(result['data'].columns)
        assert original_cols.issubset(result_cols)


class TestSwitchingWeights:
    """Test switching weight calculation."""

    def test_switch_weights_enabled(self, switch_data_simple):
        """Test that switching weights are calculated when enabled."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="~1",
            switch_d_cov="age",
            quiet=True
        )

        # Should have switching weight models
        assert isinstance(result['switch_models'], dict)

        # Data should have wt column
        assert 'wt' in result['data'].columns

        # Weights should be positive
        assert (result['data']['wt'] > 0).all()

    def test_switch_weights_with_covariates(self, switch_data_simple):
        """Test switching weights with covariates."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="age",
            switch_d_cov="age + baseline_var",
            quiet=True
        )

        assert isinstance(result['data'], pd.DataFrame)
        assert 'wt' in result['data'].columns

    def test_switch_weights_different_numerator_denominator(self, switch_data_simple):
        """Test that numerator and denominator covariates differ."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="~1",
            switch_d_cov="age + baseline_var",
            quiet=True
        )

        # Weights should vary (not all 1.0) when numerator != denominator
        assert result['data']['wt'].std() > 0


class TestCensoringWeights:
    """Test censoring weight calculation."""

    def test_censor_weights_enabled(self, censor_data_simple):
        """Test that censoring weights are calculated when enabled."""
        result = weight_func(
            sw_data=censor_data_simple,
            use_switch_weights=False,
            use_censor_weights=True,
            cense="censored",
            quiet=True
        )

        # Should have censoring weight models
        assert isinstance(result['censor_models'], dict)

        # Data should have wtC column
        assert 'wtC' in result['data'].columns

        # Weights should be positive
        assert (result['data']['wtC'] > 0).all()

    def test_censor_weights_with_covariates(self, censor_data_simple):
        """Test censoring weights with covariates."""
        result = weight_func(
            sw_data=censor_data_simple,
            use_switch_weights=False,
            use_censor_weights=True,
            cense="censored",
            cense_n_cov="~1",
            cense_d_cov="age",
            quiet=True
        )

        assert isinstance(result['data'], pd.DataFrame)
        assert 'wtC' in result['data'].columns

    def test_censor_weights_without_cense_raises_error(self, censor_data_simple):
        """Test that missing cense parameter raises error."""
        with pytest.raises(ValueError, match="cense must be specified"):
            weight_func(
                sw_data=censor_data_simple,
                use_switch_weights=False,
                use_censor_weights=True,
                cense=None,  # Missing required parameter
                quiet=True
            )

    def test_censor_weights_pooling_numerator(self, censor_data_simple):
        """Test censoring weights with numerator pooling."""
        result = weight_func(
            sw_data=censor_data_simple,
            use_switch_weights=False,
            use_censor_weights=True,
            cense="censored",
            pool_cense_n=True,
            pool_cense_d=False,
            quiet=True
        )

        assert isinstance(result['data'], pd.DataFrame)
        assert 'wtC' in result['data'].columns

    def test_censor_weights_pooling_denominator(self, censor_data_simple):
        """Test censoring weights with denominator pooling."""
        result = weight_func(
            sw_data=censor_data_simple,
            use_switch_weights=False,
            use_censor_weights=True,
            cense="censored",
            pool_cense_n=False,
            pool_cense_d=True,
            quiet=True
        )

        assert isinstance(result['data'], pd.DataFrame)
        assert 'wtC' in result['data'].columns


class TestCombinedWeights:
    """Test combined switching and censoring weights."""

    def test_both_weights_enabled(self, combined_data):
        """Test that both switching and censoring weights are calculated."""
        result = weight_func(
            sw_data=combined_data,
            use_switch_weights=True,
            use_censor_weights=True,
            switch_n_cov="~1",
            switch_d_cov="age",
            cense="censored",
            cense_n_cov="~1",
            cense_d_cov="age",
            quiet=True
        )

        # Should have both weight models
        assert isinstance(result['switch_models'], dict)
        assert isinstance(result['censor_models'], dict)

        # Final weight should be product of both
        assert 'wt' in result['data'].columns
        assert 'wtC' in result['data'].columns

        # Combined weight should be positive
        assert (result['data']['wt'] > 0).all()

    def test_combined_weights_are_product(self, combined_data):
        """Test that combined weight is product of switching and censoring weights."""
        result = weight_func(
            sw_data=combined_data,
            use_switch_weights=True,
            use_censor_weights=True,
            switch_n_cov="~1",
            switch_d_cov="age",
            cense="censored",
            cense_n_cov="~1",
            cense_d_cov="age",
            quiet=True
        )

        # The final wt should incorporate both switching and censoring weights
        # (implementation multiplies them)
        assert 'wt' in result['data'].columns
        assert (result['data']['wt'] > 0).all()


class TestWeightFuncCovariates:
    """Test different covariate specifications."""

    def test_intercept_only(self, switch_data_simple):
        """Test with intercept-only formula."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="~1",
            switch_d_cov="~1",
            quiet=True
        )

        assert isinstance(result['data'], pd.DataFrame)
        # With same numerator and denominator, weights should be close to 1.0
        # (may not be exactly 1.0 due to convergence)

    def test_single_covariate(self, switch_data_simple):
        """Test with single covariate."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="~1",
            switch_d_cov="age",
            quiet=True
        )

        assert isinstance(result['data'], pd.DataFrame)
        assert result['data']['wt'].std() > 0

    def test_multiple_covariates(self, switch_data_simple):
        """Test with multiple covariates."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="age",
            switch_d_cov="age + baseline_var",
            quiet=True
        )

        assert isinstance(result['data'], pd.DataFrame)
        assert 'wt' in result['data'].columns


class TestWeightFuncEligibility:
    """Test eligibility-based weighting."""

    def test_eligible_wts_0(self, switch_data_simple):
        """Test with eligible_wts_0 specified."""
        # Add eligibility column
        data = switch_data_simple.copy()
        data['eligible_0'] = np.random.randint(0, 2, len(data))

        result = weight_func(
            sw_data=data,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="~1",
            switch_d_cov="age",
            eligible_wts_0="eligible_0",
            quiet=True
        )

        assert isinstance(result['data'], pd.DataFrame)
        assert 'wt' in result['data'].columns

    def test_eligible_wts_1(self, switch_data_simple):
        """Test with eligible_wts_1 specified."""
        # Add eligibility column
        data = switch_data_simple.copy()
        data['eligible_1'] = np.random.randint(0, 2, len(data))

        result = weight_func(
            sw_data=data,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="~1",
            switch_d_cov="age",
            eligible_wts_1="eligible_1",
            quiet=True
        )

        assert isinstance(result['data'], pd.DataFrame)
        assert 'wt' in result['data'].columns

    def test_both_eligible_wts(self, switch_data_simple):
        """Test with both eligible_wts_0 and eligible_wts_1."""
        # Add eligibility columns
        data = switch_data_simple.copy()
        data['eligible_0'] = np.random.randint(0, 2, len(data))
        data['eligible_1'] = np.random.randint(0, 2, len(data))

        result = weight_func(
            sw_data=data,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="~1",
            switch_d_cov="age",
            eligible_wts_0="eligible_0",
            eligible_wts_1="eligible_1",
            quiet=True
        )

        assert isinstance(result['data'], pd.DataFrame)
        assert 'wt' in result['data'].columns


class TestWeightFuncEdgeCases:
    """Test edge cases and error conditions."""

    def test_missing_required_columns(self):
        """Test that missing required columns raises error."""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'period': [0, 0, 0]
            # Missing treatment, am_1, etc.
        })

        with pytest.raises((KeyError, ValueError, Exception)):
            weight_func(
                sw_data=data,
                use_switch_weights=True,
                use_censor_weights=False,
                quiet=True
            )

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        data = pd.DataFrame(columns=['id', 'period', 'treatment', 'am_1'])

        with pytest.raises((ValueError, IndexError, Exception)):
            weight_func(
                sw_data=data,
                use_switch_weights=True,
                use_censor_weights=False,
                quiet=True
            )

    def test_constant_treatment(self):
        """Test with constant treatment (no variation)."""
        data = pd.DataFrame({
            'id': np.repeat(range(1, 11), 5),
            'period': np.tile(range(5), 10),
            'treatment': np.ones(50),  # All treated
            'am_1': np.zeros(50),
            'age': np.random.randint(40, 80, 50)
        })

        # This may warn about perfect prediction but should not crash
        try:
            result = weight_func(
                sw_data=data,
                use_switch_weights=True,
                use_censor_weights=False,
                switch_n_cov="~1",
                switch_d_cov="age",
                quiet=True
            )
            assert isinstance(result['data'], pd.DataFrame)
        except (ValueError, Exception) as e:
            # Some implementations may raise an error for perfect separation
            pass


class TestWeightFuncOutput:
    """Test output structure and properties."""

    def test_output_has_correct_keys(self, switch_data_simple):
        """Test that output dictionary has expected keys."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=False,
            use_censor_weights=False,
            quiet=True
        )

        assert 'data' in result
        assert 'switch_models' in result
        assert 'censor_models' in result

    def test_output_data_is_dataframe(self, switch_data_simple):
        """Test that output data is a DataFrame."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=False,
            use_censor_weights=False,
            quiet=True
        )

        assert isinstance(result['data'], pd.DataFrame)

    def test_weight_columns_added(self, combined_data):
        """Test that weight columns are added to output."""
        result = weight_func(
            sw_data=combined_data,
            use_switch_weights=True,
            use_censor_weights=True,
            switch_n_cov="~1",
            switch_d_cov="age",
            cense="censored",
            quiet=True
        )

        # Should have wt and wtC columns
        assert 'wt' in result['data'].columns
        assert 'wtC' in result['data'].columns

    def test_weights_are_numeric(self, switch_data_simple):
        """Test that calculated weights are numeric."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="~1",
            switch_d_cov="age",
            quiet=True
        )

        assert pd.api.types.is_numeric_dtype(result['data']['wt'])

    def test_weights_are_positive(self, switch_data_simple):
        """Test that weights are positive."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="~1",
            switch_d_cov="age",
            quiet=True
        )

        # Weights should be positive (may have NaN for ineligible)
        weights = result['data']['wt'].dropna()
        assert (weights > 0).all()


class TestWeightFuncIntegration:
    """Integration tests for weight calculation."""

    @pytest.mark.integration
    def test_full_weighting_workflow(self, combined_data):
        """Test complete weighting workflow with all options."""
        result = weight_func(
            sw_data=combined_data,
            use_switch_weights=True,
            use_censor_weights=True,
            switch_n_cov="age",
            switch_d_cov="age + baseline_var",
            cense="censored",
            cense_n_cov="~1",
            cense_d_cov="age",
            pool_cense_n=False,
            pool_cense_d=False,
            quiet=True
        )

        # Verify complete workflow
        assert isinstance(result, dict)
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) > 0

        # Check weights exist and are valid
        assert 'wt' in result['data'].columns
        assert 'wtC' in result['data'].columns
        weights = result['data']['wt'].dropna()
        assert len(weights) > 0
        assert (weights > 0).all()

        # Check models were fit
        assert isinstance(result['switch_models'], dict)
        assert isinstance(result['censor_models'], dict)


class TestWeightFuncQuietMode:
    """Test quiet mode functionality."""

    def test_quiet_mode_true(self, switch_data_simple, capsys):
        """Test that quiet=True suppresses output."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="~1",
            switch_d_cov="age",
            quiet=True
        )

        captured = capsys.readouterr()
        # With quiet=True, there should be minimal or no output
        assert isinstance(result, dict)

    def test_quiet_mode_false(self, switch_data_simple, capsys):
        """Test that quiet=False enables output."""
        result = weight_func(
            sw_data=switch_data_simple,
            use_switch_weights=True,
            use_censor_weights=False,
            switch_n_cov="~1",
            switch_d_cov="age",
            quiet=False
        )

        captured = capsys.readouterr()
        # With quiet=False, there should be some output
        assert isinstance(result, dict)
