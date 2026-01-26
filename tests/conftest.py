"""
Pytest configuration and shared fixtures for trial_emulation tests.

This module provides reusable test fixtures including synthetic datasets
of varying complexity for testing the trial emulation functionality.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data_simple():
    """
    Generate a simple synthetic dataset for basic tests.

    Dataset characteristics:
    - 10 patients (IDs)
    - 5 time periods
    - Simple treatment and outcome patterns
    - No missing data

    Returns:
        pd.DataFrame: Simple synthetic dataset
    """
    np.random.seed(42)

    n_ids = 10
    n_periods = 5

    data = []
    for patient_id in range(1, n_ids + 1):
        for period in range(n_periods):
            # Simple patterns for testing
            treatment = 1 if period >= 2 and patient_id % 2 == 0 else 0
            outcome = 1 if treatment == 1 and period >= 3 and np.random.random() > 0.7 else 0
            censored = 1 if outcome == 1 else 0

            data.append({
                'id': patient_id,
                'period': period,
                'treatment': treatment,
                'outcome': outcome,
                'censored': censored,
                'age': 50 + patient_id * 2,
                'baseline_var': patient_id / 10.0
            })

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_data_complex():
    """
    Generate a more complex synthetic dataset for advanced tests.

    Dataset characteristics:
    - 100 patients
    - 10 time periods
    - More realistic treatment and outcome patterns
    - Time-varying covariates
    - Realistic censoring patterns

    Returns:
        pd.DataFrame: Complex synthetic dataset
    """
    np.random.seed(123)

    n_ids = 100
    n_periods = 10

    data = []
    for patient_id in range(1, n_ids + 1):
        # Baseline characteristics
        age = np.random.randint(40, 80)
        baseline_risk = np.random.random()

        # Track patient state
        treated = False
        has_outcome = False

        for period in range(n_periods):
            # Skip if already has outcome (censored)
            if has_outcome:
                continue

            # Treatment initiation probability increases with age and baseline risk
            if not treated:
                treatment_prob = 0.1 + 0.01 * (age - 40) + 0.2 * baseline_risk
                treated = np.random.random() < treatment_prob

            treatment = 1 if treated else 0

            # Outcome probability depends on treatment, age, and time
            outcome_prob = 0.05 + 0.01 * period
            if treatment:
                outcome_prob *= 0.6  # Treatment reduces outcome risk
            outcome_prob *= (1 + baseline_risk)

            outcome = 1 if np.random.random() < outcome_prob else 0
            if outcome:
                has_outcome = True

            # Time-varying covariate
            time_var_covariate = np.random.normal(100, 15)

            data.append({
                'id': patient_id,
                'period': period,
                'treatment': treatment,
                'outcome': outcome,
                'censored': outcome,
                'age': age,
                'baseline_risk': baseline_risk,
                'time_var_covariate': time_var_covariate,
                'eligible': 1  # All eligible for simplicity
            })

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_data_with_known_effect():
    """
    Generate synthetic data with a known treatment effect for validation.

    The treatment effect is set to reduce outcome risk by exactly 50%
    (hazard ratio = 0.5), which can be verified in the analysis.

    Dataset characteristics:
    - 200 patients
    - 8 time periods
    - Known treatment effect (HR = 0.5)
    - Balanced treatment groups

    Returns:
        pd.DataFrame: Dataset with known treatment effect
    """
    np.random.seed(456)

    n_ids = 200
    n_periods = 8
    baseline_hazard = 0.05  # 5% baseline risk per period
    hazard_ratio = 0.5  # Known treatment effect

    data = []
    for patient_id in range(1, n_ids + 1):
        # Randomize treatment (50% treated)
        treated = patient_id <= n_ids // 2
        has_outcome = False

        for period in range(n_periods):
            if has_outcome:
                continue

            treatment = 1 if treated else 0

            # Calculate outcome probability based on known hazard ratio
            hazard = baseline_hazard
            if treatment:
                hazard = baseline_hazard * hazard_ratio

            outcome = 1 if np.random.random() < hazard else 0
            if outcome:
                has_outcome = True

            data.append({
                'id': patient_id,
                'period': period,
                'treatment': treatment,
                'outcome': outcome,
                'censored': outcome,
                'age': np.random.randint(45, 75),
                'eligible': 1
            })

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_data_single_period():
    """
    Generate a minimal dataset with a single period for edge case testing.

    Returns:
        pd.DataFrame: Single period dataset
    """
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'period': [0, 0, 0, 0, 0],
        'treatment': [0, 1, 0, 1, 1],
        'outcome': [0, 0, 1, 0, 0],
        'censored': [0, 0, 1, 0, 0],
        'age': [50, 55, 60, 65, 70],
        'eligible': [1, 1, 1, 1, 1]
    })


@pytest.fixture
def synthetic_data_with_ineligible():
    """
    Generate dataset with some ineligible observations for testing eligibility logic.

    Returns:
        pd.DataFrame: Dataset with eligibility variations
    """
    np.random.seed(789)

    data = []
    for patient_id in range(1, 21):
        for period in range(5):
            # Some patients become ineligible after period 2
            eligible = 1 if period < 3 or patient_id % 3 == 0 else 0

            data.append({
                'id': patient_id,
                'period': period,
                'treatment': np.random.randint(0, 2),
                'outcome': 0,
                'censored': 0,
                'age': 50 + patient_id,
                'eligible': eligible
            })

    return pd.DataFrame(data)


@pytest.fixture
def mock_weights():
    """
    Generate mock weight arrays for testing weighting functionality.

    Returns:
        dict: Dictionary containing different weight arrays
    """
    np.random.seed(42)
    n = 100

    return {
        'uniform': np.ones(n),
        'random': np.random.uniform(0.5, 2.0, n),
        'extreme': np.random.uniform(0.01, 10.0, n),
        'with_zeros': np.concatenate([np.zeros(10), np.random.uniform(0.5, 2.0, 90)])
    }


@pytest.fixture
def sample_formula_data():
    """
    Generate sample data for testing formula utilities.

    Returns:
        pd.DataFrame: Data with various column types
    """
    np.random.seed(42)

    return pd.DataFrame({
        'outcome': np.random.randint(0, 2, 50),
        'treatment': np.random.randint(0, 2, 50),
        'continuous_var': np.random.normal(0, 1, 50),
        'categorical_var': np.random.choice(['A', 'B', 'C'], 50),
        'time': np.repeat(range(5), 10)
    })


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "cpp: marks tests that require C++ extension"
    )
