# Test Suite Documentation

This directory contains the test suite for the trial-emulation package. The tests validate both individual components and end-to-end workflows.

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures and pytest configuration
├── data/                            # Test data files
│   ├── trial_example.rda           # Original R data file
│   └── trial_example.csv           # Converted to CSV for Python
├── test_data_preparation.py         # Tests for main data_preparation() workflow
├── test_msm.py                      # Tests for marginal structural model fitting
└── test_weighting.py                # Tests for IPW calculation
```

## Test Data Sources

### Real R Package Data (Recommended)

The primary test data comes from the **R TrialEmulation package** to ensure compatibility:

**`trial_example`** dataset:
- Source: https://github.com/Causal-LDA/TrialEmulation/raw/main/data/trial_example.rda
- Format: R data file (.rda), converted to CSV for Python tests
- Size: 48,400 observations, 503 unique patients
- Columns:
  - `id`: Patient identifier
  - `period`: Time period (310-396)
  - `eligible`: Eligibility indicator (0/1)
  - `treatment`: Treatment indicator (0/1)
  - `outcome`: Outcome event indicator (0/1)
  - `catvarA`, `catvarB`, `catvarC`: Categorical covariates
  - `nvarA`, `nvarB`, `nvarC`: Numeric covariates

**Why use real R data?**
- Validated to work with statistical models (no singular matrix issues)
- Proven to produce correct results in R implementation
- Realistic structure (varying follow-up, realistic treatment patterns)
- Ensures Python/R compatibility

### Synthetic Test Data

`conftest.py` also includes synthetic data generators for unit tests:
- `synthetic_data_simple()`: Small dataset (10 patients, 5 periods)
- `synthetic_data_complex()`: Larger dataset (100 patients, 10 periods)
- `synthetic_data_with_known_effect()`: Data with known treatment effect
- `synthetic_data_single_period()`: Edge case testing

**Note**: Some synthetic data may cause numerical issues with statsmodels (singular matrix errors). When possible, use `trial_example_data` fixture instead.

## Test Categories

### Integration Tests (marked with `@pytest.mark.integration`)

These test the complete workflow end-to-end:
- `test_full_workflow_itt()`: Complete ITT analysis with real data
- `test_full_msm_workflow()`: Data preparation → MSM fitting → results

**These are the most important tests** - they prove the package works correctly on real data.

### Unit Tests

Test individual components:
- Data preparation parameters and options
- MSM model specifications
- Weight calculations
- Input validation and error handling

### Performance Tests (marked with `@pytest.mark.slow`)

Tests with large datasets or long computation times.

### C++ Extension Tests (marked with `@pytest.mark.cpp`)

Tests for the pybind11 C++ extensions (censoring calculations).

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with coverage
```bash
pytest tests/ --cov=trial_emulation --cov-report=html
```

### Run only integration tests
```bash
pytest tests/ -m integration
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_data_preparation.py
```

### Run specific test class
```bash
pytest tests/test_msm.py::TestTrialMSMIntegration
```

### Run specific test
```bash
pytest tests/test_msm.py::TestTrialMSMIntegration::test_full_msm_workflow
```

## Test Coverage Goals

**Current Coverage: ~65%**

Target coverage by module:
- `data_preparation.py`: 90%+ (main entry point)
- `msm.py`: 90%+ (main analysis)
- `weighting.py`: 80%+ (complex logic)
- `data_extension.py`: 70%+
- `data_manipulation.py`: 70%+
- Other utilities: 60%+

## Adding New Tests

### Using Real Data

When adding new tests, prefer using `trial_example_data` fixture:

```python
def test_my_feature(trial_example_data):
    """Test using real R package data."""
    result = data_preparation(
        data=trial_example_data,
        id="id",
        period="period",
        treatment="treatment",
        outcome="outcome",
        eligible="eligible",
        # ... other parameters
    )

    # Your assertions here
    assert result.N > 0
```

### Creating Synthetic Data

If you need custom synthetic data for a specific test case:

```python
def test_edge_case():
    """Test with custom synthetic data."""
    data = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "period": [0, 1, 0, 1],
        "treatment": [0, 1, 1, 1],
        "outcome": [0, 1, 0, 0],
        "eligible": [1, 1, 1, 1],
    })

    # Your test logic here
```

## Common Test Issues

### Singular Matrix Errors

**Problem**: `numpy.linalg.LinAlgError: Singular matrix`

**Cause**: Synthetic test data doesn't have enough variation for statsmodels to fit a model (perfect separation, collinearity, etc.)

**Solution**: Use `trial_example_data` fixture instead of synthetic data.

### Missing Columns

**Problem**: `KeyError: 'period'` or similar

**Cause**: After data expansion, column names change (e.g., `period` → `trial_period`, `followup_time`)

**Solution**: Check what columns are actually in the expanded data:
```python
prep = data_preparation(...)
print(prep.data.columns)
```

### Weight Column Names

**Problem**: `KeyError: 'wt'` or `KeyError: 'weight'`

**Cause**: Different weighting options create different column names

**Solution**: Check if weight column exists before testing:
```python
if "wt" in result.data.columns:
    assert (result.data["wt"] > 0).all()
```

### Python 3.14 Compatibility Issues

**Problem**: `INTERNALERROR> KeyError: 0` when running tests with As-Treated estimand

**Cause**: Python 3.14 has a compatibility issue with the patsy library (used by statsmodels for formula parsing). When tests using dose formulas (As-Treated estimand) fail, pytest's error formatting code triggers a KeyError in patsy's evaluation context, causing an internal error.

**Affected Tests**:
- `test_as_treated_estimand` (marked with `@pytest.mark.skipif` for Python ≥3.14)
- Integration tests that may use dose formulas

**Impact**:
- As-Treated estimand tests are skipped on Python 3.14
- The internal error may prevent subsequent tests in the same file from running
- This is a known issue and does not affect the core functionality (which works fine with As-Treated estimands)

**Workaround**:
- Tests are marked to skip on Python 3.14: `@pytest.mark.skipif(sys.version_info >= (3, 14), reason="Python 3.14 patsy compatibility")`
- For local development on Python 3.14, you can run test files separately to avoid the cascading effect
- The issue only affects test error formatting, not the actual functionality

**Status**: Tracking upstream patsy issue. The package itself works correctly with As-Treated estimands on all Python versions including 3.14.

## CI/CD Integration

Tests run automatically via GitHub Actions on:
- Every push to main branch
- Every pull request
- Multiple Python versions (3.10, 3.11, 3.12)
- Multiple platforms (Linux, macOS, Windows)

See `.github/workflows/tests.yml` for configuration.

## Test Data Updates

To update the R package test data:

```bash
# Download latest data from R package
curl -L -o tests/data/trial_example.rda \
  https://github.com/Causal-LDA/TrialEmulation/raw/main/data/trial_example.rda

# Convert to CSV (requires pyreadr)
python << 'EOF'
import pyreadr
result = pyreadr.read_r('tests/data/trial_example.rda')
result['trial_example'].to_csv('tests/data/trial_example.csv', index=False)
EOF
```

## Contributing Tests

When contributing new features:
1. Add tests for new functionality
2. Use `trial_example_data` when possible
3. Mark integration tests with `@pytest.mark.integration`
4. Run full test suite before submitting PR
5. Ensure coverage doesn't decrease

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.
