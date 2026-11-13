# TrialEmulation (Python)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Build Status](https://github.com/csainsbury/Trial-Emulation-R2P/workflows/Tests/badge.svg)](https://github.com/csainsbury/Trial-Emulation-R2P/actions)
[![Code Coverage](https://img.shields.io/badge/coverage-83%25-green.svg)](https://github.com/csainsbury/Trial-Emulation-R2P)
[![PyPI version](https://badge.fury.io/py/trial-emulation.svg)](https://badge.fury.io/py/trial-emulation)

<!-- Badges to add when infrastructure is ready:
[![Documentation Status](https://readthedocs.org/projects/trial-emulation/badge/?version=latest)](https://trial-emulation.readthedocs.io/en/latest/?badge=latest)
-->

Python port of the R [TrialEmulation](https://github.com/Causal-LDA/TrialEmulation) package for causal inference in observational time-to-event data.

## Overview

The **target trial emulation** framework provides a principled approach to causal inference from observational data by explicitly specifying the hypothetical randomized trial (the "target trial") that would answer the research question of interest. This package implements methods to emulate such target trials using observational data in the person-time format, commonly found in electronic health records and administrative databases.

The core methodology involves expanding longitudinal data into a sequence of "nested" trials, where each eligible person-time contributes to one or more emulated trials depending on when they became eligible for treatment. Marginal structural models (MSMs) are then used to estimate treatment effects while accounting for time-varying confounding through inverse probability weighting.

This Python implementation provides the same functionality as the R TrialEmulation package, with a focus on performance through C++ extensions for computationally intensive operations, comprehensive type hints for better IDE support, and integration with the scientific Python ecosystem (pandas, NumPy, statsmodels).

## Key Features

- **Target Trial Emulation**: Expand observational person-time data into a sequence of emulated trials
- **Multiple Estimands**: Support for intention-to-treat (ITT), per-protocol (PP), and as-treated analyses
- **Inverse Probability Weighting (IPW)**:
  - Treatment switching weights for per-protocol and as-treated estimands
  - Censoring weights to account for informative censoring
  - Flexible covariate specification for numerator and denominator models
- **Marginal Structural Models**: Fit weighted pooled logistic regression for time-to-event outcomes or Gaussian identity-link models for continuous outcomes
- **Causal Analysis Weights**: Apply inverse-probability or overlap weights as estimating-equation weights
- **Robust Variance Estimation**: Patient-clustered estimating-equation sandwich standard errors, with fitted-weight uncertainty handled by full-pipeline bootstrap
- **Case-Control Sampling**: Efficient sampling of controls for large datasets
- **Performance Optimized**: C++ extensions (pybind11) for computationally intensive operations
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Flexible Interface**: Support for R-style formulas and Python-style specifications

## Installation

### From PyPI

```bash
pip install trial-emulation
```

### From Source

```bash
git clone https://github.com/csainsbury/Trial-Emulation-R2P.git
cd Trial-Emulation-R2P
pip install -e .
```

### Development Installation

To install with development dependencies (testing, linting):

```bash
pip install -e ".[dev]"
```

## Requirements

- Python 3.10 or higher
- NumPy >= 1.20.0
- pandas >= 1.3.0
- statsmodels >= 0.13.0
- scipy >= 1.7.0
- pybind11 >= 2.10.0 (for C++ extensions)

## Quick Start

Here's a minimal example demonstrating the core workflow:

```python
import trial_emulation as te
import pandas as pd

# Load your data in long (person-time) format
# Required columns: id, period, treatment, outcome, eligible
data = pd.read_csv("your_data.csv")

# Step 1: Prepare data for trial emulation
# This expands the data into a sequence of emulated trials and calculates weights
prep = te.data_preparation(
    data=data,
    id="id",                    # Patient identifier
    period="period",            # Time period
    treatment="treatment",      # Treatment indicator (0/1)
    outcome="outcome",          # Outcome event indicator (0/1)
    eligible="eligible",        # Eligibility indicator (0/1)
    estimand_type="ITT",        # Intention-to-treat analysis
    outcome_cov=["age", "sex", "comorbidities"],  # Time-varying covariates
    use_censor_weights=True,    # Use censoring weights
    cense="censored",           # Censoring indicator
)

# Step 2: Fit marginal structural model
# This fits a weighted pooled logistic regression with robust standard errors
msm = te.trial_msm(
    data=prep,                   # Prepared data from step 1
    outcome_cov=["age", "sex"],  # Covariates to adjust for
    estimand_type="ITT",
)

# Step 3: View results with robust standard errors
print(msm.robust["summary"])

# Extract treatment effect estimate
treatment_effect = msm.model.params["assigned_treatment"]
robust_se = msm.robust["bse"]["assigned_treatment"]
print(f"Treatment effect: {treatment_effect:.3f} (SE: {robust_se:.3f})")
```

### Understanding the Results

The MSM provides:
- **Coefficients**: Log odds ratios for the effect of treatment on the outcome
- **Robust Standard Errors**: Account for clustering by patient ID
- **Confidence Intervals**: Based on robust standard errors
- **Model Summary**: Full regression output with all covariates

For ITT analyses, the coefficient for `assigned_treatment` represents the effect of being assigned to treatment at trial baseline, regardless of subsequent adherence.

### Continuous outcomes

For a continuous endpoint, provide one or more outcome rows per patient and select the Gaussian identity family. A direct one-index trial frame is appropriate when cohort construction has already selected one baseline trial per patient:

```python
continuous = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "assigned_treatment": [0, 0, 1, 1],
    "outcome": [-1.2, -0.7, -2.1, -1.8],
    "weight": [0.8, 1.1, 0.9, 1.2],
    "followup_time": [0, 0, 0, 0],
    "trial_period": [0, 0, 0, 0],
})

msm = te.trial_msm(
    data=continuous,
    outcome_family="gaussian",
    include_followup_time="~1",
    include_trial_period="~1",
    use_sample_weights=False,
)
```

`weight` is treated as a causal/estimating-equation weight, not an inverse residual-variance weight. The package reports a patient-clustered sandwich covariance conditional on fitted weights. Use `msm.robust["summary"]` and `msm.robust["matrix"]` for fixed-weight inference; `msm.model.bse` and the ordinary GLM summary are working-model quantities, not causal standard errors. If treatment, censoring, or measurement weights were estimated, refit the full nuisance-to-outcome pipeline in a patient bootstrap for final uncertainty.

## Data Format

Your input data should be in **long (person-time) format** with one row per person-period:

| id  | period | treatment | outcome | eligible | age | ... |
|-----|--------|-----------|---------|----------|-----|-----|
| 1   | 0      | 0         | 0       | 1        | 45  | ... |
| 1   | 1      | 1         | 0       | 1        | 45  | ... |
| 1   | 2      | 1         | 0       | 1        | 45  | ... |
| 2   | 0      | 0         | 0       | 1        | 52  | ... |
| 2   | 1      | 0         | 1       | 1        | 52  | ... |

Key requirements:
- **Unique identifier** (`id`): Patient or unit identifier
- **Time period** (`period`): Sequential integer starting at 0
- **Treatment** (`treatment`): Binary indicator (0 = untreated, 1 = treated)
- **Outcome** (`outcome`): Binary indicator for the event of interest
- **Eligibility** (`eligible`): Indicator for whether the person is eligible at that time
- **Covariates**: Time-varying or baseline characteristics

## Documentation

Full documentation is available at [https://trial-emulation.readthedocs.io](https://trial-emulation.readthedocs.io)

### Additional Resources

- [API Reference](https://trial-emulation.readthedocs.io/en/latest/api/)
- [User Guide](https://trial-emulation.readthedocs.io/en/latest/user_guide/)
- [Examples](https://trial-emulation.readthedocs.io/en/latest/examples/)
- [Original R Package Documentation](https://causal-lda.github.io/TrialEmulation/)

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.py` - Simple ITT analysis workflow
- `itt_analysis.py` - Detailed intention-to-treat example
- `per_protocol_analysis.py` - Per-protocol analysis with artificial censoring

## Citation

If you use this package in your research, please cite:

```bibtex
@software{trial_emulation_python,
  title = {TrialEmulation: Target Trial Emulation for Causal Inference},
  author = {Vermeulen, Xander and Sainsbury, Chris},
  year = {2026},
  url = {https://github.com/csainsbury/Trial-Emulation-R2P},
  version = {0.0.5.0}
}
```

For the methodology, please also cite the original R package and key papers:

- Danaei G, García Rodríguez LA, Cantero OF, Logan RW, Hernán MA. Observational data for comparative effectiveness research: An emulation of randomised trials of statins and primary prevention of coronary heart disease. *Statistical Methods in Medical Research*. 2013;22(1):70-96.

- Hernán MA, Robins JM. Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available. *American Journal of Epidemiology*. 2016;183(8):758-764.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to:
- Report bugs
- Suggest features
- Submit pull requests
- Set up a development environment

For major changes, please open an issue first to discuss what you would like to change.

## Development Status

This package is in **alpha** (v0.0.x series). The API may change as we gather user feedback. The package is functional and tested internally but should be considered experimental for production use.

Current priorities:
- Expanding test coverage
- Adding more examples and tutorials
- Performance optimization
- Validation against R package results

## Testing & Validation

### Test Data Sources

The package test suite uses **validated data from the original R TrialEmulation package** to ensure compatibility and correctness. Specifically, we use the `trial_example` dataset which contains:
- 48,400 observations across 503 patients
- Longitudinal structure with realistic treatment patterns
- Known to work correctly with target trial emulation methods
- Same data used in the R package documentation and vignettes

This approach ensures that our Python implementation produces results consistent with the established R implementation.

### Test Coverage

Current test status (as of v0.0.5.2):
- **R parity suite** (`tests/test_r_parity.py`) against TrialEmulation 0.0.4.9 for ITT, PP and As-Treated preparation, switching and censoring weights, weight models and MSM coefficients with robust standard errors (reference outputs and generator in `tests/data/r_reference/`)
- Simulation checks that the weighted Gaussian MSM recovers a known continuous effect (ITT and PP) and that the patient-bootstrap mode is consistent
- ✓ **Multi-platform testing** via GitHub Actions (Linux, macOS, Windows; Python 3.10–3.12)

Key tested functionality:
- Data preparation and trial expansion
- Multiple estimand types (ITT, PP, As-Treated)
- Inverse probability weighting (treatment and censoring)
- Binary and continuous marginal structural model fitting
- Weighted estimating equations and patient-clustered sandwich variance
- Various analysis options (weight truncation, period filtering, etc.)

### Running Tests

To run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=trial_emulation --cov-report=html

# Run only integration tests
pytest tests/ -m integration
```

The test suite includes:
- **Unit tests**: Individual function and module testing
- **Integration tests**: End-to-end workflow validation with real data
- **Edge case tests**: Handling of unusual inputs and boundary conditions

### Validation Against R Package

The Python implementation has been validated against the R package using:
1. Same example datasets (`trial_example`)
2. Comparison of key outputs (expanded data structure, weight calculations)
3. Integration tests that verify the complete workflow produces reasonable results

While minor numerical differences may exist due to differences in optimization algorithms and random number generation, the overall methodology and results are consistent with the R implementation.

## Acknowledgments

This Python implementation is based on the R [TrialEmulation](https://github.com/Causal-LDA/TrialEmulation) package developed by the Causal-LDA team. We thank the original developers for their methodological contributions and open-source implementation.

The target trial emulation framework is built on seminal work by:
- Miguel Hernán and James Robins (Harvard T.H. Chan School of Public Health)
- The CAUSALab team

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/csainsbury/Trial-Emulation-R2P/issues)
- **Discussions**: Ask questions or share ideas in [GitHub Discussions](https://github.com/csainsbury/Trial-Emulation-R2P/discussions)

## Links

- **GitHub**: https://github.com/csainsbury/Trial-Emulation-R2P
- **Original R Package**: https://github.com/Causal-LDA/TrialEmulation

<!-- Links to add when published:
- **PyPI**: https://pypi.org/project/trial-emulation/
- **Documentation**: https://trial-emulation.readthedocs.io
-->
