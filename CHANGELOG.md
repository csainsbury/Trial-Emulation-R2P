# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.4.0] - 2026-01-26

### Added

- Initial Python port of R TrialEmulation package for causal inference in observational time-to-event data
- Data preparation module with trial expansion functionality for target trial emulation
- Inverse probability weighting (IPW) for treatment and censoring
- Marginal structural model (MSM) fitting with statsmodels integration
- Robust variance estimation using sandwich estimators
- Case-control sampling for computational efficiency on large datasets
- C++ extension for censoring calculations (performance optimization with pybind11)
- Support for multiple estimands:
  - Intention-to-treat (ITT)
  - Per-protocol (PP)
  - As-treated
- Comprehensive type hints throughout codebase for better IDE support
- Complete data manipulation utilities for trial emulation workflow
- Formula utilities for converting between R-style and Python statsmodels formulas

### Technical

- Python 3.10+ support with modern type annotations
- Built with pybind11 for high-performance C++ integration
- Uses pandas for efficient data manipulation
- Uses statsmodels for generalized linear model (GLM) fitting
- NumPy integration for numerical computations
- Well-structured modular architecture with clear separation of concerns
- Dataclass-based result objects for type-safe outputs

### Documentation

- Basic README with usage examples
- Comprehensive ARCHITECTURE.md documenting design patterns
- Working example script demonstrating core functionality
- Apache 2.0 license
- Contributing guidelines and code of conduct
- Sphinx documentation structure with ReadTheDocs integration
- Comprehensive test suite documentation (tests/README.md)

### Testing

- Test suite with 65%+ code coverage across core modules
- Integration tests using validated data from R TrialEmulation package
- Real `trial_example` dataset (48,400 observations, 503 patients) from R package
- 36 tests passing covering:
  - Data preparation and trial expansion workflow
  - Multiple estimand types (ITT, PP, As-Treated)
  - Inverse probability weighting
  - Marginal structural model fitting
  - Robust variance estimation
  - Various analysis options and edge cases
- pytest configuration with markers for integration, slow, and C++ tests
- GitHub Actions CI/CD with multi-platform testing (Linux, macOS, Windows)
- Testing on Python 3.10, 3.11, and 3.12

### Notes

This is an alpha release (0.0.x series). The API may change in future releases as we gather user feedback and refine the interface. The package is functional and tested internally but should be considered experimental for production use.

For questions or issues, please visit: https://github.com/csainsbury/Trial-Emulation-R2P/issues

## [Unreleased]

## [0.0.5.2] - 2026-11-13

First PyPI release of the 0.0.5 series (PyPI previously had 0.0.4.0 only), so
it also ships the 0.0.5.0 continuous-outcome MSMs and the 0.0.5.1
bootstrap mode described below.

### Fixed

- **Trial expansion assigned follow-up rows to the wrong trials.** `expand()`
  computed the per-trial "expand until switch" flags in `groupby` key order and
  wrote them back in row order, so the retained rows were a permutation of the
  correct ones. On the R `trial_example` ITT expansion 1,217,947 of 1,939,053
  rows had a missing `weight` and 1,285,822 a missing `assigned_treatment`;
  ITT MSMs only ran because complete-case filtering silently dropped those
  rows. The expansion now follows R's `expand()` exactly (inner join of the
  trial-start attributes, ordered cumulative switch censoring), and is
  vectorised.
- `cumA` (cumulative treatment) only counted the first period, so As-Treated
  `dose` was wrong; it is now the within-patient cumulative sum, as in R.
- Censoring-weight models predicted P(censored) instead of P(uncensored): the
  `1 - cense` response was rewritten by string replacement into `cense`
  (which could also corrupt covariate names containing the censoring variable
  name). Models now use an explicit uncensored indicator, as R's `1 - cense ~ .`.
- Switching and censoring weights used statsmodels `Logit.fittedvalues`,
  which is the linear predictor, not the fitted probability. Weight models are
  now binomial GLMs (as R's `glm`), with fitted probabilities aligned to the
  rows each model used.
- `eligible_wts_0` / `eligible_wts_1` were ignored by `data_preparation`
  because the renamed column was looked up under the user's original name.
- ITT with a pooled numerator and separate denominator censoring models now
  keeps the rows of the pooled numerator model, as R does.
- `p99` analysis weights use R's type-1 quantiles (`inverted_cdf`).
- `case_control_sampling_trials()` with several `p_control` values returns one
  data frame per probability (it previously returned one per trial period with
  a mislabelled `sample_id`).
- Preprocessing: stable sort by patient, `time_on_regime = 0` on the first
  row, event time from each patient's last row, and data-removal messages are
  `warnings` rather than printed text. `keeplist` keeps R's column order.

### Changed

- Weight-model summaries (`switch_models`, `censor_models`) are coefficient
  tables in the layout of R's `broom::tidy` (`term`, `estimate`, `std.error`,
  `statistic`, `p.value`), plus `fit_summary` statistics as in `broom::glance`.
- Requires numpy >= 1.22.

### Validation

- New R parity suite (`tests/test_r_parity.py`) against TrialEmulation 0.0.4.9
  (git f432103, the `R_source` pin) on `trial_example` (ITT, PP with switching
  weights, As-Treated) and `data_censored` (ITT with pooled-numerator censoring
  weights, PP with switching and censoring weights, As-Treated with pooled
  censoring weights). Expanded data match R exactly in row count, keys,
  outcome, treatment, assigned treatment, dose and covariates. Tested
  tolerances (largest observed difference in brackets): weights relative 1e-5
  (1.3e-7); weight-model and MSM coefficients 1e-4 standard errors (1e-6);
  robust standard errors relative 1e-3 against R's HC0 cluster sandwich
  (5.7e-4). R's reported robust SE additionally includes the
  finite-cluster factor sqrt(G/(G - 1)) (1.001 for 503 patients); the Python
  default still omits it (see 0.0.5.0 notes). Reference generator and outputs:
  `tests/data/r_reference/`.
- Simulation tests for continuous outcomes through the full pipeline
  (`tests/test_continuous_simulation.py`): a known effect is recovered by the
  weighted Gaussian MSM for ITT (baseline weights or adjustment) and PP
  (switching weights fitted by `data_preparation`; the unweighted fit is
  biased), and a patient bootstrap using `compute_robust_variance=False`
  reproduces the point estimate and gives a standard error of the sandwich's
  magnitude.
- The As-Treated MSM test now prepares data with `estimand_type="As-Treated"`
  (as R requires, since only that preparation creates `dose`) instead of
  adding a synthetic `dose` column, and no longer skips on Python 3.14.

## [0.0.5.1] - 2026-09-18

### Added

- `compute_robust_variance=False` for outer-resampling workflows that need fitted MSM predictions but obtain uncertainty from the resampling distribution. The GLM, weights, and predictions are unchanged; `TEMSM.robust` is empty and `variance_scope` is `not_computed`.

## [0.0.5.0] - 2026-09-17

### Added

- Gaussian identity-link marginal structural models for continuous outcomes via `trial_msm(outcome_family="gaussian")`
- Explicit patient-clustered estimating-equation sandwich covariance with fixed causal weights
- Formula-row, weight, cluster, outcome-domain, rank, and convergence validation
- Tests for weighted arm-mean identity, manual sandwich agreement, weight-scale invariance, within-cluster row-splitting invariance, and binary compatibility

### Fixed

- Analysis weights are now applied by statsmodels GLM rather than being passed to discrete `Logit.fit()`, where they were not reliable estimation weights
- Robust scores and expected-information bread use a fixed working scale, preserving inference with zero or normalized causal weights
- Missing values in unrelated columns no longer trigger complete-case exclusion; only formula variables determine retained model rows

### Notes

- The sandwich covariance is conditional on fitted weights. Full-pipeline uncertainty for estimated treatment/censoring/measurement weights requires patient bootstrap or stacked estimating equations.
- The robust covariance now defaults to no finite-cluster correction. Earlier releases delegated to statsmodels `cov_cluster`, whose default applied a row/cluster-count correction; set `small_sample_correction=True` to request the explicit conventional factor.
- Use `TEMSM.robust` for fixed-weight inference. `TEMSM.model.bse` and the ordinary statsmodels GLM summary remain working-model quantities, not causal standard errors.
- No likelihood-ratio or information-criterion interpretation is claimed for causally weighted models.

### Planned beyond v0.0.5.0

- Improve test coverage to 80%+ (currently 65%)
- Complete API documentation with docstring examples
- Jupyter notebook tutorials demonstrating real-world use cases
- Performance benchmarks comparing Python vs R implementation
- Additional usage examples (dose-response, time-varying treatments)
- Enhanced error messages and input validation
- Support for additional weight truncation strategies
