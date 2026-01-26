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

### Planned for v0.0.5.0

- Improve test coverage to 80%+ (currently 65%)
- Complete API documentation with docstring examples
- Jupyter notebook tutorials demonstrating real-world use cases
- Performance benchmarks comparing Python vs R implementation
- Additional usage examples (dose-response, time-varying treatments)
- Enhanced error messages and input validation
- Support for additional weight truncation strategies
