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

### Notes

This is an alpha release (0.0.x series). The API may change in future releases as we gather user feedback and refine the interface. The package is functional and tested internally but should be considered experimental for production use.

For questions or issues, please visit: https://github.com/csainsbury/Trial-Emulation-R2P/issues

## [Unreleased]

### Planned

- Comprehensive test suite with 70%+ coverage
- Sphinx documentation and ReadTheDocs integration
- Additional usage examples and tutorials
- Jupyter notebook tutorials
- Performance benchmarks
- Validation against R package results
