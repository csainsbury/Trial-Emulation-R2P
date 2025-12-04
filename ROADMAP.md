# TrialEmulation Python Port - Development Roadmap

## Executive Summary

This roadmap outlines the prioritized steps to transform the current Python port of TrialEmulation from a functional prototype into a robust, validated, production-ready library that faithfully reproduces the R package's behavior.

**Current Status**: Core functionality is implemented, but lacks comprehensive testing, validation against R outputs, and production hardening.

---

## 1. Current State Assessment

### 1.1 Repository Structure ✅
- **Python modules**: Located in `trial_emulation/` with all core modules present
- **C++ extension**: `cpp/censor_func.cpp` with pybind11 bindings exists
- **Example script**: `example_usage.py` demonstrates basic workflow
- **Packaging**: `pyproject.toml` and `setup.py` configured

### 1.2 Incomplete/Problematic Areas ⚠️

#### Critical Issues Found:
1. **No test infrastructure**: Zero test files (`test*.py`) exist
2. **C++ fallback**: Python fallback for censoring function raises `NotImplementedError` if C++ fails
3. **Missing validation**: No systematic comparison with R outputs
4. **Incomplete error handling**: Some edge cases may not be handled gracefully
5. **Documentation gaps**: Limited docstrings, no usage guide matching R vignettes

#### Potential Issues:
- `data_manipulation.py` line 102: Uses deprecated `fillna(method="ffill")` (should use `ffill()`)
- `data_extension.py` line 27: Type hint uses lowercase `any` instead of `Any`
- `sampling.py` line 148: Potential bug with `p_control.index(p)` (should be list index)
- No handling for edge cases (empty data, single ID, etc.)

### 1.3 Example Usage Status
- `example_usage.py` runs but may have bugs/mismatches with R
- No validation that outputs match R reference

---

## 2. Prioritized Roadmap

### Phase 1: Foundation & Validation (HIGH PRIORITY)

#### 2.1 Fix Critical Bugs and Code Quality Issues
**Priority**: HIGH  
**Why**: Ensures code runs correctly before building tests

**Actions**:
1. Fix deprecated pandas API usage:
   - `data_manipulation.py` line 102: Replace `fillna(method="ffill")` with `ffill()`
   - Check for other deprecated pandas patterns
2. Fix type hints:
   - `data_extension.py` line 27: Change `any` to `Any`
3. Fix potential bug in sampling:
   - `sampling.py` line 148: Fix `p_control.index(p)` logic
4. Add input validation:
   - Check for empty DataFrames
   - Validate column existence
   - Check for required columns before processing
5. Improve C++ fallback:
   - Ensure Python fallback actually works (currently raises NotImplementedError)
   - Add graceful degradation

**Deliverable**: Bug-free `example_usage.py` that runs end-to-end

---

#### 2.2 Create Test Infrastructure
**Priority**: HIGH  
**Why**: Tests are essential for validation and regression prevention

**Actions**:
1. Set up pytest framework:
   ```python
   # tests/__init__.py (create)
   # tests/conftest.py (create) - fixtures for test data
   ```
2. Create test directory structure:
   ```
   tests/
   ├── __init__.py
   ├── conftest.py
   ├── unit/
   │   ├── test_formula_utils.py
   │   ├── test_data_utils.py
   │   ├── test_data_manipulation.py
   │   ├── test_weighting.py
   │   ├── test_data_extension.py
   │   ├── test_msm.py
   │   ├── test_robust.py
   │   └── test_sampling.py
   ├── integration/
   │   ├── test_data_preparation.py
   │   ├── test_full_workflow.py
   │   └── test_vignette_replication.py
   └── comparison/
       ├── test_r_parity.py
       └── fixtures/  # R output CSVs/JSONs
   ```
3. Add pytest to `pyproject.toml` dependencies (already in dev extras)
4. Create minimal test fixtures:
   - Small synthetic datasets (10-20 IDs, 5-10 periods)
   - Known-good outputs for unit tests

**Deliverable**: Test framework with at least one test per module

---

#### 2.3 Unit Tests - Core Modules
**Priority**: HIGH  
**Why**: Ensures each component works correctly in isolation

**Actions**:

**2.3.1 Formula Utils** (`tests/unit/test_formula_utils.py`):
- Test `parse_formula_like()`:
  - `None` → `("1", [])`
  - `"~1"` → `("1", [])`
  - `"~ age + sex"` → `("age + sex", ["age", "sex"])`
  - `["age", "sex"]` → `("age + sex", ["age", "sex"])`
  - `"age + I(age^2)"` → extracts `age` correctly
- Test `add_rhs()`:
  - `"a + b"` + `"c"` → `"a + b + c"`
  - `"1"` + `"a"` → `"a"`
- Test `rhs_vars()` extraction

**2.3.2 Data Manipulation** (`tests/unit/test_data_manipulation.py`):
- Test eligibility filtering:
  - Removes rows before first eligibility
  - Removes rows after outcome event
- Test lagged treatment (`am_1`):
  - First row has `am_1 = 0`
  - Subsequent rows have correct lag
- Test cumulative treatment (`cumA`):
  - Accumulates correctly within ID
- Test switch detection:
  - `switch = 1` when treatment changes
- Test censoring state machine:
  - Compare C++ vs Python fallback outputs
  - Test edge cases (no switches, multiple switches)

**2.3.3 Weighting** (`tests/unit/test_weighting.py`):
- Test switching weight calculation:
  - Numerator/denominator models fit correctly
  - Weight formula: `p_n / p_d` for treatment=1, `(1-p_n)/(1-p_d)` for treatment=0
  - Handles `eligible_wts_0` and `eligible_wts_1` correctly
- Test censoring weight calculation:
  - Pooled vs separate models
  - Weight formula: `pC_n / pC_d`
- Test combined weights: `wt = wtS * wtC`
- Test edge cases: no eligible rows, perfect separation

**2.3.4 Data Extension** (`tests/unit/test_data_extension.py`):
- Test expansion logic:
  - Each person-period expands to `period+1` rows (trial_period 0 to period)
  - `trial_period` and `followup_time` calculated correctly
  - `assigned_treatment` (init) forward-filled correctly
- Test weight calculation:
  - `weight0` (cumulative product) correct
  - Final `weight = weight0 / wtprod` correct
- Test outcome assignment:
  - `case = 1` when `time_of_event == period_new` and `outcome_new == 1`
- Test censoring at switch (PP analysis):
  - `expand = 0` after first switch when `use_censor=True`

**2.3.5 MSM** (`tests/unit/test_msm.py`):
- Test formula building:
  - Combines `model_var`, `include_trial_period`, `include_followup_time`, `outcome_cov`
- Test weight application:
  - `analysis_weights="asis"`, `"p99"`, `"weight_limits"`, `"unweighted"`
- Test filtering:
  - `first_followup` and `last_followup` work correctly
  - `where_case` conditions applied
- Test model fitting:
  - Model converges
  - Coefficients reasonable

**2.3.6 Robust Variance** (`tests/unit/test_robust.py`):
- Test sandwich estimator:
  - Cluster-robust SEs calculated
  - Covariance matrix symmetric and positive semi-definite
- Test summary table:
  - Columns: `estimate`, `robust_se`, `2.5%`, `97.5%`, `z`, `p_value`
  - CIs calculated correctly (1.96 * SE)

**Deliverable**: Comprehensive unit test suite with >80% coverage

---

#### 2.4 R-Python Parity Testing Framework
**Priority**: HIGH  
**Why**: Ensures Python outputs match R (the ground truth)

**Actions**:
1. Create R script runner utility (`tests/comparison/r_runner.py`):
   ```python
   # Runs R scripts and captures outputs
   # Requires rpy2 or subprocess R execution
   ```
2. Create comparison test framework (`tests/comparison/test_r_parity.py`):
   - Load same test data in R and Python
   - Run `data_preparation()` in both
   - Compare:
     - Row counts (`N`)
     - Weight distributions (mean, median, percentiles)
     - Model coefficients (within tolerance)
     - Robust SEs (within tolerance)
3. Create small synthetic dataset generator:
   - Generate data that can be used in both R and Python
   - Save as CSV for both languages
4. Create R reference outputs:
   - Run R package on test data
   - Save key outputs (weights, model coefficients, SEs) as JSON/CSV
   - Use as "golden" reference in tests

**Test Cases**:
- **Simple ITT analysis** (no weights):
  - Compare expanded data row counts
  - Compare MSM coefficients
- **ITT with switching weights**:
  - Compare weight model coefficients
  - Compare final weights distribution
  - Compare MSM coefficients
- **PP analysis** (censoring at switch):
  - Compare censoring logic outputs
  - Compare final row counts
- **With censoring weights**:
  - Compare censoring weight models
  - Compare combined weights

**Tolerance Guidelines**:
- Row counts: Exact match
- Weights: Mean absolute difference < 0.01, correlation > 0.99
- Coefficients: Relative difference < 1e-3 or absolute < 0.01
- SEs: Relative difference < 5% (due to solver differences)

**Deliverable**: Automated R-Python comparison tests with clear pass/fail criteria

---

### Phase 2: Integration & Vignette Replication (HIGH PRIORITY)

#### 2.5 Replicate R Vignettes
**Priority**: HIGH  
**Why**: Vignettes demonstrate real-world usage and validate end-to-end workflows

**Actions**:
1. Port "Getting Started" vignette (`tests/integration/test_getting_started.py`):
   - Load `R_source/data/trial_example.rda` (or equivalent CSV)
   - Run full workflow: `data_preparation()` → `trial_msm()`
   - Compare outputs with R vignette outputs
   - Document any differences and their causes
2. Port "New Interface" vignette (`tests/integration/test_new_interface.py`):
   - Test new API patterns
   - Compare results
3. Port "Extending TrialEmulation" if applicable
4. Create Python equivalents of vignette scripts:
   - `examples/getting_started.py`
   - `examples/new_interface.py`
   - Include comments explaining each step

**Deliverable**: Python scripts that replicate R vignettes with matching outputs

---

#### 2.6 End-to-End Integration Tests
**Priority**: HIGH  
**Why**: Validates complete workflows work correctly together

**Actions**:
1. Create `tests/integration/test_full_workflow.py`:
   - Test complete workflow: data → prep → msm → results
   - Test with different `estimand_type` values: "ITT", "PP", "As-Treated"
   - Test with and without weights
   - Test with case-control sampling
2. Test edge cases:
   - Single ID
   - Single period
   - No events
   - All events
   - Perfect treatment adherence (no switches)
   - All switches
3. Test error handling:
   - Missing columns
   - Invalid data types
   - Empty data
   - Invalid formulas

**Deliverable**: Comprehensive integration test suite

---

### Phase 3: API Review & Documentation (MEDIUM PRIORITY)

#### 2.7 API Consistency Review
**Priority**: MEDIUM  
**Why**: Ensures Python API matches R expectations or is clearly documented

**Actions**:
1. Create API comparison matrix (`docs/API_COMPARISON.md`):
   - Function-by-function comparison: R vs Python
   - Parameter names and defaults
   - Return value structures
   - Document intentional differences
2. Review function signatures:
   - Check all parameters match R defaults
   - Check parameter names are intuitive
   - Add missing parameters if needed
3. Review return value structures:
   - Ensure `TEDataPrep` and `TEMSM` match R class structures
   - Check attribute names match R where possible
4. Add deprecation warnings for any intentional differences

**Deliverable**: API comparison document and aligned function signatures

---

#### 2.8 Documentation
**Priority**: MEDIUM  
**Why**: Users need clear guidance to use the package effectively

**Actions**:
1. **Docstrings**:
   - Add comprehensive docstrings to all public functions
   - Include Args, Returns, Examples, Raises sections
   - Use NumPy/SciPy docstring format
2. **User Guide** (`docs/getting_started.md`):
   - Port R "Getting Started" vignette to Python
   - Include step-by-step examples
   - Show expected outputs
3. **API Reference** (`docs/api_reference.md`):
   - Auto-generate from docstrings (using Sphinx or similar)
   - Or manually create function reference
4. **Migration Guide** (`docs/migrating_from_r.md`):
   - For users familiar with R package
   - Highlight differences
   - Show side-by-side code examples
5. **README updates**:
   - Add more examples
   - Add troubleshooting section
   - Add contribution guidelines

**Deliverable**: Complete documentation suite

---

### Phase 4: Packaging & Distribution (MEDIUM PRIORITY)

#### 2.9 Build System & Packaging
**Priority**: MEDIUM  
**Why**: Package must be installable and distributable

**Actions**:
1. Verify C++ extension builds correctly:
   - Test on Windows, Linux, macOS
   - Ensure pybind11 setup works
   - Test Python fallback when C++ unavailable
2. Update `setup.py`:
   - Ensure all dependencies listed
   - Add build requirements
   - Test `pip install -e .` works
3. Create `MANIFEST.in` (already exists, verify):
   - Include all necessary files
   - Exclude test files from distribution
4. Version management:
   - Use semantic versioning
   - Update version in `pyproject.toml` and `__init__.py`
5. Create source distribution:
   - Test `python setup.py sdist`
   - Test `python setup.py bdist_wheel`
   - Verify installation from source

**Deliverable**: Package builds and installs correctly on all platforms

---

#### 2.10 Continuous Integration (CI)
**Priority**: MEDIUM (Nice-to-have initially)  
**Why**: Automated testing prevents regressions

**Actions**:
1. Create GitHub Actions workflow (`.github/workflows/test.yml`):
   ```yaml
   # Test on Python 3.10, 3.11, 3.12
   # Test on Windows, Linux, macOS
   # Build C++ extension
   # Run pytest
   # Run R comparison tests (if R available)
   ```
2. Add linting:
   - Black formatting
   - mypy type checking (if types added)
   - flake8 or ruff
3. Add coverage reporting:
   - Use pytest-cov
   - Set coverage threshold (e.g., >80%)
   - Upload to Codecov or similar
4. Add R comparison tests to CI:
   - Install R and TrialEmulation R package
   - Run comparison tests
   - Fail if tolerances exceeded

**Deliverable**: CI pipeline that runs on every PR

---

### Phase 5: Performance & Robustness (NICE-TO-HAVE)

#### 2.11 Performance Optimization
**Priority**: NICE-TO-HAVE  
**Why**: Large datasets require efficient processing

**Actions**:
1. Profile hotspots:
   - Use `cProfile` or `py-spy` to identify bottlenecks
   - Focus on `data_extension.expand()` (likely bottleneck)
   - Profile weight calculation loops
2. Optimize data extension:
   - Vectorize operations where possible
   - Consider numba JIT compilation for inner loops
   - Test chunked processing (`separate_files=True`)
3. Optimize weight calculation:
   - Batch model fitting if possible
   - Cache intermediate results
4. Memory optimization:
   - Test on large datasets (10k+ IDs, 100+ periods)
   - Identify memory leaks
   - Optimize DataFrame operations

**Deliverable**: Performance benchmarks and optimizations

---

#### 2.12 Advanced Features
**Priority**: NICE-TO-HAVE  
**Why**: Match R package's advanced capabilities

**Actions**:
1. DuckDB backend (like R version):
   - Implement `te_datastore_duckdb` equivalent
   - Support large-scale data storage
2. Extended formula syntax:
   - Support more R formula features
   - Better handling of interactions, splines
3. Additional model backends:
   - Support for XGBoost, sklearn models for weights
   - Custom model interfaces
4. Parallel processing:
   - Parallelize weight model fitting
   - Parallelize data extension chunks

**Deliverable**: Advanced features matching R package

---

## 3. Recommended Execution Order (5-10 Sessions)

### Session 1-2: Foundation & Critical Fixes
**Goal**: Get `example_usage.py` running correctly

**Tasks**:
1. Fix deprecated pandas API (`fillna(method="ffill")`)
2. Fix type hints (`any` → `Any`)
3. Fix sampling bug (`p_control.index(p)`)
4. Add basic input validation
5. Test `example_usage.py` runs end-to-end
6. Fix any runtime errors

**Deliverable**: Working `example_usage.py`

---

### Session 3-4: Test Infrastructure & Unit Tests (Part 1)
**Goal**: Set up tests and test core utilities

**Tasks**:
1. Create test directory structure
2. Set up `conftest.py` with fixtures
3. Write `test_formula_utils.py` (complete)
4. Write `test_data_utils.py` (if needed)
5. Write `test_data_manipulation.py` (basic cases)
6. Run tests, fix bugs found

**Deliverable**: Test framework + 2-3 unit test modules

---

### Session 5-6: Unit Tests (Part 2) & R Parity Framework
**Goal**: Complete unit tests and set up R comparison

**Tasks**:
1. Complete `test_data_manipulation.py`
2. Write `test_weighting.py`
3. Write `test_data_extension.py`
4. Create R comparison framework (`r_runner.py`)
5. Create small synthetic test dataset
6. Run R and Python on same data, compare outputs
7. Fix discrepancies found

**Deliverable**: Complete unit tests + R comparison framework

---

### Session 7-8: R Parity Testing & Vignette Replication
**Goal**: Validate against R outputs

**Tasks**:
1. Create comprehensive R comparison tests
2. Test ITT analysis (no weights)
3. Test ITT with switching weights
4. Test PP analysis
5. Port "Getting Started" vignette
6. Compare vignette outputs
7. Fix any major discrepancies

**Deliverable**: R-Python parity validated + vignette replicated

---

### Session 9-10: Integration Tests & Documentation
**Goal**: Complete testing and document

**Tasks**:
1. Write `test_full_workflow.py`
2. Test all `estimand_type` values
3. Test edge cases
4. Write comprehensive docstrings
5. Create `docs/getting_started.md`
6. Update README with examples
7. Create API comparison document

**Deliverable**: Complete test suite + documentation

---

### Future Sessions (Optional):
- **Session 11-12**: CI setup, packaging improvements
- **Session 13+**: Performance optimization, advanced features

---

## 4. Success Criteria

### Minimum Viable Product (MVP):
- ✅ All unit tests pass
- ✅ R comparison tests pass (within tolerance)
- ✅ `example_usage.py` runs without errors
- ✅ Basic documentation exists
- ✅ Package installs correctly

### Production Ready:
- ✅ >80% test coverage
- ✅ All R vignettes replicated
- ✅ CI pipeline running
- ✅ Comprehensive documentation
- ✅ Performance acceptable on medium datasets (1k+ IDs)

---

## 5. Risk Mitigation

### Risks:
1. **R-Python discrepancies**: May find systematic differences
   - **Mitigation**: Document differences, investigate root causes, fix if possible
2. **C++ build issues**: May fail on some platforms
   - **Mitigation**: Ensure Python fallback works correctly
3. **Performance issues**: May be slow on large datasets
   - **Mitigation**: Profile early, optimize bottlenecks, document limitations
4. **Missing R features**: Some R features may not be ported
   - **Mitigation**: Document in API comparison, prioritize based on usage

---

## 6. Notes

- **R as Ground Truth**: Always treat R outputs as correct; Python must match
- **Incremental Approach**: Fix and test one module at a time
- **Documentation**: Write docs as you go, not at the end
- **Test-Driven**: Write tests before/while fixing bugs
- **Version Control**: Commit frequently with clear messages

---

## Appendix: Quick Reference

### Key Files to Modify:
- `trial_emulation/data_manipulation.py` - Fix deprecated pandas
- `trial_emulation/data_extension.py` - Fix type hint
- `trial_emulation/sampling.py` - Fix index bug
- `tests/` - Create all test files
- `docs/` - Create documentation

### Key Commands:
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=trial_emulation tests/

# Build package
pip install -e .

# Run example
python example_usage.py
```

### Test Data Sources:
- `R_source/data/` - R package example data
- `R_source/vignettes/` - Vignette code and outputs
- Create synthetic data in `tests/fixtures/`

---

**Last Updated**: 2024  
**Status**: Ready for execution

