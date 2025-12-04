# TrialEmulation Python Port - Roadmap Summary

## Quick Start Guide

This is a condensed summary of the full roadmap. See `ROADMAP.md` for detailed plans.

---

## Current Status ✅

- **Core modules**: All implemented (`data_preparation`, `weighting`, `msm`, etc.)
- **C++ extension**: Built and working
- **Example script**: `example_usage.py` exists but needs validation
- **Tests**: **NONE** - This is the #1 priority
- **Documentation**: Basic README only

---

## Critical Bugs Fixed ✅

1. ✅ Fixed deprecated pandas `fillna(method="ffill")` → `ffill()`
2. ✅ Fixed type hint `any` → `Any` 
3. ✅ Fixed sampling bug (use enumerate instead of `index()`)

---

## Top 5 Priorities (Next 5-10 Sessions)

### 1. **Fix example_usage.py** (Session 1-2)
- Run it end-to-end
- Fix any runtime errors
- Validate outputs make sense

### 2. **Create Test Infrastructure** (Session 3-4)
- Set up `tests/` directory with pytest
- Create `conftest.py` with fixtures
- Write first unit tests (`test_formula_utils.py`, `test_data_manipulation.py`)

### 3. **R-Python Parity Testing** (Session 5-6)
- Create framework to run R and Python side-by-side
- Compare outputs on same test data
- Fix discrepancies

### 4. **Replicate R Vignettes** (Session 7-8)
- Port "Getting Started" vignette
- Compare outputs
- Create Python examples

### 5. **Documentation** (Session 9-10)
- Add docstrings to all functions
- Create `docs/getting_started.md`
- Update README

---

## Test Structure to Create

```
tests/
├── __init__.py
├── conftest.py                    # Fixtures
├── unit/
│   ├── test_formula_utils.py      # Formula parsing
│   ├── test_data_manipulation.py  # Preprocessing
│   ├── test_weighting.py          # IPW calculation
│   ├── test_data_extension.py     # Trial expansion
│   ├── test_msm.py                # MSM fitting
│   └── test_robust.py             # Robust variance
├── integration/
│   ├── test_full_workflow.py      # End-to-end
│   └── test_vignette_replication.py
└── comparison/
    ├── test_r_parity.py           # R vs Python
    └── fixtures/                  # R outputs
```

---

## Key Test Cases

### Unit Tests
- **Formula utils**: Parse `None`, `"~1"`, `"~ age + sex"`, `["age", "sex"]`
- **Data manipulation**: Eligibility filtering, lagged treatment, switches, censoring
- **Weighting**: Switching weights, censoring weights, combined weights
- **Data extension**: Expansion logic, weight calculation, outcome assignment
- **MSM**: Formula building, weight application, model fitting

### R Parity Tests
- **Simple ITT**: Compare row counts, coefficients
- **ITT with weights**: Compare weight distributions, coefficients
- **PP analysis**: Compare censoring logic, row counts

**Tolerance**: 
- Row counts: Exact
- Weights: Mean abs diff < 0.01, correlation > 0.99
- Coefficients: Relative diff < 1e-3 or abs < 0.01
- SEs: Relative diff < 5%

---

## Execution Checklist

### Session 1-2: Foundation
- [ ] Fix deprecated pandas API
- [ ] Fix type hints
- [ ] Fix sampling bug
- [ ] Add input validation
- [ ] Test `example_usage.py` runs

### Session 3-4: Test Infrastructure
- [ ] Create test directory structure
- [ ] Set up `conftest.py`
- [ ] Write `test_formula_utils.py`
- [ ] Write `test_data_manipulation.py` (basic)

### Session 5-6: R Parity
- [ ] Create R comparison framework
- [ ] Create test dataset
- [ ] Run R and Python, compare
- [ ] Fix discrepancies

### Session 7-8: Vignettes
- [ ] Port "Getting Started" vignette
- [ ] Compare outputs
- [ ] Create Python examples

### Session 9-10: Documentation
- [ ] Add docstrings
- [ ] Create `docs/getting_started.md`
- [ ] Update README

---

## Success Criteria

### MVP (Minimum Viable Product)
- ✅ All unit tests pass
- ✅ R comparison tests pass (within tolerance)
- ✅ `example_usage.py` runs without errors
- ✅ Basic documentation exists

### Production Ready
- ✅ >80% test coverage
- ✅ All R vignettes replicated
- ✅ CI pipeline running
- ✅ Comprehensive documentation

---

## Quick Commands

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

---

## Key Files

- **Roadmap**: `ROADMAP.md` (detailed plan)
- **Architecture**: `ARCHITECTURE.md` (current design)
- **R Source**: `R_source/` (ground truth reference)
- **Python Code**: `trial_emulation/` (port)
- **Example**: `example_usage.py`

---

**Remember**: R code in `R_source/` is the ground truth. Python must match it.

