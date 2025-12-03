# TrialEmulation Python Port - Architecture Overview

## Overview

This is a Python/C++ port of the R TrialEmulation package, maintaining the same functionality and API semantics while using idiomatic Python and modern C++.

## Package Structure

```
trial_emulation/
├── __init__.py              # Public API exports
├── formula_utils.py         # Formula parsing and handling
├── utils.py                 # Utility functions (printing, weight truncation)
├── types.py                 # Type definitions and result classes
├── data_utils.py            # Data column selection and validation
├── data_manipulation.py     # Preprocessing (eligibility, event times, censoring)
├── weighting.py             # IPW calculation (switching and censoring)
├── data_extension.py        # Trial expansion logic
├── data_preparation.py     # Main entry point
├── msm.py                   # Marginal structural model fitting
├── robust.py                # Robust variance estimation
└── sampling.py              # Case-control sampling

cpp/
└── censor_func.cpp          # C++ censoring state machine (pybind11)
```

## Component Mapping

### R → Python Mapping

| R Function/Class | Python Module | Purpose |
|-----------------|---------------|---------|
| `as_formula()` | `formula_utils.parse_formula_like()` | Convert formulas to strings |
| `data_preparation()` | `data_preparation.data_preparation()` | Main entry point |
| `data_manipulation()` | `data_manipulation.data_manipulation()` | Preprocessing |
| `weight_func()` | `weighting.weight_func()` | IPW calculation |
| `fit_switch_weights()` | `weighting.fit_switch_weights()` | Switching weight models |
| `fit_censor_weights()` | `weighting.fit_censor_weights()` | Censoring weight models |
| `data_extension()` | `data_extension.data_extension()` | Trial expansion |
| `expand()` | `data_extension.expand()` | Core expansion logic |
| `trial_msm()` | `msm.trial_msm()` | MSM fitting |
| `robust_calculation()` | `robust.robust_calculation()` | Robust variance |
| `case_control_sampling_trials()` | `sampling.case_control_sampling_trials()` | Case-control sampling |
| `censor_func()` (Rcpp) | `_censor_func.censor_func()` (C++) | Censoring state machine |

## Key Design Decisions

### 1. Formula Handling
- R formulas are converted to strings compatible with `statsmodels.formula.api`
- Supports `~1` (intercept-only), variable lists, and formula strings
- Variable extraction handles basic transformations like `I(x^2)`

### 2. Data Structures
- `pandas.DataFrame` replaces R's `data.table`
- Result objects use `dataclasses` to mirror R's S3/S4 classes
- Type hints throughout for better IDE support

### 3. C++ Integration
- Censoring state machine ported to C++17 with pybind11
- Python fallback implementation available if C++ module fails to build
- Maintains exact same logic as R Rcpp version

### 4. Statistical Models
- `statsmodels.Logit` replaces R's `glm()`
- Robust variance uses `statsmodels.stats.sandwich_covariance` when available
- Falls back to basic sandwich estimator if needed

### 5. Weight Calculation
- Exact same formulas as R version
- Supports pooled and separate models for switching/censoring
- Handles eligibility conditions for weight models

## Workflow

1. **Data Preparation** (`data_preparation.py`)
   - Validates inputs and converts formulas
   - Selects/renames columns
   - Calls data manipulation
   - Calculates weights if needed
   - Expands data into trials

2. **Data Manipulation** (`data_manipulation.py`)
   - Filters pre-eligibility and post-event observations
   - Calculates event times
   - Computes lagged treatment, cumulative treatment, switches
   - Applies censoring state machine (C++)

3. **Weighting** (`weighting.py`)
   - Fits logistic models for switching weights
   - Fits logistic models for censoring weights
   - Calculates IPW = (numerator / denominator) * (censoring_n / censoring_d)

4. **Data Extension** (`data_extension.py`)
   - Expands each person-period into multiple trial-period rows
   - Creates `trial_period` and `followup_time` variables
   - Handles censoring at switch for PP analysis
   - Calculates final weights and outcomes

5. **MSM Fitting** (`msm.py`)
   - Builds model formula from components
   - Fits weighted pooled logistic regression
   - Calculates robust variance using sandwich estimator

## Differences from R Version

1. **No S4 Classes**: Python uses dataclasses and regular classes instead
2. **Formula Strings**: Formulas are strings, not R formula objects
3. **Error Handling**: Python exceptions instead of R errors/warnings
4. **File I/O**: Uses pandas CSV I/O instead of data.table's fwrite
5. **Model Objects**: statsmodels objects instead of R glm objects

## Testing Strategy

1. **Unit Tests**: Test each module independently
2. **Integration Tests**: Test full workflow on example data
3. **Comparison Tests**: Compare results with R package on same data
4. **Numerical Tolerance**: Accept small differences due to solver differences

## Performance Considerations

- C++ censoring function for performance-critical inner loop
- Vectorized pandas operations where possible
- Chunked processing for large datasets (`separate_files=True`)
- Case-control sampling for computational efficiency

## Future Enhancements

- DuckDB backend for large-scale data storage (like R version)
- Parallel processing for weight model fitting
- Additional model backends (e.g., XGBoost for weights)
- Extended formula syntax support
- Better error messages and validation

