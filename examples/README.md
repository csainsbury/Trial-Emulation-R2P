# TrialEmulation Examples

This directory contains working examples demonstrating how to use the TrialEmulation package for various types of analyses.

## Running the Examples

Each example is a standalone Python script that generates synthetic data and performs a complete analysis. To run an example:

```bash
cd examples
python basic_itt_analysis.py
```

## Available Examples

### 1. basic_itt_analysis.py

**Purpose:** Demonstrates a basic intention-to-treat (ITT) analysis workflow.

**What it covers:**
- Generating synthetic longitudinal data
- Data preparation for trial emulation
- Fitting a marginal structural model
- Interpreting results with robust standard errors

**Best for:** First-time users wanting to understand the basic workflow.

**Runtime:** < 10 seconds

### 2. per_protocol_analysis.py

**Purpose:** Shows how to perform a per-protocol analysis with inverse probability weighting.

**What it covers:**
- Handling treatment switching in observational data
- Calculating censoring weights
- Weight truncation strategies
- Interpreting per-protocol effects

**Best for:** Users analyzing treatment adherence and protocol deviations.

**Runtime:** < 15 seconds

### 3. comparing_estimands.py

**Purpose:** Compares three different causal estimands (ITT, PP, As-Treated) side-by-side.

**What it covers:**
- Differences between ITT, per-protocol, and as-treated estimands
- When to use each estimand
- How results differ based on the causal question
- Interpreting differences across estimands

**Best for:** Understanding which estimand to use for your research question.

**Runtime:** < 30 seconds

## Example Data Format

All examples generate synthetic data in the required format:

- **Long (person-time) format**: One row per person-period
- **Required columns**: `id`, `period`, `treatment`, `outcome`, `eligible`
- **Optional columns**: Covariates like `age`, `sex`, etc.

Example structure:
```python
id  period  treatment  outcome  eligible  age
1   0       0          0        1         45
1   1       1          0        1         45
1   2       1          0        1         45
2   0       0          0        1         52
```

## Customizing Examples

You can modify the examples to:

1. **Change data generation parameters:**
   - Adjust `n_patients` and `n_periods` for different dataset sizes
   - Modify treatment effects by changing hazard ratios
   - Add or remove covariates

2. **Try different model specifications:**
   - Change `outcome_cov` to include/exclude covariates
   - Adjust weight truncation with `analysis_weights` parameter
   - Modify follow-up time windows with `first_followup`/`last_followup`

3. **Compare with your own data:**
   - Replace synthetic data generation with your data loading
   - Ensure your data matches the required format
   - Adjust column names as needed

## Common Issues

**Issue:** Import error for `trial_emulation`
- **Solution:** Make sure the package is installed: `pip install -e .` from the project root

**Issue:** "No module named scipy"
- **Solution:** Install dependencies: `pip install scipy`

**Issue:** C++ extension build errors
- **Solution:** See installation instructions in main README for platform-specific guidance

## Further Reading

- **Documentation:** https://trial-emulation.readthedocs.io
- **User Guide:** See `docs/source/user_guide/` for detailed methodology
- **API Reference:** Full function documentation in `docs/source/api/`

## Questions or Issues?

- Open an issue on GitHub: https://github.com/csainsbury/Trial-Emulation-R2P/issues
- Check existing discussions: https://github.com/csainsbury/Trial-Emulation-R2P/discussions
