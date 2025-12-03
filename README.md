# TrialEmulation (Python)

Python port of the R [TrialEmulation](https://github.com/Causal-LDA/TrialEmulation) package.

This package implements target trial emulation methods to apply randomized clinical trial design and analysis in an observational setting. Using marginal structural models, it can estimate intention-to-treat and per-protocol effects in emulated trials using electronic health records.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import trial_emulation as te
import pandas as pd

# Load your data in long format
data = pd.read_csv("your_data.csv")

# Prepare data for trial emulation
prep = te.data_preparation(
    data=data,
    id="id",
    period="period",
    treatment="treatment",
    outcome="outcome",
    eligible="eligible",
    estimand_type="ITT",
    outcome_cov=["age", "sex", "bmi"],
    use_censor_weights=False,
)

# Fit marginal structural model
msm = te.trial_msm(
    data=prep,
    outcome_cov=["age", "sex", "bmi"],
    estimand_type="ITT",
)

# View results
print(msm.robust["summary"])
```

## Features

- **Data Preparation**: Expand observational data into a sequence of emulated trials
- **Inverse Probability Weighting**: Calculate weights for treatment switching and censoring
- **Marginal Structural Models**: Fit MSMs for time-to-event outcomes
- **Robust Variance Estimation**: Calculate cluster-robust standard errors
- **Case-Control Sampling**: Sample controls for computational efficiency

## Documentation

See the R package documentation for detailed methodology:
- [Getting Started](https://causal-lda.github.io/TrialEmulation/articles/Getting-Started.html)
- [New Interface](https://causal-lda.github.io/TrialEmulation/articles/new-interface.html)

## License

Apache License 2.0

