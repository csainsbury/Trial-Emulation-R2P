Quick Start Guide
=================

This guide will walk you through a basic example of using TrialEmulation to perform a target trial emulation analysis.

Data Format
-----------

Your input data should be in **long (person-time) format** with one row per person-period. Required columns:

* ``id``: Patient or unit identifier
* ``period``: Sequential time period (integer starting at 0)
* ``treatment``: Binary treatment indicator (0 = untreated, 1 = treated)
* ``outcome``: Binary outcome indicator (0 = no event, 1 = event occurred)
* ``eligible``: Binary eligibility indicator (0 = not eligible, 1 = eligible)
* Additional covariates as needed

Example data structure:

.. code-block:: python

    import pandas as pd

    data = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'period': [0, 1, 2, 0, 1, 2],
        'treatment': [0, 1, 1, 0, 0, 1],
        'outcome': [0, 0, 1, 0, 0, 0],
        'eligible': [1, 1, 1, 1, 1, 1],
        'age': [45, 45, 45, 52, 52, 52],
    })

Basic Workflow
--------------

Step 1: Data Preparation
~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to prepare your data for trial emulation. This expands the data into a sequence of emulated trials and calculates inverse probability weights:

.. code-block:: python

    import trial_emulation as te

    prep = te.data_preparation(
        data=data,
        id="id",
        period="period",
        treatment="treatment",
        outcome="outcome",
        eligible="eligible",
        estimand_type="ITT",  # Intention-to-treat analysis
        outcome_cov=["age"],  # Covariates to include
        quiet=True
    )

Step 2: Fit Marginal Structural Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, fit a marginal structural model using weighted pooled logistic regression:

.. code-block:: python

    msm = te.trial_msm(
        data=prep,
        outcome_cov=["age"],
        estimand_type="ITT",
        quiet=True
    )

Step 3: View Results
~~~~~~~~~~~~~~~~~~~~

Extract and interpret the results:

.. code-block:: python

    # View full summary with robust standard errors
    print(msm.robust["summary"])

    # Extract treatment effect estimate
    treatment_coef = msm.model.params["assigned_treatment"]
    robust_se = msm.robust["bse"]["assigned_treatment"]

    print(f"Treatment effect (log OR): {treatment_coef:.3f}")
    print(f"Robust SE: {robust_se:.3f}")

Understanding Estimands
-----------------------

The package supports three types of estimands:

**Intention-to-Treat (ITT)**
    Effect of being assigned to treatment at baseline, regardless of adherence.

**Per-Protocol (PP)**
    Effect of adhering to the assigned treatment protocol. Requires artificial censoring at treatment changes.

**As-Treated**
    Effect of actual treatment received, accounting for time-varying treatment.

Example with weights:

.. code-block:: python

    # Per-protocol analysis with censoring weights
    prep = te.data_preparation(
        data=data,
        id="id",
        period="period",
        treatment="treatment",
        outcome="outcome",
        eligible="eligible",
        estimand_type="PP",
        use_censor_weights=True,
        cense="censored",  # Censoring indicator
        outcome_cov=["age"],
    )

Next Steps
----------

* Explore the :doc:`user_guide/index` for detailed explanations
* See :doc:`examples/index` for complete working examples
* Check the :doc:`api/index` for full function documentation
