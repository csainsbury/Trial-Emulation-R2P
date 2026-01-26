TrialEmulation Documentation
============================

Welcome to the documentation for **TrialEmulation**, a Python package for target trial emulation and causal inference in observational time-to-event data.

Overview
--------

The target trial emulation framework provides a principled approach to causal inference from observational data by explicitly specifying the hypothetical randomized trial (the "target trial") that would answer the research question of interest.

This package implements methods to emulate target trials using observational data in the person-time format, commonly found in electronic health records and administrative databases.

Key Features
------------

* **Target Trial Emulation**: Expand observational person-time data into a sequence of emulated trials
* **Multiple Estimands**: Support for intention-to-treat (ITT), per-protocol (PP), and as-treated analyses
* **Inverse Probability Weighting (IPW)**: Calculate weights for treatment switching and censoring
* **Marginal Structural Models**: Fit weighted pooled logistic regression for time-to-event outcomes
* **Robust Variance Estimation**: Cluster-robust standard errors using sandwich estimators
* **Performance Optimized**: C++ extensions for computationally intensive operations

Quick Start
-----------

Installation::

    pip install trial-emulation

Basic usage:

.. code-block:: python

    import trial_emulation as te
    import pandas as pd

    # Prepare data for trial emulation
    prep = te.data_preparation(
        data=data,
        id="id",
        period="period",
        treatment="treatment",
        outcome="outcome",
        eligible="eligible",
        estimand_type="ITT",
    )

    # Fit marginal structural model
    msm = te.trial_msm(
        data=prep,
        estimand_type="ITT",
    )

    # View results
    print(msm.robust["summary"])

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   api/index
   examples/index
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Links
-----

* GitHub: https://github.com/csainsbury/Trial-Emulation-R2P
* PyPI: https://pypi.org/project/trial-emulation/
* Original R Package: https://github.com/Causal-LDA/TrialEmulation

License
-------

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
