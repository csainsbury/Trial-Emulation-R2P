Installation
============

Requirements
------------

* Python 3.10 or higher
* NumPy >= 1.20.0
* pandas >= 1.3.0
* statsmodels >= 0.13.0
* scipy >= 1.7.0
* pybind11 >= 2.10.0 (for C++ extensions)

From PyPI
---------

The easiest way to install TrialEmulation is using pip::

    pip install trial-emulation

From Source
-----------

To install the latest development version from GitHub::

    git clone https://github.com/csainsbury/Trial-Emulation-R2P.git
    cd Trial-Emulation-R2P
    pip install -e .

Development Installation
------------------------

To install with development dependencies for testing and contributing::

    pip install -e ".[dev]"

This will install additional packages including pytest, black, and mypy.

Verifying Installation
----------------------

You can verify that TrialEmulation is installed correctly by importing it::

    python -c "import trial_emulation; print(trial_emulation.__version__)"

If you see the version number printed without errors, the installation was successful.

Troubleshooting
---------------

C++ Extension Build Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package includes C++ extensions for performance. If you encounter build errors:

**macOS**: Install Xcode Command Line Tools::

    xcode-select --install

**Linux**: Install a C++ compiler::

    # Ubuntu/Debian
    sudo apt-get install build-essential

    # Fedora/RHEL
    sudo dnf install gcc-c++

**Windows**: Install Visual Studio Build Tools from Microsoft.
