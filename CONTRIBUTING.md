# Contributing to TrialEmulation

Thank you for your interest in contributing to the TrialEmulation Python package! We welcome contributions from the community and are grateful for your support in making this project better.

This document provides guidelines for contributing to the project. Please take a moment to read through it before submitting your contribution.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Documentation](#documentation)
- [Questions](#questions)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Ways to Contribute

There are many ways to contribute to TrialEmulation:

- **Report bugs**: Help us identify and fix issues
- **Suggest features**: Propose new functionality or improvements
- **Submit pull requests**: Contribute code, documentation, or examples
- **Improve documentation**: Fix typos, clarify explanations, or add examples
- **Write tutorials**: Create guides or blog posts about using the package
- **Answer questions**: Help other users in GitHub Discussions or Issues
- **Review pull requests**: Provide feedback on proposed changes

All contributions, large or small, are valued and appreciated!

## Development Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- A C++ compiler (for building the pybind11 extension)
  - macOS: Xcode Command Line Tools
  - Linux: GCC or Clang
  - Windows: MSVC (Visual Studio)

### Setting Up Your Environment

1. **Fork the repository** on GitHub

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Trial-Emulation-R2P.git
   cd Trial-Emulation-R2P
   ```

3. **Add the upstream repository** as a remote:
   ```bash
   git remote add upstream https://github.com/csainsbury/Trial-Emulation-R2P.git
   ```

4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install the package in development mode** with all dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

6. **Verify installation** by running the tests:
   ```bash
   pytest tests/
   ```

### Keeping Your Fork Updated

Before starting new work, sync your fork with upstream:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

## Coding Standards

We follow Python community best practices to ensure code quality and consistency.

### Code Style

- **Black** for code formatting (100 character line length)
- **Type hints** required for all public functions and methods
- **Docstrings** in NumPy style for all public APIs
- **Import organization**: Standard library, third-party, local imports (separated by blank lines)

### Running Code Formatters

Format your code before committing:

```bash
# Format code with Black
black trial_emulation tests

# Check type hints with mypy
mypy trial_emulation --ignore-missing-imports
```

### Docstring Style

Use NumPy-style docstrings for consistency with scientific Python packages:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.

    More detailed description if needed, explaining the purpose,
    behavior, and any important notes.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> my_function(42, "hello")
    True
    """
    ...
```

### Code Organization

- Keep functions focused and single-purpose
- Prefer small, well-named functions over large monolithic ones
- Use descriptive variable names (avoid single-letter names except for mathematical notation)
- Add comments for complex logic, but prefer self-documenting code

## Testing Requirements

All new features and bug fixes must include tests. We strive for **70% minimum code coverage**.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=trial_emulation --cov-report=term-missing

# Run specific test file
pytest tests/test_data_preparation.py

# Run tests matching a pattern
pytest tests/ -k "test_weighting"

# Run tests and stop at first failure
pytest tests/ -x
```

### Writing Tests

- Use `pytest` for all tests
- Organize tests by module (e.g., `test_data_preparation.py` for `data_preparation.py`)
- Use fixtures from `conftest.py` for common test data
- Test both typical use cases and edge cases
- Include docstrings explaining what each test verifies

Example test structure:

```python
def test_function_basic_behavior(synthetic_data_simple):
    """Test that function works with typical inputs."""
    result = my_function(synthetic_data_simple)

    assert isinstance(result, ExpectedType)
    assert result.some_property == expected_value

def test_function_edge_case():
    """Test that function handles edge case correctly."""
    with pytest.raises(ValueError, match="expected error message"):
        my_function(invalid_input)
```

### Test Categories

Use markers to categorize tests:

```python
@pytest.mark.integration
def test_full_workflow():
    """Integration test for complete workflow."""
    ...

@pytest.mark.slow
def test_large_dataset():
    """Test with large dataset (takes >5 seconds)."""
    ...
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, logical commits

3. **Write or update tests** to cover your changes

4. **Ensure all tests pass**:
   ```bash
   pytest tests/
   ```

5. **Format your code**:
   ```bash
   black trial_emulation tests
   ```

6. **Check type hints**:
   ```bash
   mypy trial_emulation --ignore-missing-imports
   ```

7. **Update documentation** if needed (docstrings, README, CHANGELOG)

### Commit Messages

Write clear, concise commit messages:

- Use present tense ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Add detailed explanation in body if needed

Example:
```
Add support for custom weight truncation

Implements user-specified weight limits in trial_msm() function.
Adds weight_limits parameter and updates documentation.

Fixes #123
```

### Opening a Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a pull request** on GitHub with:
   - Clear title describing the change
   - Description explaining what and why
   - Reference to related issues (e.g., "Fixes #123")
   - List of changes made
   - Screenshots if relevant (UI changes, plots, etc.)

3. **PR Checklist** (will be in PR template):
   - [ ] Tests added/updated and passing
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] Code formatted with Black
   - [ ] Type hints added
   - [ ] All CI checks passing

### Code Review

- Maintainers will review your PR within 1 week
- Be responsive to feedback and questions
- Make requested changes in new commits (don't force-push)
- Be patient and collaborative

### After Approval

Once approved, a maintainer will merge your PR. Congratulations and thank you for your contribution!

## Reporting Bugs

Found a bug? Please help us fix it!

### Before Reporting

1. **Check existing issues** to see if it's already reported
2. **Update to the latest version** to see if it's already fixed
3. **Gather information** about the bug:
   - Python version
   - Package version
   - Operating system
   - Steps to reproduce
   - Expected vs. actual behavior
   - Error messages or tracebacks

### Submitting a Bug Report

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) on GitHub Issues and include:

- Clear, descriptive title
- Detailed description
- Minimal code example to reproduce
- Environment information
- Any relevant error messages

## Suggesting Features

Have an idea for a new feature or improvement?

### Before Suggesting

1. **Check existing issues** for similar proposals
2. **Consider if it fits the scope** of the project
3. **Think about implementation** - how would it work?

### Submitting a Feature Request

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

- Clear description of the feature
- Use case: why is it needed?
- Proposed API or interface
- Example usage code
- Any implementation ideas

## Documentation

Good documentation is crucial for usability.

### Updating Documentation

- **Docstrings**: Update when modifying functions
- **README.md**: Update for new features or changed workflows
- **CHANGELOG.md**: Add entry for user-facing changes
- **Examples**: Add or update example scripts
- **Sphinx docs**: Update if you modify public APIs

### Building Documentation Locally

```bash
cd docs
make html
# Open docs/_build/html/index.html in your browser
```

## Questions

Have questions about contributing?

- **GitHub Discussions**: For general questions and design discussions
- **GitHub Issues**: For specific technical questions related to development
- **Email maintainers**: For private or sensitive questions

We're here to help and want to make contributing as smooth as possible!

## Recognition

All contributors will be recognized in:
- GitHub contributors list
- CHANGELOG.md for specific contributions
- Future publications or presentations about the package

## License

By contributing to TrialEmulation, you agree that your contributions will be licensed under the Apache License 2.0, the same license as the project.

---

Thank you for contributing to TrialEmulation! Your efforts help advance open-source tools for causal inference in epidemiology and health research.
