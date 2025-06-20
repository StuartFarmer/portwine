# Contributing to Portwine

Thank you for your interest in contributing to portwine! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Poetry (recommended) or pip

### Setting Up Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/portwine.git
   cd portwine
   ```

2. **Install dependencies**
   ```bash
   # Using Poetry (recommended)
   poetry install
   
   # Or using pip
   pip install -e .
   pip install -r requirements-dev.txt
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write your code
- Add tests for new functionality
- Update documentation if needed
- Ensure all tests pass

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=portwine

# Run specific test file
pytest tests/test_backtester.py
```

### 4. Code Quality Checks

```bash
# Run linting
flake8 portwine/

# Run type checking
mypy portwine/

# Run formatting
black portwine/
isort portwine/
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style

### Python Code

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and concise

### Example

```python
from typing import Dict, List, Optional
import pandas as pd

def calculate_returns(
    prices: pd.DataFrame,
    method: str = "simple"
) -> pd.Series:
    """
    Calculate returns from price data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with price data
    method : str, default "simple"
        Return calculation method ("simple" or "log")
        
    Returns
    -------
    pd.Series
        Calculated returns
    """
    if method == "simple":
        return prices.pct_change()
    elif method == "log":
        return prices.apply(np.log).diff()
    else:
        raise ValueError(f"Unknown method: {method}")
```

### Documentation

- Use clear, concise language
- Include code examples
- Update API documentation when adding new features
- Follow the existing documentation structure

## Testing

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Test both success and error cases
- Use fixtures for common test data

### Example Test

```python
import pytest
import pandas as pd
from portwine import Backtester

def test_backtester_initialization():
    """Test Backtester initialization with valid parameters."""
    data_loader = MockDataLoader()
    backtester = Backtester(market_data_loader=data_loader)
    
    assert backtester.market_data_loader == data_loader
    assert backtester.calendar is None

def test_backtester_invalid_benchmark():
    """Test Backtester raises error for invalid benchmark."""
    data_loader = MockDataLoader()
    backtester = Backtester(market_data_loader=data_loader)
    
    with pytest.raises(InvalidBenchmarkError):
        backtester.run_backtest(
            strategy=MockStrategy(),
            benchmark="invalid_benchmark"
        )
```

## Pull Request Guidelines

### Before Submitting

1. **Ensure all tests pass**
2. **Update documentation** if adding new features
3. **Add tests** for new functionality
4. **Run code quality checks**
5. **Update CHANGELOG.md** if applicable

### Pull Request Description

Include:

- **Summary**: Brief description of changes
- **Motivation**: Why this change is needed
- **Changes**: Detailed list of changes
- **Testing**: How you tested the changes
- **Breaking Changes**: Any API changes

### Example

```markdown
## Summary
Add support for custom benchmark functions in Backtester

## Motivation
Users need more flexibility in benchmark selection beyond built-in options.

## Changes
- Add `CUSTOM_METHOD` benchmark type
- Update `get_benchmark_type()` to detect callable benchmarks
- Add validation for custom benchmark functions
- Update documentation with examples

## Testing
- Added unit tests for custom benchmark detection
- Tested with sample strategy and custom benchmark function
- All existing tests pass

## Breaking Changes
None
```

## Issue Reporting

### Bug Reports

When reporting bugs, include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, portwine version
- **Code example**: Minimal code to reproduce the issue

### Feature Requests

When requesting features, include:

- **Description**: Clear description of the feature
- **Use case**: Why this feature would be useful
- **Proposed implementation**: Any ideas for implementation
- **Alternatives**: Any existing workarounds

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release branch
4. Run full test suite
5. Update documentation
6. Create GitHub release
7. Publish to PyPI

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check the docs first

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details.

Thank you for contributing to portwine! 🍷 