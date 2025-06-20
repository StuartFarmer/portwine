# Installation

## Prerequisites

Portwine requires Python 3.8 or higher. We recommend using a virtual environment to manage dependencies.

## Installing Portwine

### Using pip

The easiest way to install portwine is using pip:

```bash
pip install portwine
```

### Using Poetry

If you prefer using Poetry for dependency management:

```bash
poetry add portwine
```

### From Source

To install from the latest development version:

```bash
git clone https://github.com/yourusername/portwine.git
cd portwine
pip install -e .
```

## Dependencies

Portwine has the following key dependencies:

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **cvxpy** - Convex optimization (for some benchmarks)
- **tqdm** - Progress bars
- **pandas-market-calendars** - Trading calendar support

## Optional Dependencies

Some features require additional packages:

- **matplotlib** - For plotting and visualization
- **seaborn** - Enhanced plotting capabilities
- **plotly** - Interactive plots

## Verification

To verify your installation, run:

```python
import portwine
print(portwine.__version__)
```

## Next Steps

Once installed, head over to the [Quick Start](quick-start.md) guide to run your first backtest! 