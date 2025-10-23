# Data Refactor Plan

**Branch**: `loader`
**Status**: 80-85% Complete
**Created**: 2025-10-23
**Target**: Merge to `main`

---

## Quick Links

| Document | Purpose | Priority |
|----------|---------|----------|
| [01-critical-blockers.md](01-critical-blockers.md) | Issues that MUST be fixed before merge | 🚨 HIGH |
| [02-architecture-overview.md](02-architecture-overview.md) | Complete system architecture documentation | 📚 Reference |
| [03-branch-cleanup.md](03-branch-cleanup.md) | Branch management and cleanup strategy | 🧹 Medium |
| [04-merge-readiness.md](04-merge-readiness.md) | Merge checklist and timeline | ✅ High |

---

## Overview

This plan documents a massive data layer refactor that's currently 80-85% complete on the `loader` branch. The refactor separates concerns into three layers:

```
DataProvider (source) → DataStore (storage) → DataInterface (access) → Backtester
```

### Current State

**Branch**: 28 commits ahead of main, 78 files changed

**What Works**:
- ✅ New architecture implemented
- ✅ Provider system (Alpaca, EODHD, FRED, Polygon)
- ✅ Store implementations (CSV, Parquet, Noisy)
- ✅ Data interfaces (single, multi-source, restricted)
- ✅ Backward compatibility via adapters
- ✅ Documentation and examples
- ✅ Performance optimizations (10x speedup)

**What's Blocking Merge**:
- ❌ Missing `earliest()` method in stores → runtime crash
- ❌ Missing cvxpy dependency → tests won't run
- ❌ TODO cleanup in backtester initialization
- ❌ Full test validation needed

---

## Start Here

### If You're New to This Refactor

1. Read: [02-architecture-overview.md](02-architecture-overview.md) - Understand the design
2. Read: [01-critical-blockers.md](01-critical-blockers.md) - Know what needs fixing
3. Read: [04-merge-readiness.md](04-merge-readiness.md) - See the path forward

### If You Need to Fix Blockers

1. Start with: [01-critical-blockers.md](01-critical-blockers.md)
2. Follow the solution sections for each blocker
3. Check off items in: [04-merge-readiness.md](04-merge-readiness.md)

### If You're Ready to Merge

1. Complete all items in: [04-merge-readiness.md](04-merge-readiness.md)
2. Review: [03-branch-cleanup.md](03-branch-cleanup.md) - Clean up old branches
3. Create PR following the template in merge-readiness doc

### If You Need Context

1. Review git history: `git log main..loader --oneline`
2. See file changes: `git diff main...loader --stat`
3. Read architecture doc: [02-architecture-overview.md](02-architecture-overview.md)

---

## Critical Blockers Summary

### 1. Missing `earliest()` Method
**Files**: [stores/base.py](../portwine/data/stores/base.py), [stores/csvstore.py](../portwine/data/stores/csvstore.py), [stores/parquet.py](../portwine/data/stores/parquet.py)

**Impact**: Runtime crash when backtester tries to determine start dates

**Solution**: Implement method in all DataStore classes

**Estimated Time**: 2-3 hours

---

### 2. Missing cvxpy Dependency
**File**: [pyproject.toml](../pyproject.toml)

**Impact**: Tests can't even run

**Solution**: Add `cvxpy>=1.4.0` to dependencies

**Estimated Time**: 15 minutes

---

### 3. Backtester TODO
**File**: [backtester/core.py:181](../portwine/backtester/core.py#L181)

**Impact**: Code quality issue, messy initialization logic

**Solution**: Extract factory method for cleaner code

**Estimated Time**: 1-2 hours

---

## Timeline to Merge

### Fast Track (5 days)
Focus only on critical blockers and basic testing.

### Recommended (7-10 days)
Includes code quality, thorough testing, and documentation review.

### Conservative (14 days)
Comprehensive review, extensive testing, team review cycles.

See [04-merge-readiness.md](04-merge-readiness.md) for detailed timeline.

---

## Key Files Reference

### New Data Layer
```
portwine/data/
├── interface.py              # DataInterface, MultiDataInterface
├── providers/
│   ├── base.py              # DataProvider abstract base
│   ├── alpaca.py            # Alpaca provider
│   ├── eodhd.py             # EODHD provider
│   ├── fred.py              # FRED economic data
│   ├── polygon.py           # Polygon.io provider
│   └── loader_adapters.py   # Backward compatibility
└── stores/
    ├── base.py              # DataStore abstract base
    ├── csvstore.py          # CSV storage
    ├── parquet.py           # Parquet storage
    ├── noisy.py             # Noise injection decorator
    └── adapter.py           # Legacy loader adapter
```

### Modified Core Files
- [portwine/backtester/core.py](../portwine/backtester/core.py) - Refactored backtester
- [portwine/backtester/benchmarks.py](../portwine/backtester/benchmarks.py) - Benchmarking
- [portwine/universe.py](../portwine/universe.py) - Universe integration
- [portwine/strategies/base.py](../portwine/strategies/base.py) - Strategy base class

### Documentation
- [docs/user-guide/data-management.md](../docs/user-guide/data-management.md) - User guide
- [examples/loader_migration_example.py](../examples/loader_migration_example.py) - Migration examples
- [docs/performance_optimization_guide.md](../docs/performance_optimization_guide.md) - Performance tips

### Tests
- [tests/test_stores.py](../tests/test_stores.py) - Store tests
- [tests/test_adapter.py](../tests/test_adapter.py) - Adapter tests
- [tests/test_loader_adapters_compatibility.py](../tests/test_loader_adapters_compatibility.py) - Compatibility
- [tests/test_multidata_interface.py](../tests/test_multidata_interface.py) - Interface tests
- [tests/test_backtester_integration.py](../tests/test_backtester_integration.py) - Integration tests

---

## Branch Status

### Active Development
- **loader** (current) - Data refactor work
- **main** - Production branch

### Investigate
- **backtest-refactor** - May overlap, needs investigation
- **feeds** - Recent work, check relationship
- **loader_refactor** - Likely superseded by loader

### Safe to Delete
See [03-branch-cleanup.md](03-branch-cleanup.md) for full list of 14+ branches that can be deleted.

---

## Quick Commands

### Setup
```bash
# Install dependencies
pip install cvxpy

# Run tests
pytest -v

# Check for TODOs
rg "TODO|FIXME" --type py portwine/
```

### Development
```bash
# See what changed
git diff main...loader --stat

# See commit history
git log main..loader --oneline

# Check test status
pytest --collect-only
```

### Investigation
```bash
# Check backtest-refactor relationship
git log loader..backtest-refactor --oneline
git diff loader...backtest-refactor

# Find branches to clean up
git branch --merged main
```

### Pre-Merge
```bash
# Verify no secrets
git log -p | grep -i "api_key\|secret" | grep -v "example\|test"

# Check for debug code
rg "print\(|pdb\.set_trace" --type py portwine/

# Verify tests pass
pytest -v --tb=short
```

---

## Success Criteria

Before merging to main, verify:

- [ ] All tests pass
- [ ] No critical blockers remain
- [ ] Documentation is accurate
- [ ] Migration path is clear
- [ ] Performance is maintained/improved
- [ ] Backward compatibility verified
- [ ] Code review complete

---

## Questions?

### Architecture Questions
See [02-architecture-overview.md](02-architecture-overview.md) for:
- System design rationale
- Component interactions
- Data flow diagrams
- Design patterns used

### Technical Issues
See [01-critical-blockers.md](01-critical-blockers.md) for:
- Known issues
- Solutions
- Code examples
- Test requirements

### Process Questions
See [04-merge-readiness.md](04-merge-readiness.md) for:
- Merge checklist
- Timeline estimates
- Risk assessment
- Post-merge actions

### Cleanup Questions
See [03-branch-cleanup.md](03-branch-cleanup.md) for:
- Branch deletion plan
- Investigation commands
- Risk levels
- Recovery procedures

---

## Plan Maintenance

### When to Update This Plan

- After completing any blocker
- After significant progress
- When discovering new issues
- Before major milestones

### How to Update

```bash
# Edit relevant file
vim plan/01-critical-blockers.md

# Commit changes
git add plan/
git commit -m "Update plan: completed blocker X"
```

### Version History

| Date | Update | Changed By |
|------|--------|------------|
| 2025-10-23 | Initial plan created | Stuart |

---

## Contact

**Project**: Portwine Backtesting Framework
**Repository**: /Users/stuart/Developer/PycharmProjects/portwine
**Branch**: loader
**Last Updated**: 2025-10-23

For questions or help, refer to the specific plan documents linked above.
