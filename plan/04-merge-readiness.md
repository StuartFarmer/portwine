# Merge Readiness & Next Steps

**Current Branch**: `loader`
**Target Branch**: `main`
**Status**: ⚠️ Not Ready - Blockers Present

---

## Merge Readiness Assessment

### ❌ Cannot Merge Yet

**Reason**: Critical blockers prevent successful merge and testing.

### Completion Estimate: 80-85%

**What's Done**:
- ✅ New architecture implemented
- ✅ Provider system complete
- ✅ Store implementations (CSV, Parquet)
- ✅ Data interfaces working
- ✅ Backward compatibility layer
- ✅ Migration examples and documentation
- ✅ Major backtester refactor
- ✅ Performance optimizations

**What's Blocking**:
- ❌ Missing `earliest()` method in stores (runtime crash)
- ❌ Missing cvxpy dependency (tests won't run)
- ❌ TODO cleanup in backtester initialization
- ❌ Full test suite validation needed

---

## Pre-Merge Checklist

### 1. Critical Fixes (MUST DO)

- [ ] **Install cvxpy dependency**
  - File: [pyproject.toml](../pyproject.toml)
  - Action: Add `cvxpy>=1.4.0` to dependencies
  - Verify: `pip install cvxpy && python -c "import cvxpy"`

- [ ] **Implement `earliest()` in DataStore classes**
  - Files:
    - [portwine/data/stores/base.py](../portwine/data/stores/base.py)
    - [portwine/data/stores/csvstore.py](../portwine/data/stores/csvstore.py)
    - [portwine/data/stores/parquet.py](../portwine/data/stores/parquet.py)
    - [portwine/data/stores/noisy.py](../portwine/data/stores/noisy.py)
  - Test: Create unit tests for new method

- [ ] **Run full test suite**
  - Command: `pytest -v`
  - Fix any failures
  - Ensure all existing tests pass

### 2. Code Quality (SHOULD DO)

- [ ] **Refactor Backtester.__init__ TODO**
  - File: [portwine/backtester/core.py:181](../portwine/backtester/core.py#L181)
  - Extract factory method for cleaner initialization

- [ ] **Implement MarketDataLoaderAdapter.identifiers()**
  - File: [portwine/data/stores/adapter.py:110](../portwine/data/stores/adapter.py#L110)
  - Low priority but improves API completeness

- [ ] **Review error handling**
  - Check try/except blocks in interface.py
  - Add logging for fallback paths

- [ ] **Code review pass**
  - Review all 28 commits
  - Check for debug print statements
  - Verify no sensitive data in commits

### 3. Testing (SHOULD DO)

- [ ] **Integration tests**
  - Test backtester with each store type
  - Test MultiDataInterface scenarios
  - Test backward compatibility paths

- [ ] **Performance benchmarks**
  - Compare before/after performance
  - Verify "10x speedup" claims
  - Document any regressions

- [ ] **Edge cases**
  - Empty datasets
  - Missing tickers
  - Date range boundaries
  - Alternative data with missing dates

### 4. Documentation (NICE TO HAVE)

- [ ] **Update CHANGELOG**
  - Summarize major changes
  - Note breaking changes (if any)
  - Credit any contributors

- [ ] **Review documentation**
  - Ensure [data-management.md](../docs/user-guide/data-management.md) is accurate
  - Check migration example works
  - Add troubleshooting section

- [ ] **API documentation**
  - Ensure docstrings are complete
  - Generate API docs if using Sphinx/MkDocs

### 5. Commit Cleanup (OPTIONAL)

**Current**: 28 commits on loader branch

**Options**:
1. **Keep as-is** - Full history preserved
2. **Interactive rebase** - Squash/reorder for clarity
3. **Squash merge** - Single commit on main

**Recommendation**: Interactive rebase to 5-8 logical commits:
- Commit 1: Core data provider/store architecture
- Commit 2: Data interfaces implementation
- Commit 3: Backward compatibility layer
- Commit 4: Backtester integration
- Commit 5: Tests and documentation
- Commit 6: Performance optimizations
- Commit 7: Bug fixes and polish
- Commit 8: CSV store addition

**Command**:
```bash
git rebase -i HEAD~28
# Mark commits as 'squash' or 'fixup' in editor
```

---

## Step-by-Step Merge Plan

### Phase 1: Fix Blockers (Day 1)

**Morning**: Fix cvxpy dependency
```bash
# 1. Add to pyproject.toml
# 2. Install: pip install cvxpy
# 3. Verify: pytest --collect-only
# 4. Commit: git commit -am "Add cvxpy dependency"
```

**Afternoon**: Implement `earliest()` method
```bash
# 1. Add to base.py as abstract method
# 2. Implement in csvstore.py
# 3. Implement in parquet.py
# 4. Implement in noisy.py (delegate)
# 5. Write tests
# 6. Run: pytest tests/test_stores.py -v
# 7. Commit: git commit -am "Implement earliest() in DataStore classes"
```

**Evening**: Run full test suite
```bash
# 1. Run: pytest -v
# 2. Fix any failures
# 3. Document results
```

### Phase 2: Code Quality (Day 2)

**Morning**: Refactor backtester initialization
```bash
# 1. Extract factory method
# 2. Test backtester with different interface types
# 3. Commit: git commit -am "Refactor backtester data interface initialization"
```

**Afternoon**: Code review and cleanup
```bash
# 1. Review all changes: git log main..loader
# 2. Check for TODOs: rg "TODO|FIXME|XXX" --type py
# 3. Remove debug code
# 4. Check for sensitive data: git log -p | grep -i "api_key\|secret\|password"
```

### Phase 3: Testing (Day 3)

**Morning**: Integration testing
```bash
# 1. Test with CSVDataStore
# 2. Test with ParquetDataStore
# 3. Test with MultiDataInterface
# 4. Test backward compatibility
# 5. Document any issues
```

**Afternoon**: Performance validation
```bash
# 1. Run performance benchmarks
# 2. Compare with main branch
# 3. Document results
```

### Phase 4: Pre-Merge (Day 4)

**Morning**: Documentation review
```bash
# 1. Review all docs
# 2. Test migration example
# 3. Update CHANGELOG
```

**Afternoon**: Optional commit squashing
```bash
# 1. Backup: git branch loader-backup
# 2. Interactive rebase: git rebase -i HEAD~28
# 3. Verify: pytest -v
# 4. Force push if needed: git push -f origin loader
```

### Phase 5: Merge (Day 5)

**Create Pull Request**:
```bash
# 1. Push to remote: git push origin loader
# 2. Create PR on GitHub/GitLab
# 3. Add description from plan docs
# 4. Request reviews
```

**PR Description Template**:
```markdown
## Data Layer Refactor

### Summary
Complete refactor of data layer to separate providers, stores, and interfaces.
Provides flexible, extensible architecture for multiple data sources.

### Changes
- New provider system for data sources
- Pluggable storage backends (CSV, Parquet)
- Multi-source data interfaces
- Backward compatible with legacy loaders
- 10x performance improvements in backtester
- Comprehensive test coverage

### Breaking Changes
None - fully backward compatible via adapter layer.

### Migration
See examples/loader_migration_example.py and docs/user-guide/data-management.md

### Testing
- 79 test files updated
- All tests passing
- Integration tests added
- Performance benchmarks included

### Closes Issues
- #XXX - Multiple data sources support
- #YYY - Performance optimization
- #ZZZ - Alternative data integration
```

**Merge Strategy**:
```bash
# Option 1: Squash merge (recommended for cleaner history)
# Via GitHub/GitLab UI: "Squash and merge"

# Option 2: Regular merge (preserves all commits)
git checkout main
git merge loader --no-ff
git push origin main

# Option 3: Rebase merge (linear history)
git checkout loader
git rebase main
git checkout main
git merge loader --ff-only
git push origin main
```

---

## Post-Merge Actions

### Immediate (Day of Merge)

- [ ] **Tag release**
  ```bash
  git tag -a v2.0.0 -m "Data layer refactor"
  git push origin v2.0.0
  ```

- [ ] **Update documentation site**
  - Rebuild docs: `mkdocs build`
  - Deploy: `mkdocs gh-deploy`

- [ ] **Notify team**
  - Send migration guide to users
  - Note any API changes
  - Provide support channel

### First Week

- [ ] **Monitor issues**
  - Watch for bug reports
  - Be available for questions
  - Fix critical issues quickly

- [ ] **Gather feedback**
  - User experience with new API
  - Performance observations
  - Feature requests

- [ ] **Clean up branches**
  - Delete merged branches (see [03-branch-cleanup.md](03-branch-cleanup.md))
  - Archive old branches if needed

### First Month

- [ ] **Write blog post** (optional)
  - Explain refactor rationale
  - Show before/after examples
  - Discuss lessons learned

- [ ] **Plan next phase**
  - Additional store types (database, cloud)
  - More data providers
  - Advanced features

---

## Risk Assessment

### Low Risk
✅ Backward compatibility maintained
✅ Comprehensive test coverage
✅ Gradual rollout possible (adapters allow old code to work)

### Medium Risk
⚠️ Large PR (79 files changed)
⚠️ Multiple subsystems affected
⚠️ Performance changes need validation

### High Risk
❌ None identified (after blockers fixed)

### Mitigation Strategies

1. **Feature flag** (if supported by framework)
   ```python
   USE_NEW_DATA_LAYER = os.getenv('USE_NEW_DATA_LAYER', 'false') == 'true'
   ```

2. **Gradual rollout**
   - Week 1: Internal testing only
   - Week 2: Beta users
   - Week 3: General availability

3. **Rollback plan**
   ```bash
   # If critical issues found
   git revert <merge-commit-sha>
   git push origin main
   ```

4. **Hotfix branch**
   ```bash
   # For urgent fixes post-merge
   git checkout -b hotfix/data-layer-fix main
   # Make fixes
   git checkout main
   git merge hotfix/data-layer-fix
   ```

---

## Success Metrics

### Technical Metrics

- [ ] **All tests pass** (baseline: current passing rate)
- [ ] **Performance maintained or improved** (target: ≥10x from commits)
- [ ] **Code coverage** (target: ≥80% for new code)
- [ ] **No regressions** in existing features

### User Metrics

- [ ] **Migration success rate** (target: 100% of users can migrate)
- [ ] **Issue reports** (target: <5 bugs in first week)
- [ ] **User satisfaction** (gather feedback)

### Project Metrics

- [ ] **Extensibility** - Can add new providers easily
- [ ] **Maintainability** - Clear separation of concerns
- [ ] **Documentation** - Complete and accurate

---

## Timeline Estimate

### Optimistic (5 days)
- Day 1: Fix blockers
- Day 2: Code quality
- Day 3: Testing
- Day 4: Documentation
- Day 5: Merge

### Realistic (7-10 days)
- Days 1-2: Fix blockers + buffer
- Days 3-4: Code quality + review
- Days 5-6: Testing + fixes
- Days 7-8: Documentation + polish
- Days 9-10: PR review + merge

### Conservative (14 days)
- Days 1-3: Fix blockers thoroughly
- Days 4-6: Code quality + multiple reviews
- Days 7-10: Extensive testing
- Days 11-12: Documentation + examples
- Days 13-14: PR review + merge

**Recommendation**: Plan for realistic timeline (7-10 days) to ensure quality.

---

## Key Contacts & Resources

### Code Owners
- Data layer: [Your name/team]
- Backtester: [Your name/team]
- Testing: [Your name/team]

### Documentation
- Architecture: [plan/02-architecture-overview.md](02-architecture-overview.md)
- Blockers: [plan/01-critical-blockers.md](01-critical-blockers.md)
- Branch cleanup: [plan/03-branch-cleanup.md](03-branch-cleanup.md)

### References
- User guide: [docs/user-guide/data-management.md](../docs/user-guide/data-management.md)
- Migration example: [examples/loader_migration_example.py](../examples/loader_migration_example.py)
- Performance guide: [docs/performance_optimization_guide.md](../docs/performance_optimization_guide.md)

---

## Final Pre-Merge Verification

Before clicking "Merge", verify:

```bash
# 1. All blockers resolved
grep -r "TODO.*earliest" portwine/data/
grep "cvxpy" pyproject.toml

# 2. Tests pass
pytest -v --tb=short

# 3. No debug code
rg "print\(|console\.log|debugger|pdb\.set_trace" --type py

# 4. No secrets
git log -p | grep -i "api_key\|secret\|password\|token" | grep -v "example\|test"

# 5. Documentation current
git diff main..loader -- docs/

# 6. Clean working directory
git status

# 7. Up to date with main
git fetch origin main
git log loader..origin/main  # Should be empty or acceptable

# 8. CI/CD passes (if configured)
# Check GitHub Actions / GitLab CI status
```

All checks pass? **Ready to merge!** 🚀

---

## Emergency Contacts

If something goes wrong post-merge:

1. **Immediate**: Post in team chat
2. **Within 1 hour**: Page on-call if production impacted
3. **Rollback decision**: Coordinate with team lead

**Remember**: The adapter layer means old code still works. Most issues will be isolated to new API users.
