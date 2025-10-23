# Branch Cleanup Plan

**Current Branch**: `loader`
**Last Updated**: 2025-08-26

---

## Branch Overview

### Total Branches
- **Local**: 19 branches
- **Remote (origin)**: 12 branches
- **Remote (pro)**: 16 branches

---

## Safe to Delete - Already Merged into Main

These branches have been fully merged and can be safely deleted.

### Local Branches

| Branch | Last Commit Date | Status | Notes |
|--------|-----------------|--------|-------|
| `alternative-data` | 2025-08-26 | ✅ Merged | Alternative data sources |
| `alternative-execution` | 2025-04-18 | ✅ Merged | Alt date unions |
| `docs` | 2025-06-20 | ✅ Merged | Documentation updates |
| `execution` | 2025-04-18 | ✅ Merged | Calendar support |
| `intraday` | 2025-04-18 | ✅ Merged | Intraday data modifier |
| `logging` | 2025-04-24 | ✅ Merged | Logging fixes |
| `new_loaders` | 2025-04-24 | ✅ Merged | More fixes |
| `order_bugs` | 2025-04-24 | ✅ Merged | More fixes |
| `private/main` | 2025-04-23 | ✅ Merged | Python version update |
| `pro_main` | 2025-07-26 | ✅ Merged | Untrack files |
| `rotating-universe` | 2025-07-27 | ✅ Merged | Tests pass |
| `scheduler_fixes` | 2025-04-28 | ✅ Merged | Executor warmup improvements |
| `universe_fixes` | 2025-08-06 | ✅ Merged | Fixes |
| `loader-refactor` | 2025-04-28 | ✅ Merged/Superseded | Superseded by current `loader` branch |

**Delete Command**:
```bash
git branch -d alternative-data alternative-execution docs execution intraday \
  logging new_loaders order_bugs rotating-universe scheduler_fixes \
  universe_fixes loader-refactor
```

If any refuse deletion with `-d`, use `-D` to force (only if you're sure):
```bash
git branch -D private/main pro_main
```

### Remote Branches (origin)

| Branch | Status | Notes |
|--------|--------|-------|
| `origin/alternative-data` | ✅ Merged | Can be deleted |
| `origin/docs` | ✅ Merged | Can be deleted |
| `origin/rotating-universe` | ✅ Merged | Can be deleted |
| `origin/universe_fixes` | ✅ Merged | Can be deleted |

**Delete Command** (requires push permission):
```bash
git push origin --delete alternative-data docs rotating-universe universe_fixes
```

---

## Keep - Active or Recent Work

### Active Development

| Branch | Last Commit Date | Status | Action |
|--------|-----------------|--------|--------|
| `loader` | 2025-08-26 | 🟢 **CURRENT** | Keep - active development |
| `backtest-refactor` | 2025-08-09 | ⚠️ Investigate | May overlap with loader |
| `feeds` | 2025-08-04 | ⚠️ Recent | Check if still needed |

### Recent Remote Branches

| Branch | Last Commit Date | Location | Status |
|--------|-----------------|----------|--------|
| `origin/loader` | 2025-08-26 | origin | 🟢 Keep - matches current |
| `origin/backtest-refactor` | 2025-08-09 | origin | ⚠️ Investigate overlap |
| `origin/feeds` | 2025-08-04 | origin | ⚠️ Check if needed |
| `origin/execution-timing` | 2025-04-17 | origin | ⚠️ Check if needed |
| `origin/loader_refactor` | 2025-04-28 | origin | ⚠️ Different from loader? |

### Pro Remote Branches

Most pro/ branches appear to be synced from origin. Review with pro remote owner:
- `pro/loader-refactor`
- `pro/alternative-execution`
- `pro/execution`
- `pro/intraday`
- `pro/logging`
- `pro/scheduler_fixes`
- etc.

---

## Investigate Before Action

### 1. backtest-refactor Branch

**Status**: Needs investigation - potential conflict with current work

**Last Commit**: 2025-08-09 - "old backtester deprecated"

**Concern**:
- Your `loader` branch also deprecates old backtester
- May have overlapping changes
- Need to check for conflicts

**Investigation Commands**:
```bash
# See commits in backtest-refactor not in loader
git log loader..backtest-refactor --oneline

# See commits in loader not in backtest-refactor
git log backtest-refactor..loader --oneline

# See diff between branches
git diff loader...backtest-refactor --stat

# Check if backtest-refactor is fully contained in loader
git merge-base --is-ancestor backtest-refactor loader && echo "Fully merged" || echo "Has unique commits"
```

**Possible Outcomes**:
1. **Fully contained in loader** → Delete backtest-refactor
2. **Has unique valuable changes** → Cherry-pick into loader
3. **Has conflicts** → Need manual merge/resolution
4. **Outdated approach** → Delete after review

### 2. feeds Branch

**Last Commit**: 2025-08-04 - "feed"

**Questions**:
- What feed functionality does this add?
- Is it independent of loader refactor?
- Should it be merged to main separately?

**Investigation**:
```bash
git log main..feeds --oneline
git diff main...feeds --stat
```

### 3. loader_refactor vs loader

**Two branches with similar names**:
- `loader_refactor` (underscore) - Last commit 2025-04-28
- `loader` (current) - Last commit 2025-08-26

**Question**: Is `loader` the successor to `loader_refactor`?

**Investigation**:
```bash
# Check relationship
git merge-base loader loader_refactor

# See unique commits
git log loader_refactor..loader --oneline
```

**Likely**: `loader` is the continuation, `loader_refactor` can be deleted.

---

## Special Branches - Do Not Delete

### main
The primary branch - obviously keep.

### gh-pages
GitHub Pages deployment branch - keep for documentation hosting.

---

## Recommended Cleanup Workflow

### Phase 1: Safe Deletions (Low Risk)

```bash
# 1. Delete obviously merged local branches
git branch -d alternative-data alternative-execution docs execution intraday \
  logging new_loaders order_bugs rotating-universe scheduler_fixes universe_fixes

# 2. If any fail, review them individually before using -D
```

### Phase 2: Investigations (Medium Risk)

```bash
# 1. Check backtest-refactor relationship
git log loader..backtest-refactor --oneline
git diff loader...backtest-refactor

# 2. Check feeds branch
git log main..feeds --oneline
git diff main...feeds

# 3. Check loader_refactor vs loader
git log loader_refactor..loader --oneline
```

### Phase 3: Remote Cleanup (High Risk - Cannot Undo)

**⚠️ Warning**: Only delete remote branches if you have backups and team agreement.

```bash
# Only after confirming with team and verifying local copies exist
git push origin --delete alternative-data docs rotating-universe universe_fixes
```

### Phase 4: Pro Remote (Coordinate with Team)

Coordinate with whoever manages the `pro` remote:
- Review which pro/ branches are stale
- Check if they're synced with origin
- Decide on deletion strategy

---

## Branch Naming Convention Issues

**Inconsistent naming found**:
- `loader-refactor` (hyphen)
- `loader_refactor` (underscore)
- `loader` (no suffix)
- `backtest-refactor` (hyphen)

**Recommendation**: Standardize future branch names:
- Feature branches: `feature/name-here`
- Bug fixes: `fix/issue-description`
- Refactors: `refactor/component-name`
- Use hyphens, not underscores

---

## Post-Cleanup Verification

After cleanup, verify:

```bash
# 1. List remaining branches
git branch -a

# 2. Verify main is up to date
git checkout main
git pull origin main

# 3. Verify loader branch is clean
git checkout loader
git status

# 4. Check for any stray refs
git remote prune origin
git remote prune pro
```

---

## Summary Checklist

**Before Cleanup**:
- [ ] Create backup: `git bundle create backup.bundle --all`
- [ ] Document branch purposes (if not obvious from commits)
- [ ] Get team consensus on remote deletions
- [ ] Verify main branch is fully up to date

**Safe to Delete Immediately**:
- [ ] Local merged branches (14 branches)
- [ ] Remote merged branches on origin (4 branches)

**Investigate First**:
- [ ] backtest-refactor (potential overlap with loader)
- [ ] feeds (independent feature?)
- [ ] loader_refactor (superseded by loader?)

**Keep**:
- [ ] loader (current work)
- [ ] main (primary branch)
- [ ] gh-pages (docs deployment)

**Coordinate with Team**:
- [ ] Pro remote cleanup strategy
- [ ] Remote branch deletion permissions
- [ ] Any branches others may be using

---

## Recovery Plan (If Needed)

If you accidentally delete something important:

```bash
# 1. Find the commit SHA from reflog
git reflog

# 2. Recreate branch
git branch recovered-branch <commit-sha>

# 3. Or restore from bundle
git fetch backup.bundle <branch-name>:<branch-name>
```

---

## Estimated Impact

**Disk space saved**: ~100-500 MB (depending on branch sizes)

**Reduced complexity**:
- From 19 local branches → ~5-7 active branches
- Clearer `git branch` output
- Less confusion about which branch to use

**Time saved**:
- Faster branch listing
- Less mental overhead deciding which branch to use
- Reduced risk of working on wrong branch
