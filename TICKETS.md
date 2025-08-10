## Tickets for Loader → Data (Providers + Stores) Refactor

This tracks implementable tickets derived from `LOADER_REFACTOR.md`. Tickets are grouped by milestone. All new runtime code should prefer `numpy.datetime64` internally for time handling.

---

### Milestone N — Foundations and Dual-Path Support

1) DataInterface v2: prefix→store mapping and point-in-time access
- Scope:
  - Update `portwine/data/interface.py` to accept `{prefix | None: DataStoreLike}` instead of loaders.
  - Maintain `set_current_timestamp(dt)` and `__getitem__` with `store.get(symbol, dt)` single-point lookup.
  - Preserve `MultiDataInterface`/`RestrictedDataInterface` behavior with prefix routing and restrictions.
- Acceptance:
  - `DataInterface`, `MultiDataInterface`, `RestrictedDataInterface` work with any object implementing the `DataStore` API (`get`, `add`, etc.).
  - Existing strategy code using `data['SYM']` continues to work.
  - Unit tests added/updated to cover prefixed symbols and restricted tickers.

2) DataSource: read-through cache that conforms to DataStore API
- Scope:
  - Add `portwine/data/source.py` with `DataSource(provider, store)` implementing `add`, `get`, `get_latest`, `latest`, `exists`, `identifiers`.
  - Behavior: On `get` and `get_all`, read Store first; if missing/stale, fetch via Provider, `store.add(...)`, then serve via Store.
- Acceptance:
  - Can be passed to `DataInterface` in place of a store.
  - Unit tests demonstrate cache hit/miss, backfill, and identical return shape to `DataStore.get` and `DataStore.get_all`.

3) ParquetDataStore: efficient single-date lookups and numpy-first IO boundaries
- Scope:
  - Ensure `portwine/data/store.py:ParquetDataStore.get` performs efficient single-date lookups by index.
  - Normalize to UTC and `numpy.datetime64` internally; convert only at IO edges to/from pandas for parquet.
- Acceptance:
  - Benchmarks show no significant regression for single-date reads.
  - Unit tests verify correct single-date retrieval and type consistency.

4) Providers extracted from loaders
- Scope:
  - Create provider classes under `portwine/data/providers/` (new package): `PolygonProvider`, `AlpacaProvider`, `EODHDProvider`, `FREDProvider`.
  - Move network/auth/rate-limit/pagination logic out of `portwine/loaders/*.py` into providers.
  - Keep providers free of persistence/caching.
- Acceptance:
  - Minimal tests hitting mocked HTTP to validate URL construction, pagination, and date filtering.
  - Existing example configs can instantiate providers without loader dependencies.

5) Deprecation warnings in loaders
- Scope:
  - Add `DeprecationWarning` on import/instantiation for modules in `portwine/loaders/` and in `MarketDataLoader`.
  - Update module-level docstrings to point to Providers + Stores.
- Acceptance:
  - Importing any loader emits a single `DeprecationWarning` (safely suppressible in tests).

6) Backtester integration with interface timestep
- Scope:
  - Confirm `portwine/backtester/core.py` uses `DataInterface.set_current_timestamp(dt)` and passes the interface into the strategy without any `next()` calls.
  - Remove any remaining references to `MarketDataLoader` in backtester internals (keep import compatibility if needed for a release).
- Acceptance:
  - Backtester compiles and runs with `DataInterface` backed by Store/DataSource.
  - All backtester-related tests pass unchanged or with minimal fixture updates.

---

### Milestone N+1 — Migration of Examples and Tests

7) Update examples to Providers + Stores
- Scope:
  - Replace loader usage in `examples/*.py` with construction of `ParquetDataStore` or `DataSource(provider, store)` and `DataInterface({None: store_like, ...})`.
- Acceptance:
  - All examples execute (offline where applicable) using locally cached parquet or mocked providers.

8) Ingestion utility for legacy caches
- Scope:
  - Add CLI/script (e.g., `portwine ingest`) to backfill a store from a Provider over a date range and symbols.
  - Optional adapter to ingest from legacy loaders for one-time migration.
- Acceptance:
  - Running the CLI writes parquet files and subsequent runs of examples/tests consume the store without network calls.

9) Test suite migration and coverage
- Scope:
  - Update tests to instantiate `DataStore`/`DataSource` in place of loaders.
  - Add tests for prefixed datasets (`ECON:...`, `INDEX:...`).
- Acceptance:
  - All tests in `tests/` pass locally with the new data stack.
  - Coverage added for `DataSource` cache paths and single-date retrieval.

10) Documentation updates
- Scope:
  - Update `docs/user-guide/data-management.md`, `docs/api/*` to describe Providers, Stores, optional DataSource, and the new backtester flow.
  - Mark loader docs as deprecated and link to migration guidance.
- Acceptance:
  - Docs build cleanly and reflect the new APIs with examples.

---

### Milestone N+2 — Decommission Loaders

11) Remove loader imports from runtime
- Scope:
  - Remove `from portwine.loaders...` imports across the codebase.
  - Delete loader-specific branches in any modules that still account for them.
- Acceptance:
  - Static search shows no runtime imports of `portwine.loaders`.

12) Delete `@loaders/` package
- Scope:
  - Remove `portwine/loaders/` modules and update packaging metadata.
  - Ensure changelog notes deprecation window and removal.
- Acceptance:
  - Project installs and runs without loaders, tests pass, docs updated.

---

### Cross-cutting (apply throughout milestones)

13) Time handling consistency
- Scope:
  - Use `numpy.datetime64` for in-memory timestamps in `DataInterface`, `DataSource`, `ParquetDataStore` interactions; confine conversions at IO.
- Acceptance:
  - Spot checks and unit tests confirm `numpy.datetime64` usage in internal paths.

14) Performance guardrails
- Scope:
  - Benchmarks for single-date get and typical backtest run comparing pre/post refactor.
- Acceptance:
  - No significant regressions; document any tradeoffs and mitigations if found.


