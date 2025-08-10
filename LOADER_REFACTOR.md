## Loader → Data (Providers + Stores) Refactor

This document outlines the deprecation of `@loaders/` in favor of a `@data/` paradigm that composes:

- **Providers**: fetch data from external/live sources.
- **Stores**: persist and serve data (parquet, SQL, S3, memory, etc.).
- **Optional Source Orchestrator**: read-through cache that fetches on miss and persists to a Store.

It also updates backtester integration to remove iterator-style stepping and to use store lookups keyed by a single timestep.

---

## What changes for the Backtester

- **No iterator/next()**: The backtester no longer pulls bars via `loader.next(...)`.
- **Single-timestep reads**: The backtester sets the interface's current timestep and then passes the interface to the strategy. When the strategy does `data_interface['AAPL']`, the interface uses the underlying Store to fetch a single observation by calling:
  - `store.get(identifier='AAPL', start_date=dt, end_date=dt)` → single element.
 - **Multiple datasets**: Prefixes like `INDEX:SPY` or `ECON:GDP` are still supported via a prefix→store mapping (see Interface below). A `DataSource` may also be used anywhere a store is accepted if it implements the same API.

Implications:
- The backtester composes time; the interface reads the point at that time.
- Loader `next` semantics are deprecated and removed from the runtime path.

---

## Target Architecture

### Providers (live/external access)
- Abstract base: `DataProvider` with sync/async `get_data(identifier, start_date, end_date=None)`.
- Concrete providers: `PolygonProvider`, `AlpacaProvider`, `EODHDProvider`, `FREDProvider`, etc.
- Responsibility: network/auth, rate limits, pagination. No persistence.

### Stores (persistence and serving)
- Abstract base: `DataStore` with `add`, `get`, `get_latest`, `latest`, `exists`, `identifiers`.
- Semantics for `get`:
  - Range queries return `OrderedDict[datetime64 -> dict]` in chronological order.
  - Passing the same datetime for `start_date == end_date` yields a single-element result (used by the backtester/strategies for point-in-time reads).
- Concrete stores: `ParquetDataStore` (existing), future: `SqlDataStore`, `S3DataStore`.

### Optional: DataSource (orchestrator)
- Composes a `provider` and a `store` to offer read-through cache behavior:
  - On `get(identifier, start, end)`: read from Store; if missing or stale, fetch from Provider, `store.add(...)`, then serve from Store.
  - Policies: backfill, TTL/refresh, chunking, retries, async ingestion.
- In the backtester path, reads are still satisfied by the Store; the DataSource is used to prefill or fetch-on-miss transparently.

---

## DataInterface v2 (strategy-facing)

The interface encapsulates one or more stores (or sources that conform to the store API) and exposes the same ergonomic bracket API.

- State:
  - `current_timestamp: numpy.datetime64`
  - `stores: Dict[prefix | None, DataStoreLike]`
    - `DataStoreLike` is any object implementing the `DataStore` API (`get`, `add`, etc.). `DataSource` conforms to this.

- Usage in strategies remains:
  ```python
  data.set_current_timestamp(dt)
  close = data['AAPL']['close']
  ```

- How `__getitem__` resolves data for a symbol at the current time:
  1. Parse `prefix:symbol` → `(prefix, symbol)`.
  2. Resolve the store for `prefix`.
  3. Call `store.get(symbol, dt, dt)` and convert the single-element result to `{field: value}`. When a `DataSource` is used, it provides the same `get` behavior and may fetch-on-miss before serving.

Type policy: the interface holds and passes `numpy.datetime64` internally; conversions are confined to IO edges.

---

## API Surfaces (concise)

### DataProvider
```python
class DataProvider(abc.ABC):
    def get_data(self, identifier: str, start_date: datetime, end_date: datetime | None = None): ...
    async def get_data_async(self, identifier: str, start_date: datetime, end_date: datetime | None = None): ...
```

### DataStore
```python
class DataStore(abc.ABC):
    def add(self, identifier: str, data: dict): ...
    # Single point-in-time lookup
    def get(self, identifier: str, dt: datetime) -> dict | None: ...
    # Range lookup (chronological order)
    def get_all(self, identifier: str, start_date: datetime, end_date: datetime | None = None) -> OrderedDict[datetime, dict] | None: ...
    def get_latest(self, identifier: str) -> dict | None: ...
    def latest(self, identifier: str) -> datetime | None: ...
    def exists(self, identifier: str, start_date: datetime | None = None, end_date: datetime | None = None) -> bool: ...
    def identifiers(self) -> list[str]: ...
```

### DataSource (optional)
```python
class DataSource:
    def __init__(self, provider: DataProvider, store: DataStore, *, refresh_policy=None): ...
    def add(self, identifier: str, data: dict): ...
    # Single point-in-time lookup
    def get(self, identifier: str, dt: np.datetime64) -> dict | None: ...
    # Range lookup (chronological order)
    def get_all(self, identifier: str, start: np.datetime64, end: np.datetime64 | None = None) -> OrderedDict[np.datetime64, dict] | None: ...
    # Conforms to the DataStore API so it can be used interchangeably by the interface/backtester
```

---

## Deprecation of `@loaders/`

- `MarketDataLoader` and any `next`-based data access are deprecated.
- Backtester and strategies must not rely on iterator-style data access.
- Replacement path:
  - Convert each loader to a **Provider** (network logic only) and pair with a **Store**.
  - Use a **Source** when you want automatic fetch-on-miss + persistence.

Runtime Compat: we will not ship a runtime `LoaderCompat` adapter; instead, provide an ingestion utility that uses old loaders to prefill a Store (see Migration Playbooks).

---

## Migration Playbooks

### 1) Strategy authors
- No code changes to the strategy body if it already uses the interface:
  - Backtester sets `data.set_current_timestamp(dt)` and passes `data` to `strategy.step(dt_python, data)`.
  - Strategy reads `data['SYM']` and obtains a point-in-time dict.
- Avoid direct loader usage in strategies.

### 2) Tests/examples
- Replace constructions of loaders with either:
  - `DataStore` instances when data is already present, or
  - `DataSource(provider, store)` when you want fetch-on-miss.
- Build the `DataInterface` with a mapping of prefixes to stores (or sources conforming to the store API):
  ```python
  market_store = ParquetDataStore('...')
  econ_store = ParquetDataStore('.../econ')
  data = DataInterface({None: market_store, 'ECON': econ_store})
  ```

### 3) From Loader → Provider
- Extract network/auth/rate-limit code into a `DataProvider`.
- Do not implement caching or persistence in Providers.
- Use a `DataSource(provider, store)` to orchestrate ingestion and serve data via the Store.

### 4) One-time ingestion from legacy loaders
- Provide a script/CLI to prefill Stores using existing loaders until they are removed:
  1. Build the loader.
  2. Enumerate date ranges and tickers.
  3. Convert each bar to the Store schema and `store.add(identifier, data)`.
  4. Migrate tests/examples to use Stores afterwards.

---

## Live-data semantics

- Providers are the only live component. For real-time reads:
  - Either poll the Provider and `store.add(...)` continuously,
  - Or use a `DataSource` and rely on fetch-on-miss for the current time.
- Async parity: support `get_data_async` and batched ingestion.

---

## Time Types

- Use `numpy.datetime64` internally in interfaces and accessors for consistency and performance.
- IO boundaries (parquet/SQL/network) may convert to/from `pandas.Timestamp` as needed. Keep conversions local to IO.

---

## Rollout Plan

- Release N
  - `DataInterface` v2 using `store.get(dt, dt)` directly.
  - Add Providers (extracted from loaders) for major sources; keep `ParquetDataStore` and ensure fast single-date reads.
  - Add optional `DataSource` for read-through cache behavior.
  - Mark `@loaders/` deprecated in docs.

- Release N+1
  - Update all examples/tests to Providers+Stores.
  - Provide ingestion utility to help users migrate historical caches.

- Release N+2
  - Remove `@loaders/` modules and any imports in runtime code.

---

## Implementation Checklist

- [ ] `DataInterface` v2 that accepts `{prefix: store_like}` and manages `current_timestamp`.
- [ ] Ensure `ParquetDataStore.get` path is efficient for single-date lookups (index-based).
- [ ] Providers extracted: Polygon, Alpaca, EODHD, FRED.
- [ ] Optional `DataSource` orchestrator (sync + async paths).
- [ ] Update backtester to use only the interface time step; remove any direct `next()` usage.
- [ ] Docs and examples updated; loader-based examples marked as deprecated.


