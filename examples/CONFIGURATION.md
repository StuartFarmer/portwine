# Portwine Configuration Guide

This document explains the structure and options of your `config.yaml` (or JSON) file to drive Portwine's `ExecutionBase` via `run_from_config.py`. You can swap strategies, data loaders, brokers, schedules, and more without modifying code—just edit the config.

---

## Table of Contents

1. [File Formats](#file-formats)
2. [Top‑Level Sections](#top-level-sections)
3. [Strategy Section](#strategy-section)
4. [Data Loader Section](#data-loader-section)
5. [Broker Section](#broker-section)
6. [Alternative Data Section](#alternative-data-section)
7. [Execution Section](#execution-section)
8. [Schedule Section](#schedule-section)
9. [Minimal Config Example](#minimal-config-example)
10. [CLI Invocation](#cli-invocation)
11. [JSON Variant](#json-variant)

---

## File Formats

You can write your config in either **YAML** or **JSON**. The file extension (`.yaml`, `.yml`, or `.json`) determines which parser is used.

Supported extensions:

- `.yaml` / `.yml` → parsed with `PyYAML`
- `.json`         → parsed with Python's built‑in `json`


## Top‑Level Sections

Your config file supports the following **required** and **optional** sections:

| Section             | Required? | Description                                              |
|---------------------|-----------|----------------------------------------------------------|
| `strategy`          | Yes       | Strategy class FQCN and constructor parameters.          |
| `data_loader`       | Yes       | Market data loader class and parameters.                |
| `broker`            | Yes       | Broker class and credentials/config.                    |
| `alternative_data`  | No        | One or more alternative data loader definitions.        |
| `execution`         | No        | Overrides for `ExecutionBase` defaults.                |
| `schedule`          | Yes       | When to call `ExecutionBase.step()`, via `daily_schedule` presets or explicit args. |


## Strategy Section

Configures your strategy implementation (must subclass `StrategyBase`).

```yaml
strategy:
  class: portwine.strategies.simple_moving_average.SimpleMovingAverageStrategy
  params:
    tickers: ["AAPL", "MSFT", "AMZN"]
    short_window: 20
    long_window: 50
    position_size: 0.1
```

- **`class`**: Fully‑qualified class name (FQCN) of your strategy.
- **`params`**: A mapping of keyword args passed into the strategy's `__init__`.
  - `tickers`: `List[str]` of symbols.
  - Strategy‑specific arguments (e.g. moving‑average windows).


## Data Loader Section

Defines a `MarketDataLoader` implementation for historical/real‑time OHLCV data.

```yaml
data_loader:
  class: portwine.loaders.alpaca.AlpacaMarketDataLoader
  params:
    api_key: ${ALPACA_API_KEY}         # environment var interpolation
    api_secret: ${ALPACA_API_SECRET}
    start_date: "2020-01-01"         # ISO date or datetime string
    end_date:   "2023-12-31"         # optional, defaults to today
    cache_dir:  "./cache/"            # optional directory
    paper_trading: true                # use Alpaca paper endpoints
```

- **`class`**: FQCN of a `MarketDataLoader` subclass.
- **`params`**: Constructor kwargs:
  - Credentials, file paths, date ranges, caching, etc.
  - Values can be literal or `${ENV_VAR}` to pickup environment variables.


## Broker Section

Specifies the `BrokerBase` implementation for order placement:

```yaml
broker:
  class: portwine.brokers.alpaca.AlpacaBroker
  params:
    api_key:    ${ALPACA_API_KEY}
    api_secret: ${ALPACA_API_SECRET}
    base_url:   https://paper-api.alpaca.markets
```

- **`class`**: FQCN of your broker adapter.
- **`params`**: Credentials, endpoints, or any broker‑specific config.


## Alternative Data Section

(Optional) One or more secondary loaders. Useful for sentiment, fundamentals, or custom intraday transforms.

```yaml
alternative_data:
  - class: portwine.loaders.dailytoopenclose.DailyToOpenCloseLoader
    params:
      base_loader: portwine.loaders.alpaca.AlpacaMarketDataLoader
```

- This example wraps the primary Alpaca loader to emit 09:30 open bars and 16:00 close bars.
- The code will special‑case a `base_loader` string to reference the already‑constructed primary loader.


## Execution Section

Override defaults for `ExecutionBase`. Omit entirely to use:

- `min_change_pct=0.01`
- `min_order_value=1.0`
- `fractional=True`

```yaml
execution:
  min_change_pct: 0.02    # require 2% weight change to trade
  min_order_value: 10     # skip orders under $10
  fractional: false       # floor quantities to ints
```


## Schedule Section

Controls when to run your strategy via the `daily_schedule` helper. Two modes:

1. **Explicit**: Supply `after_open_minutes` and/or `before_close_minutes` plus optional intraday `interval_seconds`.  
2. **Preset**  : (Future) Use a named shortcut (e.g. `daily_close`).

```yaml
schedule:
  # Currently ignore preset; use explicit args:
  # preset: daily_close

  after_open_minutes: null       # skip on‑open scheduling
  before_close_minutes: 0        # exactly at market close
  calendar_name: NYSE            # Exchange calendar (via pandas_market_calendars)

  # Optional:
  start_date: null               # ISO date (defaults to today)
  end_date: null                 # ISO date (defaults to today)
  interval_seconds: null         # e.g. 600 for 10‑minute bars after open
  inclusive: false               # if last interval < end_dt, append end_dt
```

- At least one of `after_open_minutes` or `before_close_minutes` **must** be non‑null.
- If you supply `interval_seconds`, it only applies when `after_open_minutes` is set.
- Example schedules:
  - **Open only**: `after_open_minutes: 5`, `before_close_minutes: null`
  - **Close only**: `after_open_minutes: null`, `before_close_minutes: 0`
  - **Both**: `after_open_minutes: 5`, `before_close_minutes: 5`


## Minimal Config Example

A bare‑bones config that runs 5 minutes after open on NYSE:

```yaml
strategy:
  class: mypkg.Strategies.Momentum
  params:
    tickers: ["SPY", "QQQ"]

data_loader:
  class: portwine.loaders.alpaca.AlpacaMarketDataLoader
  params:
    api_key: ${ALPACA_API_KEY}
    api_secret: ${ALPACA_API_SECRET}

broker:
  class: portwine.brokers.alpaca.AlpacaBroker
  params:
    api_key: ${ALPACA_API_KEY}
    api_secret: ${ALPACA_API_SECRET}

# execution: omitted → use defaults

schedule:
  after_open_minutes: 5
  before_close_minutes: null
  calendar_name: NYSE
```


## CLI Invocation

Once your config is ready, just call:

```bash
pip install pyyaml  # if using YAML
python examples/run_from_config.py --config path/to/config.yaml
```

All logging goes to stdout.  Customize logging via environment variables or by editing `logging.basicConfig()`.


## JSON Variant

You can use the exact same structure in `config.json`:

```jsonc
{
  "strategy": { ... },
  "data_loader": { ... },
  "broker": { ... },
  "alternative_data": [],
  "execution": { ... },
  "schedule": { ... }
}
```

The CLI script auto‑detects `.json` vs `.yaml` and parses accordingly.

---

With this guide, you should be able to configure Portwine end‑to‑end simply by editing your `config.yaml`—no code changes required. Happy trading! 