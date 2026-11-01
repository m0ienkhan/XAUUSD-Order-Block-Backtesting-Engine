# XAUUSD Order Block Backtesting Engine

Research and execution backtesting tools for XAUUSD 15-minute order-block strategies.

## Project contents

- `src/` - market-data, structure, analytics, strategy, and execution modules
- `scripts/` - research, walk-forward, execution, and robustness runners
- `config/` - market-data and strategy configuration
- `data/` - source and test price data
- `outputs/reports/` - generated research and backtest results
- `tests/` - unit tests

## Run

From the project root, with the project dependencies installed:

```powershell
python -m unittest discover -s tests -v
python scripts/run_execution_simulation.py
```

The execution runner uses the default historical data and report paths under
`data/raw/` and `outputs/reports/`. See each script's `--help` for configurable
inputs and outputs.

## Important

Backtest results are historical simulations, not a guarantee of future
performance or investment advice. Verify data licensing and broker contract
specifications before redistribution or live use. Spread, slippage, financing,
and account-level capital and margin constraints may not be included in a given
report.
