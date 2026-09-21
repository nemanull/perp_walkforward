# perp_walkforward

This project tests whether 5-minute data can predict short-horizon returns of crypto perpetual futures, with each model retrained every month on everything before that month and scored only on that month.

It runs one method on five Binance USD-M perpetuals, HYPE, TRX, DOGE, UNI and AAVE, with BTC, ETH and SOL as context.
The protocol is in [docs/paper.md](./docs/paper.md).

## Running it

```
uv sync
uv run walkforward download --from 2025-05 --to 2026-08
uv run walkforward build
uv run walkforward run audit
uv run walkforward run horizon-sweep
uv run walkforward run feature-sources
uv run walkforward run retraining
uv run walkforward run economics
uv run walkforward run forward
uv run walkforward run volatility
uv run walkforward run volatility-strategy
uv run pytest -q
```

The order matters, because later experiments read the choices of earlier ones from `results/`.

The download and the built data go to `data/`, which is not committed.
`run` writes tables and figures to `results/<experiment>/`, and `plot <experiment>` redraws the figures from the tables.
