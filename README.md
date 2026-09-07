# perp_walkforward

This project tests whether 5-minute data can predict short-horizon returns of crypto perpetual futures, with each model retrained every month on everything before that month and scored only on that month.

It runs one method on five Binance USD-M perpetuals, HYPE, TRX, DOGE, UNI and AAVE, with BTC, ETH and SOL as context.

## Running it

```
uv sync
uv run walkforward download --from 2025-05 --to 2026-08
uv run walkforward build
uv run pytest -q
```

The download and the built data go to `data/`, which is not committed.
