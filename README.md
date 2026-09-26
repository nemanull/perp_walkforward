# perp_walkforward

This project tests whether 5-minute data can predict short-horizon returns of crypto perpetual futures, with each model retrained every month on everything before that month and scored only on that month.

It runs one method on five Binance USD-M perpetuals, HYPE, TRX, DOGE, UNI and AAVE, with BTC, ETH and SOL as context.
The protocol is in [docs/paper.md](./docs/paper.md).

## Running it

```
uv sync
uv run main.py download --from 2025-05 --to 2026-08
uv run main.py build
uv run main.py run audit
uv run main.py run horizon-sweep
uv run main.py run feature-sources
uv run main.py run retraining
uv run main.py run economics
uv run main.py run forward
uv run main.py run volatility
uv run main.py run volatility-strategy
uv run main.py run pooled
uv run pytest -q
```

The order matters, because later experiments read the choices of earlier ones from `results/`.

The download and the built data go to `data/`, which is not committed.
`run` writes tables and figures to `results/<experiment>/`, and `plot <experiment>` redraws the figures from the tables.
