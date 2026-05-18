# perp_walkforward

Short-horizon return models for Binance USD-M perpetuals on 5-minute bars, tested walk-forward.
For now it is HYPE and DOGE, with BTC, ETH and SOL as context.

`download.py` fetches the monthly 5-minute klines from data.binance.vision into `data/`.
`features.py` builds the feature table and writes it to `out/`.
The features are a port of my old TypeScript builder.
`evaluate.py` fits LightGBM to the volatility-scaled forward return in a purged k-fold and prints the daily rank IC for each horizon.

```
uv run download.py
uv run features.py
uv run evaluate.py
```
