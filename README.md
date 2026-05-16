# perp_walkforward

Short-horizon return models for Binance USD-M perpetuals on 5-minute bars, tested walk-forward.
For now it is HYPE, with BTC, ETH and SOL as context.

`download.py` fetches the monthly 5-minute klines from data.binance.vision into `data/`.
`features.py` builds the feature table and writes it to `out/`.
The features are a port of my old TypeScript builder.

```
uv run download.py
uv run features.py
```
