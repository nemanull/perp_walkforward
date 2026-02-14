# perp_walkforward

Short-horizon return models for Binance USD-M perpetuals on 5-minute bars, tested walk-forward.
For now it is HYPE, with BTC, ETH and SOL as context.

`features.py` builds the feature table from the CSVs in `Datasets/` and writes it to `out/`.
The features are a port of my old TypeScript builder.

```
uv run features.py
```
