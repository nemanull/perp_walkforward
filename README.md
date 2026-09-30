# Binance Walk-Forward

This project tests whether 5-minute data can predict short-horizon returns of crypto perpetual futures, with each model retrained every month on everything before that month and scored only on that month.

It runs one method on five Binance USD-M perpetuals, HYPE, TRX, DOGE, UNI and AAVE, with BTC, ETH and SOL as context.
Every modelling choice was made on six research months, December 2025 to May 2026, and frozen before the forward months, June to August 2026, were scored.
The write-up is [docs/paper.md](./docs/paper.md), and notes on the code and the data are in [docs/design.md](./docs/design.md).

## Results

Four of the five coins show a 20-minute reversal in the six research months and again in the three forward months.
Traded as a five-coin portfolio, the signal loses money after fees in both periods.

| | Research, Dec 2025 to May 2026 | Forward, Jun to Aug 2026 |
| --- | ---: | ---: |
| Pooled rank IC of the frozen 20-minute rule | 0.038 (t 8.0) | 0.029 (t 4.7) |
| Breakeven fee, bps per side | 0.74 | 0.45 |
| Portfolio net at maker fees, bps a day | -20.7 | -28.3 |
| Portfolio net at taker fees, bps a day | -69.9 | -83.3 |

![Detectability against the fee](./results/horizon-sweep/detectability.png)

- The signal is a 20-minute reversal, a bet that a coin's last 20-minute move partly comes back.
  UNI, DOGE and AAVE carry it with ICs of 0.05 to 0.07, HYPE more weakly, and TRX not at all.
- At 3 and 6 hours every pooled IC is within 0.01 of zero.
- BTC does not lead any of the coins at 5 minutes, and a model on the coin's own 16 inputs does as well as one on all 75.
- A one-input rule matches or beats ridge and LightGBM, with a pooled 20-minute IC of 0.042 for momentum against 0.029 for both.
- A model trained once on June to November 2025 scores the same as a monthly refit over the next six months.
- The portfolio's edge is under 1 bp per side, while Binance charges 2 bps to a maker and 5 bps to a taker, so the portfolio loses money after fees in both periods.
  In the forward months HYPE alone made money at maker fees, and its interval includes zero.
- The best-looking coins shrank in the forward months, as selection predicts.
  DOGE fell from 0.067 to 0.037 and UNI from 0.057 to 0.026.
- Volatility is far easier to predict than direction, with a pooled rank IC of 0.47 to 0.57 against 0.04, and a three-input HAR model nearly matches LightGBM.
- Trading only when forecast volatility makes the expected edge cover the fee cuts the research taker loss from 70 bps a day to about 2 because it rarely trades, and the Sharpe ratio stays negative at -0.97.
  In the forward months the gate traded on 7 of 92 days, so its positive forward Sharpe ratio is not evidence.
- One model trained on all five coins is no better than five separate ones.

I wrote the protocol, decision rules included, before computing any result on the research months, and every configuration that ran has a row in the CSV files under `results/`.

## Limitations

The study covers one venue, five coins and about 15 months.
The coins were chosen by type, and HYPE was chosen after it became a large coin, which is a selection effect.
DOGE and AAVE move together closely, so five coins carry less independent evidence than five unrelated ones would.

Positions fill at the close of the signal bar, and costs are Binance's regular fees plus funding.
There is no order book, so spread and market impact are not modelled beyond the fee, and the maker scenario is an upper bound because a passive order may not fill.

The forward months had already happened when I wrote the protocol, so only the protocol kept them out of the research.

## Method

Bars come from Binance's public monthly archive, checked against its SHA-256 files, and sit on one 5-minute UTC grid.
A trading halt is written by Binance as a flat bar with zero trades, and such bars are treated as missing.

Each coin has the 111 features of the project's earlier TypeScript version, listed in appendix A of the paper.
They cover returns and volatility, volume surges, taker aggression, and the coin's moves against BTC, ETH and SOL.
75 of them are model inputs, after dropping levels that drift over months and columns that copy other columns.
The target is the forward log return over 20 minutes, 1 hour, 3 hours or 6 hours, divided by trailing volatility and bounded with tanh.

Four model families are compared: a momentum rule, a one-feature rule, ridge and LightGBM.
A model for month M trains only on rows whose labels are known when M starts, picks its settings on the last month of its window, and is refit before predicting M.

The information coefficient is a Spearman correlation ranked over the whole period, with a stationary bootstrap over days for its standard error, because neighbouring labels overlap.
A Spearman correlation computed inside each day is biased by about minus H divided by 288 for any prediction built from recent returns, and on a random walk it finds mean reversion at 6 hours with a t-statistic near 12 over 182 days.
`tests/test_metrics.py` reproduces this bias on simulated random walks.

Signals become positions through 1/H tranches, and profit is charged taker or maker fees on turnover and funding on the position held at each settlement.

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

The download is about 125 MB and the built data about 840 MB, both under `data/`, which is not committed.
The prediction cache in `data/predictions/` adds about 335 MB.
A full run takes about 40 minutes on a 20-core machine.
`run` writes tables and figures to `results/<experiment>/`, and `plot <experiment>` redraws the figures from the tables.
The exploration notebook reruns with `uv run --group notebook jupyter nbconvert --to notebook --execute --inplace notebooks/exploration.ipynb`.

## Layout

`main.py` has the four commands.
`src/data/` downloads the archive and builds the bars and features, and `src/experiments/` holds the nine experiments, phase 1 in `phase1.py` and phase 2 in `volatility.py` and `pooled.py`.
The rest of `src/` holds the settings, the folds, the models, the metrics, the backtest and the figures.
Tables and figures are committed under `results/`, one folder per experiment.
