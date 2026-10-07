# Design notes

How the code and the data fit together.
The research questions, the method and the results are in [`README.md`](../README.md).

## Data

Everything comes from Binance's public archive at data.binance.vision.
`download` fetches monthly 5-minute klines for the eight pairs and monthly funding rates for the five targets, and checks every zip against its `.CHECKSUM` file before unpacking it into `data/raw/`.
`build` puts the eight pairs on one grid in `data/bars.parquet`, writes the funding settlements to `data/funding.parquet`, and writes one feature file per target to `data/features/{coin}.parquet`.
Nothing under `data/` is committed, since those two commands recreate it.

HYPEUSDT's first 5-minute bar opens at 2025-05-30 10:30 UTC, so the grid starts there for every pair.
The other seven pairs have years of history, but with one calendar no coin's models train on more data than another's.

Things in the archive that caused trouble or nearly did:

- Binance writes a trading halt as a flat bar with zero trades, not as a missing row.
  It happened on 2025-08-29 from 06:20 to 06:30 UTC on all eight pairs.
  A zero-trade bar is set to missing, because its zero return is not an observation.
- A ticker can be reused.
  PUMPUSDT held another token until mid-2025, then a run of flat zero-trade bars, then a new token at a price about nine times lower.
  A pair whose real history starts later gets an entry in `FIRST_BAR` in `config.py`, and the audit lists every close that moves by a ratio above 1.3 or below 1/1.3 across a gap or a run of zero-trade bars.
- Funding times carry a few milliseconds of jitter, so they are floored to 5 minutes.
  One HYPEUSDT settlement, 2026-06-24 04:00 UTC, is missing from both the archive and the API, and the backtest charges only settlements that exist.
  HYPE settles every 4 hours and the other four targets every 8, which the backtest reads from the archive.
- The old TypeScript build joined pairs on close time and lagged by row, so one missing bar shifted every later lag.
  Here every pair sits on one complete 5-minute grid, nothing is forward-filled, and a lag counts time rather than rows.

Every rolling feature whose window touches a missing bar comes out as NaN, while a one-bar return only needs its two ends.
The backtest carries a halted bar's price forward, so the move lands on the first bar after the halt.

## Features and labels

The 111 features are a port of the TypeScript build, 25 per asset and 11 cross-asset, with its population variance and its `1e-12` guard.
Before the TypeScript code was removed, both versions ran on HYPE for October to December 2025, and the largest absolute difference was 4.9e-12.
The target coin is always called `x`, so the same code serves all five targets and the input names are identical for every coin.

`NOT_INPUTS` in `data/features.py` lists the nine per-asset columns that stay in the feature files but never reach a model, which leaves 75 inputs.
The method section of the README says why each was dropped and defines the label.

## Code

`main.py` is the command line, and the layout is in the README.
`folds.py` holds the purge rule, that a row is used only if its label is known when the step's data ends.
`predict_out_of_sample` runs the monthly loop for any fitter of the form `fit_<name>(train, valid, inputs, target, horizon)`, which returns the model, its validation predictions and a short string with the chosen setting.
Each experiment writes its tables to `results/<experiment>/` and then draws its figures from those tables, so `plot <experiment>` redraws them without recomputing anything.
Phase 2 reuses the folds and the label rule.
`experiments/volatility.py` runs the same loop on a realised-volatility target with the HAR model added, and `experiments/pooled.py` stacks the five coins' rows inside each fold.

pandas holds the time-indexed tables and computes the rolling moments, and numpy does the arithmetic of signals, positions, least squares and the bootstrap.

## Tests

Three tests check the method rather than the code, the lookahead test in `test_features.py` and the null and planted tests in `test_metrics.py`.
The lookahead test overwrites every bar after a cut and requires every earlier feature to stay the same.
The null test runs the IC on simulated random walks and requires that nothing is found, and it also shows a per-day Spearman correlation finding mean reversion that is not there.
The planted test adds a known signal and requires that it is found.

## Caching and reproducibility

Seeds are fixed and `uv.lock` pins every version.
Predictions are cached in `data/predictions/<tag>/`, where the tag is a hash of `folds.py`, `models.py`, `data/features.py` and the model settings, and the phase 2 caches also hash `volatility.py` or `pooled.py`.
A change anywhere else, for example to `experiments/common.py`, to the periods in `config.py` or to the built data, leaves the tag alone, so the cache has to be deleted by hand after such a change.
A full run from the raw archive takes about 40 minutes on a 20-core machine, most of it in the horizon sweep and the volatility models.
CI runs ruff and pytest on every push.

## What I decided against

I did not build an event-driven or C++ backtester, because the signals are bar-level with fixed holding periods and a vectorised position series gives the same PnL in a few dozen lines.
Shuffled cross-validation was out, since labels overlap and the market drifts, so shuffled folds put the future into training.
Purged k-fold would have been fine statistically, but all its folds except the last train on months after the test month, and the question here is how a model deployed month by month would have done.
An LSTM or another sequence model would have been the hardest part of the project to explain, and about 130,000 noisy rows per coin suit a tree model better.
Five separate studies, each tuned on its own coin, would have multiplied the choices and the false positives without asking a new question.

## Not done

Live trading, order book and trade data, other venues and deep learning are out of scope.
Ideas for later are a cross-sectional long and short strategy across the five coins, the funding rate as a feature, hour-of-day inputs for the direction models, Binance's 5-minute open interest metrics, and a longer history for the coins that have it.
