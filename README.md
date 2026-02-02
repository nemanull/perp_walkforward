# Binance Data Combinator

Small CLI utility to clean Binance CSV datasets and build feature CSVs
for model training.

## Setup

1) Install Node.js (v20+ recommended) and pnpm.
2) Install dependencies:

```
pnpm install
```

## Clean raw datasets

This reads from `Datasets/` and writes cleaned CSVs to `Datasets_cleaned/`.

```
pnpm run clean
```

## Build feature dataset

This reads from `Datasets_cleaned/` and writes to `Outputs/`.

Examples:

```
pnpm run build -- --HYPE --5 --4
```

```
pnpm run build -- --symbol HYPE --tf 5 --horizon 20 --cost-bps 0
```

Notes:
- Horizon is in minutes unless it matches a schema bar horizon
  (4/12/36/72).
- Horizon minutes must be divisible by `tf_minutes`.

