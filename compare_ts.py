from pathlib import Path

import pandas as pd

from features import build

# Outputs/ comes from the ts build: pnpm run clean, then pnpm run build -- --HYPE --5 --4 (and 12, 36, 72)
py = build("HYPE")
worst = 0.0
for path in sorted(Path("Outputs").glob("HYPE_5m_H*.csv")):
    ts = pd.read_csv(path)
    both = ts.merge(py, left_on="ts_close_ms", right_on="close_time", suffixes=("_ts", "_py"))
    cols = [c for c in ts.columns if c in py.columns and c not in ("open_time", "close_time")]
    diff = pd.Series({c: (both[c + "_ts"] - both[c + "_py"]).abs().max() for c in cols})
    print(path.name, len(ts), "rows,", len(both), "matched,", len(cols), "columns")
    print(diff.sort_values().tail(5).to_string())
    worst = max(worst, diff.max())

print("max abs diff", worst)
