import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://data.binance.vision/data/futures/um/monthly/klines/{pair}/5m/{pair}-5m-{month}.zip"
# no PUMPUSDT: the ticker was reused, the new token starts 2025-07-10 about 9x lower (a fake -89% bar)
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "DOGEUSDT"]
MONTHS = [str(m) for m in pd.period_range("2025-07", "2026-04", freq="M")]


def download(pair: str, month: str) -> None:
    out = Path("data") / pair / f"{pair}-5m-{month}.csv"
    if out.exists():
        return
    url = URL.format(pair=pair, month=month)
    print(url)
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(z.read(z.namelist()[0]))


if __name__ == "__main__":
    for pair in PAIRS:
        for month in MONTHS:
            download(pair, month)
