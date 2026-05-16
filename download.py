import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://data.binance.vision/data/futures/um/monthly/klines/{pair}/5m/{pair}-5m-{month}.zip"
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT"]
MONTHS = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]


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
