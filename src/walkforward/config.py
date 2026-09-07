from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

TARGETS = {
    "hype": "HYPEUSDT",
    "trx": "TRXUSDT",
    "doge": "DOGEUSDT",
    "uni": "UNIUSDT",
    "aave": "AAVEUSDT",
}
CONTEXT = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT"}
PAIRS = TARGETS | CONTEXT
FIRST_BAR = {"hype": "2025-05-30 10:30"}  # a reused ticker gets an entry here too

GRID_START = FIRST_BAR["hype"]
DOWNLOAD_MONTHS = ("2025-05", "2026-08")

SEED = 7
