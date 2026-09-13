from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

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

HORIZONS = (4, 12, 36, 72)
TRAIN_START = "2025-06-01 10:30"
RESEARCH_MONTHS = ("2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05")
FORWARD_MONTHS = ("2026-06", "2026-07", "2026-08")
RESEARCH_END = "2026-06-01"  # exclusive, every label scored in research is known by then
FORWARD_END = "2026-09-01"
CRASH = ("2025-10-10", "2025-10-12")  # left out of the tail and lag statistics

FEES_BPS = {"taker": 5.0, "maker": 2.0}

SEED = 7
RIDGE_ALPHAS = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 10.0,
    "n_estimators": 2000,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 4,
    "random_state": SEED,
    "verbose": -1,
}
ROWS_PER_LEAF_PER_HORIZON_BAR = 100
EARLY_STOPPING_ROUNDS = 100

POOLED_T = 3.0
PER_COIN_T = 3.2
PAIRED_T = 2.0
BOOTSTRAP_MEAN_BLOCK_DAYS = 5
BOOTSTRAP_DRAWS = 10_000
