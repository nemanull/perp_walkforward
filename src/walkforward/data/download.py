import hashlib
import io
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from walkforward.config import DATA_DIR, FIRST_BAR, PAIRS, TARGETS

ARCHIVE = "https://data.binance.vision/data/futures/um/monthly"

log = logging.getLogger(__name__)


def month_range(start: str, end: str) -> list[str]:
    first = int(start[:4]) * 12 + int(start[5:7]) - 1
    last = int(end[:4]) * 12 + int(end[5:7]) - 1
    if first > last:
        raise ValueError(f"{start} is after {end}")
    return [f"{month // 12}-{month % 12 + 1:02d}" for month in range(first, last + 1)]


def months_for(asset: str, start: str, end: str) -> list[str]:
    first_month = FIRST_BAR.get(asset, start)[:7]
    return [month for month in month_range(start, end) if month >= first_month]


def kline_url(pair: str, month: str) -> str:
    return f"{ARCHIVE}/klines/{pair}/5m/{pair}-5m-{month}.zip"


def funding_url(pair: str, month: str) -> str:
    return f"{ARCHIVE}/fundingRate/{pair}/{pair}-fundingRate-{month}.zip"


def kline_path(pair: str, month: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "raw" / "klines" / pair / f"{pair}-5m-{month}.csv"


def funding_path(pair: str, month: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "raw" / "funding" / pair / f"{pair}-fundingRate-{month}.csv"


def expected_sha256(checksum_text: str) -> str:
    return checksum_text.split()[0]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def download_month(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        return
    try:
        archive = fetch(url)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            log.warning("not published %s", url)
            return
        raise
    # every zip has a .CHECKSUM file next to it with its sha256
    if hashlib.sha256(archive).hexdigest() != expected_sha256(fetch(url + ".CHECKSUM").decode()):
        raise ValueError(f"checksum mismatch for {url}")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        [name] = zipped.namelist()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(zipped.read(name))
    log.debug("saved %s", path.name)


def download_all(start: str, end: str, force: bool = False) -> None:
    for asset, pair in PAIRS.items():
        months = months_for(asset, start, end)
        for month in months:
            download_month(kline_url(pair, month), kline_path(pair, month), force)
            if asset in TARGETS:
                download_month(funding_url(pair, month), funding_path(pair, month), force)
        log.info("%s %s to %s", pair, months[0], months[-1])
