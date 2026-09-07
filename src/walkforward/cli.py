import argparse
import logging

from walkforward import config
from walkforward.data import bars, download


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="fetch monthly klines and funding")
    fetch.add_argument("--from", dest="start", default=config.DOWNLOAD_MONTHS[0])
    fetch.add_argument("--to", dest="end", default=config.DOWNLOAD_MONTHS[1])
    fetch.add_argument("--force", action="store_true")
    commands.add_parser("build", help="align bars on one grid")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S"
    )
    if args.command == "download":
        download.download_all(args.start, args.end, args.force)
    elif args.command == "build":
        bars.build()
