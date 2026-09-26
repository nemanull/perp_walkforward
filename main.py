import argparse
import logging

from src import config, plots
from src.data import bars, download, features
from src.experiments import phase1, pooled, volatility

EXPERIMENTS = phase1.EXPERIMENTS | volatility.EXPERIMENTS | pooled.EXPERIMENTS


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="fetch monthly klines and funding")
    fetch.add_argument("--from", dest="start", default=config.DOWNLOAD_MONTHS[0])
    fetch.add_argument("--to", dest="end", default=config.DOWNLOAD_MONTHS[1])
    fetch.add_argument("--force", action="store_true")
    commands.add_parser("build", help="align bars and compute features")
    run = commands.add_parser("run", help="run one experiment and draw its figures")
    run.add_argument("experiment", choices=list(EXPERIMENTS))
    plot = commands.add_parser("plot", help="redraw the figures of one experiment")
    plot.add_argument("experiment", choices=list(EXPERIMENTS))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S"
    )
    if args.command == "download":
        download.download_all(args.start, args.end, args.force)
    elif args.command == "build":
        bars.build()
        features.build()
    elif args.command == "run":
        EXPERIMENTS[args.experiment]()
        plots.render(args.experiment)
    else:
        plots.render(args.experiment)


if __name__ == "__main__":
    main()
