import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, StrMethodFormatter

from walkforward.config import PAIRS, RESULTS_DIR

log = logging.getLogger(__name__)

COIN_COLOURS = {
    "hype": "#008300",
    "trx": "#2a78d6",
    "doge": "#eda100",
    "uni": "#e87ba4",
    "aave": "#4a3aa7",
}
TEXT = "#222222"
GREY = "#767676"
LIGHT_GREY = "#ababab"
GRID = "#e5e5e5"
NEUTRALS = (TEXT, GREY, LIGHT_GREY)
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#ec7f7e", "#f4f4f2", "#6fa6e6"])

STYLE = {
    "figure.dpi": 150,
    "figure.labelsize": 9,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    # "savefig.bbox": "tight",
}


def render(experiment: str, results_dir: Path = RESULTS_DIR) -> None:
    RENDERERS[experiment](results_dir / experiment)
    log.info("figures for %s done", experiment)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def draw(folder: Path, name: str, plot: Callable[..., Figure], *sources: str, **options) -> None:
    tables = [read_table(folder / source) for source in sources]
    with mpl.rc_context(STYLE):
        plot(*tables, **options).savefig(folder / name)


def new_figure(size: tuple[float, float], title: str) -> Figure:
    figure = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle(title, x=0.01, ha="left")
    return figure


def single_plot(title: str, size: tuple[float, float] = (7, 4)) -> tuple[Figure, Axes]:
    figure = new_figure(size, title)
    return figure, figure.subplots()


def legend_beside(ax: Axes) -> Legend:
    return ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)


def display_name(name: str) -> str:
    return name.upper() if name in PAIRS else name.replace("_", " ")


def by_day(table: pd.DataFrame) -> pd.DataFrame:
    return table.set_index(pd.to_datetime(table["day"])).drop(columns="day")


def zero_line(ax: Axes, vertical: bool = False) -> None:
    line = ax.axvline if vertical else ax.axhline
    line(0, color=LIGHT_GREY, linewidth=0.8, zorder=1)


def date_axis(ax: Axes) -> None:
    locator = AutoDateLocator(maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator, show_offset=False))


def coin_lines(ax: Axes, table: pd.DataFrame, **style) -> list[Line2D]:
    return [
        ax.plot(
            table.index, table[coin], color=COIN_COLOURS[coin], label=display_name(coin), **style
        )[0]
        for coin in table.columns
    ]


def plot_heatmap(
    ax: Axes, values: pd.DataFrame, text: pd.DataFrame, cmap: Colormap, vmin: float, vmax: float
) -> AxesImage:
    image = ax.imshow(values.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for (row, column), cell in np.ndenumerate(text.to_numpy()):
        ax.text(column, row, cell, ha="center", va="center", color=TEXT, fontsize=8)
    ax.set_xticks(range(values.shape[1]), values.columns)
    ax.set_yticks(range(values.shape[0]), values.index)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    return image


def colour_scale(figure: Figure, image: AxesImage, axes: Axes | list[Axes], label: str) -> None:
    scale = figure.colorbar(image, ax=axes, shrink=0.8, aspect=25)
    scale.outline.set_visible(False)
    scale.ax.tick_params(length=0)
    scale.set_label(label)


def plot_correlation(table: pd.DataFrame, title: str, label: str) -> Figure:
    matrix = table.set_index(table.columns[0])
    figure, ax = single_plot(title, (7, 6) if len(matrix) > 5 else (6, 4.6))
    named = matrix.rename(index=display_name, columns=display_name)
    image = plot_heatmap(ax, named, matrix.map("{:.2f}".format), DIVERGING, -1.0, 1.0)
    ax.set_aspect("equal")
    colour_scale(figure, image, ax, label)
    return figure


def plot_rolling_btc_correlation(table: pd.DataFrame) -> Figure:
    figure, ax = single_plot("30-day correlation of 5-minute returns with BTC")
    coin_lines(ax, by_day(table))
    date_axis(ax)
    ax.set_ylabel("correlation with BTC")
    legend_beside(ax)
    return figure


def plot_lead_lag(table: pd.DataFrame) -> Figure:
    lags = table.set_index("btc_lead_bars")
    lags.loc[0] = np.nan
    figure, ax = single_plot("Correlation of each coin's 5-minute return with BTC's at a lag")
    coin_lines(ax, lags, marker="o")
    zero_line(ax)
    ax.set_xticks(lags.index)
    ax.set_xlabel("BTC lead in 5-minute bars (negative: the coin leads, lag 0 left out)")
    ax.set_ylabel("correlation of 5-minute returns")
    legend_beside(ax)
    return figure


def plot_autocorrelation(table: pd.DataFrame) -> Figure:
    measures = {
        "lag1": "all bars",
        "lag1_without_crash": "without Oct 10 to 11",
        "lag1_rank_without_crash": "ranks, without Oct 10 to 11",
    }
    positions = np.arange(len(table))
    figure, ax = single_plot("Lag-1 autocorrelation of 5-minute returns, June to November 2025")
    for offset, (column, label), colour in zip((-0.24, 0, 0.24), measures.items(), NEUTRALS):
        ax.bar(positions + offset, table[column], width=0.2, color=colour, label=label)
    zero_line(ax)
    ax.set_xticks(positions, table["asset"].map(display_name))
    ax.set_ylabel("lag-1 autocorrelation")
    legend_beside(ax)
    return figure


def plot_daily_volatility(table: pd.DataFrame) -> Figure:
    figure, ax = single_plot("Daily volatility of each coin")
    coin_lines(ax, by_day(table) * 100)
    date_axis(ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.minorticks_off()
    ax.set_ylabel("daily volatility (%, log scale)")
    legend_beside(ax)
    return figure


def render_audit(folder: Path) -> None:
    draw(
        folder,
        "return_correlation.png",
        plot_correlation,
        "return_correlation.csv",
        title="Correlation of 5-minute returns, 2025-05-30 to 2026-05-31",
        label="correlation of 5-minute returns",
    )
    draw(
        folder,
        "rolling_btc_correlation.png",
        plot_rolling_btc_correlation,
        "rolling_btc_correlation.csv",
    )
    draw(folder, "lead_lag.png", plot_lead_lag, "lead_lag.csv")
    draw(folder, "autocorrelation.png", plot_autocorrelation, "autocorrelation.csv")
    draw(folder, "daily_volatility.png", plot_daily_volatility, "daily_volatility.csv")


RENDERERS = {
    "audit": render_audit,
}
