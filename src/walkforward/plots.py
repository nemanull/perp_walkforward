import json
import logging
from collections.abc import Callable
from pathlib import Path
import math

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
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, StrMethodFormatter

from walkforward.config import PAIRS, PER_COIN_T, POOLED_T, RESULTS_DIR
from walkforward.data.bars import BAR

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


def read_table(path: Path) -> pd.DataFrame | dict:
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


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


def small_multiples(
    count: int, columns: int, size: tuple[float, float], title: str, **share: bool
) -> tuple[Figure, list[Axes]]:
    figure = new_figure(size, title)
    axes = figure.subplots(math.ceil(count / columns), columns, squeeze=False, **share).ravel()
    for unused in axes[count:]:
        unused.remove()
    return figure, list(axes[:count])


def legend_beside(ax: Axes, handles: list | None = None, **options) -> Legend:
    return ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, **options
    )


def figure_legend(figure: Figure, handles: list) -> None:
    figure.legend(handles=handles, loc="outside right upper")


def legend_line(label: str, colour: str, linestyle: str = "-", marker: str = "") -> Line2D:
    return Line2D([], [], color=colour, linestyle=linestyle, marker=marker, label=label)


def outline(corner: tuple[float, float], side: float, label: str = "") -> Rectangle:
    return Rectangle(corner, side, side, fill=False, edgecolor=TEXT, linewidth=1.5, label=label)


def display_name(name: str) -> str:
    return name.upper() if name in PAIRS else name.replace("_", " ")


def horizon_label(horizon: int) -> str:
    minutes = horizon * BAR // pd.Timedelta(minutes=1)
    return f"{minutes} min" if minutes < 60 else f"{minutes / 60:g} h"


def recipe_label(selection: dict) -> str:
    return f"{display_name(selection['family'])}, {horizon_label(selection['horizon'])}"


def by_day(table: pd.DataFrame) -> pd.DataFrame:
    return table.set_index(pd.to_datetime(table["day"])).drop(columns="day")


def coin_table(frame: pd.DataFrame, columns: str, values: str) -> pd.DataFrame:
    table = frame.pivot(index="coin", columns=columns, values=values)
    return table.loc[frame["coin"].unique(), frame[columns].unique()]


def frozen_recipe_rows(frame: pd.DataFrame, selection: dict) -> pd.DataFrame:
    chosen = (frame["horizon"] == selection["horizon"]) & (frame["family"] == selection["family"])
    return frame[chosen]


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


def ic_text(ic: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    return ic.map("{:.3f}".format) + "\nt " + t.map("{:.1f}".format)


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


def plot_ic_heatmap(configurations: pd.DataFrame) -> Figure:
    families = configurations["family"].unique()
    limit = configurations["ic"].abs().max()
    title = "Rank IC by coin, horizon and model family, research months"
    figure, axes = small_multiples(len(families), len(families), (11, 3.8), title, sharey=True)
    for ax, family in zip(axes, families, strict=True):
        rows = configurations[configurations["family"] == family]
        ic, t = coin_table(rows, "horizon", "ic"), coin_table(rows, "horizon", "t")
        named = ic.rename(index=display_name, columns=horizon_label)
        image = plot_heatmap(ax, named, ic_text(ic, t), DIVERGING, -limit, limit)
        detected = coin_table(rows, "horizon", "detected").to_numpy(dtype=bool)
        for row, column in zip(*np.nonzero(detected), strict=True):
            ax.add_patch(outline((column - 0.45, row - 0.45), 0.9))
        ax.set_title(display_name(family))
    colour_scale(figure, image, axes, "rank IC")
    figure.supxlabel("forecast horizon")
    figure_legend(figure, [outline((0, 0), 1, f"detected,\nt ≥ {PER_COIN_T:g}")])
    return figure


def plot_pooled_ic(pooled: pd.DataFrame) -> Figure:
    families = pooled["family"].unique()
    title = "Pooled rank IC by horizon and model family, research months"
    figure, axes = small_multiples(len(families), len(families), (10, 3.6), title, sharey=True)
    for ax, family in zip(axes, families, strict=True):
        rows = pooled[pooled["family"] == family].reset_index(drop=True)
        for detected, colour in ((True, TEXT), (False, GREY)):
            chosen = rows[rows["detected"] == detected]
            ax.errorbar(chosen.index, chosen["ic"], yerr=1.96 * chosen["se"], fmt="o", color=colour)
        zero_line(ax)
        ax.set_xticks(rows.index, rows["horizon"].map(horizon_label))
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_title(display_name(family))
    axes[0].set_ylabel("pooled rank IC, 1.96 se bars")
    figure.supxlabel("forecast horizon")
    detected = legend_line(f"detected,\npooled t ≥ {POOLED_T:g}", TEXT, "none", "o")
    figure_legend(figure, [detected, legend_line("not detected", GREY, "none", "o")])
    return figure


def plot_ic_by_month(table: pd.DataFrame, selection: dict) -> Figure:
    by_month = coin_table(frozen_recipe_rows(table, selection), "month", "ic").T
    by_month.index = pd.to_datetime(by_month.index)
    figure, ax = single_plot(f"Rank IC by month, frozen recipe ({recipe_label(selection)})")
    coin_lines(ax, by_month, marker="o")
    zero_line(ax)
    date_axis(ax)
    ax.set_ylabel("rank IC")
    legend_beside(ax)
    return figure


def plot_deciles(table: pd.DataFrame, selection: dict) -> Figure:
    recipe = frozen_recipe_rows(table, selection)
    means = recipe.groupby(["coin", "decile"], sort=False)["mean_return"].mean() * 1e4
    coins = recipe["coin"].unique()
    title = f"Mean forward return by prediction decile, frozen recipe ({recipe_label(selection)})"
    figure, axes = small_multiples(len(coins), len(coins), (11, 3.4), title, sharey=True)
    for ax, coin in zip(axes, coins, strict=True):
        ax.bar(means[coin].index, means[coin], width=0.6, color=COIN_COLOURS[coin])
        zero_line(ax)
        ax.set_xticks([1, 5, 10])
        ax.set_title(display_name(coin))
    axes[0].set_ylabel("mean forward return (bps)")
    figure.supxlabel("prediction decile, mean over months")
    return figure


def plot_model_agreement(table: pd.DataFrame) -> Figure:
    values = coin_table(table, "horizon", "ridge_lightgbm_spearman")
    figure, ax = single_plot("Agreement between ridge and LightGBM predictions")
    named = values.rename(index=display_name, columns=horizon_label)
    image = plot_heatmap(ax, named, values.map("{:.2f}".format), DIVERGING, -1.0, 1.0)
    ax.set_xlabel("forecast horizon")
    colour_scale(figure, image, ax, "Spearman correlation of predictions")
    return figure


def plot_detectability(table: pd.DataFrame) -> Figure:
    positions = np.arange(len(table))
    needed = table.filter(like="needed_ic_").set_index(positions)
    needed.columns = needed.columns.str.removeprefix("needed_ic_")
    figure, ax = single_plot("Detectable rank IC and taker breakeven IC by horizon")
    breakeven = coin_lines(ax, needed, marker="o")
    detectable = f"detectable on\none coin at t = {POOLED_T:g}"
    references = ax.plot(positions, table["detectable_ic"], "--", color=TEXT, label=detectable)
    best = ax.plot(positions, table["best_pooled_ic"], "D", color=TEXT, label="best pooled IC")
    zero_line(ax)
    ax.set_xticks(positions, table["horizon"].map(horizon_label))
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("rank IC")
    ax.add_artist(legend_beside(ax, breakeven, title="taker breakeven IC", alignment="left"))
    ax.legend(
        handles=references + best, loc="lower left", bbox_to_anchor=(1.02, 0), borderaxespad=0
    )
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


def render_horizon_sweep(folder: Path) -> None:
    draw(folder, "ic_heatmap.png", plot_ic_heatmap, "configurations.csv")
    draw(folder, "pooled_ic.png", plot_pooled_ic, "pooled.csv")
    draw(folder, "ic_by_month.png", plot_ic_by_month, "ic_by_month.csv", "selection.json")
    draw(folder, "deciles.png", plot_deciles, "deciles.csv", "selection.json")
    draw(folder, "model_agreement.png", plot_model_agreement, "model_agreement.csv")
    draw(
        folder,
        "prediction_correlation.png",
        plot_correlation,
        "prediction_correlation.csv",
        title="Correlation of the coins' predictions, frozen recipe",
        label="Spearman correlation of predictions",
    )
    draw(
        folder,
        "daily_ic_correlation.png",
        plot_correlation,
        "daily_ic_correlation.csv",
        title="Correlation of the coins' daily rank IC, frozen recipe",
        label="correlation of daily rank IC",
    )
    draw(folder, "detectability.png", plot_detectability, "detectability.csv")


RENDERERS = {
    "audit": render_audit,
    "horizon-sweep": render_horizon_sweep,
}
