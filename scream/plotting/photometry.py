"""Photometric CMD plots: Gaia G vs BP-RP and DECaLS r₀ vs g₀-r₀ / r₀-z₀."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_RC = {
    "font.size": 14,
    "axes.titlesize": 26,
    "axes.labelsize": 24,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.linewidth": 1.0,
}

_COLORS = ["#15097C", "#bca006", "#c50b27"]
_MARKERS = ["o", "s", "D"]


def _cmd_plot(
    x: np.ndarray,
    y: np.ndarray,
    true_mask: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    xlabel: str,
    ylabel: str,
    xlim: tuple,
    output_path: Path,
) -> None:
    """Internal helper: single-panel CMD with hist2d background and TP/FP/FN scatter."""
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.hist2d(x[~true_mask], y[~true_mask], bins=200, cmap="gray_r", alpha=0.5)
        ax.scatter(x[tp], y[tp], s=40, alpha=0.6, label="True Positive",
                   color=_COLORS[0], marker=_MARKERS[0])
        ax.scatter(x[fp], y[fp], s=40, alpha=0.6, label="False Positive",
                   color=_COLORS[1], marker=_MARKERS[1])
        ax.scatter(x[fn], y[fn], s=40, alpha=0.6, label="False Negative",
                   color=_COLORS[2], marker=_MARKERS[2])

        ax.set_xlabel(xlabel, fontsize=35)
        ax.set_ylabel(ylabel, fontsize=35)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.legend(fontsize=26)

        fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def plot_cmd_gaia(
    color: np.ndarray,
    gmag: np.ndarray,
    true_mask: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    output_path: Path,
) -> None:
    """G vs BP-RP colour-magnitude diagram.

    Parameters
    ----------
    color       : BP-RP values for all real test stars (N,)
    gmag        : G-band magnitudes (N,)
    true_mask   : boolean mask of true stream members (N,)
    tp, fp, fn  : boolean masks for classification outcomes (N,)
    output_path : destination PNG path
    """
    _cmd_plot(
        x=color,
        y=gmag,
        true_mask=true_mask,
        tp=tp,
        fp=fp,
        fn=fn,
        xlabel="BP - RP",
        ylabel="G",
        xlim=(0.2, 1.0),
        output_path=output_path,
    )


def plot_cmd_decals_gr(
    g_r: np.ndarray,
    rmag0: np.ndarray,
    true_mask: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    output_path: Path,
) -> None:
    """r₀ vs g₀-r₀ colour-magnitude diagram.

    Parameters
    ----------
    g_r         : g₀-r₀ colour values (N,)
    rmag0       : r₀ magnitudes (N,)
    true_mask   : boolean mask of true stream members (N,)
    tp, fp, fn  : boolean masks for classification outcomes (N,)
    output_path : destination PNG path
    """
    _cmd_plot(
        x=g_r,
        y=rmag0,
        true_mask=true_mask,
        tp=tp,
        fp=fp,
        fn=fn,
        xlabel=r"g$_0$-r$_0$",
        ylabel=r"r$_0$",
        xlim=(0.2, 0.6),
        output_path=output_path,
    )


def plot_cmd_decals_rz(
    r_z: np.ndarray,
    rmag0: np.ndarray,
    true_mask: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    output_path: Path,
) -> None:
    """r₀ vs r₀-z₀ colour-magnitude diagram.

    Parameters
    ----------
    r_z         : r₀-z₀ colour values (N,)
    rmag0       : r₀ magnitudes (N,)
    true_mask   : boolean mask of true stream members (N,)
    tp, fp, fn  : boolean masks for classification outcomes (N,)
    output_path : destination PNG path
    """
    _cmd_plot(
        x=r_z,
        y=rmag0,
        true_mask=true_mask,
        tp=tp,
        fp=fp,
        fn=fn,
        xlabel=r"r$_0$-z$_0$",
        ylabel=r"r$_0$",
        xlim=(-0.1, 0.3),
        output_path=output_path,
    )
