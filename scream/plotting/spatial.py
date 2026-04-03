"""Spatial plots: Φ1 vs Φ2 with TP/FP/FN classification overlay."""
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


def plot_phi1_phi2_preds(
    phi1: np.ndarray,
    phi2: np.ndarray,
    true_mask: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    output_path: Path,
) -> None:
    """Φ1 vs Φ2 with hist2d background density and TP/FP/FN scatter.

    Parameters
    ----------
    phi1, phi2  : coordinate arrays for all real test stars (N,)
    true_mask   : boolean mask of true stream members (N,)
    tp, fp, fn  : boolean masks for true/false positives and false negatives (N,)
    output_path : Path (with or without .png extension — .png is added if absent)
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.hist2d(
            phi1[~true_mask],
            phi2[~true_mask],
            bins=100,
            cmap="gray_r",
            alpha=0.5,
        )
        ax.scatter(phi1[tp], phi2[tp], s=40, alpha=0.6, label="True Positives",
                   color=_COLORS[0], marker=_MARKERS[0])
        ax.scatter(phi1[fp], phi2[fp], s=40, alpha=0.6, label="False Positives",
                   color=_COLORS[1], marker=_MARKERS[1])
        ax.scatter(phi1[fn], phi2[fn], s=40, alpha=0.6, label="False Negatives",
                   color=_COLORS[2], marker=_MARKERS[2])

        ax.tick_params(axis="both", labelsize=16)
        ax.set_xlabel(r"$\Phi$1", fontsize=35)
        ax.set_ylabel(r"$\Phi$2", fontsize=35)
        ax.legend(fontsize=26)

        fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
