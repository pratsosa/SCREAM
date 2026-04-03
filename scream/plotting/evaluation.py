"""Evaluation plots: confusion matrix (raw counts and pred-normalised)."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

_CLASS_LABELS = ["Background", "Stream"]


def plot_confusion_matrix(
    true_labels: np.ndarray,
    preds: np.ndarray,
    output_path: Path,
) -> None:
    """Write two confusion-matrix PNGs.

    Parameters
    ----------
    true_labels : array of int/bool, shape (N,)
    preds       : array of int/bool, shape (N,)
    output_path : Path without extension.
        Writes:
            {output_path}_norm.png  — pred-normalised (fmt=".2%")
            {output_path}_raw.png   — raw counts      (fmt="d")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_raw = confusion_matrix(true_labels, preds, normalize=None)
    cm_norm = confusion_matrix(true_labels, preds, normalize="pred")

    with matplotlib.rc_context(_RC):
        # --- pred-normalised ---
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            cbar=True,
            annot_kws={"fontsize": 30},
            cbar_kws={"label": "Fraction of predicted class"},
            square=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=_CLASS_LABELS,
            yticklabels=_CLASS_LABELS,
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.tick_params(axis="both", labelsize=22)
        ax.set_xlabel("Predicted label", fontsize=26)
        ax.set_ylabel("SF label", fontsize=26)
        fig.savefig(str(output_path) + "_norm.png", dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # --- raw counts ---
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_raw,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            annot_kws={"fontsize": 30},
            square=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=_CLASS_LABELS,
            yticklabels=_CLASS_LABELS,
            ax=ax,
        )
        ax.tick_params(axis="both", labelsize=22)
        ax.set_xlabel("Predicted label", fontsize=26)
        ax.set_ylabel("SF label", fontsize=26)
        fig.savefig(str(output_path) + "_raw.png", dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
