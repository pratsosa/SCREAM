"""Kinematic track plots: Φ1 vs proper-motion μ₁ and μ₂."""
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

_HIST_KWARGS = dict(bins=100, cmap="gray_r", alpha=0.4, rasterized=True)
_SCATTER_SF = dict(s=40, c="tab:red", edgecolors="k", linewidths=0.4, alpha=0.9, zorder=3)
_SCATTER_MD = dict(s=40, cmap="viridis", edgecolors="k", linewidths=0.4, alpha=0.9, zorder=3)


def _single_track(
    phi1: np.ndarray,
    feature_data: np.ndarray,
    feature_label: str,
    true_mask: np.ndarray,
    preds_bool: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
) -> None:
    """Two-panel track plot (SF labels top, model predictions bottom) for one PM component."""
    with matplotlib.rc_context(_RC):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(18, 10), sharex=True, constrained_layout=True
        )

        ax1.hist2d(phi1, feature_data, **_HIST_KWARGS)
        ax2.hist2d(phi1, feature_data, **_HIST_KWARGS)

        ax1.scatter(phi1[true_mask], feature_data[true_mask], label="Stream", **_SCATTER_SF)

        sc = ax2.scatter(
            phi1[preds_bool],
            feature_data[preds_bool],
            c=probs[preds_bool],
            **_SCATTER_MD,
        )

        for ax in (ax1, ax2):
            ax.tick_params(axis="both")
            ax.set_ylabel(feature_label, fontsize=30)

        ax2.set_xlabel(r"$\Phi_1$", fontsize=30)
        ax1.set_title("SF Labels", fontsize=40, pad=12)
        ax2.set_title("Model Predictions", fontsize=40, pad=12)

        cbar = fig.colorbar(sc, ax=ax2, orientation="horizontal", pad=0.12, fraction=0.9)
        cbar.set_label("Model Probability", fontsize=24)
        cbar.ax.tick_params(labelsize=16)

        ax1.set_ylim(
            feature_data[true_mask].min() - 0.1,
            feature_data[true_mask].max() + 0.1,
        )
        ax2.set_ylim(
            feature_data[preds_bool].min() - 0.1,
            feature_data[preds_bool].max() + 0.1,
        )

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_phi1_pm_tracks(
    phi1: np.ndarray,
    pm_mu1: np.ndarray,
    pm_mu2: np.ndarray,
    true_mask: np.ndarray,
    preds_bool: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
) -> None:
    """Produce all three PM-track plots in one call.

    Writes to `output_dir`:
        phi1_muphi1_track.png       — two-panel (SF labels / model probs) for μ₁
        phi1_muphi2_track.png       — two-panel for μ₂
        phi1_mu_tracks_combined.png — 2×2 grid, both μ₁ and μ₂; shared vertical colorbar

    Parameters
    ----------
    phi1       : Φ₁ coordinates for all real test stars (N,)
    pm_mu1     : μ₁ proper-motion component (N,)
    pm_mu2     : μ₂ proper-motion component (N,)
    true_mask  : boolean mask of true stream members (N,)
    preds_bool : boolean mask of model-predicted members (N,)
    probs      : model probability scores (N,)
    output_dir : directory in which to write the three PNG files (created if absent)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _single_track(
        phi1=phi1,
        feature_data=pm_mu1,
        feature_label=r"$\mu_1$",
        true_mask=true_mask,
        preds_bool=preds_bool,
        probs=probs,
        output_path=output_dir / "phi1_muphi1_track.png",
    )

    _single_track(
        phi1=phi1,
        feature_data=pm_mu2,
        feature_label=r"$\mu_2$",
        true_mask=true_mask,
        preds_bool=preds_bool,
        probs=probs,
        output_path=output_dir / "phi1_muphi2_track.png",
    )

    # --- Combined 2×2 grid ---
    with matplotlib.rc_context(_RC):
        fig, axes = plt.subplots(
            2, 2, figsize=(22, 10), sharex=True, constrained_layout=True
        )
        ax_sf_mu1, ax_sf_mu2 = axes[0]
        ax_md_mu1, ax_md_mu2 = axes[1]

        for ax in axes.flat:
            ax.tick_params(axis="both", labelsize=18)

        # μ₁ column
        ax_sf_mu1.hist2d(phi1, pm_mu1, **_HIST_KWARGS)
        ax_md_mu1.hist2d(phi1, pm_mu1, **_HIST_KWARGS)
        ax_sf_mu1.scatter(phi1[true_mask], pm_mu1[true_mask], **_SCATTER_SF)
        sc = ax_md_mu1.scatter(
            phi1[preds_bool], pm_mu1[preds_bool], c=probs[preds_bool], **_SCATTER_MD
        )

        # μ₂ column
        ax_sf_mu2.hist2d(phi1, pm_mu2, **_HIST_KWARGS)
        ax_md_mu2.hist2d(phi1, pm_mu2, **_HIST_KWARGS)
        ax_sf_mu2.scatter(phi1[true_mask], pm_mu2[true_mask], **_SCATTER_SF)
        ax_md_mu2.scatter(
            phi1[preds_bool], pm_mu2[preds_bool], c=probs[preds_bool], **_SCATTER_MD
        )

        ax_sf_mu1.set_ylabel(r"$\mu_1$", fontsize=30)
        ax_md_mu1.set_ylabel(r"$\mu_1$", fontsize=30)
        ax_sf_mu2.set_ylabel(r"$\mu_2$", fontsize=30)
        ax_md_mu2.set_ylabel(r"$\mu_2$", fontsize=30)
        ax_md_mu1.set_xlabel(r"$\Phi_1$", fontsize=26)
        ax_md_mu2.set_xlabel(r"$\Phi_1$", fontsize=26)

        ax_sf_mu1.set_title("SF Labels", fontsize=40, pad=10)
        ax_sf_mu2.set_title("SF Labels", fontsize=40, pad=10)
        ax_md_mu1.set_title("Model Predictions", fontsize=40, pad=10)
        ax_md_mu2.set_title("Model Predictions", fontsize=40, pad=10)

        cbar = fig.colorbar(sc, ax=axes, orientation="vertical", pad=0.02, fraction=0.035)
        cbar.set_label("Model Probability", fontsize=26)
        cbar.ax.tick_params(labelsize=18)

        ax_sf_mu1.set_ylim(pm_mu1[true_mask].min() - 0.1, pm_mu1[true_mask].max() + 0.1)
        ax_md_mu1.set_ylim(*ax_sf_mu1.get_ylim())
        ax_sf_mu2.set_ylim(pm_mu2[true_mask].min() - 0.1, pm_mu2[true_mask].max() + 0.1)
        ax_md_mu2.set_ylim(*ax_sf_mu2.get_ylim())

        fig.savefig(output_dir / "phi1_mu_tracks_combined.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
