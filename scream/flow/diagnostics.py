"""Diagnostic plots for the normalizing flow training and sample generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_loss_curves(
    train_losses: list,
    test_losses: list,
    out_path: Path,
) -> None:
    """Save a plot of NF train and test loss over epochs.

    Parameters
    ----------
    train_losses : list
        Per-epoch training losses returned by ``pzflow.Flow.train``.
    test_losses : list
        Per-epoch test losses returned by ``pzflow.Flow.train``.
    out_path : Path
        File path to save the figure (e.g. ``.../plots/loss_curves.png``).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    # Skip epoch 0 — pzflow reports an untrained baseline loss there
    ax.plot(train_losses[1:], label="Train")
    ax.plot(test_losses[1:], label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Normalizing Flow — Training Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to: {out_path}")


def plot_feature_histograms(
    full_embeddings: np.ndarray,
    signal_mask: np.ndarray,
    col_names: list[str],
    generated_df: pd.DataFrame,
    flow_data_columns: list[str],
    out_dir: Path,
) -> None:
    """Save 1D histograms comparing real signal-region data vs generated samples.

    Also overlays the sideband (training) distribution for reference.

    Parameters
    ----------
    full_embeddings : np.ndarray
        Unscaled full feature matrix (post percentile cut), shape (N, D).
        Columns correspond to ``col_names``.
    signal_mask : np.ndarray of bool
        Boolean mask identifying the signal-region rows.
    col_names : list[str]
        Column names for ``full_embeddings`` (conditioning column first).
    generated_df : pd.DataFrame
        Combined signal + generated dataframe from ``sampler.generate_samples``.
        Generated rows are identified by ``CWoLa_Label == 0``.
    flow_data_columns : list[str]
        Subset of ``col_names`` to plot (excludes the conditioning column).
    out_dir : Path
        Directory to save one PNG per feature.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract data arrays for the flow data columns only
    data_col_indices = [col_names.index(c) for c in flow_data_columns]
    sig_data = full_embeddings[signal_mask][:, data_col_indices]
    sideband_data = full_embeddings[~signal_mask][:, data_col_indices]
    gen_data = generated_df[generated_df["CWoLa_Label"] == 0][flow_data_columns].values

    for i, col in enumerate(flow_data_columns):
        sig_col = sig_data[:, i]
        side_col = sideband_data[:, i]
        gen_col = gen_data[:, i]

        lo = np.percentile(sig_col, 1)
        hi = np.percentile(sig_col, 99)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(sig_col, density=True, bins=250, alpha=0.5,
                range=(lo, hi), label="Real Signal Region")
        ax.hist(side_col, density=True, bins=250, alpha=0.4,
                color="black", histtype="step",
                range=(lo, hi), label="Sideband (train)")
        ax.hist(gen_col, density=True, bins=250, alpha=0.5,
                range=(lo, hi), label="Generated")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.set_title(f"{col}: Real vs Generated")
        ax.legend()
        fig.tight_layout()

        out_path = out_dir / f"hist_{col}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"Feature histograms ({len(flow_data_columns)}) saved to: {out_dir}")
