"""Background sample generation for SCREAM CATHODE.

Uses a trained pzflow Flow to generate synthetic background samples in the
signal region, conditioned on pm_ra values drawn from a KDE fit to the
observed signal-region pm_ra distribution.

pzflow and sklearn are the only extra dependencies here (beyond numpy/pandas).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from scream.config.schema import StreamConfig


def generate_samples(
    flow,
    scaler: StandardScaler,
    signal_mask: np.ndarray,
    full_embeddings: np.ndarray,
    source_id: np.ndarray,
    stream: np.ndarray,
    col_names: list[str],
    cfg: StreamConfig,
    n_multiplier: int = 4,
    kde_bandwidth: float = 0.001,
    seed: int = 12345,
) -> pd.DataFrame:
    """Generate background samples and combine with real signal-region data.

    Parameters
    ----------
    flow : pzflow.Flow
        Trained normalizing flow (output of ``trainer.train_flow``).
    scaler : StandardScaler
        Scaler fit on the full feature matrix before training.
    signal_mask : np.ndarray of bool
        Boolean mask identifying signal-region rows in ``full_embeddings``.
    full_embeddings : np.ndarray
        Raw (unscaled) full feature matrix after percentile cut, shape (N, D).
    source_id : np.ndarray
        Source IDs aligned with rows of ``full_embeddings``.
    stream : np.ndarray
        Stream membership flags aligned with rows of ``full_embeddings``.
    col_names : list[str]
        Column names corresponding to columns of ``full_embeddings``.
        First entry must be the conditioning column (pm_ra).
    cfg : StreamConfig
        Stream configuration — used to get ``flow_cond_columns`` and the
        output path.
    n_multiplier : int
        Number of generated samples per signal-region star (default 4, as
        recommended in the CATHODE paper).
    kde_bandwidth : float
        Bandwidth for the KDE fit to the signal-region pm_ra distribution.
    seed : int
        Random seed for KDE sampling.

    Returns
    -------
    pd.DataFrame
        Combined dataframe of real signal-region stars and generated background
        samples, with columns:
        [*col_names, stream, CWoLa_Label, source_id]
        Ready to be written to ``cfg.generated_data_path``.
    """
    cond_col = cfg.flow_cond_columns[0]  # "pm_ra"
    cond_idx = col_names.index(cond_col)

    # Fit KDE to signal-region pm_ra (unscaled)
    pm_ra_signal = full_embeddings[signal_mask, cond_idx].reshape(-1, 1)
    kde = KernelDensity(bandwidth=kde_bandwidth)
    kde.fit(pm_ra_signal)
    n_samples = n_multiplier * int(np.sum(signal_mask))
    rng = np.random.default_rng(seed)
    kde_samples = kde.sample(n_samples=n_samples,
                             random_state=rng.integers(2**31))

    # Scale the KDE samples the same way the flow was trained
    # (flow was trained on scaled data; conditioning values must also be scaled)
    cond_scale = scaler.scale_[cond_idx]
    cond_mean = scaler.mean_[cond_idx]
    kde_samples_scaled = (kde_samples - cond_mean) / cond_scale

    samples_df = flow.sample(
        nsamples=1,
        conditions=pd.DataFrame(
            data=kde_samples_scaled.astype("float64"),
            columns=cfg.flow_cond_columns,
        ),
        save_conditions=True,
    )

    # Invert scaling: reconstruct the full 11-column array for inverse_transform
    # samples_df has columns flow_cond_columns + flow_data_columns = col_names
    samples_df = samples_df[col_names]
    samples_arr = scaler.inverse_transform(samples_df.values)
    samples_df = pd.DataFrame(data=samples_arr, columns=col_names)

    # Build signal-region real-data dataframe
    signal_arr = full_embeddings[signal_mask]
    signal_df = pd.DataFrame(data=signal_arr, columns=col_names)
    signal_df["stream"] = stream[signal_mask].astype(int)
    signal_df["CWoLa_Label"] = 1
    signal_df["source_id"] = source_id[signal_mask]

    # Generated background: stream=2 (convention from original script), label=0
    samples_df["stream"] = 2
    samples_df["CWoLa_Label"] = 0
    samples_df["source_id"] = -1

    return pd.concat([signal_df, samples_df], ignore_index=True)


def save_samples(df: pd.DataFrame, cfg: StreamConfig) -> Path:
    """Write the combined dataframe to ``cfg.generated_data_path``.

    Creates parent directories as needed.

    Returns
    -------
    Path
        The path the CSV was written to.
    """
    out_path = Path(cfg.generated_data_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
