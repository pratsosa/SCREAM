"""Normalizing flow training for SCREAM.

Trains a pzflow RollingSplineCoupling flow on the sideband region of the raw
FITS data. The trained Flow object is returned and optionally saved to disk.

pzflow and JAX are optional dependencies — only imported when this module is
actually used, so MLP-only workflows do not require them.
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scream.config.schema import StreamConfig


def _load_fits(path: str) -> dict:
    """Read the raw GD-1 FITS file and return a dict of named arrays."""
    df = Table.read(path)

    # Stream-frame astrometry
    phi1    = np.array(df["phi1"]).astype("float64")
    phi2    = np.array(df["phi2"]).astype("float64")
    pm_phi1 = np.array(df["pm_phi1"]).astype("float64")   # conditioning column
    pm_phi2 = np.array(df["pm_phi2"]).astype("float64")

    # Raw (non-extinction-corrected) photometric magnitudes
    G_mag  = np.array(df["phot_g_mean_mag"]).astype("float64")
    Bp_mag = np.array(df["phot_bp_mean_mag"]).astype("float64")
    Rp_mag = np.array(df["phot_rp_mean_mag"]).astype("float64")
    g_mag  = 22.5 - 2.5 * np.log10(np.array(df["ls_flux_g"]).astype("float64"))
    r_mag  = 22.5 - 2.5 * np.log10(np.array(df["ls_flux_r"]).astype("float64"))
    z_mag  = 22.5 - 2.5 * np.log10(np.array(df["ls_flux_z"]).astype("float64"))

    # Error columns — fed to NF as data columns
    phot_g_flux_err  = np.array(df["phot_g_mean_flux_error"]).astype("float64")
    phot_bp_flux_err = np.array(df["phot_bp_mean_flux_error"]).astype("float64")
    phot_rp_flux_err = np.array(df["phot_rp_mean_flux_error"]).astype("float64")
    flux_err_g       = np.array(df["flux_err_g"]).astype("float64")
    flux_err_r       = np.array(df["flux_err_r"]).astype("float64")
    flux_err_z       = np.array(df["flux_err_z"]).astype("float64")
    pmra_error       = np.array(df["pmra_error"]).astype("float64")
    pmdec_error      = np.array(df["pmdec_error"]).astype("float64")
    ra_error         = np.array(df["ra_error"]).astype("float64")   # degrees (converted in data_prep)
    dec_error        = np.array(df["dec_error"]).astype("float64")  # degrees (converted in data_prep)

    # EBV — NOT a NF column; passed through separately for sampler
    ebv = np.array(df["ls_ebv"]).astype("float64")

    stream    = np.array(df["stream"])
    source_id = np.array(df["source_id"])

    if "signal_region" in df.colnames:
        signal_region = np.array(df["signal_region"], dtype=bool)
    else:
        signal_region = None

    return dict(
        phi1=phi1, phi2=phi2, pm_phi1=pm_phi1, pm_phi2=pm_phi2,
        G_mag=G_mag, Bp_mag=Bp_mag, Rp_mag=Rp_mag,
        g_mag=g_mag, r_mag=r_mag, z_mag=z_mag,
        phot_g_flux_err=phot_g_flux_err,
        phot_bp_flux_err=phot_bp_flux_err,
        phot_rp_flux_err=phot_rp_flux_err,
        flux_err_g=flux_err_g, flux_err_r=flux_err_r, flux_err_z=flux_err_z,
        pmra_error=pmra_error, pmdec_error=pmdec_error,
        ra_error=ra_error, dec_error=dec_error,
        ebv=ebv,
        stream=stream, source_id=source_id,
        signal_region=signal_region,
    )


def _build_signal_mask(arrays: dict, cfg: StreamConfig) -> np.ndarray:
    """Return a boolean mask for the signal region."""
    if cfg.pm_ra_signal_range is None:
        if arrays["signal_region"] is None:
            raise ValueError(
                "pm_ra_signal_range is None but the FITS file has no "
                "'signal_region' column. Set pm_ra_signal_range in the YAML."
            )
        return arrays["signal_region"]
    lo, hi = cfg.pm_ra_signal_range
    return (arrays["pm_phi1"] >= lo) & (arrays["pm_phi1"] <= hi)


def _apply_percentile_mask(full_embeddings: np.ndarray, skip_cols: list[int],
                            perc_low: float = 0.05,
                            perc_high: float = 99.95) -> np.ndarray:
    """Return a boolean row mask removing extreme outliers column-wise."""
    mask = np.ones(full_embeddings.shape[0], dtype=bool)
    for i in range(full_embeddings.shape[1]):
        if i in skip_cols:
            continue
        col = full_embeddings[:, i]
        mask &= (col >= np.percentile(col, perc_low)) & \
                (col <= np.percentile(col, perc_high))
    return mask


def train_flow(cfg: StreamConfig, num_epochs: int = 200, batch_size: int = 512,
               max_lr: float = 3e-4, seed: int = 12345, patience: int | None = 20):
    """Train a normalizing flow on the sideband region.

    Parameters
    ----------
    cfg : StreamConfig
        Stream configuration (paths, feature lists, signal region).
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    max_lr : float
        Peak learning rate for the one-cycle schedule.
    seed : int
        Random seed passed to train/test split.

    Returns
    -------
    flow : pzflow.Flow
        Trained flow object.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler fit on the full (pre-split) feature matrix — needed by
        ``sampler.generate_samples`` to invert the scaling on generated data.
    signal_mask : np.ndarray of bool
        Boolean mask identifying the signal region rows (post percentile cut).
    full_embeddings : np.ndarray
        Unscaled full feature matrix (signal + sideband) after percentile cut,
        shape (N, D) with columns in col_names order. Sampler uses this to
        build the signal-region dataframe and to fit the KDE.
    train_losses : list
        Per-epoch training losses from ``pzflow.Flow.train``.
    test_losses : list
        Per-epoch test losses from ``pzflow.Flow.train``.
    """
    import optax
    import jax.numpy as jnp
    from pzflow import Flow
    from pzflow.bijectors import Chain, ShiftBounds, RollingSplineCoupling
    from pzflow.distributions import CentBeta13

    col_names = cfg.flow_cond_columns + cfg.flow_data_columns  # pm_phi1 first
    data_col_names = cfg.flow_data_columns
    cond_col_names = cfg.flow_cond_columns

    arrays = _load_fits(cfg.raw_data_path)

    # Stack full feature matrix in col_names order for scaling
    col_map = {
        "phi1": arrays["phi1"], "phi2": arrays["phi2"],
        "pm_phi1": arrays["pm_phi1"], "pm_phi2": arrays["pm_phi2"],
        "G_mag": arrays["G_mag"], "Bp_mag": arrays["Bp_mag"], "Rp_mag": arrays["Rp_mag"],
        "g_mag": arrays["g_mag"], "r_mag": arrays["r_mag"], "z_mag": arrays["z_mag"],
        "phot_g_flux_err": arrays["phot_g_flux_err"],
        "phot_bp_flux_err": arrays["phot_bp_flux_err"],
        "phot_rp_flux_err": arrays["phot_rp_flux_err"],
        "flux_err_g": arrays["flux_err_g"],
        "flux_err_r": arrays["flux_err_r"],
        "flux_err_z": arrays["flux_err_z"],
        "pmra_error": arrays["pmra_error"],
        "pmdec_error": arrays["pmdec_error"],
        "ra_error": arrays["ra_error"],
        "dec_error": arrays["dec_error"],
    }
    full_embeddings = np.column_stack([col_map[c] for c in col_names])

    # Fit scaler before any masking so it sees the full distribution
    scaler = StandardScaler()
    scaler.fit(full_embeddings)

    # Percentile mask: skip phi1, phi2, G_mag (no natural outlier boundary)
    skip_cols = [col_names.index("phi1"),
                 col_names.index("phi2"),
                 col_names.index("G_mag")]

    signal_mask_raw = _build_signal_mask(arrays, cfg)
    perc_mask = _apply_percentile_mask(full_embeddings, skip_cols)

    full_embeddings = full_embeddings[perc_mask]
    signal_mask = signal_mask_raw[perc_mask]
    stream = arrays["stream"][perc_mask]
    source_id = arrays["source_id"][perc_mask]
    ebv = arrays["ebv"][perc_mask]

    # Scale and split sideband into train/test
    full_embeddings_scaled = scaler.transform(full_embeddings)
    sideband = full_embeddings_scaled[~signal_mask]

    train_split, test_split = train_test_split(sideband, test_size=0.5,
                                               random_state=seed)
    df_train = pd.DataFrame(data=train_split, columns=col_names)
    df_test = pd.DataFrame(data=test_split, columns=col_names)

    # Build flow architecture
    data = df_train[data_col_names].values
    mins = jnp.array(data.min(axis=0))
    maxs = jnp.array(data.max(axis=0))
    ndim = data.shape[1]
    B = 5
    shift_B = B - 1.0

    bijector = Chain(
    ShiftBounds(mins, maxs, B=shift_B), # Does the shifting
    RollingSplineCoupling(nlayers=ndim,  # nlayers: number of (NeuralSplineCoupling(), Roll()) pairs in the chain - default = ndim
                          hidden_layers = 4, # hidden_layers: number of hidden layers used to parametrize each Spline - default = 2
                          hidden_dim = 128, B=B, # hidden_dim: number of neurons in each hidden layer. B: Same as above - default = 128
                          n_conditions = 1, # n_conditions: leave as 1 since we are conditioning on pm_phi1 only
                          K=16) # K: Spline resolution? Paper states "In the limit of high spline resolution (i.e. K → ∞) [...] [flow] can model  model arbitrarily complex distributions"
                                # default = 16
    )
    

    latent = CentBeta13(input_dim=ndim, B=B)
    flow = Flow(data_col_names, bijector=bijector, latent=latent,
                conditional_columns=cond_col_names)

    # Learning rate schedule
    total_steps = len(df_train) // batch_size * num_epochs
    lr_schedule = optax.cosine_onecycle_schedule(
        peak_value=max_lr,
        transition_steps=total_steps,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=1000,
    )
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule),
    )

    train_losses, test_losses = flow.train(
        df_train, df_test,
        verbose=True, epochs=num_epochs,
        progress_bar=False, batch_size=batch_size,
        optimizer=opt, patience = patience
    )

    return flow, scaler, signal_mask, full_embeddings, ebv, source_id, stream, col_names, train_losses, test_losses
