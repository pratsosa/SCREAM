"""GD-1 diagnostic visualizations.

Loads the cleaned FITS from GD1_data_prep.py and produces four plots:
  1. phi1 vs phi2          — stream track on sky
  2. pm_phi1 vs pm_phi2    — proper motion distribution
  3. G vs BP-RP            — Gaia colour-magnitude diagram
  4. r0 vs g0-r0           — DECaLS colour-magnitude diagram

Each plot shows all stars as a 2-D density histogram with
StreamFinder-labelled GD-1 members overlaid as a red scatter.
"""

import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Style — mirrors scream/plotting conventions
# ---------------------------------------------------------------------------
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
_HIST_KW   = dict(bins=150, cmap="gray_r", alpha=0.5, rasterized=True)
_STREAM_KW = dict(s=30, c="tab:red", edgecolors="k", linewidths=0.3, alpha=0.9, zorder=3)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# INPUT_FITS = "/pscratch/sd/p/pratsosa/GD-1_gaia_x_decals_full.fits"
INPUT_FITS = "/pscratch/sd/p/pratsosa/GD-1_gaia_x_decals_040726.fits"
OUTPUT_DIR = Path("/pscratch/sd/p/pratsosa/GD-1_plots")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"Loading {INPUT_FITS} ...")
t = Table.read(INPUT_FITS)
stream = np.array(t["stream"], dtype=bool)
print(f"  Total stars : {len(t):,}")
print(f"  Stream stars: {stream.sum():,}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: clip axis limits to 1st–99th percentile to suppress outliers
# ---------------------------------------------------------------------------
def _pct_lim(arr, lo=1, hi=99, pad=0.05):
    lo_val, hi_val = np.percentile(arr, [lo, hi])
    span = hi_val - lo_val
    return lo_val - pad * span, hi_val + pad * span


# ---------------------------------------------------------------------------
# Plot 1 — phi1 vs phi2
# ---------------------------------------------------------------------------
with matplotlib.rc_context(_RC):
    fig, ax = plt.subplots(figsize=(12, 5))

    phi1 = np.array(t["phi1"])
    phi2 = np.array(t["phi2"])
    xlim = _pct_lim(phi1)
    ylim = _pct_lim(phi2)

    ax.hist2d(phi1[~stream], phi2[~stream],
              range=[xlim, ylim], **_HIST_KW)
    ax.scatter(phi1[stream], phi2[stream], label="GD-1 (SF)", **_STREAM_KW)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\Phi_1$ [deg]", fontsize=28)
    ax.set_ylabel(r"$\Phi_2$ [deg]", fontsize=28)
    ax.legend()

    out = OUTPUT_DIR / "phi1_phi2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

# ---------------------------------------------------------------------------
# Plot 2 — pm_phi1 vs pm_phi2
# ---------------------------------------------------------------------------
with matplotlib.rc_context(_RC):
    fig, ax = plt.subplots(figsize=(10, 8))

    pm1 = np.array(t["pm_phi1"])
    pm2 = np.array(t["pm_phi2"])
    xlim = _pct_lim(pm1)
    ylim = _pct_lim(pm2)

    ax.hist2d(pm1[~stream], pm2[~stream],
              range=[xlim, ylim], **_HIST_KW)
    ax.scatter(pm1[stream], pm2[stream], label="GD-1 (SF)", **_STREAM_KW)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\mu_{\Phi_1}$ [mas yr$^{-1}$]", fontsize=28)
    ax.set_ylabel(r"$\mu_{\Phi_2}$ [mas yr$^{-1}$]", fontsize=28)
    ax.legend()

    out = OUTPUT_DIR / "pm_phi1_phi2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

# ---------------------------------------------------------------------------
# Plot 3 — G vs BP-RP  (Gaia CMD)
# ---------------------------------------------------------------------------
with matplotlib.rc_context(_RC):
    fig, ax = plt.subplots(figsize=(10, 8))

    color = np.array(t["phot_bp_mean_mag"] - t["phot_rp_mean_mag"])
    gmag  = np.array(t["phot_g_mean_mag"])
    xlim  = _pct_lim(color)
    ylim  = _pct_lim(gmag)

    ax.hist2d(color[~stream], gmag[~stream],
              range=[xlim, ylim], **_HIST_KW)
    ax.scatter(color[stream], gmag[stream], label="GD-1 (SF)", **_STREAM_KW)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_xlabel("BP - RP", fontsize=28)
    ax.set_ylabel("G", fontsize=28)
    ax.legend()

    out = OUTPUT_DIR / "cmd_gaia.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

# ---------------------------------------------------------------------------
# Plot 4 — r0 vs g0-r0  (DECaLS CMD)
# ---------------------------------------------------------------------------
with matplotlib.rc_context(_RC):
    fig, ax = plt.subplots(figsize=(10, 8))

    g_r   = np.array(t["gmag0"] - t["rmag0"])
    rmag0 = np.array(t["rmag0"])
    xlim  = _pct_lim(g_r)
    ylim  = _pct_lim(rmag0)

    ax.hist2d(g_r[~stream], rmag0[~stream],
              range=[xlim, ylim], **_HIST_KW)
    ax.scatter(g_r[stream], rmag0[stream], label="GD-1 (SF)", **_STREAM_KW)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_xlabel(r"g$_0$ - r$_0$", fontsize=28)
    ax.set_ylabel(r"r$_0$", fontsize=28)
    ax.legend()

    out = OUTPUT_DIR / "cmd_ls_gr.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

print("Done.")
