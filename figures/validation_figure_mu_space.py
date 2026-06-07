# %% [markdown]
# # Validation μ-space Figure
# Generates validation_mu_space.pdf/.png for the GD-1 SCREAM paper.
# Panels: (a) φ1 vs μ₁ (pm_phi1), (b) φ1 vs μ₂ (pm_phi2).
# TP/FP/FN defined w.r.t. SF labels.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table

from matplotlib import font_manager

font_path = '/pscratch/sd/p/pratsosa/cmunrm.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

# %% [markdown]
# ## Constants

# %%
MODEL_PROB_THRESHOLD = 0.878
CROSSMATCH_PATH = '/pscratch/sd/p/pratsosa/GD-1_gaia_x_decals_VRAD2.fits'
OUTPUT_PATH     = '/global/homes/p/pratsosa/SCREAM/figures/V2_validation_mu_space_poster.pdf'

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# %% [markdown]
# ## Shared style

# %%
_COLORS      = ["C0", "C1", "C2"]
_COLORS_DESI = ["C3", "C4", "C9"]
_MARKERS     = ["o", "X", "^"]
_LABELS_SF   = ["TP (SF)", "FP (SF)", "FN (SF)"]
_LABELS_DESI = ["TP (DESI)", "FP (DESI)", "FN (DESI)"]
_EDGE_COLORS = ['k', 'k', 'k']
_LINEWIDTHS  = [0.5, 0.5, 0.5]
_ZORDER      = [3, 2, 1]
s     = 100
alpha = 0.6

plt.rcParams.update({
    'font.family':    'serif',
    'axes.labelsize': 45,
    'xtick.labelsize': 40,
    'ytick.labelsize': 40,
    'legend.fontsize': 35,
})

# %% [markdown]
# ## Load catalog

# %%
cat = Table.read(CROSSMATCH_PATH)

# %% [markdown]
# ## Derived arrays

# %%
sf_label   = np.array(cat['stream']).astype(bool)
pred_label = np.array(cat['model_prob']) > MODEL_PROB_THRESHOLD

TP = sf_label  &  pred_label
FP = ~sf_label &  pred_label
FN = sf_label  & ~pred_label

phi1   = np.array(cat['phi1'])
mu1    = np.array(cat['pm_phi1'])
mu2    = np.array(cat['pm_phi2'])

visible = TP | FP | FN

# DESI-labeled subset
has_desi_label    = np.array(cat['desi_label']) != -1
cat_desi          = cat[has_desi_label]
scream_label_desi = np.array(cat_desi['model_prob']) > MODEL_PROB_THRESHOLD
spec_label_desi   = np.array(cat_desi['desi_label']).astype(bool)

desi_TP = spec_label_desi  &  scream_label_desi
desi_FP = ~spec_label_desi &  scream_label_desi
desi_FN = spec_label_desi  & ~scream_label_desi
desi_TN = ~spec_label_desi & ~scream_label_desi

desi_phi1 = np.array(cat_desi['phi1'])
desi_mu1  = np.array(cat_desi['pm_phi1'])
desi_mu2  = np.array(cat_desi['pm_phi2'])

desi_visible = desi_TP | desi_FP | desi_FN

# %% [markdown]
# ## Figure

# %%
fig, (ax_mu1, ax_mu2) = plt.subplots(
    2, 1, figsize=(15.5, 10), sharex=True, constrained_layout=True
)

# ── Panel A: φ1 vs μ₁ ────────────────────────────────────────────────────────
ax_mu1.hist2d(phi1, mu1, bins=100, norm=LogNorm(), cmap='gray_r', alpha=0.4, rasterized=True)
for mask, color, marker, zorder, ec, lw, label in zip(
        [TP, FP, FN], _COLORS, _MARKERS, _ZORDER, _EDGE_COLORS, _LINEWIDTHS, _LABELS_SF):
    ax_mu1.scatter(phi1[mask], mu1[mask],
                   c=color, marker=marker, s=s, alpha=alpha,
                   edgecolors=ec, linewidths=lw, zorder=zorder, label=label)
ax_mu1.set_ylabel(r'$\mu_{\Phi_1}\ (\mathrm{mas/yr})$')
ax_mu1.set_ylim(mu1[visible].min() - 0.1, mu1[visible].max() + 0.1)
plt.setp(ax_mu1.get_xticklabels(), visible=False)

# ── Panel B: φ1 vs μ₂ ────────────────────────────────────────────────────────
ax_mu2.hist2d(phi1, mu2, bins=100, norm=LogNorm(), cmap='gray_r', alpha=0.4, rasterized=True)
for mask, color, marker, zorder, ec, lw, label in zip(
        [TP, FP, FN], _COLORS, _MARKERS, _ZORDER, _EDGE_COLORS, _LINEWIDTHS, _LABELS_SF):
    ax_mu2.scatter(phi1[mask], mu2[mask],
                   c=color, marker=marker, s=s, alpha=alpha,
                   edgecolors=ec, linewidths=lw, zorder=zorder, label=label)
ax_mu2.set_xlabel(r'$\Phi_1\ (\mathrm{deg})$')
ax_mu2.set_ylabel(r'$\mu_{\Phi_2}\ (\mathrm{mas/yr})$')
# ax_mu2.set_ylim(mu2[visible].min() - 0.1, mu2[visible].max() + 0.1)

ax_mu2.set_ylim(mu2[visible].min() - 0.1, 0)

# ax_mu2.legend(loc='lower right', frameon=False, markerscale=1, handletextpad=0.05)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT_PATH.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f'Saved to {OUTPUT_PATH}')
plt.close(fig)

# %% [markdown]
# ## DESI-label version (sanity check)

# %%
DESI_OUTPUT_PATH = OUTPUT_PATH.replace('mu_space', 'mu_space_desi')

fig, (ax_mu1, ax_mu2) = plt.subplots(
    2, 1, figsize=(8, 5), sharex=True, constrained_layout=True
)

# ── Panel A: φ1 vs μ₁ ────────────────────────────────────────────────────────
ax_mu1.hist2d(desi_phi1[desi_TN], desi_mu1[desi_TN],
              bins=100, norm=LogNorm(), cmap='gray_r', alpha=0.4, rasterized=True)
for mask, color, marker, zorder, ec, lw, label in zip(
        [desi_TP, desi_FP, desi_FN], _COLORS_DESI, _MARKERS, _ZORDER, _EDGE_COLORS, _LINEWIDTHS, _LABELS_DESI):
    ax_mu1.scatter(desi_phi1[mask], desi_mu1[mask],
                   c=color, marker=marker, s=s, alpha=alpha,
                   edgecolors=ec, linewidths=lw, zorder=zorder, label=label)
ax_mu1.set_ylabel(r'$\mu_{\Phi_1}\ (\mathrm{mas\,yr}^{-1})$')
ax_mu1.set_ylim(desi_mu1[desi_visible].min() - 0.1, desi_mu1[desi_visible].max() + 0.1)
plt.setp(ax_mu1.get_xticklabels(), visible=False)

# ── Panel B: φ1 vs μ₂ ────────────────────────────────────────────────────────
ax_mu2.hist2d(desi_phi1[desi_TN], desi_mu2[desi_TN],
              bins=100, norm=LogNorm(), cmap='gray_r', alpha=0.4, rasterized=True)
for mask, color, marker, zorder, ec, lw, label in zip(
        [desi_TP, desi_FP, desi_FN], _COLORS_DESI, _MARKERS, _ZORDER, _EDGE_COLORS, _LINEWIDTHS, _LABELS_DESI):
    ax_mu2.scatter(desi_phi1[mask], desi_mu2[mask],
                   c=color, marker=marker, s=s, alpha=alpha,
                   edgecolors=ec, linewidths=lw, zorder=zorder, label=label)
ax_mu2.set_xlabel(r'$\Phi_1\ (\mathrm{deg})$')
ax_mu2.set_ylabel(r'$\mu_{\Phi_2}\ (\mathrm{mas/yr})$')
ax_mu2.set_ylim(desi_mu2[desi_visible].min() - 0.1, desi_mu2[desi_visible].max() + 0.1)
ax_mu2.legend(loc='upper right', frameon=False, markerscale=2, handletextpad=0.05)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(DESI_OUTPUT_PATH, dpi=300, bbox_inches='tight')
fig.savefig(DESI_OUTPUT_PATH.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f'Saved to {DESI_OUTPUT_PATH}')
