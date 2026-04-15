# %% [markdown]
# # Validation 3-Panel Figure
# Generates validation_3panel.pdf/.png for the GD-1 SCREAM paper.
# Panels: (a) sky φ1/φ2, (b) DESI VRAD vs φ1, (c) CMD r₀ vs g₀−r₀.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table

from matplotlib import font_manager

# 1. Register the font file
font_path = '/pscratch/sd/p/XXXXa/cmunrm.ttf'
font_manager.fontManager.addfont(font_path)

# 2. Get the font name and set it globally
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()


# %% [markdown]
# ## Constants

# %%
# MODEL_PROB_THRESHOLD = 0.91
MODEL_PROB_THRESHOLD = 0.878
CROSSMATCH_PATH = '/pscratch/sd/p/XXXXa/GD-1_gaia_x_decals_VRAD2.fits'
OUTPUT_PATH     = f'/global/homes/p/XXXXa/SCREAM/notebooks/V2_validation_3panel.pdf'

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# %% [markdown]
# ## Shared style

# %%
# _COLORS  = ["C0", "C3", "C8"]   # TP, FP, FN
_COLORS  = ["C0", "C1", "C2"]   # TP, FP, FN
_COLORS_DESI = ["C3", "C4", "C9"]  # TP, FP, FN for DESI-labeled subset
# _MARKERS = ["o", "s", "D"]
_MARKERS = ["o", "X", "^"]
_LABELS_SF  = ["TP (SF)", "FP (SF)", "FN (SF)"]
_LABELS_DESI = ["TP (DESI)", "FP (DESI)", "FN (DESI)"]
# _EDGE_COLORS = [None, None, None]
_EDGE_COLORS = ['k', 'k', 'k']
_LINEWIDTHS = [0.25, 0.25, 0.25]
_ZORDER = [3, 2, 1]
s=13
alpha=0.6
plt.rcParams.update({
    'font.family':      'serif',
    # 'font.serif': ['Nimbus Roman No9 L'],
    # 'font.serif': ['Liberation Serif'],
    'axes.labelsize':   10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
})

# %% [markdown]
# ## Load crossmatched catalog
# Run crossmatch_DESI.py once to generate this file.

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

phi1 = np.array(cat['phi1'])
phi2 = np.array(cat['phi2'])
g_r  = np.array(cat['gmag0']) - np.array(cat['rmag0'])
r0   = np.array(cat['rmag0'])

has_desi_label = np.array(cat['desi_label']) != -1
cat_desi = cat[has_desi_label]
print(len(cat_desi))
scream_label_desi = np.array(cat_desi['model_prob']) > MODEL_PROB_THRESHOLD
spec_label_desi   = np.array(cat_desi['desi_label']).astype(bool)

desi_phi1 = np.array(cat_desi['phi1'])
desi_phi2 = np.array(cat_desi['phi2'])

desi_g_r  = np.array(cat_desi['gmag0']) - np.array(cat_desi['rmag0'])
desi_r0   = np.array(cat_desi['rmag0'])
desi_TP = spec_label_desi  &  scream_label_desi
desi_FP = ~spec_label_desi &  scream_label_desi
desi_FN = spec_label_desi  & ~scream_label_desi
desi_TN = ~spec_label_desi & ~scream_label_desi
# Print metrics (accuracy, precision, recall) for DESI-labeled subset.
accuracy  = np.mean(scream_label_desi == spec_label_desi)
precision = np.sum(desi_TP) / np.sum(scream_label_desi)
recall    = np.sum(desi_TP) / np.sum(spec_label_desi)
f1        = 2 * (precision * recall) / (precision + recall)
print(f'  Accuracy:  {accuracy:.3f}')
print(f'  Precision: {precision:.3f}')
print(f'  Recall:    {recall:.3f}')
print(f'  F1 Score:  {f1:.3f}')

vrad = np.array(cat_desi['DESI_VRAD'])
vrad_err = np.array(cat_desi['DESI_VRAD_ERR'])

# has_vrad  = np.isfinite(np.array(cat['DESI_VRAD']))
# vrad      = np.array(cat['DESI_VRAD'])
# vrad_err  = np.array(cat['DESI_VRAD_ERR'])
# vrad_mask = pred_label & has_vrad

# %% [markdown]
# ## Figure

# %%
fig = plt.figure(figsize=(5.5, 3.5), constrained_layout=True)
gs  = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
ax_sky  = fig.add_subplot(gs[0, 0])              # Panel A: φ1 vs φ2
ax_vrad = fig.add_subplot(gs[1, 0], sharex=ax_sky)  # Panel B: VRAD vs φ1
ax_cmd  = fig.add_subplot(gs[:, 1])   # Panel C: r₀ vs g₀−r₀ (full height)

# ── Panel A: φ1 vs φ2 ────────────────────────────────────────────────────────
ax_sky.hist2d(phi1[~sf_label], phi2[~sf_label],
              bins=150, norm=LogNorm(), cmap='gray_r', alpha=0.4)
for mask, color, marker, zorder, edgecolor, linewidth in zip([TP, FP, FN], _COLORS, _MARKERS, _ZORDER, _EDGE_COLORS, _LINEWIDTHS):
    ax_sky.scatter(phi1[mask], phi2[mask],
                   c=color, marker=marker, s=s, alpha=alpha, edgecolors=edgecolor, linewidths=linewidth, zorder=zorder)
ax_sky.set_ylabel(r'$\Phi_2$ (deg)')
ax_sky.set_ylim(-5.25, None)
plt.setp(ax_sky.get_xticklabels(), visible=False)
# ax_sky.text(0.03, 0.95, '(a)', transform=ax_sky.transAxes, va='top')

# ── Panel B: VRAD vs φ1 ──────────────────────────────────────────────────────
# ax_vrad.errorbar(phi1[vrad_mask], vrad[vrad_mask],
#                  yerr=vrad_err[vrad_mask],
#                  fmt='o', color=_COLORS[0],
#                  linewidth=0.8, capsize=2, ms=4, alpha=0.7)
# ax_vrad.scatter(phi1[vrad_mask], vrad[vrad_mask], c=_COLORS[0],  s=s, alpha=alpha, 
#                 marker='o', edgecolors='k', linewidths=0.5, zorder=3)
ax_vrad.hist2d(desi_phi1[desi_TN], vrad[desi_TN], bins=150, norm=LogNorm(), cmap='gray_r', alpha=0.4)
for mask, color, marker, zorder, edgecolor, linewidth, label in zip([desi_TP, desi_FP, desi_FN], _COLORS_DESI, _MARKERS, _ZORDER, _EDGE_COLORS, _LINEWIDTHS, _LABELS_DESI):
    ax_vrad.scatter(desi_phi1[mask], vrad[mask],
                   c=color, marker=marker, s=s, alpha=alpha, edgecolors=edgecolor, linewidths=linewidth, zorder=zorder, label=label)
ax_vrad.set_ylim(-300, 300)
# Make the legend be 3 columns
# ax_vrad.legend(loc='upper right', ncol=3)
# ax_vrad.legend(loc='upper right', frameon=True, borderpad=0.1)
# I need to move the markers closer to the text, no frame
ax_vrad.legend(loc='upper right', frameon=False, markerscale=2, handletextpad=0.05)


ax_vrad.set_xlabel(r'$\Phi_1$ (deg)')
ax_vrad.set_ylabel(r'$V_\mathrm{rad}\ (\mathrm{km\,s}^{-1})$')
# ax_vrad.text(0.03, 0.95, '(b)', transform=ax_vrad.transAxes, va='top')

# ── Panel C: r₀ vs g₀−r₀ CMD ─────────────────────────────────────────────────
ax_cmd.hist2d(g_r[~sf_label], r0[~sf_label],
              bins=150, norm=LogNorm(), cmap='gray_r', alpha=0.4)
for mask, color, marker, label, zorder, edgecolor, linewidth in zip([TP, FP, FN], _COLORS, _MARKERS, _LABELS_SF, _ZORDER, _EDGE_COLORS, _LINEWIDTHS):
    ax_cmd.scatter(g_r[mask], r0[mask],
                   c=color, marker=marker, s=s, alpha=alpha, edgecolors=edgecolor, linewidths=linewidth,
                   label=label, zorder=zorder)
ax_cmd.invert_yaxis()
ax_cmd.set_xlim(0, .7)
ax_cmd.set_xlabel(r'$g - r$')
ax_cmd.set_ylabel(r'$r$')
ax_cmd.legend(loc='upper right', frameon=False, markerscale=2, handletextpad=0.05)
# ax_cmd.text(0.03, 0.95, '(c)', transform=ax_cmd.transAxes, va='top')

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT_PATH.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f'Saved to {OUTPUT_PATH}')


# Make 2 other figures for the DESI-labeled subset only, with TP/FP/FN colored by the DESI label instead of the SF label. 
# These are for internal use only, not in the paper.
# Let's just do phi1 vs phi2 and CMR for the DESI-labeled subset, colored by DESI label.

# Clear the previous figure
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist2d(desi_phi1[desi_TN], desi_phi2[desi_TN], bins=150, norm=LogNorm(), cmap='gray_r', alpha=0.4)
for mask, color, marker, label, zorder, edgecolor, linewidth in zip([desi_TP, desi_FP, desi_FN], _COLORS_DESI, _MARKERS, _LABELS_DESI, _ZORDER, _EDGE_COLORS, _LINEWIDTHS):
    ax.scatter(desi_phi1[mask], desi_phi2[mask],
               c=color, marker=marker, s=30, alpha=alpha, edgecolors=edgecolor, linewidths=linewidth,
               label=label, zorder=zorder)
ax.set_xlabel(r'$\Phi_1$ (deg)')
ax.set_ylabel(r'$\Phi_2$ (deg)')
ax.legend(loc='upper right', frameon=False, markerscale=2, handletextpad=0.05)
fig.savefig(OUTPUT_PATH.replace('.pdf', '_desi_phi1_phi2.png'), dpi=300, bbox_inches='tight')

plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist2d(desi_g_r[desi_TN], desi_r0[desi_TN], bins=150, norm=LogNorm(), cmap='gray_r', alpha=0.4)
for mask, color, marker, label, zorder, edgecolor, linewidth in zip([desi_TP, desi_FP, desi_FN], _COLORS_DESI, _MARKERS, _LABELS_DESI, _ZORDER, _EDGE_COLORS, _LINEWIDTHS):
    ax.scatter(desi_g_r[mask], desi_r0[mask],
               c=color, marker=marker, s=30, alpha=alpha, edgecolors=edgecolor, linewidths=linewidth,
               label=label, zorder=zorder)
ax.set_xlim(0, .7)
ax.invert_yaxis()
ax.set_xlabel(r'$g - r$')
ax.set_ylabel(r'$r$')
ax.legend(loc='upper right', frameon=False, markerscale=2, handletextpad=0.05)
fig.savefig(OUTPUT_PATH.replace('.pdf', '_desi_cmd.png'), dpi=300, bbox_inches='tight')