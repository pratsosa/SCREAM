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
font_path = '/pscratch/sd/p/pratsosa/cmunrm.ttf'
font_manager.fontManager.addfont(font_path)

# 2. Get the font name and set it globally
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()


# %% [markdown]
# ## Constants

# %%
MODEL_PROB_THRESHOLD = 0.91
CROSSMATCH_PATH = '/pscratch/sd/p/pratsosa/GD-1_gaia_x_decals_VRAD.fits'
OUTPUT_PATH     = f'/global/homes/p/pratsosa/SCREAM/notebooks/validation_3panel.pdf'

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# %% [markdown]
# ## Shared style

# %%
_COLORS  = ["C0", "C3", "C8"]   # TP, FP, FN
_MARKERS = ["o", "s", "D"]
_LABELS  = ["True Positives", "False Positives", "False Negatives"]
s=10
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

has_vrad  = np.isfinite(np.array(cat['DESI_VRAD']))
vrad      = np.array(cat['DESI_VRAD'])
vrad_err  = np.array(cat['DESI_VRAD_ERR'])
vrad_mask = pred_label & has_vrad

# %% [markdown]
# ## Figure

# %%
fig = plt.figure(figsize=(5.5, 4), constrained_layout=True)
gs  = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
ax_sky  = fig.add_subplot(gs[0, 0])              # Panel A: φ1 vs φ2
ax_vrad = fig.add_subplot(gs[1, 0], sharex=ax_sky)  # Panel B: VRAD vs φ1
ax_cmd  = fig.add_subplot(gs[:, 1])   # Panel C: r₀ vs g₀−r₀ (full height)

# ── Panel A: φ1 vs φ2 ────────────────────────────────────────────────────────
ax_sky.hist2d(phi1[~sf_label], phi2[~sf_label],
              bins=150, norm=LogNorm(), cmap='gray_r', alpha=0.4)
for mask, color, marker in zip([TP, FP, FN], _COLORS, _MARKERS):
    ax_sky.scatter(phi1[mask], phi2[mask],
                   c=color, marker=marker, s=s, alpha=alpha, edgecolors='k', linewidths=0.5,)
ax_sky.set_ylabel(r'$\Phi_2$ (deg)')
plt.setp(ax_sky.get_xticklabels(), visible=False)
ax_sky.text(0.03, 0.95, '(a)', transform=ax_sky.transAxes, va='top')

# ── Panel B: VRAD vs φ1 ──────────────────────────────────────────────────────
ax_vrad.errorbar(phi1[vrad_mask], vrad[vrad_mask],
                 yerr=vrad_err[vrad_mask],
                 fmt='o', color=_COLORS[0],
                 elinewidth=0.8, capsize=2, ms=4, alpha=0.7)
ax_vrad.set_xlabel(r'$\Phi_1$ (deg)')
ax_vrad.set_ylabel(r'$V_\mathrm{rad}\ (\mathrm{km\,s}^{-1})$')
ax_vrad.text(0.03, 0.95, '(b)', transform=ax_vrad.transAxes, va='top')

# ── Panel C: r₀ vs g₀−r₀ CMD ─────────────────────────────────────────────────
ax_cmd.hist2d(g_r[~sf_label], r0[~sf_label],
              bins=150, norm=LogNorm(), cmap='gray_r', alpha=0.4)
for mask, color, marker, label in zip([TP, FP, FN], _COLORS, _MARKERS, _LABELS):
    ax_cmd.scatter(g_r[mask], r0[mask],
                   c=color, marker=marker, s=s, alpha=alpha, edgecolors='k', linewidths=0.5,
                   label=label)
ax_cmd.invert_yaxis()
ax_cmd.set_xlim(0, .7)
ax_cmd.set_xlabel(r'$g - r$')
ax_cmd.set_ylabel(r'$r$')
ax_cmd.legend(loc='upper right')
ax_cmd.text(0.03, 0.95, '(c)', transform=ax_cmd.transAxes, va='top')

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT_PATH.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f'Saved to {OUTPUT_PATH}')