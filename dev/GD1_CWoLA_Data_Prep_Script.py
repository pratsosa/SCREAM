# %% [markdown]
# Adapted from GD1_CWoLA_Data_prep.ipynb

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import table
from tqdm import tqdm
import seaborn as sns
import pickle
from datetime import datetime, timezone, timedelta
import warnings

import torch
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset
from torch import nn
from torch.nn import Linear, LayerNorm, ReLU
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_cluster import knn_graph
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.transforms import KNNGraph
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
import torchvision
from torchvision.ops import sigmoid_focal_loss
from torch_geometric.nn import DeepGCNLayer, GENConv

from torchmetrics.classification import BinaryMatthewsCorrCoef

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb

from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, roc_auc_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.exceptions import UndefinedMetricWarning

from utils import count_parameters, get_mask_splits, DeeperGCN, LitDeepGCN, GaiaDataModule, GaiaDataModuleCustom
from sklearn.metrics import precision_recall_curve, average_precision_score

# %% [markdown]
# ## Data Preparation:
# Use SF labels --> crossmatch w/ Gaia ID  
# Drop data points where G0, R0, Z0 undefined or create mask column

# %%
t = table.Table.read('/pscratch/sd/p/pratsosa/GD-1_gaia_x_decals.fits')
t

# %%
gmag0 = np.array(t['gmag0'])
rmag0 = np.array(t['rmag0'])
zmag0 = np.array(t['zmag0'])

grz_mask = ~(np.isinf(gmag0) | np.isinf(rmag0) | np.isinf(zmag0))

# %%
temp_t = t[grz_mask]


# %%
streamfinder = Table(fits.getdata("/pscratch/sd/p/pratsosa/general_stream_data/streamfinder_gaiadr3.fits"))

stream = streamfinder['Stream']
stream_mask = (stream == 53)
gaia_IDS = streamfinder['Gaia']
stream_IDS = np.array(gaia_IDS[stream_mask])

plt.scatter(temp_t['ra'], temp_t['dec'], s=1, alpha=0.1)
plt.scatter(streamfinder['RAdeg'][stream_mask], streamfinder['DEdeg'][stream_mask], s=10)

# %%
GD1_source_ids = np.array(temp_t['source_id'])
common_IDS, comm1, comm2 = np.intersect1d(GD1_source_ids, stream_IDS, return_indices=True)
stream_mask_final = np.full(len(temp_t), False, dtype=bool)
stream_mask_final[comm1] = True
print(np.sum(stream_mask_final))
temp_t['stream'] = stream_mask_final


# %%
plt.scatter(temp_t['ra'], temp_t['dec'], s=.1)
plt.scatter(temp_t['ra'][temp_t['stream']], temp_t['dec'][temp_t['stream']], s=10)

# %%
plt.scatter(temp_t['pmra'], temp_t['pmdec'], s=.1)
plt.scatter(temp_t['pmra'][temp_t['stream']], temp_t['pmdec'][temp_t['stream']], s=10)

# %%
import galstreams
mws = galstreams.MWStreams(verbose=False, implement_Off=True)
stream_track = 'GD-1-I21'

# %%
from adql_utils import ra_dec_to_phi1_phi2, phi1_phi2_to_ra_dec, pmra_pmdec_to_pmphi12
stream_frame = mws[stream_track].stream_frame
phi1, phi2 = ra_dec_to_phi1_phi2(stream_frame, temp_t['ra']*u.deg, temp_t['dec']*u.deg)
pm_phi1, pm_phi2 = pmra_pmdec_to_pmphi12(stream_frame, temp_t['ra']*u.deg, temp_t['dec']*u.deg,
                                        temp_t['pmra']*u.mas/u.yr, temp_t['pmdec']*u.mas/u.yr)

# %%
plt.scatter(phi1, phi2, s=.1)
plt.scatter(phi1[temp_t['stream']], phi2[temp_t['stream']], s=10)

# %%
plt.scatter(pm_phi1, pm_phi2, s=.1)
plt.scatter(pm_phi1[temp_t['stream']], pm_phi2[temp_t['stream']], s=10)

# %%
temp_t['phi1'] = phi1
temp_t['phi2'] = phi2
temp_t['pm_phi1'] = pm_phi1
temp_t['pm_phi2'] = pm_phi2

# %%
phi1_phi2_mask = (phi1 < 60) & (phi2 < 10) & (phi2 > -10)
plt.scatter(temp_t['phi1'][phi1_phi2_mask], temp_t['phi2'][phi1_phi2_mask], s=.1)
plt.scatter(temp_t['phi1'][temp_t['stream']], temp_t['phi2'][temp_t['stream']], s=10)

# %%
temp_t = temp_t[phi1_phi2_mask]

# %%
mws.summary.loc[stream_track, :]['distance_mid']

# %%
temp_t['stream'].sum()

# %%
parallax = np.array(temp_t['parallax'])
parallax_err = np.array(temp_t['parallax_error'])

par_mask = parallax - 3 * np.abs(parallax_err) < 1 / mws.summary.loc[stream_track, :]['distance_mid']

# %%
temp_t['stream'][par_mask].sum()

# %%
temp_t = temp_t[par_mask]
len(temp_t)

# %%
temp_t[0]

# %%
color = temp_t['phot_bp_mean_mag'] - temp_t['phot_rp_mean_mag']
color_mask = (color > 0.0) & (color < 1.0)
g_mask = temp_t['phot_g_mean_mag'] < 20.2
full_mask = color_mask & g_mask
temp_t['stream'][full_mask].sum(), len(temp_t[full_mask])

# %%
temp_t = temp_t[full_mask]
temp_t['stream'].sum(), len(temp_t)

# %%
pm_phi1 = np.array(temp_t['pm_phi1'])
pm_phi2 = np.array(temp_t['pm_phi2'])
phi1 = np.array(temp_t['phi1'])
phi2 = np.array(temp_t['phi2'])
stream = np.array(temp_t['stream'])
stream_pm_phi1 = pm_phi1[stream]

# %%
# Going to try stream cuts in different coordinates: pmra*cos(dec) and parallax

parallax = np.array(temp_t['parallax'])
dec = np.array(temp_t['dec'])
pmra = np.array(temp_t['pmra'])
pmra_cosdec = pmra * np.cos(np.radians(dec))
stream = np.array(temp_t['stream'])

stream_parallax = parallax[stream]
stream_pmra_cosdec = pmra_cosdec[stream]

par_med, par_std = np.median(parallax[stream]), np.std(parallax[stream])
par_lower_perc = np.percentile(stream_parallax, 5)
par_upper_perc = np.percentile(stream_parallax, 95)
par_lower_bound = par_med - 5 * par_std
par_upper_bound = par_med + 5 * par_std

par_regions = [[par_lower_perc, par_upper_perc], [par_lower_bound, par_upper_bound]]


pmra_med, pmra_std = np.median(pmra_cosdec[stream]), np.std(pmra_cosdec[stream])
pmra_lower_perc = np.percentile(stream_pmra_cosdec, 0)
pmra_upper_perc = np.percentile(stream_pmra_cosdec, 100)
pmra_lower_bound = pmra_med - 5 * pmra_std
pmra_upper_bound = pmra_med + 5 * pmra_std
pmra_regions = [[pmra_lower_perc, pmra_upper_perc], [pmra_lower_bound, pmra_upper_bound]]

par_signal_region = (parallax > par_regions[0][0]) & (parallax < par_regions[0][1])
par_sideband_region = (parallax > par_regions[1][0]) & (parallax < par_regions[1][1]) & ~par_signal_region

pmra_signal_region = (pmra_cosdec > pmra_regions[0][0]) & (pmra_cosdec < pmra_regions[0][1])
pmra_sideband_region = (pmra_cosdec > pmra_regions[1][0]) & (pmra_cosdec < pmra_regions[1][1]) & ~pmra_signal_region

par_stream_prop_signal = np.sum(par_signal_region[stream]) / (np.sum(par_signal_region[stream]) + np.sum(par_sideband_region[~stream]))
print(f'The proportion of stream stars in the parallax signal region is : {par_stream_prop_signal * 100}%')
par_stream_prop_sideband = np.sum(par_sideband_region[stream]) / (np.sum(par_signal_region[~stream]) + np.sum(par_sideband_region[~stream]))
print(f'The proportion of stream stars in the parallax sideband region is : {par_stream_prop_sideband * 100}%')

pmra_stream_prop_signal = np.sum(pmra_signal_region[stream]) / (np.sum(pmra_signal_region[stream]) + np.sum(pmra_sideband_region[~stream]))
print(f'The proportion of stream stars in the pmra*cos(dec) signal region is : {pmra_stream_prop_signal * 100}%')
pmra_stream_prop_sideband = np.sum(pmra_sideband_region[stream]) / (np.sum(pmra_signal_region[~stream]) + np.sum(pmra_sideband_region[~stream]))
print(f'The proportion of stream stars in the pmra*cos(dec) sideband region is : {pmra_stream_prop_sideband * 100}%')



# %%
pm_p1_med, pm_p1_std= np.median(stream_pm_phi1), np.std(stream_pm_phi1)


#regions = [[pm_p1_med - 1*pm_p1_std, pm_p1_med + 1*pm_p1_std] , [pm_p1_med - 3*pm_p1_std, pm_p1_med + 3*pm_p1_std]]
# Define regions using 5 and 95 percentiles
lower_perc = np.percentile(stream_pm_phi1, 5)
upper_perc = np.percentile(stream_pm_phi1, 95)
# lower_perc = pm_p1_med - 1*pm_p1_std
# upper_perc = pm_p1_med + 1*pm_p1_std
lower_bound = pm_p1_med - 3*pm_p1_std
upper_bound = pm_p1_med + 3*pm_p1_std

regions = [[lower_perc, upper_perc], [lower_bound, upper_bound]]

sig_low, sig_high = regions[0]
side_low, side_high = regions[1]
print(f"sig_low: {sig_low}, sig_high: {sig_high}, side_low: {side_low}, side_high: {side_high}")

signal_region = (pm_phi1 > sig_low) & (pm_phi1 < sig_high)
sideband_region = (pm_phi1 > side_low) & (pm_phi1 < side_high) & ~signal_region

# %%
stream_prop_signal = np.sum(signal_region[stream]) / (np.sum(signal_region[stream]) + np.sum(signal_region[~stream]))
print(f'The proportion of stream stars in the signal region is : {stream_prop_signal * 100}%')

# %%
stream_prop_side = np.sum(sideband_region[stream]) / (np.sum(sideband_region[stream]) + np.sum(sideband_region[~stream]))
print(f'The proportion of stream stars in the sideband region is : {stream_prop_side * 100}%')

# %%
# Colorblind-safe colors
stream_color = '#0072B2'     # Blue
background_color = '#E69F00' # Orange
signal_shade = '#56B4E9'     # Light Blue
sideband_shade = '#999999'   # Grey
signal_line = '#0072B2'
sideband_line = '#333333'

plt.figure(figsize=(10, 6))

# Plot histograms
plt.hist(pm_phi1, density=True, bins=1000, range=(-20, 20), alpha=0.6, color=background_color, label='Background Stars')
plt.hist(stream_pm_phi1, density=True, bins=100, alpha=0.6, color=stream_color, label='Stream Stars')

# Highlight signal region (light blue)
plt.axvspan(sig_low, sig_high, color=signal_shade, alpha=0.3, label='Signal Region')

# Highlight sideband regions (light grey)
if side_low < sig_low:
    plt.axvspan(side_low, sig_low, color=sideband_shade, alpha=0.2, label='Sideband Region')
if sig_high < side_high:
    plt.axvspan(sig_high, side_high, color=sideband_shade, alpha=0.2)

# Region boundaries
plt.axvline(sig_low, color=signal_line, linestyle='--', linewidth=1)
plt.axvline(sig_high, color=signal_line, linestyle='--', linewidth=1)
plt.axvline(side_low, color=sideband_line, linestyle='--', linewidth=1)
plt.axvline(side_high, color=sideband_line, linestyle='--', linewidth=1)

# Labels and styling
plt.xlabel(r'Proper Motion $\Phi_1$ (mas/yr)', fontsize=30)
plt.ylabel('Density', fontsize=30)
plt.title(r'Stream vs Background Proper Motion $\Phi_1$', fontsize=30)
plt.tick_params(axis='both', labelsize=12)
plt.legend(fontsize=20)
plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
# plt.savefig('Stream vs Background PM Phi1.png', dpi=600)
plt.show()


# %%
roi = sideband_region | signal_region
# signal_region = signal_region[roi]
# sideband_region = sideband_region[roi]   
np.sum(roi)

# %%
temp_t['roi'] = roi
temp_t['signal_region'] = signal_region
temp_t['sideband_region'] = sideband_region

# %%
(temp_t['stream'].sum())

# %%
temp_t = table.unique(temp_t, keys='source_id', keep='first')

# %%
df = temp_t.to_pandas()

ra_actual = np.array(df['ra'])
dec_actual = np.array(df['dec'])

ra = np.array(df['phi1']).astype('float64')
dec = np.array(df['phi2'])

pm_ra = np.array(df['pm_phi1']).astype('float64')
pm_dec = np.array(df['pm_phi2'])

pm_ra_error = np.array(df['pmra_error']).astype('float64')
pm_dec_error = np.array(df['pmdec_error']).astype('float64')

gmag  = np.array(df['phot_g_mean_mag'])
color = np.array(df['phot_bp_mean_mag']) - np.array(df['phot_rp_mean_mag'])
parallax = np.array(df['parallax'])
parallax_error = np.array(df['parallax_error'])

gmag0, rmag0, zmag0 = np.array(df['gmag0']), np.array(df['rmag0']), np.array(df['zmag0'])

# %%
full_mask = ~(np.isnan(ra) | np.isnan(dec) | np.isnan(pm_ra) | np.isnan(pm_dec) | np.isnan(gmag) | np.isnan(color) 
| np.isnan(parallax) | np.isnan(parallax_error) | np.isnan(gmag0)| np.isnan(rmag0) | np.isnan(zmag0) 
| np.isnan(pm_ra_error) | np.isnan(pm_dec_error) | np.isnan(ra_actual) | np.isnan(dec_actual) )
temp_t = temp_t[full_mask]

# %%
# temp_t.write('/pscratch/sd/p/pratsosa/GD-1_gaia_x_decals_stream_prep.fits', overwrite=True)


