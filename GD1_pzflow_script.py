# %%
import pzflow
import optax
from pzflow import Flow
import jax
import jax.numpy as jnp
from pzflow.bijectors import Chain, ShiftBounds, RollingSplineCoupling
from pzflow.distributions import Uniform, CentBeta13
from sklearn.neighbors import KernelDensity

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
import astropy.units as u
from tqdm import tqdm
import seaborn as sns
import pickle
from datetime import datetime, timezone, timedelta
import warnings

import torch
from torch import nn
from torch.nn import Linear, LayerNorm, ReLU
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import GCNConv, radius_graph, DeepGCNLayer, GENConv, GAT, BatchNorm
from torch_geometric.data import Data
from torch_cluster import knn_graph

from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.transforms import KNNGraph
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
import torchvision
from torchvision.ops import sigmoid_focal_loss
from torchmetrics.classification import BinaryMatthewsCorrCoef

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb

from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

# %%
seed_everything(12345)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(device))    

# %%
import importlib
import utils
importlib.reload(utils)

# Re-import the function to reflect changes
from utils import count_parameters, get_mask_splits, DeeperGCN, LitDeepGCN, GaiaDataModule, GaiaDataModuleCustom, GaiaDataModuleGD1, GaiaDataModuleGD1CATHODE

# %% [markdown]
# ## Setting up Data for Cathode
# #### 1. First separate data into signal and sideband using pm_ra and 3σ cuts
# #### 2. Train flow on sideband data, condition on pm_ra
# #### 3. Fit KDE to pm_ra distribution in the signal region
# #### 4. Sample N values from $p_{KDE}$(pm_ra)
# #### 5. Generate N samples from flow conditioned on the N sampled pm_ra values

# %%
load_data_dir = '/pscratch/sd/p/pratsosa/GD-1_gaia_x_decals_stream_prep.fits'
df = Table.read(load_data_dir)


ra = np.array(df['phi1']).astype('float64')
dec = np.array(df['phi2']).astype('float64')

pm_ra  = np.array(df['pm_phi1']).astype('float64')
pm_dec = np.array(df['pm_phi2']).astype('float64')

pm_ra_error = np.array(df['pmra_error']).astype('float64')
pm_dec_error = np.array(df['pmdec_error']).astype('float64')

gmag  = np.array(df['phot_g_mean_mag'])
color = np.array(df['phot_bp_mean_mag']) - np.array(df['phot_rp_mean_mag'])


gmag0, rmag0, zmag0 = np.array(df['gmag0']), np.array(df['rmag0']), np.array(df['zmag0'])
g_r = gmag0-rmag0
r_z = rmag0-zmag0
g_z = gmag0-zmag0

stream = np.array(df['stream'])

roi = np.array(df['roi'], dtype=bool)
signal_region = np.array(df['signal_region'], dtype=bool)
sideband_region = np.array(df['sideband_region'], dtype=bool)


embeddings = np.column_stack((ra, dec, pm_dec, pm_ra_error, pm_dec_error, gmag, color, rmag0, g_r, r_z))
full_embeddings = np.column_stack((ra, dec, pm_ra, pm_dec, pm_ra_error, pm_dec_error, gmag, color, rmag0, g_r, r_z))

scaler = StandardScaler()
scaler.fit(full_embeddings)

np.sum(signal_region), np.sum(~signal_region)

# %%
source_id = np.array(df['source_id'])

# %%

col_names = ['ra', 'dec', 'pm_ra', 'pm_dec', 'pm_ra_error', 'pm_dec_error', 'gmag', 'color', 'rmag0', 'g_r', 'r_z']

data_col_names = ['ra', 'dec', 'pm_dec', 'pm_ra_error', 'pm_dec_error', 'gmag', 'color', 'rmag0', 'g_r', 'r_z']

cond_col_names = ["pm_ra"]

# %%
# Create a mask which masks the .05th and 99.95th percentiles of the pm_ra, pm_dec, color, rmag0, g_r, r_z, in full_embeddings (everything except ra, dec, gmag which are columns 1, 2, 5)
# Logic is that ra and dec don't have extreme outliers, and a cut on gmag has already been applied in data prep
perc_mask = np.ones(full_embeddings.shape[0], dtype=bool)
perc_low = .05
perc_high = 99.95
for i in range(full_embeddings.shape[1]):
    if i in [0, 1, 6]:  # Skip ra, dec, gmag
        continue
    col = full_embeddings[:, i]
    lower_perc = np.percentile(col, perc_low)
    upper_perc = np.percentile(col, perc_high)
    perc_mask &= (col >= lower_perc) & (col <= upper_perc)

embeddings = embeddings[perc_mask]
full_embeddings = full_embeddings[perc_mask]
signal_region = signal_region[perc_mask]
stream = stream[perc_mask]

# %%
source_id = source_id[perc_mask]

# %%
# Shows the stream pm_ra distribution vs overall pm_ra distribution

plt.hist(full_embeddings[:, 2], density=True, bins = 100, range=(-20, 15), label='Overall PM RA')
plt.hist(full_embeddings[stream, 2], density=True, bins=10, color='black', alpha=0.7, label='Stream PM RA')

# Plot the boundaries of the signal region
plt.axvline(x=np.min(full_embeddings[signal_region, 2]), color='r', linestyle='--', label='Signal Region Boundaries')
plt.axvline(x=np.max(full_embeddings[signal_region, 2]), color='r', linestyle='--')
plt.legend()
plt.xlabel('PM RA (mas/yr)', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.title('Stream vs Overall PM RA Distribution', fontsize=18)
plt.show()

# %%
full_embeddings = scaler.transform(full_embeddings)
train_data = full_embeddings[~signal_region]

train_split, test_split = train_test_split(train_data, test_size=0.5, random_state=12345)

df_train = pd.DataFrame(data=train_split, columns = col_names)
df_train_subsampled = df_train.sample(frac=.1, random_state=12345)
df_test = pd.DataFrame(data=test_split, columns = col_names)
df_test_subsampled = df_test.sample(frac = .1, random_state = 12345)

# %% [markdown]
# ## Flow

# %%
data = df_train[data_col_names].values
mins = jnp.array(data.min(axis=0))
maxs = jnp.array(data.max(axis=0))
ndim = data.shape[1]

# Splines are defined on range [-B, B]. I kept the default value of B=5.
B = 5
shift_B = B - 1.0  # map into [-4,4] if B=5 (recommended by pzflow docs)

bijector = Chain(
    ShiftBounds(mins, maxs, B=shift_B), # Does the shifting
    RollingSplineCoupling(nlayers=ndim,  # nlayers: number of (NeuralSplineCoupling(), Roll()) pairs in the chain - default = ndim
                          hidden_layers = 4, # hidden_layers: number of hidden layers used to parametrize each Spline - default = 2
                          hidden_dim = 128, B=B, # hidden_dim: number of neurons in each hidden layer. B: Same as above - default = 128
                          n_conditions = 1, # n_conditions: leave as 1 since we are conditioning on pm_phi1 only
                          K=16) # K: Spline resolution? Paper states "In the limit of high spline resolution (i.e. K → ∞) [...] [flow] can model  model arbitrarily complex distributions"
                                # default = 16
)

# latent = Uniform(input_dim=ndim, B=6)
latent = CentBeta13(input_dim=ndim, B=B) # The default latent space used by pzflow - centered beta distribution with alpha, beta = 13

flow = Flow(data_col_names, bijector=bijector, latent=latent,
            conditional_columns=cond_col_names)

# %%
flow = Flow(data_columns=data_col_names, conditional_columns=cond_col_names)

# %%
import optax
# opt = optax.adam(learning_rate=1e-5)
num_epochs = 200
max_lr = 3e-4
batch_size = 512
total_steps = len(df_train) // batch_size * num_epochs
pct_start = 0.3
div_factor = 10
final_div_factor = 1000

# Create the one-cycle schedule function
lr_schedule = optax.cosine_onecycle_schedule(
    peak_value=max_lr,
    transition_steps=total_steps,
    pct_start=pct_start,
    div_factor=div_factor,
    final_div_factor=final_div_factor
)

opt = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping
    optax.adam(learning_rate=lr_schedule)   # Very low learning rate
)

# %%

train_losses, test_losses = flow.train(df_train, df_test, verbose=True, epochs=num_epochs, progress_bar=False, batch_size=batch_size, optimizer=opt)

# %%
plt.plot(train_losses[1:])
plt.plot(test_losses[1:])
plt.xlabel("Epoch")
plt.ylabel("Training loss")
# plt.ylim(0, 13)
plt.show()


# %%
from sklearn.neighbors import KernelDensity
kde_sig = KernelDensity(bandwidth=0.001)
kde_sig.fit(full_embeddings[:, 2][signal_region].reshape(-1, 1))
m_kde_samples = kde_sig.sample(n_samples= 4 * np.sum(signal_region))  # We sample 4x the number of signal region points as recommended in CATHODE paper

# %%
# Quick check that the sampled pm_ra distribution looks similar to the signal region pm_ra distribution

plt.hist(full_embeddings[:, 2][signal_region], bins=100, density=True)
plt.hist(m_kde_samples, bins=100, density=True, histtype='step')
plt.show()

# %%
samples_df = flow.sample(nsamples=1, 
                      conditions=pd.DataFrame(data=m_kde_samples.astype('float64'),
                      columns = ["pm_ra"]),
                      save_conditions=True)
# Invert scaling on the samples
samples_df[col_names] = scaler.inverse_transform(samples_df[col_names])
samples_df.head()

# %%
# Print 1D histograms of the generated samples vs the signal region data for each feature
# This shows us how well the flow is interpolating in the signal region
sig_data = embeddings[signal_region]
train_data = embeddings[~signal_region]
sig_generated = samples_df.to_numpy()

for i, col in enumerate(data_col_names):
    plt.figure(figsize=(8,5))
    plt.hist(sig_data[:, i], density=True, bins=250, alpha=0.5, label='Real Data', range=(np.percentile(sig_data[:, i], 1), np.percentile(sig_data[:, i], 99)))
    plt.hist(train_data[:, i], density=True, bins=250, alpha=0.4, label='Train Data', color = 'black', histtype='step',range=(np.percentile(train_data[:, i], 1), np.percentile(train_data[:, i], 99)))
    plt.hist(sig_generated[:, i], density=True, bins=250, alpha=0.5, label='Generated Data', range=(np.percentile(sig_generated[:, i], 1), np.percentile(sig_generated[:, i], 99)))
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.title(f'Histogram of {col}: Real vs Generated Data')
    plt.legend()
    plt.show()

# %%
data = embeddings[signal_region]

fig, axs = plt.subplots(2, 2, figsize=(8, 10), sharey='row')

# First row
axs[0, 0].scatter(data[:, 0], data[:, 1], s=0.1)
# axs[0, 0].scatter(data[:, 0][true_labels_test], data[:, 1][true_labels_test], s=1)

axs[0, 1].scatter(samples_df['ra'], samples_df['dec'], s=0.1)
axs[0, 0].set_ylabel("Dec")
axs[0, 0].set_xlabel("RA")
axs[0, 1].set_xlabel("RA")

# axs[0, 1].set_xlim(np.min(data[:, 0]), np.max(data[:, 0]))
# axs[0, 1].set_ylim(np.min(data[:, 1]), np.max(data[:, 1]))


# Second row
axs[1, 0].scatter(data[:, 4], data[:, 3], s=0.1)
# axs[2, 0].scatter(data[:, 5][true_labels_test], data[:, 4][true_labels_test], s=1)

axs[1, 1].scatter(samples_df['color'], samples_df['gmag'], s=0.1)
axs[1, 0].set_ylabel("G Mag")
axs[1, 0].set_xlabel("Bp - Rp")
axs[1, 1].set_xlabel("Bp - Rp")
# axs[2, 1].set_xlim(0.5, 1)

# axs[2, 1].set_ylim(np.min(data[:, 4][true_labels_test]), 20)

# Column titles
axs[0, 0].set_title("Observed Data")
axs[0, 1].set_title("Simulated Data")

# invert the axes for magnitude
axs[1, 0].invert_yaxis()
axs[1, 1].invert_yaxis()


# Optional: remove redundant y-axis labels for the right column
for i in range(2):
    axs[i, 1].tick_params(labelleft=False)

# Adjust layout
fig.tight_layout()
# plt.savefig('Observed_vs_Simulated_Data_PZFlow.png')
plt.show()

# %%
samples_df

# %%
full_embeddings = scaler.inverse_transform(full_embeddings)
signal_df = pd.DataFrame(data=full_embeddings[signal_region], columns = col_names)

# %%
signal_stream = stream[signal_region]
signal_df['stream'] = signal_stream.astype(int)
signal_df['CWoLa_Label'] = np.ones_like(signal_stream).astype(int)
signal_df['source_id'] = source_id[signal_region]

samples_df['stream'] = np.ones(len(samples_df)).astype(int) * 2
samples_df['CWoLa_Label'] = np.zeros(len(samples_df)).astype(int)
samples_df['source_id'] = np.ones(len(samples_df)) * -1

# %%
full_df = pd.concat([signal_df, samples_df], ignore_index=True)
full_df.head()

# %%
full_df.to_csv('GD1_errs_LS_CATHODE_4HL_128_HD_16K_3e4LR_200epochs.csv', index=False)


