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

# %%
import importlib
import utils
importlib.reload(utils)

# Re-import the function to reflect changes/
from utils import count_parameters, get_mask_splits, DeeperGCN, LitDeepGCN, GaiaDataModule, GaiaDataModuleCustom, GaiaDataModuleC19, CATHODELinearDataModule, LitLinearModel
from utils import mc_marginal_bce_loss, EM_CATHODEGaiaDatasetLinear, EM_CATHODELinearDataModule, EM_LitLinearModel, LinearModel

# %%
import torch, numpy as np
from torch import nn

def unit_test_mc_equals_bce():
    B, D, N_mc = 8, 5, 10
    torch.manual_seed(0)
    x = torch.randn(B, D)
    # Make errors zero
    errors = torch.zeros_like(x)
    # Small MLP that supports leading dims:
    model = LinearModel(input_dim=D)
    # create identical samples
    x_samples = x.unsqueeze(1) + torch.randn(B, N_mc, D) * errors.unsqueeze(1)  # zeros => identical
    y_pred = model(x_samples).squeeze(-1).permute(1, 0)  # shape (N_mc, B)
    labels = (torch.rand(B) > 0.5).float()

    pos_weight = torch.tensor(4)
    # compute mc loss
    mc_loss = mc_marginal_bce_loss(y_pred, labels, pos_weight=pos_weight).item()

    # compute deterministic BCE on single input (should be equal)
    logits_single = model(x).squeeze(-1)
    bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    det_loss = bce(logits_single, labels).item()

    print("mc_loss:", mc_loss, "det_loss:", det_loss)
    assert np.allclose(mc_loss, det_loss, atol=1e-6), "MC loss != deterministic BCE when errors=0"

unit_test_mc_equals_bce()


# %%
# python
import copy, torch, numpy as np
torch.manual_seed(0)
# create model copies with identical init
D = 9
model_det = LinearModel(input_dim=D, num_layers=3, hidden_units=64, dropout=0.0)
model_em  = copy.deepcopy(model_det)
opt_det = torch.optim.SGD(model_det.parameters(), lr=1e-2)
opt_em  = torch.optim.SGD(model_em.parameters(), lr=1e-2)

# one batch (errors zero)
x = torch.randn(16, D)
errors = torch.zeros_like(x)
labels = (torch.rand(16) > 0.5).float()

# deterministic forward/backward
logits_det = model_det(x).squeeze(-1)
loss_det = torch.nn.BCEWithLogitsLoss()(logits_det, labels)
loss_det.backward()
opt_det.step()
opt_det.zero_grad()

# EM forward/backward with N_mc samples but errors zero -> should match
N_mc = 10
x_samples = x.unsqueeze(1) + torch.randn(16, N_mc, D) * errors.unsqueeze(1)  # identical
logits_em = model_em(x_samples).squeeze(-1).permute(1,0)  # (N_mc, B)
loss_em = mc_marginal_bce_loss(logits_em, labels)
loss_em.backward()
opt_em.step()
opt_em.zero_grad()

# Compare parameter differences
def param_vec(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()]).numpy()

print("det change norm:", np.linalg.norm(param_vec(model_det) - param_vec(model_em)))
# They should be very close (near zero) if implementations agree

# %%
seed_everything(12345)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

print(torch.cuda.get_device_properties(device)) 

# %%
df = pd.read_csv(load_data_dir)

df1 = df[df["CWoLa_Label"] == 1]
df0 = df[df["CWoLa_Label"] == 0]

# Sample exactly 1/4 of the zeros
n0 = len(df0)
k = n0 // 4  # or math.ceil(n0/4)
df0_sampled = df0.sample(n=k, random_state=12345)
# Concatenate
df_new = pd.concat([df1, df0_sampled], axis=0).reset_index(drop=True)

# %%
df_new.to_csv('GD1_errs_LS_CATHODE_4HL_128_HD_16K_3e4LR_200epochs_1to1.csv')

# %%
name = 'EM_256HU_10N_50wiggle'

# %%
stream = 'GD1'

# %%
load_data_dir = 'GD1_errs_LS_CATHODE_4HL_128_HD_16K_3e4LR_200epochs_1to1.csv'

batch_size = 25000

data_module = EM_CATHODELinearDataModule(name = '', stream = stream, load_data_dir = load_data_dir, batch_size=batch_size, p_wiggle = .5)

data_module.setup('fit')

# %%
test_loader = torch.load(f'/pscratch/sd/p/XXXXa/{stream}_CATHODE_mlp/linear_test_loader_{name}.pth', weights_only=False)
for batch in test_loader:
    print(batch[1][:, 0].shape)
    break

# %%
num_sig, num_bkgd = 0, 0
test_loader = torch.load(f'/pscratch/sd/p/XXXXa/{stream}_CATHODE_mlp/linear_test_loader_{name}.pth', weights_only=False)
for batch in test_loader:
    _, y, _, _ = batch
    y_cwola = y[:, 0]
    num_sig += torch.sum(y_cwola.bool())
    num_bkgd += torch.sum(~y_cwola.bool())
# pos_weight = torch.tensor([num_bkgd / num_sig]).to(device)
pos_weight = torch.tensor([num_bkgd / num_sig])
print(torch.tensor([num_bkgd / num_sig]))
pos_weight = None
print(num_sig + num_bkgd)
print(pos_weight)
steps_per_epoch = len(data_module.train_dataloader())

# %%
wandb.finish()
wandb_logger = WandbLogger(log_model="all", name = name, project= f'{stream} Gaia x Decals CATHODE')

checkpoint_callback = ModelCheckpoint(
            dirpath=f'{os.environ['PSCRATCH']}/model_checkpoints/{name}',  # Directory where checkpoints will be saved
            filename='{epoch}', # Template for checkpoint names
            every_n_epochs=1, # Save every n epochs
            save_top_k=-1
        )

# %%
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

early_stop_callback = EarlyStopping(monitor="True validation f1 score (0.8 thresh)", min_delta=0.00, patience=35, verbose=False, mode="max", strict=True)

# %%
import importlib
import utils
importlib.reload(utils)

# Re-import the function to reflect changes
from utils import count_parameters, get_mask_splits, DeeperGCN, LitDeepGCN, GaiaDataModule, GaiaDataModuleCustom, GaiaDataModuleC19, CATHODELinearDataModule, LitLinearModel
from utils import mc_marginal_bce_loss, EM_CATHODEGaiaDatasetLinear, EM_CATHODELinearDataModule, EM_LitLinearModel, LinearModel

# %%

model = EM_LitLinearModel(input_dim=9, lr=1e-2, EPOCHS=0, steps_per_epoch=steps_per_epoch,
                       pos_weight=pos_weight, num_layers=4, hidden_units=256, dropout=0.0, pct_start=0.1, 
                       num_mc_samples=50, weight_decay= 0.0, layer_norm=False, activation = 'relu', residual=False)
print(count_parameters(model))
print(name)

# %%
%%time
EPOCHS = 100
noise_anneal_dict = {"noise_anneal_min": 3.0, 'f_max':  3.0, 't_peak':  .9}

model = EM_LitLinearModel(input_dim=9, lr=1e-2, EPOCHS=EPOCHS, steps_per_epoch=steps_per_epoch,
                       pos_weight=pos_weight, num_layers=4, hidden_units=256, dropout=0.0, pct_start=0.1, 
                       num_mc_samples=10, weight_decay= 0.0, layer_norm=False, activation = 'relu', residual=False,
                       anneal_noise=False, noise_anneal_type = 'cosine', noise_anneal_dict = noise_anneal_dict)

trainer = L.Trainer(accelerator='gpu', max_epochs=EPOCHS, log_every_n_steps=1,
                    precision='16-mixed', logger = wandb_logger,
                    callbacks=[checkpoint_callback, early_stop_callback], fast_dev_run=False)



trainer.fit(model=model, datamodule = data_module)

# %%
wandb.finish()

# %%
# name = 'MLP_512HU_5HL_50drop'
# name = 'EM_MLP_256HU_3HL_silu_skip_1e3lr_1N'
name = 'EM_384HU_10N_50wiggle_2'
model = EM_LitLinearModel.load_from_checkpoint(f'/pscratch/sd/p/XXXXa/model_checkpoints/{name}/epoch=25.ckpt')
model.eval()


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
test_loader_dummy = torch.load(f'/pscratch/sd/p/XXXXa/{stream}_CATHODE_mlp/linear_test_loader_{name}.pth',
                               weights_only=False)
val_loader_dummy = torch.load(f'/pscratch/sd/p/XXXXa/{stream}_CATHODE_mlp/linear_val_loader_{name}.pth',
                                 weights_only=False)
num_samples_inference = 100
with torch.inference_mode():
    model.eval()
    model.to(torch.float32)
    pred_labels_test = []
    classes = []
    probs_test = []
    for i, sub_data in enumerate(test_loader_dummy):

        x, y, errs_in, _ = sub_data

        B, D = x.shape
        # I applied data augmentation during training by adding fake noise to every features except for 3rd and 4th features
        # I want to zero out all errors except for the 3rd and 4th features (i.e., only RA and Dec errors are non-zero)

        zs = np.zeros(B).astype('float64')
        errors = torch.zeros_like(x)
        errors[:, 2] = errs_in[:, 2]
        errors[:, 3] = errs_in[:, 3]
        # errors = torch.tensor(np.column_stack((zs, zs, errors[:, 2], errors[:, 3], zs, zs, zs, zs, zs)))
        
        # x shape: (B, D)
        # errors shape: (B, D)
        # Sample from the error distributions, num_mc_samples times
        x_samples = x.unsqueeze(1) + torch.randn(B, num_samples_inference, D).to(x.device) * errors.unsqueeze(1)  # Shape: (B, num_mc_samples, D)
        # print(x_samples)
        # print(f'x_samples shape: {x_samples.shape}')
        pred_logits_test = model.model.to(device)(x_samples.to(device)).squeeze()
        # y_pred shape: (B, num_mc_samples)
        # print(f'y_pred shape before permute: {y_pred.shape}')
        
        # pred_logits_test = pred_logits_test.mean(dim=1)  # Shape: (B,) - old version
    
        y_cwola = y[:, 0]
        y_true = y[:, 1]
        sig = nn.Sigmoid()
        # pred_probs_test = sig(pred_logits_test).to('cpu').numpy()
        pred_probs_test = sig(pred_logits_test).mean(dim=1).to('cpu').numpy()
        
        pred_labels_test.extend((pred_probs_test >= 0.5).astype(int))
        classes.extend(y_cwola.numpy())
        probs_test.extend(pred_probs_test)


with torch.inference_mode():
    model.eval()
    model.to(torch.float32)
    pred_labels_val = []

    probs_val = []
    for i, sub_data in enumerate(val_loader_dummy):

        x, y, errs_in, _ = sub_data
        B, D = x.shape

        errors = torch.zeros_like(x)
        errors[:, 2] = errs_in[:, 2]
        errors[:, 3] = errs_in[:, 3]
        # zs = np.zeros(B).astype('float64')
        # errors = torch.tensor(np.column_stack((zs, zs, errors[:, 2], errors[:, 3], zs, zs, zs, zs, zs)))
        
        # x shape: (B, D)
        # errors shape: (B, D)
        # Sample from the error distributions, num_mc_samples times
       
        x_samples = x.unsqueeze(1) + torch.randn(B, num_samples_inference, D).to(x.device) * errors.unsqueeze(1)  # Shape: (B, num_mc_samples, D)
        # print(f'x_samples shape: {x_samples.shape}')
        pred_logits_val = model.model.to(device)(x_samples.to(device)).squeeze()
        # y_pred shape: (B, num_mc_samples)
        # print(f'y_pred shape before permute: {y_pred.shape}')
        # pred_logits_val = pred_logits_val.mean(dim=1)  # Shape: (B,) - old version
    
        y_cwola = y[:, 0]
        y_true = y[:, 1]
        sig = nn.Sigmoid()
        # pred_probs_val = sig(pred_logits_val).to('cpu').numpy() - old version
        pred_probs_val = sig(pred_logits_val).mean(dim=1).to('cpu').numpy()
        pred_labels_val.extend((pred_probs_val >= 0.5).astype(int))
        
        probs_val.extend(pred_probs_val)

probs_test = np.array(probs_test)
probs_val = np.array(probs_val)
preds_test = np.array(pred_labels_test).flatten()
preds_val = np.array(pred_labels_val).flatten()
print(f"Total Signal predictions: {np.sum(preds_test)}")

classes = np.array(classes)
print(f"Number of Signal stars: {np.sum(classes)}")

results = (preds_test == classes)
accuracy = np.mean(results)

print(f"Accuracy: {accuracy}")
print(f'recall score   : {recall_score(classes, preds_test):.5f}')
print(f'precision score: {precision_score(classes, preds_test):.5f}')
print(f'f1 score       : {f1_score(classes, preds_test):.5f}')

# %%


# %%
test_loader_dummy = torch.load(f'/pscratch/sd/p/XXXXa/{stream}_CATHODE_mlp/linear_test_loader_{name}.pth',
                               weights_only=False)
val_loader_dummy = torch.load(f'/pscratch/sd/p/XXXXa/{stream}_CATHODE_mlp/linear_val_loader_{name}.pth',
                                 weights_only=False)

scaled_data = []
val_mask = []
test_mask = []
mixed_labels_test = []
true_labels_test = []
mixed_labels_val = []
true_labels_val = []

sampled_data_full = []
pm_ra_errors = []
pm_dec_errors = []
for batch in test_loader_dummy:
    x, y, errors, sampled_data = batch
    # print(~sampled_data.numpy().astype(bool))
    sampled_data_full.extend(~(sampled_data.numpy().astype(bool)))

    scaled_data.extend(x.numpy())
    mixed_labels_test.extend(y[:, 0].numpy())
    true_labels_test.extend(y[:, 1].numpy())
    pm_ra_errors.extend(errors[:, 2])
    pm_dec_errors.extend(errors[:, 3])
pm_ra_errors, pm_dec_errors = np.array(pm_ra_errors), np.array(pm_dec_errors)

scaled_data = np.array(scaled_data)
scaled_data = scaled_data[sampled_data_full]

pm_ra_errors, pm_dec_errors = pm_ra_errors[sampled_data_full], pm_dec_errors[sampled_data_full]

scaler = data_module.scaler
scaled_data = scaler.inverse_transform(scaled_data)

mixed_labels_test = np.array(mixed_labels_test)[sampled_data_full]
mixed_mask_test = mixed_labels_test.astype(bool)

true_labels_test = np.array(true_labels_test)[sampled_data_full]
true_mask_test = true_labels_test.astype(bool)

preds_bool_test = preds_test.astype(bool)[sampled_data_full]
probs_test = probs_test[sampled_data_full]

sampled_data_val = []
for batch in val_loader_dummy:
    x, y, errors, sampled_data = batch
    mixed_labels_val.extend(y[:, 0].numpy())
    true_labels_val.extend(y[:, 1].numpy())
    sampled_data_val.extend(~(sampled_data.numpy().astype(bool)))

mixed_labels_val = np.array(mixed_labels_val)[sampled_data_val]
mixed_mask_val = mixed_labels_val.astype(bool)
true_labels_val = np.array(true_labels_val)[sampled_data_val]
true_mask_val = true_labels_val.astype(bool)
preds_bool_val = preds_val.astype(bool)[sampled_data_val]
probs_val = probs_val[sampled_data_val]

test_ra, test_dec = scaled_data[:, 0], scaled_data[:, 1]
test_pm_ra, test_pm_dec = scaled_data[:, 2], scaled_data[:, 3]
test_gmag, test_color = scaled_data[:, 4], scaled_data[:, 5]
test_rmag0, test_g_r = scaled_data[:, 6], scaled_data[:, 7]
test_r_z = scaled_data[:, 8]

print('Done loading data.')

# %%
fpr, tpr, thresholds = roc_curve(true_labels_val, probs_val)
#plt.scatter(fpr, tpr, c=thresholds, cmap='Reds')
plt.plot(fpr, tpr)
plt.plot(np.linspace(0, 1, 1000),np.linspace(0, 1, 1000), ls='--', c='grey')
plt.xlabel("False Positive Rate [FP / (FP + TN)]")
plt.ylabel("True Positive Rate [TP / (TP + FN)]")
plt.title('ROC Curve')

distances = np.sqrt((fpr)**2 + (1 - tpr)**2)
optimal_idx = np.argmin(distances)
optimal_threshold = thresholds[optimal_idx]

print(f"Threshold closest to (0,1): {optimal_threshold:.4f}")
print(f"FPR: {fpr[optimal_idx]:.4f}, TPR: {tpr[optimal_idx]:.4f}")
auc_score = roc_auc_score(true_labels_val, probs_val)
print(f"ROC AUC Score: {auc_score}")

# %%
precision, recall, thresholds = precision_recall_curve(true_labels_val, probs_val)
ap = average_precision_score(true_labels_val, probs_val)

distances = np.sqrt((1 - precision[:-1])**2 + (1 - recall[:-1])**2)
optimal_idx = np.argmin(distances) 
optimal_threshold = thresholds[optimal_idx]

print(f"Threshold closest to (1,1): {optimal_threshold:.4f}")
print(f"Precision: {precision[optimal_idx]:.4f}, Recall: {recall[optimal_idx]:.4f}")

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, label=f'PR curve (AP = {ap:.3f})', color='k')
# Plot an X where the optimal threshold is
plt.scatter(recall[optimal_idx], precision[optimal_idx], color='red', label='Optimal Threshold', marker='x', s=100)
plt.axhline(y=0.754, xmax=0.39, color='green', linestyle='dotted', linewidth=2, label='Best Result in Literature')
plt.axvline(x=0.379, ymax=0.73, color='green', linestyle='dotted', linewidth=2)

# I want to calculate maximum recall at 0.754 precision
max_recall_at_precision = recall[np.argmax(precision >= 0.754)]
print(f"Maximum Recall at 0.754 Precision: {max_recall_at_precision:.4f}")

plt.axvline(x=max_recall_at_precision, ymax=0.754, color='red', linestyle='dashed', linewidth=1, label='Model Recall at .754 Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(fontsize=10)
plt.show()

# %%
# Set plotting parameters
import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.linewidth': 1.0
})

# %%
# Calculate the confusion matrix
threshold = optimal_threshold
#.955
threshold = 0.96
preds_threshold = (probs_test >= threshold).astype(int)
preds_threshold_bool = preds_threshold.astype(bool)
cm = confusion_matrix(true_labels_test, preds_threshold, normalize=None)

print(f'recall score   : {recall_score(true_labels_test, preds_threshold):.5f}')
print(f'precision score: {precision_score(true_labels_test, preds_threshold):.5f}')
print(f'f1 score       : {f1_score(true_labels_test, preds_threshold):.5f}')

# Visualize the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt = 'd' , cmap='Blues', 
            xticklabels=['Background', 'Stream'], 
            yticklabels=['Background', 'Stream'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Cwola Classifier Confusion Matrix')

plt.show()

# %%
threshold = 0.96
preds_threshold = (probs_test >= threshold).astype(int)

# ---------------------------
# Metrics
# ---------------------------
recall = recall_score(true_labels_test, preds_threshold)
precision = precision_score(true_labels_test, preds_threshold)
f1 = f1_score(true_labels_test, preds_threshold)

print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1 score  : {f1:.4f}")


cm_pred_norm = confusion_matrix(
    true_labels_test,
    preds_threshold,
    normalize="pred"
)

# ---------------------------
# Plot styling
# ---------------------------


fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    cm_pred_norm,
    annot=True,
    fmt=".2%",
    cmap="Blues",
    cbar=True,
    annot_kws={"fontsize": 30},
    cbar_kws={"label": "Fraction of predicted class"},
    square=True,
    linewidths=0.5,
    linecolor="white",
    xticklabels=["Background", "Stream"],
    yticklabels=["Background", "Stream"],
    vmin=0,
    vmax=1,
    ax=ax
)
ax.tick_params(axis="both", labelsize=22)

ax.set_xlabel("Predicted label", fontsize=26)
ax.set_ylabel("SF label", fontsize=26)
# ax.set_title("CATHODE Classifier Confusion Matrix")

# plt.tight_layout()
plt.savefig("report_plots//confusion_matrix.png", bbox_inches="tight", dpi=600, pad_inches = 0)
plt.show()

# %%
plt.hist(probs_test, density=True, bins = 100)
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1)
plt.yscale('log')
plt.plot()

# %%
tp = (true_mask_test & preds_threshold_bool)
fp = (~true_mask_test & preds_threshold_bool)
fn = (true_mask_test & ~preds_threshold_bool)

# %%
e_fp = 15/17
e_fn = 25/54

# %%
np.sum(tp) / (np.sum(tp) + np.sum(fp)), (np.sum(tp) + e_fp * np.sum(fp)) / (np.sum(tp) + np.sum(fp))

# %%
np.sum(tp) / (np.sum(tp) + np.sum(fn)), (np.sum(tp) + e_fn * np.sum(fn)) / (np.sum(tp) + np.sum(fn))

# %%
preds_threshold_bool = (preds_threshold_bool) & ((test_pm_dec < -1) & (test_pm_dec > -6))

# %%
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist2d(test_ra, test_dec, bins=100, cmap='gray_r', alpha=.5)#, label='Background')

ax1.scatter(test_ra[true_mask_test], test_dec[true_mask_test], s=75, color = '#ff7f0e')#, label='GD1')
ax1.quiver(test_ra[true_mask_test], test_dec[true_mask_test], test_pm_ra[true_mask_test], test_pm_dec[true_mask_test], angles='xy', 
           scale_units='xy', scale=3, color='#000000', width=0.003, alpha=.7)


ax2.hist2d(test_ra[~preds_threshold_bool], test_dec[~preds_threshold_bool], bins=100, cmap='gray_r', alpha=.5)         
ax2.scatter(test_ra[preds_threshold_bool], test_dec[preds_threshold_bool], s=75)#, label = 'GD1')

ax2.quiver(test_ra[preds_threshold_bool], test_dec[preds_threshold_bool], 
           test_pm_ra[preds_threshold_bool], test_pm_dec[preds_threshold_bool], 
           angles='xy', scale_units='xy', scale=3, color='#000000', width=0.003, alpha=.3)

ax1.tick_params(axis='both', labelsize=16)
ax2.tick_params(axis='both', labelsize=16)
ax1.set_xlabel(r'$\Phi$1', fontsize = 35)
ax1.set_ylabel(r'$\Phi$2', fontsize = 35)
ax2.set_xlabel(r'$\Phi$1', fontsize = 35)
ax2.set_ylabel(r'$\Phi$2', fontsize = 35)

fig.set_size_inches(20, 10)
ax1.set_title('True Labels', fontsize=50)
ax2.set_title('Predicted Labels', fontsize=50)

plt.tight_layout()
# plt.savefig('C19_Plots/True_vs_Preds_Linear_cwola.png', dpi=600)
plt.show()

# %%
pm_error = (pm_ra_errors ** 2 + pm_dec_errors ** 2) ** (.5)

# %%
from scipy.optimize import curve_fit

def line(x, m, b):
    return m*x + b
err_mask = pm_error > -0.01
popt, covar = curve_fit(line, probs_test[err_mask], pm_error[err_mask])

# %%
m, b = popt

# %%
plt.scatter(probs_test, pm_error, s=1)
xs = np.arange(np.min(probs_test), np.max(probs_test), .01)
plt.plot(xs, m*xs + b, color = 'r')
plt.xlabel('Model Output')
plt.ylabel('Proper Motion Error')

# %%
np.median(pm_error[(preds_threshold_bool) & (pm_error > 0.05)]), np.median(pm_error[(~preds_threshold_bool)  & (pm_error > 0.05)])

# %%


# %%
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(test_pm_ra, test_pm_dec, s=1, color = 'gray', alpha=.5)
ax1.scatter(test_pm_ra[true_mask_test], test_pm_dec[true_mask_test], s=30, color = '#ff7f0e')

ax2.scatter(test_pm_ra[~preds_threshold_bool], test_pm_dec[~preds_threshold_bool], s=1, color = 'gray', alpha=.5)
ax2.scatter(test_pm_ra[preds_threshold_bool], test_pm_dec[preds_threshold_bool], s=30)
fig.set_size_inches(20, 10)
# ax1.set_ylim(-10, 3)
# ax2.set_ylim(-10, 3)
ax1.set_title('True Labels')
ax2.set_title('Predicted Labels')

# %%

# Create the subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Left: True labels
ax1.hist2d(test_color, test_gmag, bins=100, cmap='gray_r', alpha=0.5)
ax1.scatter(test_color[true_mask_test], test_gmag[true_mask_test], s=30, color='#ff7f0e')
ax1.set_title('True Labels', fontsize=50)
ax1.set_xlabel('Bp - Rp', fontsize=35)
ax1.set_ylabel('G Magnitude', fontsize=35)
ax1.tick_params(axis='both', which='major', labelsize=16)

# Right: Predicted labels
ax2.hist2d(test_color[~preds_threshold_bool], test_gmag[~preds_threshold_bool], bins=100, cmap='gray_r', alpha=0.5)
ax2.scatter(test_color[preds_threshold_bool], test_gmag[preds_threshold_bool], s=30, color='#1f77b4')
ax2.set_title('Predicted Labels', fontsize=50)
ax2.set_xlabel('Bp - Rp', fontsize=35)
ax2.set_ylabel('G Magnitude', fontsize=35)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax1.invert_yaxis()
ax2.invert_yaxis()


plt.tight_layout()


# plt.savefig('C19_Plots/True_vs_Predicted_CMD_Presentation.png', dpi=600)

plt.show()


# %%

# Create the subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Left: True labels
ax1.hist2d(test_g_r, test_rmag0, bins=100, cmap='gray_r', alpha=0.5)
ax1.scatter(test_g_r[true_mask_test], test_rmag0[true_mask_test], s=30, color='#ff7f0e')
ax1.set_title('True Labels', fontsize=50)
ax1.set_xlabel('G - R', fontsize=35)
ax1.set_ylabel('R Magnitude', fontsize=35)
ax1.tick_params(axis='both', which='major', labelsize=16)

# Right: Predicted labels
ax2.hist2d(test_g_r[~preds_threshold_bool], test_rmag0[~preds_threshold_bool], bins=100, cmap='gray_r', alpha=0.5)
ax2.scatter(test_g_r[preds_threshold_bool], test_rmag0[preds_threshold_bool], s=30, color='#1f77b4')
ax2.set_title('Predicted Labels', fontsize=50)
ax2.set_xlabel('G - R', fontsize=35)
ax2.set_ylabel('R Magnitude', fontsize=35)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax1.invert_yaxis()
ax2.invert_yaxis()
ax1.set_xlim(0, 1)
ax2.set_xlim(0, 1)
plt.tight_layout()


# plt.savefig('C19_Plots/True_vs_Predicted_CMD_Presentation.png', dpi=600)

plt.show()


# %%
colors = ["#15097C", "#bca006","#c50b27"]
markers = ['o', 's', 'D']

fig, (ax1) = plt.subplots(1, 1)
ax1.hist2d(test_ra[~true_mask_test], test_dec[~true_mask_test], bins=100, cmap='gray_r', alpha=.5)#, label='Background')

ax1.scatter(test_ra[tp], test_dec[tp], s=40, alpha=.6, label = 'True Positives',color = colors[0], marker=markers[0]) 
ax1.scatter(test_ra[fp], test_dec[fp], s=40, alpha=.6, label = 'False Positives',color = colors[1], marker=markers[1])
ax1.scatter(test_ra[fn], test_dec[fn], s=40, alpha=.6, label = 'False Negatives',color = colors[2], marker=markers[2])


# ax1.quiver(test_ra[true_mask_test], test_dec[true_mask_test], test_pm_ra[true_mask_test], test_pm_dec[true_mask_test], angles='xy', 
#            scale_units='xy', scale=3, color='#000000', width=0.003, alpha=.7)



ax1.tick_params(axis='both', labelsize=16)
ax1.set_xlabel(r'$\Phi$1', fontsize = 35)
ax1.set_ylabel(r'$\Phi$2', fontsize = 35)

fig.set_size_inches(10, 8)
# ax1.set_title('Predictions vs SF Labels', fontsize=45)
plt.legend(fontsize = 26)
# plt.tight_layout()
plt.savefig(f'report_plots/phi1_phi2_preds.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

# %%
fig, (ax1) = plt.subplots(1, 1)
ax1.scatter(test_pm_ra, test_pm_dec, s=1, color = 'gray', alpha=.5)
ax1.scatter(test_pm_ra[tp], test_pm_dec[tp], s=30, alpha = .75, label = 'True Positives')
ax1.scatter(test_pm_ra[fp], test_pm_dec[fp], s=30, alpha = .75, label = 'False Positives')
ax1.scatter(test_pm_ra[fn], test_pm_dec[fn], s=30, alpha = .75, label = 'False Negatives')

# Want to clip ax1 and ax2 y axes to 5, 95 percentiles of pm_dec values
ax1.set_ylim(np.percentile(test_pm_dec, 5), np.percentile(test_pm_dec, 95))

ax1.set_xlabel(r'$\mu_{\Phi 1}$', fontsize = 35)
ax1.set_ylabel(r'$\mu_{\Phi 2}$', fontsize = 35)
plt.legend(fontsize=20)
fig.set_size_inches(10, 8)
ax1.set_title('Predictions vs True Labels', fontsize=45)
# ax2.set_title('Predicted Labels')
# plt.savefig(f'GD1_Plots/TP_FP_FN_CATHODE_pmphi12{name}.png', dpi=600)
plt.show()

# %%
# Create the subplots
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))

# Left: True labels
ax1.hist2d(test_color[~true_mask_test], test_gmag[~true_mask_test], bins=100, cmap='gray_r', alpha=0.5)

ax1.scatter(test_color[tp], test_gmag[tp], s=40, alpha = .6, label='True Positive', color = colors[0], marker=markers[0])
ax1.scatter(test_color[fp], test_gmag[fp], s=40, alpha = .6, label='False Positive', color = colors[1], marker=markers[1])
ax1.scatter(test_color[fn], test_gmag[fn], s=40, alpha = .6, label='False Negative', color = colors[2], marker=markers[2])
# ax1.set_title('Predictions vs SF Labels', fontsize=45)
ax1.set_xlabel('BP - RP', fontsize=35)
ax1.set_ylabel('G', fontsize=35)
ax1.tick_params(axis='both', which='major', labelsize=16)

ax1.invert_yaxis()
ax1.set_xlim(0.2, 1)
# plt.tight_layout()
plt.legend(fontsize=26)

plt.savefig(f'report_plots/Gmag_BP_RP_preds.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()


# %%
# Create the subplots
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))

# Left: True labels
ax1.hist2d(test_g_r[~true_mask_test], test_rmag0[~true_mask_test], bins=200, cmap='gray_r', alpha=0.5)

ax1.scatter(test_g_r[tp], test_rmag0[tp], s=40, alpha = .6, label='True Positive', color = colors[0], marker=markers[0])
ax1.scatter(test_g_r[fp], test_rmag0[fp], s=40, alpha = .6, label='False Positive', color = colors[1], marker=markers[1])
ax1.scatter(test_g_r[fn], test_rmag0[fn], s=40, alpha = .6, label='False Negative', color = colors[2], marker=markers[2])

# ax1.set_title('Predictions vs SF Labels', fontsize=45)
ax1.set_xlabel('g$_0$-r$_0$', fontsize=35)
ax1.set_ylabel('r$_0$', fontsize=35)
ax1.tick_params(axis='both', which='major', labelsize=16)

ax1.invert_yaxis()
ax1.set_xlim(0.2, .6)
# plt.tight_layout()
plt.legend(fontsize=26)
# plt.savefig('C19_Plots/True_vs_Predicted_CMD_Presentation.png', dpi=600)
plt.savefig(f'report_plots/rmag_g_r_preds.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()


# %%
# Create the subplots
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))

# Left: True labels
ax1.hist2d(test_r_z[~true_mask_test], test_rmag0[~true_mask_test], bins=200, cmap='gray_r', alpha=0.5)

ax1.scatter(test_r_z[tp], test_rmag0[tp], s=40, alpha = .6, label='True Positive', color = colors[0], marker=markers[0])
ax1.scatter(test_r_z[fp], test_rmag0[fp], s=40, alpha = .6, label='False Positive', color = colors[1], marker=markers[1])
ax1.scatter(test_r_z[fn], test_rmag0[fn], s=40, alpha = .6, label='False Negative', color = colors[2], marker=markers[2])

# ax1.set_title('Predictions vs SF Labels', fontsize=45)
ax1.set_xlabel('r$_0$-z$_0$', fontsize=35)
ax1.set_ylabel('r$_0$', fontsize=35)
ax1.tick_params(axis='both', which='major', labelsize=16)

ax1.invert_yaxis()
ax1.set_xlim(-0.1, .3)
# plt.tight_layout()
plt.legend(fontsize=26)

plt.savefig(f'report_plots/rmag_r_z_preds.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()


# %%
def get_id_from_data(data_features, feature_names, data_path):
    """
    When I pass my data through the neural network, I lose track of its ID number. 
    I pass only the input features through and so I can only recover the inputs & the output value.
    This ID number is given by the "source_id" column in the data_path table. 
    What I want to do is be able to is take a (n_samples, 1) array of data points and return their original indices in the full dataset.
    USE np.intersect1d
    """

    t = Table.read(data_path)
    # print(t)
    # Now load in the equivalent features from df and the cross-match
    df_features = t[feature_names].to_pandas().values
    source_ids = np.array(t['source_id'])
    sig = np.array(t['signal_region'])

    df_features = df_features[sig]
    source_ids = source_ids[sig]
    # Find the distance to each row in data_features
    id_list = []

    for row in tqdm(data_features):
        dist = np.sqrt(np.sum((df_features - row)**2, axis=1))
        min_idx = np.argmin(dist)
        id_list.append(source_ids[min_idx])

    return id_list


# %%
t = Table.read('/pscratch/sd/p/XXXXa/GD-1_gaia_x_decals_stream_prep.fits')
df_features = t[['phi1', 'phi2', 'pm_phi1', 'pm_phi2']].to_pandas().values

# %%
test_ra

# %%
df_ra = np.array(t['phi1'])
df_dec = np.array(t['phi2'])
df_pmra = np.array(t['pm_phi1'])
df_pmdec = np.array(t['pm_phi2'])

# %%
test_ids = get_id_from_data(np.column_stack((test_ra, test_dec, test_pm_ra, test_pm_dec)), ['phi1', 'phi2', 'pm_phi1', 'pm_phi2'], '/pscratch/sd/p/XXXXa/GD-1_gaia_x_decals_stream_prep.fits')

# %%
# now use test_ids to get data from the original table
ids = np.array(t['source_id'])
pm_ra_error = np.array(t['pmra_error'])
pm_dec_error = np.array(t['pmdec_error'])


# Get the cross match indeces of ids and test_ids
id_indices = np.nonzero(np.isin(ids, test_ids))[0]

# %%
test_pm_ra_error = pm_ra_error[id_indices]
test_pm_dec_error = pm_dec_error[id_indices]

# %%
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist2d(test_ra, test_dec, bins=100, cmap='gray_r', alpha=.5)#, label='Background')

ax1.scatter(test_ra[true_mask_test], test_dec[true_mask_test], s=75, color = '#ff7f0e')#, label='GD1')

ax2.hist2d(test_ra[~preds_threshold_bool], test_dec[~preds_threshold_bool], bins=100, cmap='gray_r', alpha=.5)  
# Color these points by pmra_error and pmdec_error       
ax2.scatter(test_ra[fp], test_dec[fp], s=75, c=np.sqrt(test_pm_ra_error[fp] ** 2 + test_pm_dec_error[fp] ** 2), cmap='viridis')#, label = 'GD1')


ax1.tick_params(axis='both', labelsize=16)
ax2.tick_params(axis='both', labelsize=16)
ax1.set_xlabel(r'$\Phi$1', fontsize = 35)
ax1.set_ylabel(r'$\Phi$2', fontsize = 35)
ax2.set_xlabel(r'$\Phi$1', fontsize = 35)
ax2.set_ylabel(r'$\Phi$2', fontsize = 35)

fig.set_size_inches(20, 10)
ax1.set_title('True Labels', fontsize=50)
ax2.set_title('Predicted Labels', fontsize=50)
cbar = plt.colorbar(ax2.collections[1], ax=ax2)
cbar.set_label('PM Error (mas/yr)', fontsize=20)
plt.tight_layout()
# plt.savefig('C19_Plots/True_vs_Preds_Linear_cwola.png', dpi=600)
plt.show()

# %%
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(test_pm_ra, test_pm_dec, s=1, color = 'gray', alpha=.5)
ax1.scatter(test_pm_ra[true_mask_test], test_pm_dec[true_mask_test], s=30, color = '#ff7f0e')

ax2.scatter(test_pm_ra[~preds_threshold_bool], test_pm_dec[~preds_threshold_bool], s=1, color = 'gray', alpha=.5)
ax2.scatter(test_pm_ra[fp], test_pm_dec[fp], s=30, c=np.sqrt(test_pm_ra_error[fp] ** 2 + test_pm_dec_error[fp] ** 2), cmap='viridis')
cbar = plt.colorbar(ax2.collections[1], ax=ax2)
cbar.set_label('PM Error (mas/yr)', fontsize=20)
fig.set_size_inches(20, 10)
ax1.set_title('True Labels')
ax2.set_title('Predicted Labels')

# %%
# Create the subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Left: True labels
ax1.hist2d(test_color, test_gmag, bins=100, cmap='gray_r', alpha=0.5)
ax1.scatter(test_color[true_mask_test], test_gmag[true_mask_test], s=30, color='#ff7f0e')
ax1.set_title('True Labels', fontsize=50)
ax1.set_xlabel('Bp - Rp', fontsize=35)
ax1.set_ylabel('G Magnitude', fontsize=35)
ax1.tick_params(axis='both', which='major', labelsize=16)

# Right: Predicted labels
ax2.hist2d(test_color[~preds_threshold_bool], test_gmag[~preds_threshold_bool], bins=100, cmap='gray_r', alpha=0.5)
ax2.scatter(test_color[fp], test_gmag[fp], s=30, c = np.sqrt(test_pm_ra_error[fp] ** 2 + test_pm_dec_error[fp] ** 2), cmap='viridis')
ax2.set_title('Predicted Labels', fontsize=50)
ax2.set_xlabel('Bp - Rp', fontsize=35)
ax2.set_ylabel('G Magnitude', fontsize=35)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax1.invert_yaxis()
ax2.invert_yaxis()
plt.colorbar(ax2.collections[1], ax=ax2, label='PM Error (mas/yr)')

plt.tight_layout()


# plt.savefig('C19_Plots/True_vs_Predicted_CMD_Presentation.png', dpi=600)

plt.show()


# %%
XXXX_labels = Table.read('..//Data/gd1_members_gaia.fits')
XXXX_ids = np.array(XXXX_labels['source_id'])

# %%
XXXX_test_ids = np.isin(test_ids, XXXX_ids)

# %%
np.sum(XXXX_test_ids)

# %%
fig, 
plt.hist2d(test_ra, test_dec, bins=100, cmap='gray_r', alpha=.5)#, label='Background')
plt.scatter(test_ra[XXXX_test_ids], test_dec[XXXX_test_ids], s=4, color = '#1f77b4')

# %%
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist2d(test_ra, test_dec, bins=100, cmap='gray_r', alpha=.5)#, label='Background')
ax1.scatter(test_ra[true_mask_test], test_dec[true_mask_test], s=75, color = '#ff7f0e')#, label='GD1')

ax2.hist2d(test_ra[~XXXX_test_ids], test_dec[~XXXX_test_ids], bins=100, cmap='gray_r', alpha=.5)  
# Color these points by pmra_error and pmdec_error       
ax2.scatter(test_ra[XXXX_test_ids], test_dec[XXXX_test_ids], s=75, color = '#1f77b4')#, label = 'GD1')

ax1.tick_params(axis='both', labelsize=16)
ax2.tick_params(axis='both', labelsize=16)
ax1.set_xlabel(r'$\Phi$1', fontsize = 35)
ax1.set_ylabel(r'$\Phi$2', fontsize = 35)
ax2.set_xlabel(r'$\Phi$1', fontsize = 35)
ax2.set_ylabel(r'$\Phi$2', fontsize = 35)

fig.set_size_inches(20, 10)
ax1.set_title('Streamfinder Labels', fontsize=50)
ax2.set_title('XXXX Labels', fontsize=50)

plt.tight_layout()
# plt.savefig('C19_Plots/True_vs_Preds_Linear_cwola.png', dpi=600)
plt.show()

# %%
XXXX_test_ids

# %%
cm = confusion_matrix(XXXX_test_ids, preds_threshold, normalize=None)

print(f'recall score   : {recall_score(XXXX_test_ids, preds_threshold):.5f}')
print(f'precision score: {precision_score(XXXX_test_ids, preds_threshold):.5f}')
print(f'f1 score       : {f1_score(XXXX_test_ids, preds_threshold):.5f}')

# Visualize the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt = 'd' , cmap='Blues', 
            xticklabels=['Background', 'Stream'], 
            yticklabels=['Background', 'Stream'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Cwola Classifier Confusion Matrix')
    
plt.show()

# %%
# Save relevant data to a pandas dataframe
df = pd.DataFrame(np.column_stack((test_ids, test_ra, test_dec, test_pm_ra, test_pm_dec, pm_ra_errors,
                                    pm_dec_errors, test_color, test_gmag, test_rmag0, test_g_r, test_r_z, probs_test, true_labels_test)), 
                                    columns = ['source_id', 'phi1', 'phi2', 'pm_phi1', 'pm_phi2', 'pmra_error', 'pmdec_error',
                                               'bp_rp', 'gmag', 'rmag0', 'g_r', 'r_z', 'model_prob', 'true_label'])


# %%
name

# %%
f'/pscratch/sd/p/XXXXa/{stream}_CATHODE_mlp/results_{name}.csv'

# %%
df.to_csv(f'/pscratch/sd/p/XXXXa/{stream}_CATHODE_mlp/results_{name}.csv', index=False)

# %% [markdown]
# ## phi1 vs X plots

# %%
# I want to make plots where I show the model outputs in the phi1 vs X space where X is each of the other features. Show this as a scatter plot
# The left panel should be the true labels and the right panel should be the model outputs

features = {r'PM $\Phi1$': test_pm_ra,
            r'PM $\Phi2$': test_pm_dec,}

# %%
feature_name = r'$\mu_1$'
feature_data = test_pm_ra

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(18, 10),
    sharex=True,
    constrained_layout=True
)

# ---------------------------
# Background density (both panels)
# ---------------------------
hist_kwargs = dict(
    bins=100,
    cmap="gray_r",
    alpha=0.4,
    rasterized=True
)

ax1.hist2d(test_ra, feature_data, **hist_kwargs)
ax2.hist2d(test_ra, feature_data, **hist_kwargs)

# ---------------------------
# Top panel: Streamfinder labels
# ---------------------------
ax1.scatter(
    test_ra[true_mask_test],
    feature_data[true_mask_test],
    s=40,
    c="tab:red",
    edgecolors="k",
    linewidths=0.4,
    alpha=0.9,
    zorder=3,
    label="Stream"
)

# ---------------------------
# Bottom panel: Model predictions
# ---------------------------
sc = ax2.scatter(
    test_ra[preds_threshold_bool],
    feature_data[preds_threshold_bool],
    s=40,
    c=probs_test[preds_threshold_bool],
    cmap="viridis",
    edgecolors="k",
    linewidths=0.4,
    alpha=0.9,
    zorder=3
)

# ---------------------------
# Axis formatting
# ---------------------------
for ax in (ax1, ax2):
    ax.tick_params(axis="both")
    ax.set_ylabel(feature_name, fontsize=30)

ax2.set_xlabel(r'$\Phi_1$', fontsize=30)

# ---------------------------
# Titles (with padding)
# ---------------------------
ax1.set_title("SF Labels", fontsize=40, pad=12)
ax2.set_title("Model Predictions", fontsize=40, pad=12)

# ---------------------------
# Colorbar (dedicated axis)
# ---------------------------
cbar = fig.colorbar(
    sc,
    ax=ax2,
    orientation="horizontal",
    pad=0.12,
    fraction=0.9
)

cbar.set_label("Model Probability", fontsize=24)
cbar.ax.tick_params(labelsize=16)
# ---------------------------
# Optional y-limits for PM Φ2
# ---------------------------
ax1.set_ylim(feature_data[true_mask_test].min()-.1, feature_data[true_mask_test].max()+.1)
ax2.set_ylim(feature_data[preds_threshold_bool].min()-.1, feature_data[preds_threshold_bool].max()+.1)

plt.savefig("report_plots//phi1_muphi1_track.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
feature_name = r'$\mu_2$'
feature_data = test_pm_dec

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(18, 10),
    sharex=True,
    constrained_layout=True
)

# ---------------------------
# Background density (both panels)
# ---------------------------
hist_kwargs = dict(
    bins=100,
    cmap="gray_r",
    alpha=0.4,
    rasterized=True
)

ax1.hist2d(test_ra, feature_data, **hist_kwargs)
ax2.hist2d(test_ra, feature_data, **hist_kwargs)

# ---------------------------
# Top panel: Streamfinder labels
# ---------------------------
ax1.scatter(
    test_ra[true_mask_test],
    feature_data[true_mask_test],
    s=40,
    c="tab:red",
    edgecolors="k",
    linewidths=0.4,
    alpha=0.9,
    zorder=3,
    label="Stream"
)

# ---------------------------
# Bottom panel: Model predictions
# ---------------------------
sc = ax2.scatter(
    test_ra[preds_threshold_bool],
    feature_data[preds_threshold_bool],
    s=40,
    c=probs_test[preds_threshold_bool],
    cmap="viridis",
    edgecolors="k",
    linewidths=0.4,
    alpha=0.9,
    zorder=3
)

# ---------------------------
# Axis formatting
# ---------------------------
for ax in (ax1, ax2):
    ax.tick_params(axis="both")
    ax.set_ylabel(feature_name, fontsize = 30)

ax2.set_xlabel(r'$\Phi_1$', fontsize=30)

# ---------------------------
# Titles (with padding)
# ---------------------------
ax1.set_title("SF Labels", pad=12, fontsize=40)
ax2.set_title("Model Predictions", pad=12, fontsize=40)

# ---------------------------
# Colorbar (dedicated axis)
# ---------------------------
cbar = fig.colorbar(
    sc,
    ax=ax2,
    orientation="horizontal",
    pad=0.12,
    fraction=0.9
)

cbar.set_label("Model Probability", fontsize=24)
cbar.ax.tick_params(labelsize=16)
# ---------------------------
# Optional y-limits for PM Φ2
# ---------------------------
ax1.set_ylim(feature_data[true_mask_test].min()-.1, feature_data[true_mask_test].max()+.1)
ax2.set_ylim(feature_data[preds_threshold_bool].min()-.1, feature_data[preds_threshold_bool].max()+.1)

plt.savefig("report_plots//phi1_muphi2_track.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
fig, axes = plt.subplots(
    2, 2,
    figsize=(22, 10),
    sharex=True,
    constrained_layout=True
)

# Axes unpacking for clarity
ax_sf_mu1, ax_sf_mu2 = axes[0]
ax_md_mu1, ax_md_mu2 = axes[1]

# ---------------------------
# Common background density
# ---------------------------
hist_kwargs = dict(
    bins=100,
    cmap="gray_r",
    alpha=0.4,
    rasterized=True
)

# ---------------------------
# μ1 panels
# ---------------------------
feature_data_mu1 = test_pm_ra

ax_sf_mu1.hist2d(test_ra, feature_data_mu1, **hist_kwargs)
ax_md_mu1.hist2d(test_ra, feature_data_mu1, **hist_kwargs)

ax_sf_mu1.scatter(
    test_ra[true_mask_test],
    feature_data_mu1[true_mask_test],
    s=40,
    c="tab:red",
    edgecolors="k",
    linewidths=0.4,
    alpha=0.9,
    zorder=3
)

sc = ax_md_mu1.scatter(
    test_ra[preds_threshold_bool],
    feature_data_mu1[preds_threshold_bool],
    s=40,
    c=probs_test[preds_threshold_bool],
    cmap="viridis",
    edgecolors="k",
    linewidths=0.4,
    alpha=0.9,
    zorder=3
)

# ---------------------------
# μ2 panels
# ---------------------------
feature_data_mu2 = test_pm_dec

ax_sf_mu2.hist2d(test_ra, feature_data_mu2, **hist_kwargs)
ax_md_mu2.hist2d(test_ra, feature_data_mu2, **hist_kwargs)

ax_sf_mu2.scatter(
    test_ra[true_mask_test],
    feature_data_mu2[true_mask_test],
    s=40,
    c="tab:red",
    edgecolors="k",
    linewidths=0.4,
    alpha=0.9,
    zorder=3
)

ax_md_mu2.scatter(
    test_ra[preds_threshold_bool],
    feature_data_mu2[preds_threshold_bool],
    s=40,
    c=probs_test[preds_threshold_bool],
    cmap="viridis",
    edgecolors="k",
    linewidths=0.4,
    alpha=0.9,
    zorder=3
)

# ---------------------------
# Axis formatting
# ---------------------------
for ax in axes.flat:
    ax.tick_params(axis="both", labelsize=18)

ax_sf_mu1.set_ylabel(r"$\mu_1$", fontsize=30)
ax_md_mu1.set_ylabel(r"$\mu_1$", fontsize=30)
ax_sf_mu2.set_ylabel(r"$\mu_2$", fontsize=30)
ax_md_mu2.set_ylabel(r"$\mu_2$", fontsize=30)

ax_md_mu1.set_xlabel(r"$\Phi_1$", fontsize=26)
ax_md_mu2.set_xlabel(r"$\Phi_1$", fontsize=26)

# ---------------------------
# Titles
# ---------------------------
ax_sf_mu1.set_title("SF Labels", fontsize=40, pad=10)
ax_sf_mu2.set_title("SF Labels", fontsize=40, pad=10)
ax_md_mu1.set_title("Model Predictions", fontsize=40, pad=10)
ax_md_mu2.set_title("Model Predictions", fontsize=40, pad=10)

# ---------------------------
# Shared colorbar (single)
# ---------------------------
# cbar = fig.colorbar(
#     sc,
#     ax=axes[1, :],
#     orientation="horizontal",
#     pad=0.12,
#     fraction=0.08
# )

cbar = fig.colorbar(
    sc,
    ax=axes,
    orientation="vertical",
    pad=0.02,
    fraction=0.035
)


cbar.set_label("Model Probability", fontsize=26)
cbar.ax.tick_params(labelsize=18)

# ---------------------------
# Limits (recommended: shared per column)
# ---------------------------
ax_sf_mu1.set_ylim(
    feature_data_mu1[true_mask_test].min() - 0.1,
    feature_data_mu1[true_mask_test].max() + 0.1
)
ax_md_mu1.set_ylim(ax_sf_mu1.get_ylim())

ax_sf_mu2.set_ylim(
    feature_data_mu2[true_mask_test].min() - 0.1,
    feature_data_mu2[true_mask_test].max() + 0.1
)
ax_md_mu2.set_ylim(ax_sf_mu2.get_ylim())

# ---------------------------
# Save
# ---------------------------
plt.savefig(
    "report_plots/phi1_mu_tracks_combined.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()


# %%
# I want to plot the stream linear density as a function of phi1
# Show the true labels in top panel and the model predictions (using preds_threshold_bool mask) in the bottom panel
# Show only a KDE smoothed histogram

bin_edges = np.linspace(-50, 50, 10)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1])
# Share x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
# Top panel: true labels
true_stream_phi1 = test_ra[true_mask_test]

bw_adjust = 0.35

sns.kdeplot(true_stream_phi1, bw_adjust=bw_adjust, fill=True, ax=ax1, color='r')
ax1.set_xlabel(r'$\Phi$1', fontsize=35)
ax1.set_ylabel('Linear Density', fontsize=35)
ax1.set_title('True Stream Members', fontsize=50)

# Bottom panel: model predictions
pred_stream_phi1 = test_ra[preds_threshold_bool]

sns.kdeplot(pred_stream_phi1, bw_adjust=bw_adjust, fill=True, ax=ax2, color='b')
ax2.set_xlabel(r'$\Phi$1', fontsize=35)
ax2.set_ylabel('Linear Density', fontsize=35)
ax2.set_title('Model Predicted Members', fontsize=50)
plt.tight_layout()


# %%


# %%



