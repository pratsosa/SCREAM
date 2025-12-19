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
from torch.utils.data import DataLoader
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
from torch_geometric.loader import ClusterData, ClusterLoader
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

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold


def count_parameters(model):
    """
    Function which returns the total number of parameters in a given model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def get_mask_splits(embeddings, train_pct):
    """
    Function which splits given data into Train, Validation, and Test sets. 
    embeddings: Data to be split (only used in order to get the shape)
    train_pct: Percentage of the data to use for training (20% of training data will be allocated to validation)
    """
    N = embeddings.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    split_idx1, split_idx2 = int(N * train_pct * .2), int(N * train_pct)
    val_indices = indices[:split_idx1]
    train_indices = indices[split_idx1:split_idx2]
    test_indices = indices[split_idx2:]
    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)
    val_mask[val_indices] = True
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, val_mask, test_mask



class DeeperGCN(torch.nn.Module):
    """
    A Pytorch Module which implements the DeeperGCN architecture.
    Implementation adopted from 
    """
    def __init__(self, hidden_channels, num_layers, input_dim, bias=False):
        super().__init__()
        seed_everything(12345)
        self.node_encoder = Linear(input_dim, hidden_channels)
        
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=True)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, 1)
        if bias:
            print('Initializing bias')
            with torch.no_grad():
                self.lin.bias.fill_(bias)
            
    def forward(self, x, edge_index):
        x = self.node_encoder(x)
        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.05, training=self.training)

        return self.lin(x)
        
class LitDeepGCN(L.LightningModule):
    def __init__(self, model, hidden_channels, num_layers, lr, criterion, input_dim, EPOCHS, steps_per_epoch, pos_weight, batch_size, cluster_size):
        super().__init__()
        
        if model == 'DeeperGCN':
            self.DeepGCN_model = DeeperGCN(hidden_channels, num_layers, input_dim)
        if criterion == 'BCE':
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
       
        self.lr = lr
        self.batch_size = batch_size
        self.cluster_size = cluster_size
        self.save_hyperparameters()
        self.logits = []
        self.labels = []
        self.train_logits, self.train_labels  = [], []
        self.val_logits, self.val_labels, self.val_true_labels  = [], [], []
        self.test_logits, self.test_labels = [], []
        self.EPOCHS = EPOCHS
        self.steps_per_epoch = steps_per_epoch

    def shared_step(self, batch, mask_name: str, stage: str):
        #batch.to(device)
        out = self.DeepGCN_model(batch.x, batch.edge_index).squeeze()
        mask = getattr(batch, mask_name)
        y = batch.y[mask]
        y_true = y[:, 0]
        y_actual = y[:, 1]
        y_pred = out[mask]
    
        if y_true.numel() == 0:
            return None, None, None
    
        loss = self.criterion(y_pred, y_true)
    
        return loss, y_pred.detach().cpu(), y_true.detach().cpu(), y_actual.detach().cpu()

    def training_step(self, batch, batch_idx):
        loss, train_pred, train_true, _ = self.shared_step(batch, mask_name="train_mask", stage='train')
        if loss is not None:
            self.log("train loss", loss, on_epoch=True, batch_size=batch.num_nodes, prog_bar=True)
            self.train_logits.append(train_pred)
            self.train_labels.append(train_true)
        return loss
        # return self.shared_step(batch, mask_name="train_mask", stage="train")

    def validation_step(self, batch, batch_idx):
        loss, val_pred, val_true, val_actual = self.shared_step(batch, mask_name="val_mask", stage='validation')
        if loss is not None:
            self.log("validation loss", loss, on_epoch=True, batch_size=batch.num_nodes)
            self.val_logits.append(val_pred)
            self.val_labels.append(val_true)
            self.val_true_labels.append(val_actual)
        # self.shared_step(batch, mask_name="val_mask", stage="validation")


    def test_step(self, batch, batch_idx):
        loss, test_pred, test_true, _ = self.shared_step(batch, mask_name="test_mask", stage='test')
        if loss is not None:
            self.log("test loss", loss, on_epoch=True, batch_size=batch.num_nodes)
            print(test_pred.shape)
            print(test_true.shape)
            self.test_logits.append(test_pred)
            self.test_labels.append(test_true)
        # self.shared_step(batch, mask_name="test_mask", stage="test")

    def on_train_epoch_end(self):
        if len(self.train_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.train_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.train_labels).numpy()
    
        train_f1 = f1_score(y_true, preds)
        train_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("train f1 score", train_f1)
        self.log("train MCC score", train_mcc)
        
        self.train_logits.clear()
        self.train_labels.clear()
        
    def on_validation_epoch_start(self):
        self.val_logits.clear()
        self.val_labels.clear()
        self.val_true_labels.clear()
        
    def on_validation_epoch_end(self):
        if len(self.val_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.val_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.val_labels).numpy()
    
        val_f1 = f1_score(y_true, preds)
        val_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("validation f1 score", val_f1)
        self.log("validation MCC score", val_mcc)

        y_actual = torch.cat(self.val_true_labels).numpy()
    
        val_f1 = f1_score(y_actual, preds)
        val_mcc = matthews_corrcoef(y_actual, preds)
    
        self.log("True validation f1 score", val_f1)
        self.log("True validation MCC score", val_mcc)
    
    def on_test_epoch_start(self):
        self.test_logits.clear()
        self.test_labels.clear()
        
    def on_test_epoch_end(self):

        if len(self.test_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.test_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.test_labels).numpy()
    
        test_f1 = f1_score(y_true, preds)
        test_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("test f1 score", test_f1)
        self.log("test MCC score", test_mcc)

        
        # sig = nn.Sigmoid()
        # probs = sig(torch.tensor(self.logits)).numpy()
        # preds = np.array((probs >= 0.5).astype(int))
        # self.preds = preds 
        
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({"pr": wandb.plot.pr_curve(y_true.flatten(), np.array(np.stack((1-probs, probs), axis=1)))})
        cm = wandb.plot.confusion_matrix(y_true=y_true.flatten().tolist(), preds=preds.flatten().tolist(), class_names=["Background", "GD1"])
        #cm = wandb.plot.confusion_matrix(y_true=np.array(model.labels).flatten().tolist(), preds=preds.flatten().tolist(), class_names=["Background", "GD1"])
        
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({"conf_mat": cm})
            
        # self.cm=confusion_matrix(self.labels, preds, normalize=None)
        self.cm = cm
        self.test_logits.clear()
        self.test_labels.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = {
        "scheduler": torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,  # peak LR
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.EPOCHS,
            pct_start=0.3,  # fraction of steps to increase LR (I USUALLY HAVE THIS AT .1)
            div_factor=10.0,  # initial_lr = max_lr/div_factor
            final_div_factor=1e3  # final_lr = initial_lr/final_div_factor
        ),
        "interval": "step",  # Step per batch
        "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    #     return optimizer



class LitDeepGCN_MCMBCE(L.LightningModule):
    def __init__(self, model, hidden_channels, num_layers, lr, criterion, input_dim, EPOCHS, steps_per_epoch, pos_weight, batch_size, cluster_size):
        super().__init__()
        
        if model == 'DeeperGCN':
            self.DeepGCN_model = DeeperGCN(hidden_channels, num_layers, input_dim)
        if criterion == 'BCE':
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
       
        self.lr = lr
        self.batch_size = batch_size
        self.cluster_size = cluster_size
        self.save_hyperparameters()
        self.logits = []
        self.labels = []
        self.train_logits, self.train_labels  = [], []
        self.val_logits, self.val_labels, self.val_true_labels  = [], [], []
        self.test_logits, self.test_labels = [], []
        self.EPOCHS = EPOCHS
        self.steps_per_epoch = steps_per_epoch

    def shared_step(self, batch, mask_name: str, stage: str):
     
        out = self.DeepGCN_model(batch.x, batch.edge_index).squeeze()
        mask = getattr(batch, mask_name)
        y = batch.y[mask]
        y_true = y[:, 0]
        y_actual = y[:, 1]
        y_pred = out[mask]
    
        if y_true.numel() == 0:
            return None, None, None
    
        loss = self.criterion(y_pred, y_true)
    
        return loss, y_pred.detach().cpu(), y_true.detach().cpu(), y_actual.detach().cpu()

    def training_step(self, batch, batch_idx):
        loss, train_pred, train_true, _ = self.shared_step(batch, mask_name="train_mask", stage='train')
        if loss is not None:
            self.log("train loss", loss, on_epoch=True, batch_size=batch.num_nodes, prog_bar=True)
            self.train_logits.append(train_pred)
            self.train_labels.append(train_true)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, val_pred, val_true, val_actual = self.shared_step(batch, mask_name="val_mask", stage='validation')
        if loss is not None:
            self.log("validation loss", loss, on_epoch=True, batch_size=batch.num_nodes)
            self.val_logits.append(val_pred)
            self.val_labels.append(val_true)
            self.val_true_labels.append(val_actual)



    def test_step(self, batch, batch_idx):
        loss, test_pred, test_true, _ = self.shared_step(batch, mask_name="test_mask", stage='test')
        if loss is not None:
            self.log("test loss", loss, on_epoch=True, batch_size=batch.num_nodes)
            print(test_pred.shape)
            print(test_true.shape)
            self.test_logits.append(test_pred)
            self.test_labels.append(test_true)
     

    def on_train_epoch_end(self):
        if len(self.train_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.train_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.train_labels).numpy()
    
        train_f1 = f1_score(y_true, preds)
        train_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("train f1 score", train_f1)
        self.log("train MCC score", train_mcc)
        
        self.train_logits.clear()
        self.train_labels.clear()
        
    def on_validation_epoch_start(self):
        self.val_logits.clear()
        self.val_labels.clear()
        self.val_true_labels.clear()
        
    def on_validation_epoch_end(self):
        if len(self.val_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.val_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.val_labels).numpy()
    
        val_f1 = f1_score(y_true, preds)
        val_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("validation f1 score", val_f1)
        self.log("validation MCC score", val_mcc)

        y_actual = torch.cat(self.val_true_labels).numpy()
    
        val_f1 = f1_score(y_actual, preds)
        val_mcc = matthews_corrcoef(y_actual, preds)
    
        self.log("True validation f1 score", val_f1)
        self.log("True validation MCC score", val_mcc)
    
    def on_test_epoch_start(self):
        self.test_logits.clear()
        self.test_labels.clear()
        
    def on_test_epoch_end(self):

        if len(self.test_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.test_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.test_labels).numpy()
    
        test_f1 = f1_score(y_true, preds)
        test_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("test f1 score", test_f1)
        self.log("test MCC score", test_mcc)

        
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({"pr": wandb.plot.pr_curve(y_true.flatten(), np.array(np.stack((1-probs, probs), axis=1)))})
        cm = wandb.plot.confusion_matrix(y_true=y_true.flatten().tolist(), preds=preds.flatten().tolist(), class_names=["Background", "GD1"])
        
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({"conf_mat": cm})
            
        
        self.cm = cm
        self.test_logits.clear()
        self.test_labels.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = {
        "scheduler": torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,  # peak LR
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.EPOCHS,
            pct_start=0.1,  # fraction of steps to increase LR
            div_factor=10.0,  # initial_lr = max_lr/div_factor
            final_div_factor=1e3  # final_lr = initial_lr/final_div_factor
        ),
        "interval": "step",  # Step per batch
        "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}





class GaiaDataModule(L.LightningDataModule):
    def __init__(self, load_data_dir = None, 
                 cluster_size = 1024, batch_size = 1, train_pct = .8, 
                 load_edges = False, data_dir = 'data.pth',
                 edge_dir = 'temp_data.pickle', load_dataloaders = False):
        
        super().__init__()
        seed_everything(12345)
        self.load_data_dir = load_data_dir
        self.cluster_size = cluster_size
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.load_edges = load_edges
        self.edge_dir = edge_dir
        self.load_dataloaders = load_dataloaders
        self.data_dir = data_dir
       
    def setup(self, stage: str):
        if not self.load_dataloaders:
            df = Table.read(self.load_data_dir)

            ra = np.array(df['phi1'])
            dec = np.array(df['phi2'])
            pm_ra = np.array(df['pm_phi1'])
            pm_dec = np.array(df['pm_phi2'])
            gmag  = np.array(df['phot_g_mean_mag'])
            color = np.array(df['bp_rp'])
            parallax = np.array(df['parallax'])
            stream = np.array(df['stream'])

            
            parallax_mask =  (parallax < 1)
            color_mask = (color > 0.5) & (color < 1.5)
            color_mask = (color > 0.5) & (color < 1.0)
            g_mask = gmag < 20.2
            full_mask = parallax_mask & color_mask & g_mask

            ra, dec = ra[full_mask], dec[full_mask]
            pm_ra, pm_dec, gmag, color = pm_ra[full_mask], pm_dec[full_mask], gmag[full_mask], color[full_mask]
            parallax = parallax[full_mask]
            stream = stream[full_mask]

            pm_ra_stream = pm_ra[stream]
            std_ra_stream, med_ra_stream = np.std(pm_ra_stream), np.median(pm_ra_stream)

            
            signal_region = (pm_ra < med_ra_stream + 1*std_ra_stream) & (pm_ra > med_ra_stream - 1*std_ra_stream)
            sideband_region = (pm_ra < med_ra_stream + 3*std_ra_stream) & (pm_ra > med_ra_stream - 3*std_ra_stream) & ~signal_region

            roi = signal_region | sideband_region
            
            ra, dec, pm_ra, pm_dec, gmag, color = ra[roi], dec[roi], pm_ra[roi], pm_dec[roi], gmag[roi], color[roi]
            parallax = parallax[roi]
            stream = stream[roi]
            signal_region = signal_region[roi]
            sideband_region = sideband_region[roi]

    
            embeddings = np.column_stack((ra, dec, pm_dec, gmag, color))

            train_mask, val_mask, test_mask = get_mask_splits(embeddings, self.train_pct)
            
            self.scaler = StandardScaler()
            self.scaler.fit(embeddings)

            labels = np.column_stack((signal_region, stream))
            
            x = torch.tensor(embeddings).to(device = 'cpu', dtype = torch.float32)
            y = torch.tensor(labels).to(device = 'cpu', dtype = torch.float32)
            
            if not self.load_edges:
                pos = torch.stack((x[:, 0], x[:, 1])).T
                knn_transform = KNNGraph(k=25, force_undirected=True)
                temp_data = knn_transform(Data(x=torch.zeros_like(x), pos=pos))
                edges = temp_data.edge_index
                with open(f"{os.environ['PSCRATCH']}/general_stream_cwola/{self.edge_dir}", 'wb') as file:
                    pickle.dump(temp_data, file)
            else:
                with open(f"{os.environ['PSCRATCH']}/general_stream_cwola/{self.edge_dir}", 'rb') as file:
                    edges = pickle.load(file).to('cpu').edge_index
            
            self.data = Data(x=torch.tensor(self.scaler.transform(x.numpy())), edge_index = edges, y=y, pm_ra = torch.tensor(pm_ra, dtype=torch.float32))
            self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
            self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
            self.data.test_mask  = torch.tensor(test_mask, dtype=torch.bool)
            torch.save(self.data, f"{os.environ['PSCRATCH']}/general_stream_cwola/{self.data_dir}")
            cluster_data = ClusterData(self.data, num_parts = self.cluster_size) #had this at 1024 before
    
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=True, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/general_stream_cwola/train_loader_{self.batch_size}.pth')
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=False, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/general_stream_cwola/test_loader_{self.batch_size}.pth')
     
    def train_dataloader(self):    
        print('Calling train_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/general_stream_cwola/train_loader_{self.batch_size}.pth', 
                          weights_only = False)
    def val_dataloader(self):
        print('Calling val_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/general_stream_cwola/test_loader_{self.batch_size}.pth', 
                          weights_only = False)
    def test_dataloader(self):
        print('Calling test_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/general_stream_cwola/test_loader_{self.batch_size}.pth', 
                          weights_only = False)


class GaiaDataModuleCustom(L.LightningDataModule):
    def __init__(self, load_data_dir = None, 
                 cluster_size = 1024, batch_size = 1, train_pct = .8, 
                 load_edges = False, data_dir = 'data.pth',
                 edge_dir = 'temp_data.pickle', load_dataloaders = False):
        
        super().__init__()
        seed_everything(12345)
        self.load_data_dir = load_data_dir
        self.cluster_size = cluster_size
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.load_edges = load_edges
        self.edge_dir = edge_dir
        self.load_dataloaders = load_dataloaders
        self.data_dir = data_dir
       
    def setup(self, stage: str):
        if not self.load_dataloaders:
            df = Table.read(self.load_data_dir)

            ra_actual = np.array(df['ra'])
            dec_actual = np.array(df['dec'])
            
            ra = np.array(df['phi1']).astype('float64')
            dec = np.array(df['phi2'])

            ra = np.array(df['ra']).astype('float64')
            dec = np.array(df['dec'])
            
            pm_ra = np.array(df['pm_phi1']).astype('float64')
            pm_dec = np.array(df['pm_phi2'])
            
            pm_ra = np.array(df['pmra']).astype('float64')
            pm_dec = np.array(df['pmdec'])

            pm_ra = pm_ra * np.cos(np.radians(dec))
            
            gmag  = np.array(df['phot_g_mean_mag'])
            color = np.array(df['phot_bp_mean_mag']) - np.array(df['phot_rp_mean_mag'])
            parallax = np.array(df['parallax'])
            stream = np.array(df['stream'])

            decals_gmag = 22.5 - 2.5*np.log10(np.array(df["flux_g"]))
            decals_rmag = 22.5 - 2.5*np.log10(np.array(df["flux_r"]))
            decals_zmag = 22.5 - 2.5*np.log10(np.array(df["flux_z"]))
            g_r = decals_gmag - decals_rmag
            r_z = decals_rmag - decals_zmag
            g_z = decals_gmag - decals_zmag

            ra_mask = ra_actual < 270 # used to be 287.62
            parallax_mask =  (parallax < 1)
            color_mask = (color > 0.5) & (color < 1.5)
            color_mask = (color > 0.5) & (color < 1.0)
            g_mask = gmag < 20.2

            decals_mask = ~(np.isnan(g_r) | np.isnan(r_z) | np.isnan(g_z) | np.isinf(g_r) | np.isinf(r_z) | np.isinf(g_z))
            full_mask = parallax_mask & color_mask & g_mask & ra_mask & decals_mask

            ra, dec = ra[full_mask], dec[full_mask]
            pm_ra, pm_dec, gmag, color = pm_ra[full_mask], pm_dec[full_mask], gmag[full_mask], color[full_mask]
            parallax = parallax[full_mask]
            stream = stream[full_mask]
    
            g_r, r_z, g_z = g_r[full_mask], r_z[full_mask], g_z[full_mask]
            
            pm_ra_stream = pm_ra[stream]
            std_ra_stream, med_ra_stream = np.std(pm_ra_stream), np.median(pm_ra_stream)

            sig_low, sig_high = [np.percentile(pm_ra_stream, 10), np.percentile(pm_ra_stream, 60)]
            side_low, side_high = [np.percentile(pm_ra_stream, 0), np.percentile(pm_ra_stream, 100)]

            signal_region = (pm_ra > sig_low) & (pm_ra < sig_high)
            sideband_region = (pm_ra > side_low) & (pm_ra < side_high) & ~signal_region

            roi = signal_region | sideband_region
            
            ra, dec, pm_ra, pm_dec, gmag, color = ra[roi], dec[roi], pm_ra[roi], pm_dec[roi], gmag[roi], color[roi]
            parallax = parallax[roi]
            stream = stream[roi]
            signal_region = signal_region[roi]
            sideband_region = sideband_region[roi]
            g_r, r_z, g_z = g_r[roi], r_z[roi], g_z[roi]
    
            embeddings = np.column_stack((ra, dec, pm_dec, gmag, color, g_r, r_z, g_z))

            train_mask, val_mask, test_mask = get_mask_splits(embeddings, self.train_pct)
            
            self.scaler = StandardScaler()
            self.scaler.fit(embeddings)

            labels = np.column_stack((signal_region, stream))
            
            x = torch.tensor(embeddings).to(device = 'cpu', dtype = torch.float32)
            y = torch.tensor(labels).to(device = 'cpu', dtype = torch.float32)
            
            if not self.load_edges:
                pos = torch.stack((x[:, 0], x[:, 1])).T
                knn_transform = KNNGraph(k=25, force_undirected=True)
                temp_data = knn_transform(Data(x=torch.zeros_like(x), pos=pos))
                edges = temp_data.edge_index
                with open(f"{os.environ['PSCRATCH']}/general_stream_cwola/{self.edge_dir}", 'wb') as file:
                    pickle.dump(temp_data, file)
            else:
                with open(f"{os.environ['PSCRATCH']}/general_stream_cwola/{self.edge_dir}", 'rb') as file:
                    edges = pickle.load(file).to('cpu').edge_index
            
            self.data = Data(x=torch.tensor(self.scaler.transform(x.numpy())), edge_index = edges, y=y, pm_ra = torch.tensor(pm_ra, dtype=torch.float32))
            self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
            self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
            self.data.test_mask  = torch.tensor(test_mask, dtype=torch.bool)
            torch.save(self.data, f"{os.environ['PSCRATCH']}/general_stream_cwola/{self.data_dir}")
            cluster_data = ClusterData(self.data, num_parts = self.cluster_size) #had this at 1024 before
    
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=True, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/general_stream_cwola/train_loader_{self.batch_size}.pth')
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=False, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/general_stream_cwola/test_loader_{self.batch_size}.pth')
     
    def train_dataloader(self):    
        print('Calling train_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/general_stream_cwola/train_loader_{self.batch_size}.pth', 
                          weights_only = False)
    def val_dataloader(self):
        print('Calling val_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/general_stream_cwola/test_loader_{self.batch_size}.pth', 
                          weights_only = False)
    def test_dataloader(self):
        print('Calling test_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/general_stream_cwola/test_loader_{self.batch_size}.pth', 
                          weights_only = False)


class GaiaDataModuleC19(L.LightningDataModule):
    def __init__(self, load_data_dir = None, 
                 cluster_size = 1024, batch_size = 1, train_pct = .8, 
                 load_edges = False, data_dir = 'data.pth',
                 edge_dir = 'temp_data.pickle', load_dataloaders = False,
                 use_cwola = True):
        
        super().__init__()
        seed_everything(12345)
        self.load_data_dir = load_data_dir
        self.cluster_size = cluster_size
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.load_edges = load_edges
        self.edge_dir = edge_dir
        self.load_dataloaders = load_dataloaders
        self.data_dir = data_dir
        self.use_cwola = use_cwola
       
    def setup(self, stage: str):
        if not self.load_dataloaders:
            df = Table.read(self.load_data_dir)

            ra_actual = np.array(df['temp_ra'])
            dec_actual = np.array(df['dec'])
            
            ra = np.array(df['phi1']).astype('float64')
            dec = np.array(df['phi2']).astype('float64')
            
            pm_ra  = np.array(df['pm_phi1']).astype('float64')
            pm_dec = np.array(df['pm_phi2']).astype('float64')
            
            pm_ra_error = np.array(df['pmra_error']).astype('float64')
            pm_dec_error = np.array(df['pmdec_error']).astype('float64')

            gmag  = np.array(df['phot_g_mean_mag'])
            color = np.array(df['phot_bp_mean_mag']) - np.array(df['phot_rp_mean_mag'])
            parallax = np.array(df['parallax'])
            parallax_error = np.array(df['parallax_error'])

            gmag0, rmag0, zmag0 = np.array(df['gmag0']), np.array(df['rmag0']), np.array(df['zmag0'])
            g_r = gmag0-rmag0
            r_z = rmag0-zmag0
            g_z = gmag0-zmag0
            #g-r, r-z, g-z
            stream = np.array(df['stream'])

            roi = np.array(df['roi'], dtype=bool)
            signal_region = np.array(df['signal_region'], dtype=bool)
            sideband_region = np.array(df['sideband_region'], dtype=bool)

            if self.use_cwola:

                stream_pm_phi1 = pm_ra[stream]
                pm_p1_med, pm_p1_std= np.median(stream_pm_phi1), np.std(stream_pm_phi1)
                
                # Define regions using 5 and 95 percentiles
                lower_perc = np.percentile(stream_pm_phi1, 5)
                upper_perc = np.percentile(stream_pm_phi1, 95)
                lower_bound = pm_p1_med - 5*pm_p1_std
                upper_bound = pm_p1_med + 5*pm_p1_std

                regions = [[lower_perc, upper_perc], [lower_bound, upper_bound]]

                sig_low, sig_high = regions[0]
                side_low, side_high = regions[1]

                signal_region = (pm_ra > sig_low) & (pm_ra < sig_high)
                sideband_region = (pm_ra > side_low) & (pm_ra < side_high) & ~signal_region
                roi = signal_region | sideband_region

                ra, dec, pm_ra, pm_dec, gmag, color = ra[roi], dec[roi], pm_ra[roi], pm_dec[roi], gmag[roi], color[roi]
                pm_ra_error, pm_dec_error = pm_ra_error[roi], pm_dec_error[roi]
                parallax, parallax_error = parallax[roi], parallax_error[roi]
                stream = stream[roi]
                signal_region = signal_region[roi]
                sideband_region = sideband_region[roi]
                gmag0, rmag0, zmag0 = gmag0[roi], rmag0[roi], zmag0[roi]
                g_r, r_z, g_z = g_r[roi], r_z[roi], g_z[roi]
    
            # embeddings = np.column_stack((ra, dec, pm_dec, pm_ra_error, pm_dec_error, gmag, color, gmag0, rmag0, zmag0, parallax, parallax_error))

            if self.use_cwola:
                # embeddings = np.column_stack((ra, dec, pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))
                embeddings = np.column_stack((dec, pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))
            else:
                embeddings = np.column_stack((ra, dec, pm_ra, pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))

            # embeddings = np.column_stack((pm_dec, pm_ra_error, pm_dec_error, gmag, color, gmag0, rmag0, zmag0, parallax, parallax_error))
            
            print(f'Total nan values in embeddings: {np.sum(np.isnan(embeddings))}')
            train_mask, val_mask, test_mask = get_mask_splits(embeddings, self.train_pct)
            
            self.scaler = StandardScaler()
            self.scaler.fit(embeddings)

            labels = np.column_stack((signal_region, stream)) if self.use_cwola else np.column_stack((stream, stream))
            
            x = torch.tensor(embeddings).to(device = 'cpu', dtype = torch.float32)
            y = torch.tensor(labels).to(device = 'cpu', dtype = torch.float32)
            
            if not self.load_edges:
                pos = torch.stack((x[:, 0], x[:, 1])).T
                knn_transform = KNNGraph(k=25, force_undirected=True)
                temp_data = knn_transform(Data(x=torch.zeros_like(x), pos=pos))
                edges = temp_data.edge_index
                with open(f"{os.environ['PSCRATCH']}/C19_cwola/{self.edge_dir}", 'wb') as file:
                    pickle.dump(temp_data, file)
            else:
                with open(f"{os.environ['PSCRATCH']}/C19_cwola/{self.edge_dir}", 'rb') as file:
                    edges = pickle.load(file).to('cpu').edge_index
            
            self.data = Data(x=torch.tensor(self.scaler.transform(x.numpy())), edge_index = edges, y=y, pm_ra = torch.tensor(pm_ra, dtype=torch.float32), ra = torch.tensor(ra, dtype=torch.float32))
            # self.data = Data(x=torch.tensor(self.scaler.transform(x.numpy())), edge_index = edges, y=y, pm_ra = torch.tensor(pm_ra, dtype=torch.float32), ra = torch.tensor(ra, dtype=torch.float32), dec = torch.tensor(dec, dtype=torch.float32))
            self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
            self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
            self.data.test_mask  = torch.tensor(test_mask, dtype=torch.bool)
            torch.save(self.data, f"{os.environ['PSCRATCH']}/C19_cwola/{self.data_dir}")
            cluster_data = ClusterData(self.data, num_parts = self.cluster_size) #had this at 1024 before
    
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=True, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/C19_cwola/train_loader_{self.batch_size}.pth')
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=False, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/C19_cwola/test_loader_{self.batch_size}.pth')
     
    def train_dataloader(self):    
        print('Calling train_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/C19_cwola/train_loader_{self.batch_size}.pth', 
                          weights_only = False)
    def val_dataloader(self):
        print('Calling val_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/C19_cwola/test_loader_{self.batch_size}.pth', 
                          weights_only = False)
    def test_dataloader(self):
        print('Calling test_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/C19_cwola/test_loader_{self.batch_size}.pth', 
                          weights_only = False)
    
class GaiaDataModuleGD1(L.LightningDataModule):
    def __init__(self, name, load_data_dir = None, 
                 cluster_size = 1024, batch_size = 1, train_pct = .8, 
                 load_edges = False, data_dir = 'data.pth',
                 edge_dir = 'temp_data.pickle', load_dataloaders = False,
                 use_cwola = True):
        
        super().__init__()
        seed_everything(12345)
        self.name = name
        self.load_data_dir = load_data_dir
        self.cluster_size = cluster_size
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.load_edges = load_edges
        self.edge_dir = edge_dir
        self.load_dataloaders = load_dataloaders
        self.data_dir = data_dir
        self.use_cwola = use_cwola
       
    def setup(self, stage: str):
        if not self.load_dataloaders:
            # Check to see if file is csv or fits
            if self.load_data_dir.endswith('.csv'):
                df = pd.read_csv(self.load_data_dir)
                df = Table.from_pandas(df)
            else:
                df = Table.read(self.load_data_dir)

            ra_actual = np.array(df['ra'])
            dec_actual = np.array(df['dec'])
            
            ra = np.array(df['phi1']).astype('float64')
            dec = np.array(df['phi2']).astype('float64')
            
            pm_ra  = np.array(df['pm_phi1']).astype('float64')
            pm_dec = np.array(df['pm_phi2']).astype('float64')
            
            pm_ra_error = np.array(df['pmra_error']).astype('float64')
            pm_dec_error = np.array(df['pmdec_error']).astype('float64')

            gmag  = np.array(df['phot_g_mean_mag'])
            color = np.array(df['phot_bp_mean_mag']) - np.array(df['phot_rp_mean_mag'])
            parallax = np.array(df['parallax'])
            parallax_error = np.array(df['parallax_error'])

            gmag0, rmag0, zmag0 = np.array(df['gmag0']), np.array(df['rmag0']), np.array(df['zmag0'])
            g_r = gmag0-rmag0
            r_z = rmag0-zmag0
            g_z = gmag0-zmag0
            #g-r, r-z, g-z
            stream = np.array(df['stream'])

            roi = np.array(df['roi'], dtype=bool)
            signal_region = np.array(df['signal_region'], dtype=bool)
            sideband_region = np.array(df['sideband_region'], dtype=bool)

            if self.use_cwola:
                # Commented this out for now since I've already defined the roi I want to use in my data prep script
                # Will uncomment if I decide to play around with ROI for GD-1

                # stream_pm_phi1 = pm_ra[stream]
                # pm_p1_med, pm_p1_std= np.median(stream_pm_phi1), np.std(stream_pm_phi1)
                
                # # Define regions using 5 and 95 percentiles
                # lower_perc = np.percentile(stream_pm_phi1, 5)
                # upper_perc = np.percentile(stream_pm_phi1, 95)
                # lower_bound = pm_p1_med - 5*pm_p1_std
                # upper_bound = pm_p1_med + 5*pm_p1_std

                # regions = [[lower_perc, upper_perc], [lower_bound, upper_bound]]

                # sig_low, sig_high = regions[0]
                # side_low, side_high = regions[1]

                # signal_region = (pm_ra > sig_low) & (pm_ra < sig_high)
                # sideband_region = (pm_ra > side_low) & (pm_ra < side_high) & ~signal_region
                # roi = signal_region | sideband_region

                ra, dec, pm_ra, pm_dec, gmag, color = ra[roi], dec[roi], pm_ra[roi], pm_dec[roi], gmag[roi], color[roi]
                pm_ra_error, pm_dec_error = pm_ra_error[roi], pm_dec_error[roi]
                parallax, parallax_error = parallax[roi], parallax_error[roi]
                stream = stream[roi]
                signal_region = signal_region[roi]
                sideband_region = sideband_region[roi]
                gmag0, rmag0, zmag0 = gmag0[roi], rmag0[roi], zmag0[roi]
                g_r, r_z, g_z = g_r[roi], r_z[roi], g_z[roi]
    
            # embeddings = np.column_stack((ra, dec, pm_dec, pm_ra_error, pm_dec_error, gmag, color, gmag0, rmag0, zmag0, parallax, parallax_error))

            if self.use_cwola:
                embeddings = np.column_stack((ra, dec, pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))
                # embeddings = np.column_stack((dec, pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))
            else:
                embeddings = np.column_stack((ra, dec, pm_ra, pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))

            # embeddings = np.column_stack((pm_dec, pm_ra_error, pm_dec_error, gmag, color, gmag0, rmag0, zmag0, parallax, parallax_error))
            
            print(f'Total nan values in embeddings: {np.sum(np.isnan(embeddings))}')
            print(f'Minimum pm_dec: {np.min(embeddings[:, 2])}')
            print(f'Maximum pm_dec: {np.max(embeddings[:, 2])}')
            train_mask, val_mask, test_mask = get_mask_splits(embeddings, self.train_pct)
            
            self.scaler = StandardScaler()
            self.scaler.fit(embeddings)

            labels = np.column_stack((signal_region, stream)) if self.use_cwola else np.column_stack((stream, stream))
            
            x = torch.tensor(embeddings).to(device = 'cpu', dtype = torch.float32)
            y = torch.tensor(labels).to(device = 'cpu', dtype = torch.float32)
            
            if not self.load_edges:
                pos = torch.stack((x[:, 0], x[:, 1])).T
                knn_transform = KNNGraph(k=25, force_undirected=True)
                temp_data = knn_transform(Data(x=torch.zeros_like(x), pos=pos))
                edges = temp_data.edge_index
                with open(f"{os.environ['PSCRATCH']}/GD1_cwola/{self.edge_dir}", 'wb') as file:
                    pickle.dump(temp_data, file)
            else:
                with open(f"{os.environ['PSCRATCH']}/GD1_cwola/{self.edge_dir}", 'rb') as file:
                    edges = pickle.load(file).to('cpu').edge_index
            
            self.data = Data(x=torch.tensor(self.scaler.transform(x.numpy())), edge_index = edges, y=y, pm_ra = torch.tensor(pm_ra, dtype=torch.float32), ra = torch.tensor(ra, dtype=torch.float32))
            
            # self.data = Data(x=torch.tensor(self.scaler.transform(x.numpy())), edge_index = edges, y=y, pm_ra = torch.tensor(pm_ra, dtype=torch.float32), ra = torch.tensor(ra, dtype=torch.float32), dec = torch.tensor(dec, dtype=torch.float32))
            
            self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
            self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
            self.data.test_mask  = torch.tensor(test_mask, dtype=torch.bool)
            torch.save(self.data, f"{os.environ['PSCRATCH']}/GD1_cwola/{self.data_dir}")
            cluster_data = ClusterData(self.data, num_parts = self.cluster_size) #had this at 1024 before
    
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=True, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/GD1_cwola/train_loader_{self.name}.pth')
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=False, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/GD1_cwola/test_loader_{self.name}.pth')

    def train_dataloader(self):    
        print('Calling train_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/GD1_cwola/train_loader_{self.name}.pth', 
                          weights_only = False)
    def val_dataloader(self):
        print('Calling val_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/GD1_cwola/test_loader_{self.name}.pth', 
                          weights_only = False)
    def test_dataloader(self):
        print('Calling test_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/GD1_cwola/test_loader_{self.name}.pth', 
                          weights_only = False)


class GaiaDataModuleGD1CATHODE(L.LightningDataModule):
    def __init__(self, name, load_data_dir = None, 
                 cluster_size = 1024, batch_size = 1, train_pct = .8, 
                 load_edges = False, data_dir = 'data.pth',
                 edge_dir = 'temp_data.pickle', load_dataloaders = False,
                 use_cwola = True, no_errs = False):
        
        super().__init__()
        seed_everything(12345)
        self.name = name
        self.load_data_dir = load_data_dir
        self.cluster_size = cluster_size
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.load_edges = load_edges
        self.edge_dir = edge_dir
        self.load_dataloaders = load_dataloaders
        self.data_dir = data_dir
        self.use_cwola = use_cwola
        self.no_errs = no_errs
       
    def setup(self, stage: str):
        if not self.load_dataloaders:
            # Check to see if file is csv or fits
            if self.load_data_dir.endswith('.csv'):
                df = pd.read_csv(self.load_data_dir)
                df = Table.from_pandas(df)
            else:
                df = Table.read(self.load_data_dir)

            # ra_actual = np.array(df['ra'])
            # dec_actual = np.array(df['dec'])
            
            ra = np.array(df['ra']).astype('float64')
            dec = np.array(df['dec']).astype('float64')
            
            pm_ra  = np.array(df['pm_ra']).astype('float64')
            pm_dec = np.array(df['pm_dec']).astype('float64')

            if not self.no_errs:
                pm_ra_error = np.array(df['pm_ra_error']).astype('float64')
                pm_dec_error = np.array(df['pm_dec_error']).astype('float64')
                parallax_error = np.array(df['parallax_error']).astype('float64')

            gmag  = np.array(df['gmag'])
            color = np.array(df['color'])
            # parallax = np.array(df['parallax'])
            rmag0 = np.array(df['rmag0'])
            g_r = np.array(df['g_r'])
            r_z = np.array(df['r_z'])
            # g_z = np.array(df['g_z'])

            #g-r, r-z, g-z

            stream = np.array(df['stream'])
            cwola_label = np.array(df['CWoLa_Label'], dtype=bool)
            # Change the stream labels to be 0 for sampled data
            sampled_data = (stream == 2)
            stream[sampled_data] = 0

            if self.no_errs:
                # embeddings = np.column_stack((ra, dec, pm_ra, pm_dec, gmag, color, g_r, r_z, g_z, parallax))
                # embeddings = np.column_stack((ra, dec, pm_dec, gmag, color, g_r, r_z, g_z, parallax))
                embeddings = np.column_stack((ra, dec, pm_dec, gmag, color, rmag0, g_r, r_z))
            else:
                # embeddings = np.column_stack((ra, dec, pm_ra, pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))
                # embeddings = np.column_stack((ra, dec,  pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))
                embeddings = np.column_stack((ra, dec,  pm_dec, pm_ra_error, pm_dec_error, gmag, color))
                
            print(f'Total nan values in embeddings: {np.sum(np.isnan(embeddings))}')

            train_mask, val_mask, test_mask = get_mask_splits(embeddings, self.train_pct)
            
            self.scaler = StandardScaler()
            self.scaler.fit(embeddings)

            labels = np.column_stack((cwola_label, stream))
            
            x = torch.tensor(embeddings).to(device = 'cpu', dtype = torch.float32)
            y = torch.tensor(labels).to(device = 'cpu', dtype = torch.float32)
            
            if not self.load_edges:
                pos = torch.stack((x[:, 0], x[:, 1])).T
                knn_transform = KNNGraph(k=25, force_undirected=True)
                temp_data = knn_transform(Data(x=torch.zeros_like(x), pos=pos))
                edges = temp_data.edge_index
                with open(f"{os.environ['PSCRATCH']}/GD1_cathode/{self.edge_dir}", 'wb') as file:
                    pickle.dump(temp_data, file)
            else:
                with open(f"{os.environ['PSCRATCH']}/GD1_cathode/{self.edge_dir}", 'rb') as file:
                    edges = pickle.load(file).to('cpu').edge_index
            
            self.data = Data(x=torch.tensor(self.scaler.transform(x.numpy())), edge_index = edges, y=y, pm_ra = torch.tensor(pm_ra, dtype=torch.float32), sampled_data = torch.tensor(sampled_data, dtype=torch.bool))
            
            # self.data = Data(x=torch.tensor(self.scaler.transform(x.numpy())), edge_index = edges, y=y, pm_ra = torch.tensor(pm_ra, dtype=torch.float32), ra = torch.tensor(ra, dtype=torch.float32), dec = torch.tensor(dec, dtype=torch.float32))
            
            self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
            self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
            self.data.test_mask  = torch.tensor(test_mask, dtype=torch.bool)
            torch.save(self.data, f"{os.environ['PSCRATCH']}/GD1_cathode/{self.data_dir}")
            cluster_data = ClusterData(self.data, num_parts = self.cluster_size) #had this at 1024 before
    
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=True, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/GD1_cathode/train_loader_{self.name}.pth')
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=False, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/GD1_cathode/test_loader_{self.name}.pth')

    def train_dataloader(self):    
        print('Calling train_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/GD1_cathode/train_loader_{self.name}.pth', 
                          weights_only = False)
    def val_dataloader(self):
        print('Calling val_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/GD1_cathode/test_loader_{self.name}.pth', 
                          weights_only = False)
    def test_dataloader(self):
        print('Calling test_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/GD1_cathode/test_loader_{self.name}.pth', 
                          weights_only = False)



#  Data Module for C19 CATHODE ---------------------------------------------------------------------------------------------------------------------
class GaiaDataModuleC19CATHODE(L.LightningDataModule):
    def __init__(self, name, load_data_dir = None, 
                 cluster_size = 1024, batch_size = 1, train_pct = .8, 
                 load_edges = False, data_dir = 'data.pth',
                 edge_dir = 'temp_data.pickle', load_dataloaders = False,
                 use_cwola = True, no_errs = False):
        
        super().__init__()
        seed_everything(12345)
        self.name = name
        self.load_data_dir = load_data_dir
        self.cluster_size = cluster_size
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.load_edges = load_edges
        self.edge_dir = edge_dir
        self.load_dataloaders = load_dataloaders
        self.data_dir = data_dir
        self.use_cwola = use_cwola
        self.no_errs = no_errs
       
    def setup(self, stage: str):
        if not self.load_dataloaders:
            # Check to see if file is csv or fits
            if self.load_data_dir.endswith('.csv'):
                df = pd.read_csv(self.load_data_dir)
                df = Table.from_pandas(df)
            else:
                df = Table.read(self.load_data_dir)

            # ra_actual = np.array(df['ra'])
            # dec_actual = np.array(df['dec'])
            
            ra = np.array(df['ra']).astype('float64')
            dec = np.array(df['dec']).astype('float64')
            
            pm_ra  = np.array(df['pm_ra']).astype('float64')
            pm_dec = np.array(df['pm_dec']).astype('float64')

            if not self.no_errs:
                pm_ra_error = np.array(df['pm_ra_error']).astype('float64')
                pm_dec_error = np.array(df['pm_dec_error']).astype('float64')
                parallax_error = np.array(df['parallax_error']).astype('float64')

            gmag  = np.array(df['gmag'])
            color = np.array(df['color'])
            # parallax = np.array(df['parallax'])
            rmag0 = np.array(df['rmag0'])
            g_r = np.array(df['g_r'])
            r_z = np.array(df['r_z'])
            # g_z = np.array(df['g_z'])

            #g-r, r-z, g-z

            stream = np.array(df['stream'])
            cwola_label = np.array(df['CWoLa_Label'], dtype=bool)
            # Change the stream labels to be 0 for sampled data
            sampled_data = (stream == 2)
            stream[sampled_data] = 0

            if self.no_errs:
                # embeddings = np.column_stack((ra, dec, pm_ra, pm_dec, gmag, color, g_r, r_z, g_z, parallax))
                # embeddings = np.column_stack((ra, dec, pm_dec, gmag, color, g_r, r_z, g_z, parallax))
                embeddings = np.column_stack((ra, dec, pm_dec, gmag, color, rmag0, g_r, r_z))
            else:
                # embeddings = np.column_stack((ra, dec, pm_ra, pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))
                # embeddings = np.column_stack((ra, dec,  pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))
                embeddings = np.column_stack((ra, dec,  pm_dec, pm_ra_error, pm_dec_error, gmag, color))
                
            print(f'Total nan values in embeddings: {np.sum(np.isnan(embeddings))}')

            train_mask, val_mask, test_mask = get_mask_splits(embeddings, self.train_pct)
            
            self.scaler = StandardScaler()
            self.scaler.fit(embeddings)

            labels = np.column_stack((cwola_label, stream))
            
            x = torch.tensor(embeddings).to(device = 'cpu', dtype = torch.float32)
            y = torch.tensor(labels).to(device = 'cpu', dtype = torch.float32)
            
            if not self.load_edges:
                pos = torch.stack((x[:, 0], x[:, 1])).T
                knn_transform = KNNGraph(k=25, force_undirected=True)
                temp_data = knn_transform(Data(x=torch.zeros_like(x), pos=pos))
                edges = temp_data.edge_index
                with open(f"{os.environ['PSCRATCH']}/C19_cathode/{self.edge_dir}", 'wb') as file:
                    pickle.dump(temp_data, file)
            else:
                with open(f"{os.environ['PSCRATCH']}/C19_cathode/{self.edge_dir}", 'rb') as file:
                    edges = pickle.load(file).to('cpu').edge_index
            
            self.data = Data(x=torch.tensor(self.scaler.transform(x.numpy())), edge_index = edges, y=y, pm_ra = torch.tensor(pm_ra, dtype=torch.float32), sampled_data = torch.tensor(sampled_data, dtype=torch.bool))
            
            # self.data = Data(x=torch.tensor(self.scaler.transform(x.numpy())), edge_index = edges, y=y, pm_ra = torch.tensor(pm_ra, dtype=torch.float32), ra = torch.tensor(ra, dtype=torch.float32), dec = torch.tensor(dec, dtype=torch.float32))
            
            self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
            self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
            self.data.test_mask  = torch.tensor(test_mask, dtype=torch.bool)
            torch.save(self.data, f"{os.environ['PSCRATCH']}/C19_cathode/{self.data_dir}")
            cluster_data = ClusterData(self.data, num_parts = self.cluster_size) #had this at 1024 before
    
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=True, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/C19_cathode/train_loader_{self.name}.pth')
            torch.save(ClusterLoader(cluster_data, batch_size=self.batch_size, shuffle=False, num_workers=8), 
                       f'{os.environ['PSCRATCH']}/C19_cathode/test_loader_{self.name}.pth')

    def train_dataloader(self):    
        print('Calling train_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/C19_cathode/train_loader_{self.name}.pth', 
                          weights_only = False)
    def val_dataloader(self):
        print('Calling val_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/C19_cathode/test_loader_{self.name}.pth', 
                          weights_only = False)
    def test_dataloader(self):
        print('Calling test_dataloader func')
        return torch.load(f'{os.environ['PSCRATCH']}/C19_cathode/test_loader_{self.name}.pth', 
                          weights_only = False)

# ------------------------------------------------------------------------------------------------------------------------------
# This class will define a linear model for benchmarking

# ------------------------------------------------------------
# Helper: activation selector
# ------------------------------------------------------------
def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()           # also called Swish
    if name == "gelu":
        return nn.GELU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unknown activation '{name}'")


# ------------------------------------------------------------
# Residual MLP block (Pre-LN)
# ------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, activation="silu"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = get_activation(activation)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # Pre-LayerNorm → Linear → Activation → Linear → + skip
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        return x + out


# ------------------------------------------------------------
# Main MLP Architecture
# ------------------------------------------------------------
class LinearModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layers: int = 3,
        hidden_units: int = 256,
        dropout: float = 0.0,
        layer_norm: bool = False,
        activation: str = "relu",
        residual: bool = False,
    ):
        """
        If residual=True:
            - Uses Pre-LN ResidualBlocks for all hidden layers
            - LayerNorm= True inside blocks, independent of layer_norm flag
        """
        super().__init__()
        assert num_layers >= 1
        assert 0.0 <= dropout < 1.0

        self.residual = residual

        layers = []
        in_dim = input_dim

        # ------------------------------------------------------------
        # Residual MLP Case
        # ------------------------------------------------------------
        if residual:
            # First projection to hidden_units
            layers.append(nn.Linear(in_dim, hidden_units))

            # Residual blocks
            for _ in range(num_layers - 1):
                layers.append(ResidualBlock(hidden_units, activation=activation))

            # Output
            layers.append(nn.Linear(hidden_units, 1))
            self.net = nn.Sequential(*layers)
            return

        # ------------------------------------------------------------
        # Standard feedforward MLP (no residual)
        # ------------------------------------------------------------
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_units))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_units))

            layers.append(get_activation(activation))

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            in_dim = hidden_units

        # Final output layer
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    

# class LinearModel(torch.nn.Module):
#     def __init__(self, input_dim, num_layers: int = 3, hidden_units: int = 256, dropout: float = 0.0, layer_norm: bool = False):
#         super(LinearModel, self).__init__()
#         assert num_layers >= 1, "num_layers must be >= 1"
#         assert 0.0 <= dropout < 1.0, "dropout must be in [0.0, 1.0)"
    
#         layers = []
#         in_dim = input_dim
#         # If num_layers == 1: single linear to output
#         for i in range(num_layers - 1):
#             layers.append(torch.nn.Linear(in_dim, hidden_units))
#             if layer_norm:
#                 layers.append(torch.nn.LayerNorm(hidden_units))
#             layers.append(torch.nn.ReLU())
#             if dropout > 0.0:
#                 layers.append(torch.nn.Dropout(p=dropout))
#             in_dim = hidden_units

#         # Final layer to single logit
#         layers.append(torch.nn.Linear(in_dim, 1))

#         self.net = torch.nn.Sequential(*layers)
#     def forward(self, x):
#             return self.net(x)


# This class will define the dataset to be used for the linear model

class GaiaDatasetLinear(Dataset):
    def __init__(self, data, labels, pm_ra):
        self.data = data
        self.labels = labels
        self.pm_ra = pm_ra

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.pm_ra[idx]


class CATHODEGaiaDatasetLinear(Dataset):
    def __init__(self, data, labels, id_plus_sample):
        self.data = data
        self.labels = labels
        self.id_plus_sample = id_plus_sample
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.id_plus_sample[idx]
    
# This class will define the data module for the linear model
class LinearDataModule(L.LightningDataModule):
    def __init__(self, load_data_dir = None, 
                 batch_size = 1024, train_pct = .8, 
                load_dataloaders = False, use_cwola = True):
        
        super().__init__()
        seed_everything(12345)
        self.load_data_dir = load_data_dir
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.load_dataloaders = load_dataloaders
        self.use_cwola = use_cwola
    def setup(self, stage: str):
        if not self.load_dataloaders:
            df = Table.read(self.load_data_dir)

            # ra = np.array(df['phi1']).astype('float64')
            # dec = np.array(df['phi2']).astype('float64')
            # pm_ra  = np.array(df['pm_phi1']).astype('float64')
            # pm_dec = np.array(df['pm_phi2']).astype('float64')
            
            ra = np.array(df['temp_ra']).astype('float64')
            dec = np.array(df['dec']).astype('float64')
            pm_ra  = np.array(df['pmra']).astype('float64')
            pm_dec = np.array(df['pmdec']).astype('float64')

            pm_ra_for_storage = pm_ra.copy()
            pm_ra_cos_dec = pm_ra * np.cos(np.radians(dec))

            pm_ra_error = np.array(df['pmra_error']).astype('float64')
            pm_dec_error = np.array(df['pmdec_error']).astype('float64')

            gmag  = np.array(df['phot_g_mean_mag'])
            color = np.array(df['phot_bp_mean_mag']) - np.array(df['phot_rp_mean_mag'])
            parallax = np.array(df['parallax'])
            parallax_error = np.array(df['parallax_error'])

            gmag0, rmag0, zmag0 = np.array(df['gmag0']), np.array(df['rmag0']), np.array(df['zmag0'])
            g_r = gmag0-rmag0
            r_z = rmag0-zmag0
            g_z = gmag0-zmag0

            stream = np.array(df['stream'])

            roi = np.array(df['roi'], dtype=bool)
            signal_region = np.array(df['signal_region'], dtype=bool)
            sideband_region = np.array(df['sideband_region'], dtype=bool)

            if self.use_cwola:

                # stream_pm_phi1 = pm_ra[stream]
                # pm_p1_med, pm_p1_std= np.median(stream_pm_phi1), np.std(stream_pm_phi1)
                
                # # Define regions using 5 and 95 percentiles
                # lower_perc = np.percentile(stream_pm_phi1, 5)
                # upper_perc = np.percentile(stream_pm_phi1, 95)
                # lower_bound = pm_p1_med - 5*pm_p1_std
                # upper_bound = pm_p1_med + 5*pm_p1_std

                # regions = [[lower_perc, upper_perc], [lower_bound, upper_bound]]

                # sig_low, sig_high = regions[0]
                # side_low, side_high = regions[1]

                stream_pm_ra = pm_ra_cos_dec[stream]
                pm_ra_med, pm_ra_std= np.median(stream_pm_ra), np.std(stream_pm_ra)
                
                # Define regions using 5 and 95 percentiles
                lower_perc = np.percentile(stream_pm_ra, 5)
                upper_perc = np.percentile(stream_pm_ra, 95)
                lower_bound = pm_ra_med - 5*pm_ra_std
                upper_bound = pm_ra_med + 5*pm_ra_std

                regions = [[lower_perc, upper_perc], [lower_bound, upper_bound]]

                sig_low, sig_high = regions[0]
                side_low, side_high = regions[1]

                signal_region = (pm_ra > sig_low) & (pm_ra < sig_high)
                sideband_region = (pm_ra > side_low) & (pm_ra < side_high) & ~signal_region
                roi = signal_region | sideband_region

                ra, dec, pm_ra, pm_dec, gmag, color = ra[roi], dec[roi], pm_ra[roi], pm_dec[roi], gmag[roi], color[roi]
                pm_ra_error, pm_dec_error = pm_ra_error[roi], pm_dec_error[roi]
                parallax, parallax_error = parallax[roi], parallax_error[roi]
                stream = stream[roi]
                gmag0, rmag0, zmag0 = gmag0[roi], rmag0[roi], zmag0[roi]
                g_r, r_z, g_z = g_r[roi], r_z[roi], g_z[roi]
                signal_region = signal_region[roi]
                sideband_region = sideband_region[roi]
                pm_ra_for_storage = pm_ra_for_storage[roi]
            
            if self.use_cwola:
                embeddings = np.column_stack((ra, dec, pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))
            else:
                embeddings = np.column_stack((ra, dec, pm_ra, pm_dec, pm_ra_error, pm_dec_error, gmag, color, g_r, r_z, g_z, parallax, parallax_error))

            print(f'Total nan values in embeddings: {np.sum(np.isnan(embeddings))}')

            train_mask, val_mask, test_mask = get_mask_splits(embeddings, self.train_pct)
            
            self.scaler = StandardScaler()
            self.scaler.fit(embeddings)

            labels = np.column_stack((signal_region, stream)) if self.use_cwola else np.column_stack((stream, stream))
            
            train_dataset = GaiaDatasetLinear(torch.tensor(self.scaler.transform(embeddings[train_mask]), dtype=torch.float32),
                                                  torch.tensor(labels[train_mask], dtype=torch.float32),
                                                  torch.tensor(pm_ra_for_storage[train_mask], dtype=torch.float32))
            test_dataset = GaiaDatasetLinear(torch.tensor(self.scaler.transform(embeddings[test_mask]), dtype=torch.float32),
                                                    torch.tensor(labels[test_mask], dtype=torch.float32),
                                                    torch.tensor(pm_ra_for_storage[test_mask], dtype=torch.float32))
            val_dataset = GaiaDatasetLinear(torch.tensor(self.scaler.transform(embeddings[val_mask]), dtype=torch.float32),
                                                    torch.tensor(labels[val_mask], dtype=torch.float32),
                                                    torch.tensor(pm_ra_for_storage[val_mask], dtype=torch.float32))
            
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
            torch.save(self.train_loader, f'{os.environ["PSCRATCH"]}/C19_cwola/linear_train_loader_{self.batch_size}.pth')
            torch.save(self.test_loader, f'{os.environ["PSCRATCH"]}/C19_cwola/linear_test_loader_{self.batch_size}.pth')
            torch.save(self.val_loader, f'{os.environ["PSCRATCH"]}/C19_cwola/linear_val_loader_{self.batch_size}.pth')
            
    def train_dataloader(self):    
        print('Calling train_dataloader func')
        return self.train_loader
    def val_dataloader(self):
        print('Calling val_dataloader func')
        return self.val_loader
    def test_dataloader(self):
        print('Calling test_dataloader func')
        return self.test_loader
            
    
# This class will define the data module for the linear model
class CATHODELinearDataModule(L.LightningDataModule):
    def __init__(self, name, stream, load_data_dir = None, 
                 batch_size = 1024, train_pct = .8, 
                load_dataloaders = False):
        
        super().__init__()
        seed_everything(12345)
        self.load_data_dir = load_data_dir
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.load_dataloaders = load_dataloaders
        self.name = name
        self.stream = stream

    def setup(self, stage: str):
        if not self.load_dataloaders:
            if self.load_data_dir.endswith('.csv'):
                df = pd.read_csv(self.load_data_dir)
                df = Table.from_pandas(df)
            else:
                df = Table.read(self.load_data_dir)
            

            # id = np.array(df['source_id'])
            ra = np.array(df['ra']).astype('float64')
            dec = np.array(df['dec']).astype('float64')

            pm_ra  = np.array(df['pm_ra']).astype('float64')
            pm_dec = np.array(df['pm_dec']).astype('float64')
            
            gmag  = np.array(df['gmag'])
            color = np.array(df['color'])

            # parallax = np.array(df['parallax'])
            # pm_ra_error = np.array(df['pmra_error']).astype('float64')
            # pm_dec_error = np.array(df['pmdec_error']).astype('float64')
            # parallax_error = np.array(df['parallax_error'])

            rmag0 = np.array(df['rmag0'])
            g_r = np.array(df['g_r'])
            r_z = np.array(df['r_z'])

            stream = np.array(df['stream'])
            cwola_label = np.array(df['CWoLa_Label'], dtype=bool)

            sampled_data = (stream == 2)
            stream[sampled_data] = 0
        
            
            embeddings = np.column_stack((ra, dec, pm_ra, pm_dec, gmag, color, rmag0, g_r, r_z))
            print(f'Total nan values in embeddings: {np.sum(np.isnan(embeddings))}')

            train_mask, val_mask, test_mask = get_mask_splits(embeddings, self.train_pct)
            
            self.scaler = StandardScaler()
            self.scaler.fit(embeddings)

            labels = np.column_stack((cwola_label, stream))
            # id_plus_sample = np.column_stack((id, sampled_data.astype(bool)))
            id_plus_sample = sampled_data.astype(bool)

            train_dataset = CATHODEGaiaDatasetLinear(torch.tensor(self.scaler.transform(embeddings[train_mask]), dtype=torch.float32),
                                                  torch.tensor(labels[train_mask], dtype=torch.float32),
                                                  torch.tensor(id_plus_sample[train_mask], dtype=torch.float32))
            
            test_dataset = CATHODEGaiaDatasetLinear(torch.tensor(self.scaler.transform(embeddings[test_mask]), dtype=torch.float32),
                                                    torch.tensor(labels[test_mask], dtype=torch.float32),
                                                    torch.tensor(id_plus_sample[test_mask], dtype=torch.float32))
            
            val_dataset = CATHODEGaiaDatasetLinear(torch.tensor(self.scaler.transform(embeddings[val_mask]), dtype=torch.float32),
                                                    torch.tensor(labels[val_mask], dtype=torch.float32),
                                                    torch.tensor(id_plus_sample[val_mask], dtype=torch.float32))
            
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
            # Create directory if it doesn't exist
            os.makedirs(f'{os.environ["PSCRATCH"]}/{self.stream}_CATHODE_mlp', exist_ok=True)

            torch.save(self.train_loader, f'{os.environ["PSCRATCH"]}/{self.stream}_CATHODE_mlp/linear_train_loader_{self.name}.pth')
            torch.save(self.test_loader, f'{os.environ["PSCRATCH"]}/{self.stream}_CATHODE_mlp/linear_test_loader_{self.name}.pth')
            torch.save(self.val_loader, f'{os.environ["PSCRATCH"]}/{self.stream}_CATHODE_mlp/linear_val_loader_{self.name}.pth')
            
    def train_dataloader(self):    
        print('Calling train_dataloader func')
        return self.train_loader
    def val_dataloader(self):
        print('Calling val_dataloader func')
        return self.val_loader
    def test_dataloader(self):
        print('Calling test_dataloader func')
        return self.test_loader



# Now I will define the LightningModule for the linear model

class LitLinearModel(L.LightningModule):
    def __init__(self, lr, input_dim, EPOCHS, steps_per_epoch, pos_weight, num_layers=3, hidden_units=256, dropout=0.0):
        super().__init__()
        self.model = LinearModel(input_dim, num_layers=num_layers, hidden_units=hidden_units, dropout=dropout)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        self.lr = lr
        self.save_hyperparameters()
        self.logits = []
        self.labels = []
        self.train_logits, self.train_labels  = [], []
        self.val_logits, self.val_labels, self.val_true_labels  = [], [], []
        self.test_logits, self.test_labels = [], []
        self.EPOCHS = EPOCHS
        self.steps_per_epoch = steps_per_epoch
        
    def shared_step(self, batch, stage: str):
        #batch.to(device)
        x, y, pm_ra = batch
        y_pred = self.model(x).squeeze()
    
        y_cwola = y[:, 0]
        y_true = y[:, 1]

        loss = self.criterion(y_pred, y_cwola)

        return loss, y_pred.detach().cpu(), y_cwola.detach().cpu(), y_true.detach().cpu()

    def training_step(self, batch, batch_idx):
        loss, train_pred, train_true, _ = self.shared_step(batch, stage='train')
        if loss is not None:
            self.log("train loss", loss, on_epoch=True, prog_bar=True)
            self.train_logits.append(train_pred)
            self.train_labels.append(train_true)
        return loss
        # return self.shared_step(batch, mask_name="train_mask", stage="train")

    def validation_step(self, batch, batch_idx):
        loss, val_pred, val_true, val_actual = self.shared_step(batch,  stage='validation')
        if loss is not None:
            self.log("validation loss", loss, on_epoch=True)
            self.val_logits.append(val_pred)
            self.val_labels.append(val_true)
            self.val_true_labels.append(val_actual)
        # self.shared_step(batch, mask_name="val_mask", stage="validation")


    def test_step(self, batch, batch_idx):
        loss, test_pred, test_true, _ = self.shared_step(batch, stage='test')
        if loss is not None:
            self.log("test loss", loss, on_epoch=True)
            print(test_pred.shape)
            print(test_true.shape)
            self.test_logits.append(test_pred)
            self.test_labels.append(test_true)
        # self.shared_step(batch, mask_name="test_mask", stage="test")

    def on_train_epoch_end(self):
        if len(self.train_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.train_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.train_labels).numpy()
    
        train_f1 = f1_score(y_true, preds)
        train_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("train f1 score", train_f1)
        self.log("train MCC score", train_mcc)
        
        self.train_logits.clear()
        self.train_labels.clear()
        
    def on_validation_epoch_start(self):
        self.val_logits.clear()
        self.val_labels.clear()
        self.val_true_labels.clear()
        
    def on_validation_epoch_end(self):
        if len(self.val_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.val_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        preds_80 = (probs >= 0.8).astype(int)
        y_true = torch.cat(self.val_labels).numpy()
    
        val_f1 = f1_score(y_true, preds)
        val_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("validation f1 score", val_f1)
        self.log("validation MCC score", val_mcc)

        y_actual = torch.cat(self.val_true_labels).numpy()
    
        val_f1 = f1_score(y_actual, preds)
        val_mcc = matthews_corrcoef(y_actual, preds)

        val_f1_80 = f1_score(y_actual, preds_80)
        val_mcc_80 = matthews_corrcoef(y_actual, preds_80)
    
        self.log("True validation f1 score", val_f1)
        self.log("True validation MCC score", val_mcc)

        self.log("True validation f1 score (0.8 thresh)", val_f1_80)
        self.log("True validation MCC score (0.8 thresh)", val_mcc_80)
    
    def on_test_epoch_start(self):
        self.test_logits.clear()
        self.test_labels.clear()
        
    def on_test_epoch_end(self):

        if len(self.test_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.test_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.test_labels).numpy()
    
        test_f1 = f1_score(y_true, preds)
        test_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("test f1 score", test_f1)
        self.log("test MCC score", test_mcc)

        
        # if isinstance(self.logger, WandbLogger):
        #     self.logger.experiment.log({"pr": wandb.plot.pr_curve(y_true.flatten(), np.array(np.stack((1-probs, probs), axis=1)))})
        # cm = wandb.plot.confusion_matrix(y_true=y_true.flatten().tolist(), preds=preds.flatten().tolist(), class_names=["Background", "GD1"])
        # #cm = wandb.plot.confusion_matrix(y_true=np.array(model.labels).flatten().tolist(), preds=preds.flatten().tolist(), class_names=["Background", "GD1"])
        
        # if isinstance(self.logger, WandbLogger):
        #     self.logger.experiment.log({"conf_mat": cm})
            
        # self.cm=confusion_matrix(self.labels, preds, normalize=None)
        # self.cm = cm
        self.test_logits.clear()
        self.test_labels.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = {
        "scheduler": torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,  # peak LR
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.EPOCHS,
            pct_start=0.3,  # fraction of steps to increase LR ------- USUALLY AT 0.1
            div_factor=10.0,  # initial_lr = max_lr/div_factor
            final_div_factor=1e3  # final_lr = initial_lr/final_div_factor
        ),
        "interval": "step",  # Step per batch
        "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

    # --------------------------------------------------------------------------------------------------------------------------------------------------

# Here I define a new model which will implement monte carlo marginal BCE loss, using uncertainties in features
# --------------------------------------------------------------------------------------------------------------------------------------------------

# First I define my custom MC marginal BCE loss function
# The data will already be passed in as an array of shape (N, num_samples)

# def mc_marginal_bce_loss(y_pred, y_true):
#     """
#     Computes the Monte Carlo Marginal Binary Cross Entropy Loss.
    
#     Parameters:
#     y_pred (torch.Tensor): Predicted logits of shape (N, num_samples).
#     y_true (torch.Tensor): True labels of shape (num_samples,).
    
#     Returns:
#     torch.Tensor: The computed loss.
#     """
    
#     bce_func = nn.BCEWithLogitsLoss(reduction='none')
#     bce_losses = bce_func(y_pred, y_true.unsqueeze(0).expand_as(y_pred))  # Shape: (N, num_samples)

#     likelihoods = torch.exp(-bce_losses)  # Convert losses to likelihoods
#     log_likelihoods = torch.log(likelihoods + 1e-9)  # Add small constant for numerical stability

#     log_total_likelihood = torch.logsumexp(log_likelihoods, dim=1)  # Sum over samples in log space
#     log_num_samples = torch.log(torch.tensor(y_pred.size(1), dtype=torch.float32))
#     log_marginal_likelihood = log_total_likelihood - log_num_samples

#     loss = -torch.mean(log_marginal_likelihood)  # Negative log likelihood
#     return loss


def mc_marginal_bce_loss(y_pred, y_true, pos_weight: torch.Tensor = None):
    """
    Monte Carlo marginal likelihood loss for binary classification.

    Parameters
    ----------
    y_pred : torch.Tensor
        Logits predicted by the model with shape (N, B)
        where:
            N = number of Monte Carlo samples per input
            B = batch size (number of data points)

    y_true : torch.Tensor
        Ground truth labels with shape (B,).

    Returns
    -------
    torch.Tensor
        Scalar loss = - log [ (1/N) * sum_j p(y|x_j) ] averaged over batch.
    """

    # BCEWithLogitsLoss computes -log p(y|x) directly and is numerically stable.
    # Using reduction='none' gives shape (N, B)
    if pos_weight is None:
        bce = nn.BCEWithLogitsLoss(reduction='none')
    else:
        # pos_weight must be a 1-element tensor on the correct device
        pw = pos_weight.to(dtype=y_pred.dtype, device=y_pred.device)
        bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pw)

    # Expand y_true to match y_pred shape.
    # y_true: (B,) → (1, B) → (N, B)
    y_true_expanded = y_true.unsqueeze(0).expand_as(y_pred)

    # Compute elementwise BCE = -log p(y|x_MC_sample)
    # Shape: (N, B)
    bce_losses = bce(y_pred, y_true_expanded)


    # Compute log(sum_j p_j) stably
    # logsumexp(log p_j) = logsumexp(-bce_losses)
    log_total_likelihood = torch.logsumexp(-bce_losses, dim=0)  # sum over MC samples

    # Subtract log(N) → gives log( mean_j p_j )
    N = y_pred.size(0)
    log_marginal_likelihood = log_total_likelihood - torch.log(torch.tensor(float(N), device=y_pred.device))

    # Loss is negative log marginal likelihood, averaged over batch
    loss = -log_marginal_likelihood.mean()

    return loss


class EM_CATHODEGaiaDatasetLinear(Dataset):
    def __init__(self, data, labels, errors, id_plus_sample):
        self.data = data
        self.labels = labels
        self.errors = errors 
        self.id_plus_sample = id_plus_sample
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.errors[idx], self.id_plus_sample[idx]
    

# This class will define the data module for the linear model
class EM_CATHODELinearDataModule(L.LightningDataModule):
    def __init__(self, name, stream, load_data_dir = None, 
                 batch_size = 1024, train_pct = .8, 
                load_dataloaders = False, p_wiggle = 0):
        
        super().__init__()
        seed_everything(12345)
        self.load_data_dir = load_data_dir
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.load_dataloaders = load_dataloaders
        self.name = name
        self.stream = stream
        self.p_wiggle = p_wiggle

    def setup(self, stage: str):
        if not self.load_dataloaders:
            if self.load_data_dir.endswith('.csv'):
                df = pd.read_csv(self.load_data_dir)
                df = Table.from_pandas(df)
            else:
                df = Table.read(self.load_data_dir)

            id = np.array(df['source_id'])
            ra = np.array(df['ra']).astype('float64')
            dec = np.array(df['dec']).astype('float64')

            pm_ra  = np.array(df['pm_ra']).astype('float64')
            pm_dec = np.array(df['pm_dec']).astype('float64')
            
            gmag  = np.array(df['gmag'])
            color = np.array(df['color'])

  
            pm_ra_error = np.array(df['pm_ra_error']).astype('float64')
            pm_dec_error = np.array(df['pm_dec_error']).astype('float64')


            rmag0 = np.array(df['rmag0'])
            g_r = np.array(df['g_r'])
            r_z = np.array(df['r_z'])

            stream = np.array(df['stream'])
            cwola_label = np.array(df['CWoLa_Label'], dtype=bool)

            sampled_data = (stream == 2)
            stream[sampled_data] = 0
            
            embeddings = np.column_stack((ra, dec, pm_ra, pm_dec, gmag, color, rmag0, g_r, r_z))
            print(f'Total nan values in embeddings: {np.sum(np.isnan(embeddings))}')

            train_mask, val_mask, test_mask = get_mask_splits(embeddings, self.train_pct)
            
            self.scaler = StandardScaler()
        
            self.scaler.fit(embeddings)

            no_err = np.zeros_like(ra)
          
            # errors = np.column_stack((no_err, no_err, pm_ra_error, pm_dec_error, no_err, no_err, no_err, no_err, no_err))
            
            print(f'Percent Wiggle is {self.p_wiggle}')
            
            # ra_err, dec_err = np.ptp(ra)*self.p_wiggle*np.ones_like(ra), np.ptp(dec)*self.p_wiggle*np.ones_like(ra)
            # gmag_err, color_err = np.ptp(gmag)*self.p_wiggle*np.ones_like(ra), np.ptp(color)*self.p_wiggle*np.ones_like(ra)
            # rmag0_err, g_r_err, r_z_err = np.ptp(rmag0)*self.p_wiggle*np.ones_like(ra), np.ptp(g_r)*self.p_wiggle*np.ones_like(ra), np.ptp(r_z)*self.p_wiggle*np.ones_like(ra)

            ra_err, dec_err = np.std(ra)*self.p_wiggle*np.ones_like(ra)*.25, np.std(dec)*self.p_wiggle*np.ones_like(ra)*.25
            gmag_err, color_err = np.std(gmag)*self.p_wiggle*np.ones_like(ra), np.std(color)*self.p_wiggle*np.ones_like(ra)
            rmag0_err, g_r_err, r_z_err = np.std(rmag0)*self.p_wiggle*np.ones_like(ra), np.std(g_r)*self.p_wiggle*np.ones_like(ra), np.std(r_z)*self.p_wiggle*np.ones_like(ra)


            # Covariance stuff
            
            # pm_error = (pm_ra_error**2 + pm_dec_error **2) ** (0.5)
            # p_e_n = pm_error / np.median(pm_error)

            # ra_err, dec_err, gmag_err, color_err = ra_err * p_e_n, dec_err * p_e_n, gmag_err * p_e_n, color_err * p_e_n
            # rmag0_err, g_r_err, r_z_err = rmag0_err * p_e_n, g_r_err * p_e_n, r_z_err * p_e_n


            
            # print(np.max(pm_error))
            # print(np.percentile(pm_error, 75))
            # print(np.median(pm_error))
            # print(np.percentile(pm_error, 25))
            # print(np.min(pm_error))
            
            errors = np.column_stack((ra_err, dec_err, pm_ra_error, pm_dec_error, gmag_err, color_err, rmag0_err, g_r_err, r_z_err))
            
            # errors = np.column_stack((no_err, no_err, no_err, no_err, no_err, no_err, no_err, no_err, no_err))

            
            feature_scales = self.scaler.scale_  # numpy array, shape (D,)
            
            safe_scale = np.where(feature_scales == 0.0, 1.0, feature_scales)
            
            errors_scaled = errors / safe_scale  # shape (N, D)
            
            labels = np.column_stack((cwola_label, stream))
            # id_plus_sample = np.column_stack((id, sampled_data.astype(bool)))
            id_plus_sample = sampled_data.astype(bool)
            print(errors_scaled[0])
            # Data has shape: (N, D)
            # Errors has shape: (N, D)
            train_dataset = EM_CATHODEGaiaDatasetLinear(torch.tensor(self.scaler.transform(embeddings[train_mask]), dtype=torch.float32),
                                                  torch.tensor(labels[train_mask], dtype=torch.float32),
                                                  torch.tensor(errors_scaled[train_mask], dtype=torch.float32),
                                                  torch.tensor(id_plus_sample[train_mask], dtype=torch.float32))
            
            test_dataset = EM_CATHODEGaiaDatasetLinear(torch.tensor(self.scaler.transform(embeddings[test_mask]), dtype=torch.float32),
                                                    torch.tensor(labels[test_mask], dtype=torch.float32),
                                                    torch.tensor(errors_scaled[test_mask], dtype=torch.float32),
                                                    torch.tensor(id_plus_sample[test_mask], dtype=torch.float32))
            
            val_dataset = EM_CATHODEGaiaDatasetLinear(torch.tensor(self.scaler.transform(embeddings[val_mask]), dtype=torch.float32),
                                                    torch.tensor(labels[val_mask], dtype=torch.float32),
                                                    torch.tensor(errors_scaled[val_mask], dtype=torch.float32),
                                                    torch.tensor(id_plus_sample[val_mask], dtype=torch.float32))
            
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
            # Create directory if it doesn't exist
            os.makedirs(f'{os.environ["PSCRATCH"]}/{self.stream}_CATHODE_mlp', exist_ok=True)

            torch.save(self.train_loader, f'{os.environ["PSCRATCH"]}/{self.stream}_CATHODE_mlp/linear_train_loader_{self.name}.pth')
            torch.save(self.test_loader, f'{os.environ["PSCRATCH"]}/{self.stream}_CATHODE_mlp/linear_test_loader_{self.name}.pth')
            torch.save(self.val_loader, f'{os.environ["PSCRATCH"]}/{self.stream}_CATHODE_mlp/linear_val_loader_{self.name}.pth')
            
    def train_dataloader(self):    
        print('Calling train_dataloader func')
        return self.train_loader
    def val_dataloader(self):
        print('Calling val_dataloader func')
        return self.val_loader
    def test_dataloader(self):
        print('Calling test_dataloader func')
        return self.test_loader
    



class EM_LitLinearModel(L.LightningModule):
    # This model will implement EM with MC marginal BCE loss
    # To do so I will need to sample from the error distributions of the features during training and evaluation (before passing to the model)
    def __init__(self, lr, input_dim, EPOCHS, steps_per_epoch, pos_weight, num_layers=3, 
                hidden_units=256, dropout=0.0, num_mc_samples=10, pct_start=0.3, weight_decay = 0.0, layer_norm=False,
                activation='relu', residual=False, anneal_noise = False, noise_anneal_type = 'linear_decay', noise_anneal_dict = None):
        super().__init__()
        self.model = LinearModel(input_dim, num_layers=num_layers, hidden_units=hidden_units, dropout=dropout, 
                                 layer_norm=layer_norm, activation=activation, residual=residual)
        # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        self.pos_weight = pos_weight
        self.lr = lr
        self.num_mc_samples = num_mc_samples
        self.pct_start = pct_start
        self.weight_decay = weight_decay
        
        self.anneal_noise = anneal_noise
        self.noise_anneal_type = noise_anneal_type
        self.noise_anneal_dict = noise_anneal_dict
        self.save_hyperparameters()

        
        self.logits = []
        self.labels = []
        self.train_logits, self.train_labels  = [], []
        self.val_logits, self.val_labels, self.val_true_labels  = [], [], []
        self.test_logits, self.test_labels = [], []
        self.EPOCHS = EPOCHS
        self.steps_per_epoch = steps_per_epoch
        
        
        
    def shared_step(self, batch, stage: str):
        #batch.to(device)
        x, y, errors, _ = batch
        # x shape: (B, D)
        # errors shape: (B, D)
       
        # Sample from the error distributions, num_mc_samples times
        B, D = x.shape
        noise_factor = self.noise_scale_factor()

        err_mask = torch.ones_like(errors, dtype=torch.bool)
        err_mask[:, 3] = False
        err_mask[:, 4] = False
        errors = torch.where(err_mask, noise_factor * errors, errors)
        
        # x_samples = x.unsqueeze(1) + noise_factor * torch.randn(B, self.num_mc_samples, D).to(x.device) * errors.unsqueeze(1)  # Shape: (B, num_mc_samples, D) 
        x_samples = x.unsqueeze(1) + torch.randn(B, self.num_mc_samples, D).to(x.device) * errors.unsqueeze(1)
        
        assert x_samples.shape == (B, self.num_mc_samples, D), f"Expected shape (B, {self.num_mc_samples}, D), got {x_samples.shape}"

        
        if torch.isnan(x_samples).any() or torch.isinf(x_samples).any():
            raise RuntimeError("x_samples contains NaN/Inf")

    
        # print(f'x_samples shape: {x_samples.shape}')
        y_pred = self.model(x_samples).squeeze(-1)
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            raise RuntimeError("y_pred contains NaN/Inf")
        assert y_pred.shape == (B, self.num_mc_samples), f"Expected shape (B, {self.num_mc_samples}), got {y_pred.shape}"
        # y_pred shape: (B, num_mc_samples)
        # print(f'y_pred shape before permute: {y_pred.shape}')
        y_pred = y_pred.permute(1, 0)  # Shape: (num_mc_samples, B)
    

        y_cwola = y[:, 0]
        y_true = y[:, 1]

        loss = mc_marginal_bce_loss(y_pred, y_cwola, self.pos_weight)
        # loss has shape: scalar
        # I want to return y_pred in shape (B, 1) for later evaluation
        
        # y_pred = y_pred.permute(1, 0).mean(dim=1)  # Shape: (B,) - old code that returned logits
        probs_mc = torch.sigmoid(y_pred)  # (N_mc, B)
        p_marginal = probs_mc.mean(dim=0)    # (B,)

        # return loss, y_pred.detach().cpu(), y_cwola.detach().cpu(), y_true.detach().cpu()
        return loss, p_marginal.detach().cpu(), y_cwola.detach().cpu(), y_true.detach().cpu()

    def training_step(self, batch, batch_idx):
        loss, train_pred, train_true, _ = self.shared_step(batch, stage='train')
        if loss is not None:
            self.log("train loss", loss, on_epoch=True)
            self.train_logits.append(train_pred)
            self.train_labels.append(train_true)
        return loss
        

    def validation_step(self, batch, batch_idx):
        loss, val_pred, val_true, val_actual = self.shared_step(batch,  stage='validation')
        if loss is not None:
            self.log("validation loss", loss, on_epoch=True)
            self.val_logits.append(val_pred)
            self.val_labels.append(val_true)
            self.val_true_labels.append(val_actual)
        


    def test_step(self, batch, batch_idx):
        loss, test_pred, test_true, _ = self.shared_step(batch, stage='test')
        if loss is not None:
            self.log("test loss", loss, on_epoch=True)
            print(test_pred.shape)
            print(test_true.shape)
            self.test_logits.append(test_pred)
            self.test_labels.append(test_true)
        

    def on_train_epoch_end(self):
        if len(self.train_logits) == 0:
            return
        # probs = torch.sigmoid(torch.cat(self.train_logits)).numpy() - old code
        probs = torch.cat(self.train_logits).numpy()

        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.train_labels).numpy()
    
        train_f1 = f1_score(y_true, preds)
        train_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("train f1 score", train_f1)
        self.log("train MCC score", train_mcc)
        
        self.train_logits.clear()
        self.train_labels.clear()
        
    def on_validation_epoch_start(self):
        self.val_logits.clear()
        self.val_labels.clear()
        self.val_true_labels.clear()
        
    def on_validation_epoch_end(self):
        if len(self.val_logits) == 0:
            return
        # probs = torch.sigmoid(torch.cat(self.val_logits)).numpy() # - old
        probs = torch.cat(self.val_logits).numpy()
        assert (probs.min() >= 0 and probs.max() <=1), f'Probabilities are out of [0,1] range'
        preds = (probs >= 0.5).astype(int)
        preds_80 = (probs >= 0.8).astype(int)
        y_true = torch.cat(self.val_labels).numpy()
    
        val_f1 = f1_score(y_true, preds)
        val_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("validation f1 score", val_f1)
        self.log("validation MCC score", val_mcc)

        y_actual = torch.cat(self.val_true_labels).numpy()
    
        val_f1 = f1_score(y_actual, preds)
        val_mcc = matthews_corrcoef(y_actual, preds)

        val_f1_80 = f1_score(y_actual, preds_80)
        val_mcc_80 = matthews_corrcoef(y_actual, preds_80)
    
        self.log("True validation f1 score", val_f1)
        self.log("True validation MCC score", val_mcc)

        self.log("True validation f1 score (0.8 thresh)", val_f1_80, prog_bar = True)
        self.log("True validation MCC score (0.8 thresh)", val_mcc_80)
    
    def on_test_epoch_start(self):
        self.test_logits.clear()
        self.test_labels.clear()
        
    def on_test_epoch_end(self):

        if len(self.test_logits) == 0:
            return
        probs = torch.cat(self.test_logits).numpy()
        # probs = torch.sigmoid(torch.cat(self.test_logits)).numpy() # - old version
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.test_labels).numpy()
    
        test_f1 = f1_score(y_true, preds)
        test_mcc = matthews_corrcoef(y_true, preds)
    
        self.log("test f1 score", test_f1)
        self.log("test MCC score", test_mcc)

        self.test_logits.clear()
        self.test_labels.clear()
    
    def configure_optimizers(self):
        
        if self.weight_decay > 0.0:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        
        scheduler = {
        "scheduler": torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,  # peak LR
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.EPOCHS,
            pct_start=self.pct_start,  # fraction of steps to increase LR ------- USUALLY AT 0.1
            div_factor=10.0,  # initial_lr = max_lr/div_factor
            final_div_factor=1e3  # final_lr = initial_lr/final_div_factor
        ),
        "interval": "step",  # Step per batch
        "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
    def on_after_backward(self):
        """Check for NaN/Inf or exploding/vanishing gradients."""
        grad_norm = 0.0
        for name, p in self.named_parameters():
            if p.grad is None:
                continue

            g = p.grad
            
            # NaN / Inf detection
            if torch.isnan(g).any() or torch.isinf(g).any():
                print(f"[Gradient NaN/Inf detected] in {name}")
                self.log("debug/grad_nan_inf", 1)
                return

            # Aggregate for global norm
            grad_norm += g.norm(2).item() ** 2

        grad_norm = grad_norm ** 0.5
        self.log("debug/grad_norm", grad_norm, on_step=True, on_epoch=False)
        
        # Exploding / vanishing threshold checks
        # if grad_norm > 1e4:
            # print(f"[Exploding Gradients] Global grad norm = {grad_norm:.2e}")
            # self.log("debug/exploding_grad", grad_norm)
        if grad_norm < 1e-6:
            print(f"[Vanishing Gradients] Global grad norm = {grad_norm:.2e}")
            self.log("debug/vanishing_grad", grad_norm)

    def on_before_optimizer_step(self, optimizer):
        """Check parameter values before the optimizer updates them."""
        for name, p in self.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"[Parameter NaN/Inf detected] in {name}")
                self.log("debug/param_nan_inf", 1)
                return
    # def on_before_optimizer_step(self, optimizer):
    #     # This hook runs *after* gradient clipping, right before optimizer.step()
    #     grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf'))
    #     self.log("debug/grad_norm_post_clip", grad_norm, on_step=True)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        total = torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf'))
        self.log("debug/grad_norm_post_clip", total, on_step=True)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)

    # ----------------------------
    # FORWARD ACTIVATION CHECKS
    # ----------------------------
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Check model outputs for numerical issues."""
        # print(outputs)
        loss = outputs['loss']

        if torch.isnan(loss) or torch.isinf(loss):
            print("[Loss NaN/Inf detected]")
            self.log("debug/loss_nan_inf", 1)

    def noise_scale_factor(self):
        """Return a scalar in [noise_anneal_min, 1] depending on training progress."""

        if not self.anneal_noise or not self.training:
            return 1.0
    
        # compute fraction of training completed
        # global_step goes from 0 → steps_per_epoch*EPOCHS
        total_steps = self.steps_per_epoch * self.EPOCHS
        t = min(1.0, self.global_step / total_steps)

        noise_anneal_min = self.noise_anneal_dict['noise_anneal_min']
        
        if self.noise_anneal_type == "linear_decay":
            return 1.0 - (1.0 - noise_anneal_min) * t
    
        elif self.noise_anneal_type == "cosine":
            # return noise_anneal_min + 0.5 * (1 - noise_anneal_min) * (1 + np.cos(np.pi * t))
            f_max = self.noise_anneal_dict['f_max']
            t_peak = self.noise_anneal_dict['t_peak']
            
            if t < t_peak:
                # rising cosine bump
                return 1 + 0.5*(f_max - 1)*(1 - np.cos(np.pi * t / t_peak))
            else:
                # cosine decay
                return noise_anneal_min + 0.5*(f_max - noise_anneal_min)*(
                    1 + np.cos(np.pi * (t - t_peak) / (1 - t_peak))
                )
        
    
        elif self.noise_anneal_type == "exp":
            return noise_anneal_min + (1 - noise_anneal_min) * np.exp(-5 * t)
    
        # fallback
        return 1.0

    