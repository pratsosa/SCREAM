import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from astropy.table import Table
from sklearn.preprocessing import StandardScaler
import lightning as L
from lightning.pytorch import seed_everything

from scream.data.datasets import (
    GaiaDatasetLinear,
    CATHODEGaiaDatasetLinear,
    EM_CATHODEGaiaDatasetLinear,
)
from scream.data.transforms import get_mask_splits


class CATHODELinearDataModule(L.LightningDataModule):
    def __init__(self, name, stream, load_data_dir=None,
                 batch_size=1024, train_pct=.8,
                 load_dataloaders=False):

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

            ra = np.array(df['ra']).astype('float64')
            dec = np.array(df['dec']).astype('float64')

            pm_ra = np.array(df['pm_ra']).astype('float64')
            pm_dec = np.array(df['pm_dec']).astype('float64')

            gmag = np.array(df['gmag'])
            color = np.array(df['color'])

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
            id_plus_sample = sampled_data.astype(bool)

            train_dataset = CATHODEGaiaDatasetLinear(
                torch.tensor(self.scaler.transform(embeddings[train_mask]), dtype=torch.float32),
                torch.tensor(labels[train_mask], dtype=torch.float32),
                torch.tensor(id_plus_sample[train_mask], dtype=torch.float32))

            test_dataset = CATHODEGaiaDatasetLinear(
                torch.tensor(self.scaler.transform(embeddings[test_mask]), dtype=torch.float32),
                torch.tensor(labels[test_mask], dtype=torch.float32),
                torch.tensor(id_plus_sample[test_mask], dtype=torch.float32))

            val_dataset = CATHODEGaiaDatasetLinear(
                torch.tensor(self.scaler.transform(embeddings[val_mask]), dtype=torch.float32),
                torch.tensor(labels[val_mask], dtype=torch.float32),
                torch.tensor(id_plus_sample[val_mask], dtype=torch.float32))

            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

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


class EM_CATHODELinearDataModule(L.LightningDataModule):
    def __init__(self, name, stream, load_data_dir=None,
                 batch_size=1024, train_pct=.8,
                 load_dataloaders=False, p_wiggle=0):

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

            pm_ra = np.array(df['pm_ra']).astype('float64')
            pm_dec = np.array(df['pm_dec']).astype('float64')

            gmag = np.array(df['gmag'])
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

            print(f'Percent Wiggle is {self.p_wiggle}')

            ra_err = np.std(ra) * self.p_wiggle * np.ones_like(ra) * .25
            dec_err = np.std(dec) * self.p_wiggle * np.ones_like(ra) * .25
            gmag_err = np.std(gmag) * self.p_wiggle * np.ones_like(ra)
            color_err = np.std(color) * self.p_wiggle * np.ones_like(ra)
            rmag0_err = np.std(rmag0) * self.p_wiggle * np.ones_like(ra)
            g_r_err = np.std(g_r) * self.p_wiggle * np.ones_like(ra)
            r_z_err = np.std(r_z) * self.p_wiggle * np.ones_like(ra)

            errors = np.column_stack((ra_err, dec_err, pm_ra_error, pm_dec_error, gmag_err, color_err, rmag0_err, g_r_err, r_z_err))

            feature_scales = self.scaler.scale_
            safe_scale = np.where(feature_scales == 0.0, 1.0, feature_scales)
            errors_scaled = errors / safe_scale

            labels = np.column_stack((cwola_label, stream))
            id_plus_sample = sampled_data.astype(bool)
            print(errors_scaled[0])

            train_dataset = EM_CATHODEGaiaDatasetLinear(
                torch.tensor(self.scaler.transform(embeddings[train_mask]), dtype=torch.float32),
                torch.tensor(labels[train_mask], dtype=torch.float32),
                torch.tensor(errors_scaled[train_mask], dtype=torch.float32),
                torch.tensor(id_plus_sample[train_mask], dtype=torch.float32))

            test_dataset = EM_CATHODEGaiaDatasetLinear(
                torch.tensor(self.scaler.transform(embeddings[test_mask]), dtype=torch.float32),
                torch.tensor(labels[test_mask], dtype=torch.float32),
                torch.tensor(errors_scaled[test_mask], dtype=torch.float32),
                torch.tensor(id_plus_sample[test_mask], dtype=torch.float32))

            val_dataset = EM_CATHODEGaiaDatasetLinear(
                torch.tensor(self.scaler.transform(embeddings[val_mask]), dtype=torch.float32),
                torch.tensor(labels[val_mask], dtype=torch.float32),
                torch.tensor(errors_scaled[val_mask], dtype=torch.float32),
                torch.tensor(id_plus_sample[val_mask], dtype=torch.float32))

            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

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
