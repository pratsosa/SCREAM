from pathlib import Path

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
from scream.utils.hpc import get_scratch_dir


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

            loaders_dir = get_scratch_dir(self.stream) / "loaders"
            loaders_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.train_loader, loaders_dir / f"linear_train_loader_{self.name}.pth")
            torch.save(self.test_loader, loaders_dir / f"linear_test_loader_{self.name}.pth")
            torch.save(self.val_loader, loaders_dir / f"linear_val_loader_{self.name}.pth")

    def train_dataloader(self):
        print('Calling train_dataloader func')
        return self.train_loader

    def val_dataloader(self):
        print('Calling val_dataloader func')
        return self.val_loader

    def test_dataloader(self):
        print('Calling test_dataloader func')
        return self.test_loader


def _gaia_extinction_numpy(G, Bp, Rp, ebv, n_iter=10):
    """
    Numpy port of mw_extinction_gaia (dev/transforms.py) using 10 iterations.
    Returns (A_G, A_Bp, A_Rp) in magnitudes.
    """
    A0 = 3.1 * ebv
    curbp = np.clip(Bp - Rp, -2.0, 5.0)
    for _ in range(n_iter):
        AG  = (0.9761 + (-0.1704)*curbp + 0.0086*curbp**2 + 0.0011*curbp**3
               + (-0.0438)*A0 + 0.0013*A0**2 + 0.0099*curbp*A0) * A0
        ABp = (1.1517 + (-0.0871)*curbp + (-0.0333)*curbp**2 + 0.0173*curbp**3
               + (-0.0230)*A0 + 0.0006*A0**2 + 0.0043*curbp*A0) * A0
        ARp = (0.6104 + (-0.0170)*curbp + (-0.0026)*curbp**2 + (-0.0017)*curbp**3
               + (-0.0078)*A0 + 0.00005*A0**2 + 0.0006*curbp*A0) * A0
        curbp = np.clip((Bp - Rp) - ABp + ARp, -2.0, 5.0)
    return AG, ABp, ARp


class EM_CATHODELinearDataModule(L.LightningDataModule):
    def __init__(self, name, stream, load_data_dir=None,
                 batch_size=1024, train_pct=.8,
                 load_dataloaders=False,
                 subsample_generated_seed=None):

        super().__init__()
        seed_everything(12345)
        self.load_data_dir = load_data_dir
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.load_dataloaders = load_dataloaders
        self.name = name
        self.stream = stream
        self.subsample_generated_seed = subsample_generated_seed

    def setup(self, stage: str):
        if not self.load_dataloaders:
            if self.load_data_dir.endswith('.csv'):
                df = pd.read_csv(self.load_data_dir)
            else:
                df = Table.read(self.load_data_dir).to_pandas()

            # Drop rows with non-positive flux errors (NF can occasionally generate
            # unphysical negative values; these cause divergence in extinction_gaia).
            flux_err_cols = ['phot_g_flux_err', 'phot_bp_flux_err', 'phot_rp_flux_err',
                             'flux_err_g', 'flux_err_r', 'flux_err_z']
            n_before = len(df)
            df = df[(df[flux_err_cols] > 0).all(axis=1)].reset_index(drop=True)
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                print(f"Dropped {n_dropped} rows with non-positive flux errors")

            if self.subsample_generated_seed is not None:
                df1 = df[df["CWoLa_Label"] == 1]
                df0 = df[df["CWoLa_Label"] == 0]
                df0 = df0.sample(n=len(df1), random_state=self.subsample_generated_seed)
                df = pd.concat([df1, df0], axis=0).reset_index(drop=True)
                print(f"Subsampled generated data to 1:1 ratio — {len(df1)} observed, {len(df0)} generated")

            # --- Raw feature columns (N, 10) — unscaled, fed to dataset as-is ---
            phi1    = np.array(df['phi1']).astype('float64')
            phi2    = np.array(df['phi2']).astype('float64')
            pm_phi1 = np.array(df['pm_phi1']).astype('float64')
            pm_phi2 = np.array(df['pm_phi2']).astype('float64')
            G_mag   = np.array(df['G_mag']).astype('float64')
            Bp_mag  = np.array(df['Bp_mag']).astype('float64')
            Rp_mag  = np.array(df['Rp_mag']).astype('float64')
            g_mag   = np.array(df['g_mag']).astype('float64')
            r_mag   = np.array(df['r_mag']).astype('float64')
            z_mag   = np.array(df['z_mag']).astype('float64')

            raw_features = np.column_stack([phi1, phi2, pm_phi1, pm_phi2,
                                             G_mag, Bp_mag, Rp_mag, g_mag, r_mag, z_mag])
            print(f'Total nan values in raw_features: {np.sum(np.isnan(raw_features))}')

            # --- Error columns (N, 11) — EBV packed last, passed unscaled ---
            phot_g_flux_err  = np.array(df['phot_g_flux_err']).astype('float64')
            phot_bp_flux_err = np.array(df['phot_bp_flux_err']).astype('float64')
            phot_rp_flux_err = np.array(df['phot_rp_flux_err']).astype('float64')
            flux_err_g       = np.array(df['flux_err_g']).astype('float64')
            flux_err_r       = np.array(df['flux_err_r']).astype('float64')
            flux_err_z       = np.array(df['flux_err_z']).astype('float64')
            pmra_error       = np.array(df['pmra_error']).astype('float64')
            pmdec_error      = np.array(df['pmdec_error']).astype('float64')
            ra_error         = np.array(df['ra_error']).astype('float64')
            dec_error        = np.array(df['dec_error']).astype('float64')
            ebv              = np.array(df['ebv']).astype('float64')

            errors = np.column_stack([phot_g_flux_err, phot_bp_flux_err, phot_rp_flux_err,
                                       flux_err_g, flux_err_r, flux_err_z,
                                       pmra_error, pmdec_error, ra_error, dec_error,
                                       ebv])

            # --- Compute extinction-corrected MLP features for scaler fitting ---
            AG, ABp, ARp = _gaia_extinction_numpy(G_mag, Bp_mag, Rp_mag, ebv)
            G0    = G_mag  - AG
            Bp0   = Bp_mag - ABp
            Rp0   = Rp_mag - ARp
            g0    = g_mag  - 3.214 * ebv
            r0    = r_mag  - 2.165 * ebv
            z0    = z_mag  - 1.211 * ebv
            BpRp0 = Bp0 - Rp0
            gr0   = g0  - r0
            rz0   = r0  - z0

            mlp_features = np.column_stack([phi1, phi2, pm_phi1, pm_phi2,
                                             G0, BpRp0, r0, gr0, rz0])
            print(f'Total nan values in mlp_features: {np.sum(np.isnan(mlp_features))}')

            train_mask, val_mask, test_mask = get_mask_splits(raw_features, self.train_pct)

            self.scaler = StandardScaler()
            self.scaler.fit(mlp_features[train_mask])

            stream = np.array(df['stream'])
            cwola_label = np.array(df['CWoLa_Label'], dtype=bool)
            sampled_data = (stream == 2)
            stream[sampled_data] = 0

            labels = np.column_stack((cwola_label, stream))
            id_plus_sample = sampled_data.astype(bool)

            train_dataset = EM_CATHODEGaiaDatasetLinear(
                torch.tensor(raw_features[train_mask], dtype=torch.float32),
                torch.tensor(labels[train_mask], dtype=torch.float32),
                torch.tensor(errors[train_mask], dtype=torch.float32),
                torch.tensor(id_plus_sample[train_mask], dtype=torch.float32))

            test_dataset = EM_CATHODEGaiaDatasetLinear(
                torch.tensor(raw_features[test_mask], dtype=torch.float32),
                torch.tensor(labels[test_mask], dtype=torch.float32),
                torch.tensor(errors[test_mask], dtype=torch.float32),
                torch.tensor(id_plus_sample[test_mask], dtype=torch.float32))

            val_dataset = EM_CATHODEGaiaDatasetLinear(
                torch.tensor(raw_features[val_mask], dtype=torch.float32),
                torch.tensor(labels[val_mask], dtype=torch.float32),
                torch.tensor(errors[val_mask], dtype=torch.float32),
                torch.tensor(id_plus_sample[val_mask], dtype=torch.float32))

            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

            loaders_dir = get_scratch_dir(self.stream) / "loaders"
            loaders_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.train_loader, loaders_dir / f"linear_train_loader_{self.name}.pth")
            torch.save(self.test_loader, loaders_dir / f"linear_test_loader_{self.name}.pth")
            torch.save(self.val_loader, loaders_dir / f"linear_val_loader_{self.name}.pth")

    def train_dataloader(self):
        print('Calling train_dataloader func')
        return self.train_loader

    def val_dataloader(self):
        print('Calling val_dataloader func')
        return self.val_loader

    def test_dataloader(self):
        print('Calling test_dataloader func')
        return self.test_loader
