import torch
from torch.utils.data import Dataset


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


class EM_CATHODEGaiaDatasetLinear(Dataset):
    def __init__(self, data, labels, errors, id_plus_sample, source_id):
        self.data = data
        self.labels = labels
        self.errors = errors
        self.id_plus_sample = id_plus_sample
        self.source_id = source_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.errors[idx], self.id_plus_sample[idx], self.source_id[idx]
