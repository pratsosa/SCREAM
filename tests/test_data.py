import numpy as np
import pandas as pd
import torch
import pytest

from scream.data.datasets import EM_CATHODEGaiaDatasetLinear


# ---------------------------------------------------------------------------
# EM_CATHODEGaiaDatasetLinear
# ---------------------------------------------------------------------------

def _make_dataset(N=20, n_features=9, n_errors=9):
    torch.manual_seed(0)
    data = torch.randn(N, n_features)
    labels = torch.randint(0, 2, (N, 2)).float()
    errors = torch.rand(N, n_errors)
    id_plus_sample = torch.zeros(N)
    return EM_CATHODEGaiaDatasetLinear(data, labels, errors, id_plus_sample)


def test_dataset_len():
    ds = _make_dataset(N=20)
    assert len(ds) == 20


def test_dataset_getitem_shapes():
    ds = _make_dataset(N=20, n_features=9, n_errors=9)
    data, labels, errors, id_plus_sample = ds[0]
    assert data.shape == (9,)
    assert labels.shape == (2,)
    assert errors.shape == (9,)
    assert id_plus_sample.shape == ()


def test_dataset_getitem_types():
    ds = _make_dataset(N=20)
    data, labels, errors, id_plus_sample = ds[0]
    assert data.dtype == torch.float32
    assert labels.dtype == torch.float32


# ---------------------------------------------------------------------------
# EM_CATHODELinearDataModule — smoke test with synthetic CSV
# ---------------------------------------------------------------------------

def _make_synthetic_csv(tmp_path, N=100):
    """Write a minimal CSV with all columns EM_CATHODELinearDataModule expects."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "source_id":    np.arange(N, dtype=np.int64),
        "ra":           rng.uniform(120.0, 160.0, N),
        "dec":          rng.uniform(20.0,  50.0,  N),
        "pm_ra":        rng.uniform(-15.0, -5.0,  N),
        "pm_dec":       rng.uniform(-5.0,   5.0,  N),
        "gmag":         rng.uniform(15.0,  20.2,  N),
        "color":        rng.uniform(0.5,    1.0,  N),
        "pm_ra_error":  rng.uniform(0.01,   0.5,  N),
        "pm_dec_error": rng.uniform(0.01,   0.5,  N),
        "rmag0":        rng.uniform(15.0,  20.0,  N),
        "g_r":          rng.uniform(0.2,    0.6,  N),
        "r_z":          rng.uniform(-0.1,   0.3,  N),
        "stream":       rng.choice([0, 1, 2], N),   # 2 = generated
        "CWoLa_Label":  rng.integers(0, 2, N),
    })
    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_datamodule_setup_produces_loaders(tmp_path, monkeypatch):
    from pathlib import Path
    import scream.data.datamodules as dm_module

    # Redirect get_scratch_dir so the module writes to tmp_path
    monkeypatch.setattr(dm_module, "get_scratch_dir", lambda stream: tmp_path)

    from scream.data.datamodules import EM_CATHODELinearDataModule

    csv_path = _make_synthetic_csv(tmp_path, N=100)
    dm = EM_CATHODELinearDataModule(
        name="test_run",
        stream="gd1",
        load_data_dir=csv_path,
        batch_size=32,
        train_pct=0.8,
    )
    dm.setup("fit")

    # Loaders must exist and yield batches of the right structure
    batch = next(iter(dm.train_dataloader()))
    data, labels, errors, id_plus_sample = batch
    assert data.ndim == 2
    assert data.shape[1] == 9        # 9 embedding features
    assert labels.ndim == 2
    assert errors.shape[1] == 9
    assert id_plus_sample.ndim == 1
