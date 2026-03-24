# CLAUDE.md — SCREAM Repository Guide

## Project Overview

**SCREAM: Stream Characterization with Error-Aware Machine Learning**

SCREAM is a weakly supervised ML framework for detecting and characterizing Milky Way stellar streams, applied primarily to the GD-1 stellar stream. It is based on the CATHODE anomaly detection framework and operates in two stages:

1. **Normalizing Flow (NF)**: Learns a data-driven background distribution from sideband regions.
2. **MLP Classifier**: Trained (unsupervised/weakly supervised) to distinguish background stars from stream members by comparing the real data distribution to the NF-generated background.

Key innovation: measurement uncertainties (astrometric errors from Gaia) are explicitly incorporated into MLP training via Monte Carlo (MC) marginalization, a technique called **Error Marginalization CATHODE (EM-CATHODE)**.

The model is evaluated against STREAMFINDER labels and validated with DESI radial velocity data (held out from training entirely).

---

## Repository Status

This codebase supports a paper being prepared for conference submission. Current priorities:
- **Clean up** code and notebooks before publication
- **Refactor** for usability and extensibility

---

## Data Sources

- **Gaia DR3**: Astrometric features (RA, DEC, PM_RA, PM_DEC, parallax, G magnitude, BP-RP color) and their associated measurement errors
- **DESI Legacy Imaging Survey DR9**: Photometric data (R-band magnitude, g-r, r-z, g-z colors)
- **DESI Radial Velocities**: Used only for post-hoc validation, not in training
- **STREAMFINDER labels**: Used for evaluation

Data files live at `$PSCRATCH` (NERSC), e.g. `/pscratch/sd/p/pratsosa/GD1_errs_LS_CATHODE_*.csv`.

---

## File Structure

```
SCREAM/
├── utils.py                        # All model classes, data modules, loss functions (~2300 lines)
├── GD1_EM_CATHODE_MLP.ipynb        # Main analysis notebook: training + evaluation for GD-1
├── C19 CATHODE pzflow example.ipynb # C19 stream analysis with pzflow
├── GD1_preds_analysis.ipynb        # Post-training prediction analysis and visualization
└── README.md                       # Brief project description
```

---

## Core Module: `utils.py`

Everything lives in `utils.py`. Key components:

### Loss Function
- **`mc_marginal_bce_loss`**: Monte Carlo marginal BCE loss. Perturbs inputs by sampled errors and averages logits over MC samples before computing loss. This is the core of the error-aware training.

### Datasets
| Class | Description |
|---|---|
| `GaiaDatasetLinear` | Basic dataset |
| `CATHODEGaiaDatasetLinear` | CATHODE variant with sample IDs |
| `EM_CATHODEGaiaDatasetLinear` | Error-aware variant — stores features, labels, and per-star errors |

### Data Modules (PyTorch Lightning DataModules)
| Class | Stream | Notes |
|---|---|---|
| `GaiaDataModule` | Generic | KNN graph construction, saves to PSCRATCH |
| `GaiaDataModuleCustom` | Generic | Adds DECaLS photometry |
| `GaiaDataModuleGD1` | GD-1 | Handles CSV/FITS, includes measurement errors |
| `GaiaDataModuleGD1CATHODE` | GD-1 | CATHODE variant; `no_errs` flag to exclude errors |
| `GaiaDataModuleC19` | C19 | CWOLA-based sideband/signal regions |
| `GaiaDataModuleC19CATHODE` | C19 | CATHODE variant for C19 |
| `CATHODELinearDataModule` | Any | CATHODE linear model data module |
| `EM_CATHODELinearDataModule` | Any | Full error-marginalization pipeline; primary data module used in training |

**`EM_CATHODELinearDataModule`** is the main one used in `GD1_EM_CATHODE_MLP.ipynb`. It:
- Loads data, applies quality cuts (parallax < 1, color 0.5–1.0, mag < 20.2)
- Defines signal region by proper motion (CWOLA weak labels)
- Scales features with `StandardScaler`, scales errors proportionally (`p_wiggle` parameter)
- Saves train/val/test loaders to `$PSCRATCH/{stream}_CATHODE_mlp/`

### Neural Network Models
| Class | Description |
|---|---|
| `LinearModel` | Flexible MLP — configurable layers, dropout, layer norm, residual blocks |
| `DeeperGCN` | Graph convolutional network (GENConv layers) for spatial graphs |

`LinearModel` supports optional **Pre-LN residual blocks** (`ResidualBlock`: LayerNorm → Linear → Activation → Linear → skip).

### PyTorch Lightning Wrappers
| Class | Description |
|---|---|
| `LitLinearModel` | Standard BCE training for `LinearModel` |
| `EM_LitLinearModel` | MC marginalization training; supports **noise annealing** |
| `LitDeepGCN` | BCE training for `DeeperGCN` |
| `LitDeepGCN_MCMBCE` | MC marginal BCE training for `DeeperGCN` |

**`EM_LitLinearModel`** is the primary training class. Key features:
- `n_mc_samples`: number of MC error samples per forward pass
- **Noise annealing**: gradually reduces error perturbation over training (`linear_decay`, `cosine`, or `exponential` schedule via `noise_scale_factor()`)
- Gradient monitoring for NaN/Inf detection
- AdamW optimizer with weight decay
- OneCycleLR scheduler

---

## Training Workflow

The full pipeline (as run in `GD1_EM_CATHODE_MLP.ipynb`):

1. **Normalizing Flow (external)**: A NF (e.g. pzflow) is trained on sideband data to learn the background distribution. It generates synthetic background samples.

2. **Data preparation** via `EM_CATHODELinearDataModule`:
   - Real data = sideband + signal region (CWOLA labels distinguish them)
   - Sampled data = NF-generated background
   - Combined into train/val/test DataLoaders

3. **Model training** via `EM_LitLinearModel` + PyTorch Lightning `Trainer`:
   ```python
   Trainer(accelerator='gpu', max_epochs=100, callbacks=[EarlyStopping(monitor='val_f1')])
   ```
   - WandB logging enabled
   - Early stopping on validation F1

4. **Evaluation** (`GD1_preds_analysis.ipynb`):
   - ROC and PR curves
   - Optimal threshold via F1/recall tradeoff
   - Confusion matrix
   - Spatial (RA/DEC), proper motion, and CMD visualizations
   - Cross-match with DESI RV data
   - Comparison against STREAMFINDER and external catalogs

5. **Output**: Predictions CSV with `source_id` for external cross-matching.

---

## Feature Sets

**GD-1 with EM-CATHODE** (default ~9 features):
- RA, DEC (position)
- PM_DEC (proper motion)
- PM_RA_ERROR, PM_DEC_ERROR (astrometric uncertainties used for MC sampling)
- GMAG (G-band magnitude)
- COLOR (BP-RP)

**Extended with DECaLS photometry** (~11–13 features):
- Above + RMAG0, G_R, R_Z (and optionally G_Z, PARALLAX, PARALLAX_ERROR)

**Label structure** (`y` array):
- `y[:, 0]` = CWOLA weak label (0 = sideband, 1 = signal region)
- `y[:, 1]` = True stream label (STREAMFINDER or similar; only for evaluation)

---

## Technology Stack

| Category | Libraries |
|---|---|
| ML framework | PyTorch, PyTorch Lightning |
| Graph models | PyTorch Geometric |
| Normalizing flows | pzflow |
| Astronomy I/O | Astropy |
| Data | Pandas, NumPy |
| Metrics | scikit-learn |
| Experiment tracking | Weights & Biases (WandB) |
| Visualization | Matplotlib, Seaborn |

---

## Compute Environment

Runs on **NERSC** (HPC cluster). Training uses GPU (`accelerator='gpu'`). Large data files and model checkpoints are written to `$PSCRATCH`.

---

## Refactoring Goals

When cleaning up for publication and future extensibility, key areas to address:

- **`utils.py`** is monolithic (~2300 lines). Candidate splits:
  - `models.py` — `LinearModel`, `DeeperGCN`, `ResidualBlock`
  - `lightning_modules.py` — all `Lit*` classes
  - `data_modules.py` — all `*DataModule` classes and datasets
  - `losses.py` — `mc_marginal_bce_loss` and other loss functions
  - `utils.py` — only true utilities (`get_mask_splits`, `get_activation`, `count_parameters`)

- Multiple `DataModule` classes share logic (masking, scaling, sideband definition) that could be unified via a base class.

- Stream-specific configuration (feature lists, file paths, proper motion cuts) is scattered through data modules — could be extracted into config files or dataclasses.

- Notebooks contain duplicated evaluation/plotting code that could move to shared helper functions.
