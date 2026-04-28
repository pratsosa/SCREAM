# SCREAM

**S**tream **C**haracterization with **E**rror-**A**ware **M**achine Learning

SCREAM is a research pipeline for detecting stellar streams in Gaia + DECaLS photometric data using a combination of normalizing flows and a Monte Carlo error-marginalizing neural classifier. The codebase is currently focused on the GD-1 stream and is under active development.

---

## Scientific Context

Stellar streams are the remnants of tidally disrupted globular clusters or dwarf galaxies, stretched into thin arcs across the Milky Way halo. Their kinematics and chemistry are sensitive probes of the Galactic potential and dark matter substructure. SCREAM targets the detection of stellar stream members in the presence of a dominant Galactic foreground, using a data-driven approach that requires no simulation-based background model.

The current application is the GD-1 stream, using a cross-matched catalog of Gaia astrometry (proper motions, G-band photometry, BP−RP color) and DECaLS photometry (r₀, g₀−r₀, r₀−z₀), expressed in stream-aligned coordinates ($\phi_1$, $\phi_2$, $\mu_{1}$, $\mu_{2}$). Ground truth labels from STREAMFINDER are used for evaluation only and are not used during training. Results are further validated against DESI radial velocity data.

---

## Method

SCREAM implements **EM-CATHODE**: an extension of the CATHODE anomaly-detection framework ([Hallin et al. 2022](https://ui.adsabs.harvard.edu/abs/2022PhRvD.106e55006H/abstract)) that marginalizes over measurement errors via Monte Carlo sampling during both training and inference.

The pipeline has two sequential stages.

### Stage 1 — Normalizing Flow and Background Generation

A normalizing flow (pzflow `RollingSplineCoupling`) is trained on stars in the *sideband* region — defined as those outside the signal region in $\mu_{1}$ (proper motion along the stream). The flow models the joint distribution of all observables conditioned on $\mu_{1}$, which serves as the interpolating variable across the signal region.

After training, the flow is used to generate synthetic background samples conditioned on $\mu_{1}$ values drawn from a kernel density estimate of the signal-region stars. This produces a generated background catalog at a configurable multiplicity (default 4× the number of signal-region stars).

### Stage 2 — Error-Marginalizing MLP Classifier

The signal-region stars (label 1) and the flow-generated background (label 0) are used as a weakly supervised training set under the **Classification Without Labels (CWoLa)** paradigm. An MLP classifier is trained to distinguish the two populations using the full set of astrometric and photometric features.

During each training step, each star is perturbed *N* times using its true per-star measurement errors from the Gaia and DECaLS catalogs. Photometric features are sampled in flux space — each band's magnitude is converted to flux, perturbed by the measured per-star flux uncertainty, then converted back to a magnitude — after which extinction correction is applied iteratively to each MC sample in the forward pass. Astrometric features (φ₁, φ₂, μ₁, μ₂) are perturbed directly using their measured per-star uncertainties. The 11 error quantities used (6 photometric flux errors + 4 astrometric errors + E(B−V)) are all real catalog values; no artificial noise scaling is applied.

The loss is the **Monte Carlo marginal binary cross-entropy**:

$$\mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \log \left[ \frac{1}{N} \sum_{j=1}^{N} p(y_b \mid \tilde{x}_{b,j}) \right]$$

where $\tilde{x}_{b,j}$ is the $j$-th noise-perturbed sample of star $b$ and $p(y|\tilde{x})$ is the predicted probability. This formulation marginalizes over measurement uncertainty in a principled way, making the classifier's decisions robust to errors.

During validation and inference, 10 MC samples are drawn per star and averaged in probability space. Validation metrics include AUC (Mann-Whitney U, averaged over MC samples and computed on nominal features) and MCE (minimum classifier error), in addition to precision, recall, and F1 at a fixed threshold.

---

## Repository Structure

```
SCREAM/
├── scream/                     # Python package (pip install -e .)
│   ├── config/
│   │   └── schema.py           # StreamConfig and TrainConfig dataclasses
│   ├── data/
│   │   ├── datasets.py         # GaiaDatasetLinear, CATHODEGaiaDatasetLinear,
│   │   │                       #   EM_CATHODEGaiaDatasetLinear
│   │   ├── datamodules.py      # EM_CATHODELinearDataModule (primary),
│   │   │                       #   CATHODELinearDataModule
│   │   └── transforms.py       # get_mask_splits(), feature scaling helpers
│   ├── flow/
│   │   ├── trainer.py          # Normalizing flow training (pzflow / JAX)
│   │   ├── sampler.py          # KDE conditioning + background sample generation
│   │   └── diagnostics.py      # Flow loss curves and feature histograms
│   ├── losses/
│   │   └── mc_marginal.py      # mc_marginal_bce_loss()
│   ├── models/
│   │   ├── mlp.py              # LinearModel, ResidualBlock, get_activation()
│   │   ├── lit_em_mlp.py       # EM_LitLinearModel (PyTorch Lightning)
│   │   └── lit_mlp.py          # LitLinearModel (baseline, no error marginalisation)
│   ├── plotting/
│   │   ├── evaluation.py       # Confusion matrices
│   │   ├── spatial.py          # φ₁/φ₂ sky position plots
│   │   ├── kinematics.py       # Proper motion track plots
│   │   └── photometry.py       # Gaia and DECaLS color–magnitude diagrams
│   └── utils/
│       ├── hpc.py              # PSCRATCH path helpers (get_scratch_dir, date_string)
│       ├── io.py               # FITS/CSV I/O wrappers
│       └── metrics.py          # count_parameters()
│
├── configs/
│   ├── streams/
│   │   └── gd1.yaml            # GD-1 stream: features, error features, data paths, quality cuts
│   └── experiments/
│       └── em_cathode_mlp.yaml # MLP hyperparameters
│
├── scripts/
│   ├── train_flow.py           # CLI: train normalizing flow + generate background
│   ├── train_em_cathode.py     # CLI: single-run MLP training (exploratory)
│   ├── cross_validate.py       # CLI: K-fold cross-validation (primary paper path)
│   ├── evaluate.py             # CLI: inference + plots for a single run
│   ├── evaluate_cv.py          # CLI: aggregate CV predictions + diagnostic plots
│   └── sweep_train.py          # CLI: W&B hyperparameter sweep agent
│
├── slurm/
│   ├── train_flow.sh           # NERSC batch script for flow training (1 GPU, 1h30)
│   ├── train_em_cathode.sh     # NERSC batch script for MLP training (1 GPU, 4h)
│   └── sweep_em_cathode.sh     # NERSC batch script for sweep agents
│
├── data/                       # Data pipeline scripts
│   ├── GD1_download.py         # Query DataLab: Gaia × DECaLS → HDF5
│   ├── GD1_data_prep.py        # Preprocess HDF5 → FITS (stream coords, quality cuts)
│   ├── fetcher_new.py          # Parallel DataLab query engine (used by GD1_download.py)
│   ├── adql_utils.py           # Coordinate transforms (φ₁/φ₂, μ₁/μ₂)
│   ├── transforms.py           # Flux/magnitude conversion, extinction helpers
│   └── README.md               # Documents that real data lives in $PSCRATCH
│
├── figures/                    # Paper figure scripts and rendered outputs
│   ├── crossmatch_DESI.py      # Join CV predictions + DESI VRAD → FITS
│   ├── validation_figure.py    # Generate 3-panel validation figure
│   └── *.pdf / *.png           # Rendered figure files
│
├── tests/
│   ├── test_losses.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_transforms.py
│
├── dev/                        # Exploratory and reference material (not part of pipeline)
└── notebooks/                  # Exploratory scratch notebooks (not part of pipeline)
```

---

## Data

Raw data is stored in `$PSCRATCH` and is never committed to this repository. See [data/README.md](data/README.md) for a description of the input files. The authoritative paths are defined in `configs/streams/gd1.yaml` and consumed directly by the training scripts.

The raw input for the normalizing flow is a FITS file containing the Gaia × DECaLS cross-matched catalog in stream-aligned coordinates, with precomputed photometric quantities, per-star measurement errors, and a `signal_region` boolean column. The generated background CSV produced by `train_flow.py` is the direct input to `train_em_cathode.py` and `cross_validate.py`.

---

## Environment Setup

The package is designed for use on NERSC Perlmutter with the `myenv` conda environment.

```bash
module load conda
conda activate myenv
cd /global/homes/p/pratsosa/SCREAM
pip install -e .
```

The editable install makes `scream.*` importable from any directory without `PYTHONPATH` manipulation. The `pzflow` / JAX dependencies are only required for the flow training stage; all MLP training and evaluation steps function without them.

---

## Configuration

All stream-specific parameters and model hyperparameters are defined in YAML files and loaded into typed dataclasses at runtime.

**`configs/streams/gd1.yaml`** — stream geometry, feature lists, error feature lists, data paths, and quality cuts:

```yaml
name: gd1
raw_data_path: /pscratch/sd/p/pratsosa/GD-1_gaia_x_decals_040726.fits
generated_data_path: /pscratch/sd/p/pratsosa/GD1_SCREAM/generated/<run_id>/gd1_generated.csv

features: [phi1, phi2, pm_phi1, pm_phi2, G0, Bp0_Rp0, rmag0, g0_r0, r0_z0]

# True per-star measurement errors; EBV always last
error_features:
  [phot_g_flux_err, phot_bp_flux_err, phot_rp_flux_err,
   flux_err_g, flux_err_r, flux_err_z,
   pmra_error, pmdec_error, ra_error, dec_error, ebv]

flow_data_columns: [phi1, phi2, pm_phi2, G_mag, Bp_mag, Rp_mag, g_mag, r_mag, z_mag,
                    phot_g_flux_err, phot_bp_flux_err, phot_rp_flux_err,
                    flux_err_g, flux_err_r, flux_err_z,
                    pmra_error, pmdec_error, ra_error, dec_error]
flow_cond_columns: [pm_phi1]

pm_ra_signal_range: null   # null → use pre-computed 'signal_region' FITS column

quality_cuts:
  parallax_max: 1.0
  color_min:    0.5
  color_max:    1.0
  gmag_max:    20.2
```

**`configs/experiments/em_cathode_mlp.yaml`** — model architecture and training hyperparameters:

```yaml
hidden_units: [256, 128, 32]
num_layers: 3
activation: gelu
use_layer_norm: false
use_residual: false
dropout: 0.0

lr: 0.02389
pct_start: 0.1       # fraction of training spent in LR warmup (OneCycleLR)
max_epochs: 100
batch_size: 512
weight_decay: 0.00465
num_mc_samples: 10   # MC noise samples per star during training

early_stopping_patience: 20
seed: 12345
wandb_project: SCREAM_GD1
```

Key `TrainConfig` fields:

| Field | Description |
|---|---|
| `hidden_units` | Hidden layer widths — int (all layers equal) or list[int] |
| `num_layers` | Number of hidden layers |
| `activation` | Activation function (`relu`, `gelu`, etc.) |
| `use_layer_norm` | Apply LayerNorm after each hidden layer |
| `use_residual` | Use residual (skip) connections between layers |
| `dropout` | Dropout probability applied after each hidden layer |
| `lr` | Peak learning rate (OneCycleLR) |
| `pct_start` | Fraction of training spent in LR warmup |
| `weight_decay` | AdamW weight decay |
| `num_mc_samples` | MC noise samples per star per training step |
| `early_stopping_patience` | Val-loss patience before early stopping |

---

## Workflow

The full pipeline, from raw data download to the paper validation figure. Run all commands from `SCREAM/` unless noted. Activate the environment first:

```bash
module load conda && conda activate myenv
```

---

### Stage 0a — Download data

```bash
cd data/
python GD1_download.py
```

Queries DataLab (NOIRLab) for Gaia × DECaLS cross-matched photometry along the GD-1 track in 30 parallel chunks and saves the result to HDF5.

**Output:** `$PSCRATCH/fetcher_output_new/GD-1-I21/GD-1-I21_matched.hdf5`

---

### Stage 0b — Data preparation

```bash
cd data/
python GD1_data_prep.py
```

Applies extinction corrections, computes DECaLS magnitudes, crossmatches with StreamFinder membership labels, converts to stream-aligned coordinates (φ₁, φ₂, μ₁, μ₂), applies quality cuts, and defines the signal region.

**Input:** HDF5 from Stage 0a  
**Output:** `$PSCRATCH/GD-1_gaia_x_decals_stream_prep.fits`

---

### Stage 1 — Train normalizing flow and generate background

```bash
python scripts/train_flow.py --stream configs/streams/gd1.yaml
# or on NERSC:
sbatch slurm/train_flow.sh
```

Trains a `pzflow` normalizing flow on sideband stars and immediately generates synthetic background samples (4× the signal-region count by default). The flow is not serialized to disk; generation happens in the same process.

**Output:**
```
$PSCRATCH/GD1_SCREAM/generated/<run_id>/
    gd1.yaml
    run_params.yaml
    gd1_generated.csv          # combined signal + generated background
    plots/
        loss_curves.png
        histograms/
```

After a successful run, update `generated_data_path` in `configs/streams/gd1.yaml` to point to the new CSV before proceeding.

Key CLI options:

| Option | Default | Description |
|---|---|---|
| `--epochs` | 200 | Flow training epochs |
| `--batch-size` | 512 | Mini-batch size |
| `--max-lr` | 3e-4 | Peak LR (one-cycle schedule) |
| `--n-multiplier` | 4 | Generated samples per signal-region star |
| `--patience` | None | Early stopping patience (disabled by default) |
| `--seed` | 12345 | Random seed |

---

### Stage 2 — Train classifier

**Stage 2a — Single-run (exploratory path)**

```bash
python scripts/train_em_cathode.py \
    --stream configs/streams/gd1.yaml \
    --experiment configs/experiments/em_cathode_mlp.yaml \
    --run-name <name>
# or on NERSC:
sbatch slurm/train_em_cathode.sh
```

**Stage 2b — K-fold cross-validation (primary / paper path)**

```bash
python scripts/cross_validate.py \
    --stream configs/streams/gd1.yaml \
    --experiment configs/experiments/em_cathode_mlp.yaml \
    --run_name <cv_run_name> \
    --n_folds 5
```

To run a single fold in isolation (e.g. for parallelism):

```bash
python scripts/cross_validate.py \
    --stream configs/streams/gd1.yaml \
    --experiment configs/experiments/em_cathode_mlp.yaml \
    --run_name <cv_run_name> \
    --n_folds 5 \
    --fold <i>
```

**Output (CV):** one checkpoint directory per fold at `$PSCRATCH/GD1_SCREAM/checkpoints/<cv_run_name>_<i>/`

---

### Stage 3 — Evaluate

**Stage 3a — Single-run (exploratory path)**

```bash
python scripts/evaluate.py \
    --stream configs/streams/gd1.yaml \
    --run-name <name> \
    [--threshold 0.96]
```

**Stage 3b — CV evaluation (primary / paper path)**

```bash
python scripts/evaluate_cv.py \
    --stream configs/streams/gd1.yaml \
    --run_name <cv_run_name> \
    [--threshold 0.878]
```

Runs inference over all folds, concatenates held-out predictions, and writes a threshold sensitivity table (precision / recall / F1 at thresholds near the selected value).

**Output:** `$PSCRATCH/GD1_SCREAM/results/<cv_run_name>_cv_predictions.csv`

---

### Stage 4 — Crossmatch with DESI radial velocities

Open `figures/crossmatch_DESI.py` and set `RESULTS_PATH` and the predictions filename to point to the CSV from Stage 3b, then run:

```bash
python figures/crossmatch_DESI.py
```

Joins the GD-1 Gaia × DECaLS catalog with model predictions on `source_id`, then crossmatches against DESI DR1 (`mwsall-pix-iron.fits`) and Emma's `GD1_DESI_memprob.fits` for radial velocity labels.

**Required input files in `$PSCRATCH/DESI_data/`:** `mwsall-pix-iron.fits`, `GD1_DESI_memprob.fits`  
**Output:** `$PSCRATCH/GD-1_gaia_x_decals_VRAD2.fits`

---

### Stage 5 — Generate validation figure

Optionally edit `MODEL_PROB_THRESHOLD` in `figures/validation_figure.py`, then run:

```bash
python figures/validation_figure.py
```

Reads the crossmatched FITS from Stage 4 and renders the 3-panel paper figure: (a) φ₁/φ₂ sky map, (b) DESI VRAD vs φ₁, (c) DECaLS CMD.

**Output:** `figures/V2_validation_3panel.pdf` and `.png`

---

### Hyperparameter sweep

Before committing to a full cross-validation run, sweep hyperparameters with W&B:

```bash
# Step 1: initialize a new sweep (run once — prints the SWEEP_ID)
python scripts/sweep_train.py

# Step 2: launch one or more agents
python scripts/sweep_train.py --sweep-id <entity/project/id> --count 50
# or on NERSC:
sbatch slurm/sweep_em_cathode.sh
```

Once the sweep completes, select the best configuration from W&B, update `configs/experiments/em_cathode_mlp.yaml`, and proceed to Stage 2b.

---

## Running Tests

```bash
module load conda && conda activate myenv
cd /global/homes/p/pratsosa/SCREAM
pytest tests/
```

The test suite covers the loss function, model forward passes and parameter counts, dataset and datamodule behavior on synthetic data, and data splitting transforms. No GPU or `$PSCRATCH` access is required.

---

## Current Limitations

- **Single stream.** The configuration system (`StreamConfig` + stream YAML) is designed to generalize, but only GD-1 has been run end-to-end. Applying SCREAM to a new stream requires a new stream YAML and a corresponding data preparation script.
- **Flow not serialized.** The `pzflow` normalizing flow cannot be pickled after training due to its custom JAX bijector stack. Background generation must happen in the same process as training; the flow cannot be reloaded for further sampling.
- **Hardcoded paths in figure scripts.** `crossmatch_DESI.py` and `validation_figure.py` contain hardcoded `$PSCRATCH` paths and threshold values that must be edited manually before each run.
