# SCREAM

**S**tream **C**haracterization with **E**rror-**A**ware **M**achine Learning

SCREAM is a research pipeline for detecting stellar streams in Gaia + DECaLS photometric data using a combination of normalizing flows and a Monte Carlo error-marginalizing neural classifier. The codebase is currently focused on the GD-1 stream and is under active development.

---

## Scientific Context

Stellar streams are the remnants of tidally disrupted globular clusters or dwarf galaxies, stretched into thin arcs across the Milky Way halo. Their kinematics and chemistry are sensitive probes of the Galactic potential and dark matter substructure. SCREAM targets the detection of stellar stream members in the presence of a dominant Galactic foreground, using a data-driven approach that requires no simulation-based background model.

The current application is the GD-1 stream, using a cross-matched catalog of Gaia astrometry (proper motions, G-band photometry, BP−RP color) and DECaLS photometry (r₀, g₀−r₀, r₀−z₀), expressed in stream-aligned coordinates ($\phi_1$, $\phi_2$, $\mu_{1}$, $\mu_{2}$). Ground truth labels from STREAMFINDER are used for evaluation only and are not used during training. Results are further validated against DESI radial velocity data.

---

## Method

SCREAM implements **EM-CATHODE**: an extension of the CATHODE anomaly-detection framework ([Hallin et al. 2022](https://ui.adsabs.harvard.edu/abs/2022PhRvD.106e5006H/abstract)) that marginalizes over astrometric measurement errors via Monte Carlo sampling during both training and inference.

The pipeline has two sequential stages.

### Stage 1 — Normalizing Flow and Background Generation

A normalizing flow (pzflow `RollingSplineCoupling`) is trained on stars in the *sideband* region — defined as those outside the signal region in $\mu_{1}$ (proper motion along the stream). The flow models the joint distribution of all observables conditioned on $\mu_{1}$, which serves as the interpolating variable across the signal region.

After training, the flow is used to generate synthetic background samples conditioned on $\mu_{1}$ values drawn from a kernel density estimate of the signal-region stars. This produces a generated background catalog at a configurable multiplicity (default 4× the number of signal-region stars).

### Stage 2 — Error-Marginalizing MLP Classifier

The signal-region stars (label 1) and the flow-generated background (label 0) are used as a weakly supervised training set under the **Classification Without Labels (CWoLa)** paradigm. An MLP classifier is trained to distinguish the two populations using the full set of astrometric and photometric features.

During each training step, each star is perturbed *N* times by its Gaia proper-motion measurement errors ($\sigma_{\mu_{1}}$, $\sigma_{\mu_{2}}$), producing *N* logit predictions per star. The loss is the **Monte Carlo marginal binary cross-entropy**:

$$\mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \log \left[ \frac{1}{N} \sum_{j=1}^{N} p(y_b \mid \tilde{x}_{b,j}) \right]$$

where $\tilde{x}_{b,j}$ is the $j$-th noise-perturbed sample of star $b$ and $p(y|\tilde{x})$ is the predicted probability. This formulation marginalizes over measurement uncertainty in a principled way, making the classifier's decisions robust to astrometric errors.

During inference, 100 MC samples are drawn per star and averaged in probability space to yield a final stream membership probability.

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
│   │   ├── spatial.py          # $\phi_1$/$\phi_2$ sky position plots
│   │   ├── kinematics.py       # Proper motion track plots
│   │   └── photometry.py       # Gaia and DECaLS color–magnitude diagrams
│   └── utils/
│       ├── hpc.py              # PSCRATCH path helpers (get_scratch_dir, date_string)
│       ├── io.py               # FITS/CSV I/O wrappers
│       └── metrics.py          # count_parameters()
│
├── configs/
│   ├── streams/
│   │   └── gd1.yaml            # GD-1 stream: features, data paths, quality cuts
│   └── experiments/
│       └── em_cathode_mlp.yaml # MLP hyperparameters
│
├── scripts/
│   ├── train_flow.py           # CLI: train normalizing flow + generate background
│   ├── train_em_cathode.py     # CLI: train EM-CATHODE MLP classifier
│   └── evaluate.py             # CLI: load checkpoint, run inference, write plots
│
├── slurm/
│   ├── train_flow.sh           # NERSC batch script for flow training (1 GPU, 1h30)
│   └── train_em_cathode.sh     # NERSC batch script for MLP training (1 GPU, 4h)
│
├── tests/
│   ├── test_losses.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_transforms.py
│
├── data/
│   └── README.md               # Documents that real data lives in $PSCRATCH
│
└── archival_code/              # Pre-refactor notebooks and scripts (reference only)
```

---

## Data

Raw data is stored in `$PSCRATCH` and is never committed to this repository. See [data/README.md](data/README.md) for a description of the input files. The authoritative paths are defined in `configs/streams/gd1.yaml` and consumed directly by the training scripts.

The raw input for the normalizing flow is a FITS file containing the Gaia × DECaLS cross-matched catalog in stream-aligned coordinates, with precomputed photometric quantities and a `signal_region` boolean column. The generated background CSV produced by `train_flow.py` is the direct input to `train_em_cathode.py`.

---

## Environment Setup

The package is designed for use on NERSC Perlmutter with the `myenv` conda environment.

```bash
module load conda
conda activate myenv
cd /global/homes/p/XXXXa/SCREAM
pip install -e .
```

The editable install makes `scream.*` importable from any directory without `PYTHONPATH` manipulation. The `pzflow` / JAX dependencies are only required for the flow training stage; all MLP training and evaluation steps function without them.

---

## Configuration

All stream-specific parameters and model hyperparameters are defined in YAML files and loaded into typed dataclasses at runtime.

**`configs/streams/gd1.yaml`** — stream geometry, feature lists, data paths, and quality cuts:

```yaml
name: gd1
raw_data_path: /pscratch/sd/p/XXXXa/GD-1_gaia_x_decals_stream_prep.fits
generated_data_path: /pscratch/sd/p/XXXXa/GD1_SCREAM/generated/<run_id>/gd1_generated.csv

features: [ra, dec, pm_ra, pm_dec, gmag, color, rmag0, g_r, r_z]
error_features: [pm_ra_error, pm_dec_error]

flow_data_columns: [ra, dec, pm_dec, pm_ra_error, pm_dec_error, gmag, color, rmag0, g_r, r_z]
flow_cond_columns: [pm_ra]

pm_ra_signal_range: null   # null → use pre-computed 'signal_region' FITS column

quality_cuts:
  parallax_max: 1.0
  color_min:    0.5
  color_max:    1.0
  gmag_max:    20.2
```

**`configs/experiments/em_cathode_mlp.yaml`** — model architecture and training hyperparameters:

```yaml
hidden_units: 384
num_layers: 4
activation: relu
lr: 0.01
pct_start: 0.1       # fraction of training spent in LR warmup (OneCycleLR)
max_epochs: 100
batch_size: 25000
num_mc_samples: 10   # MC noise samples per star during training
p_wiggle: 0.5        # fractional std applied as photometric feature noise
early_stopping_patience: 35
seed: 12345
wandb_project: SCREAM_GD1
```

---

## Workflow

The pipeline proceeds in two sequential stages. Each stage produces artifacts consumed by the next.

### Stage 1: Flow Training and Background Generation

```bash
python scripts/train_flow.py --stream configs/streams/gd1.yaml
```

Or via SLURM:

```bash
sbatch slurm/train_flow.sh
```

This script trains a `pzflow` normalizing flow on the sideband stars and immediately generates background samples in the same process (the flow is not serialized to disk, as the custom bijector stack does not survive pickle round-trips). Each run is written to a timestamped directory:

```
$PSCRATCH/GD1_SCREAM/generated/<run_id>/
    gd1.yaml                  # copy of stream config
    run_params.yaml           # all CLI arguments
    gd1_generated.csv         # combined signal + generated background
    plots/
        loss_curves.png
        histograms/           # one histogram per flow data column
```

After a successful run, update `generated_data_path` in `configs/streams/gd1.yaml` to point to the new CSV before proceeding to Stage 2.

Key CLI options:

| Option | Default | Description |
|---|---|---|
| `--epochs` | 200 | Flow training epochs |
| `--batch-size` | 512 | Mini-batch size |
| `--max-lr` | 3e-4 | Peak LR (one-cycle schedule) |
| `--n-multiplier` | 4 | Generated samples per signal-region star |
| `--patience` | None | Early stopping patience (disabled by default) |
| `--seed` | 12345 | Random seed |

### Stage 2: MLP Training

```bash
python scripts/train_em_cathode.py \
    --stream configs/streams/gd1.yaml \
    --experiment configs/experiments/em_cathode_mlp.yaml \
    [--seed 42] \
    [--run-name my_run]
```

Or via SLURM:

```bash
sbatch slurm/train_em_cathode.sh
```

The data module loads the generated CSV, subsamples the background to a 1:1 ratio with signal-region stars in memory (the 4:1 CSV on disk is preserved), fits a `StandardScaler`, and saves train/val/test dataloaders and the fitted scaler to `$PSCRATCH/GD1_SCREAM/loaders/`. The scaler is required by `evaluate.py` for feature inverse-transformation.

Training artifacts are written to:

```
$PSCRATCH/GD1_SCREAM/
    checkpoints/<run_name>/
        gd1.yaml              # stream config copy
        em_cathode_mlp.yaml   # experiment config copy
        run_params.yaml       # seed and run_name
        epoch=N.ckpt          # checkpoint saved every epoch
        best.ckpt             # best checkpoint by val F1 (0.8 threshold monitor)
    loaders/
        linear_train_loader_<run_name>.pth
        linear_val_loader_<run_name>.pth
        linear_test_loader_<run_name>.pth
        scaler_<run_name>.pkl
```

Training is logged to Weights & Biases under the project specified by `wandb_project` in the experiment YAML.

### Stage 3: Inference and Evaluation

```bash
python scripts/evaluate.py \
    --stream configs/streams/gd1.yaml \
    --run-name <name> \
    [--checkpoint /path/to/epoch=N.ckpt] \
    [--threshold 0.96] \
    [--output-dir /path/to/plots/]
```

By default the script loads `best.ckpt` and determines the classification threshold as the point on the validation-set precision–recall curve closest to (precision=1, recall=1). A manual threshold can be specified with `--threshold`.

Inference runs 100 MC noise samples per star on the test set. The script outputs:

```
$PSCRATCH/GD1_SCREAM/
    results/
        <run_name>_predictions.csv   # features + model_prob + true_label
    plots/<run_name>/
        eval_config.yaml             # reproducibility metadata (written first)
        confusion_matrix_norm.png    # prediction-normalised confusion matrix
        confusion_matrix_raw.png     # raw-count confusion matrix
        phi1_phi2_preds.png          # $\phi_1$/$\phi_2$ sky map with TP/FP/FN overlay
        Gmag_BP_RP_preds.png         # Gaia color–magnitude diagram
        rmag_g_r_preds.png           # DECaLS g−r color–magnitude diagram
        rmag_r_z_preds.png           # DECaLS r−z color–magnitude diagram
        phi1_muphi1_track.png        # $\mu_{1}$ track along the stream
        phi1_muphi2_track.png        # $\mu_{2}$ track along the stream
        phi1_mu_tracks_combined.png  # 2×2 combined track plot
```

`eval_config.yaml` records the checkpoint path, threshold value and its source, number of MC samples, dataset sizes, and final metrics (recall, precision, F1) in a form sufficient to reproduce any plot exactly.

---

## Running Tests

```bash
module load conda && conda activate myenv
cd /global/homes/p/XXXXa/SCREAM
pytest tests/
```

The test suite covers the loss function, model forward passes and parameter counts, dataset and datamodule behavior on synthetic data, and data splitting transforms. No GPU or `$PSCRATCH` access is required.

---

## Current Limitations and Planned Extensions

SCREAM is under active development. The following extensions are planned:

- **Additional streams.** The configuration system (stream YAML + `StreamConfig` dataclass) is designed to support new streams by adding a single YAML file, with no changes to the Python codebase.
