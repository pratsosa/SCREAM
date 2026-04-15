"""K-fold cross-validation for the EM-CATHODE MLP.

Produces a single CSV of (source_id, model_prob) covering every real
signal-region star, by training one MLP per fold and collecting each
fold's held-out test predictions.

Usage:
    python scripts/cross_validate.py \\
        --stream configs/streams/gd1.yaml \\
        --experiment configs/experiments/em_cathode_mlp.yaml \\
        --run_name cv_run \\
        [--n_folds 5] \\
        [--fold 2]           # run a single fold only (0-indexed)
        [--wandb_project my_project]
"""
import argparse
import os
import shutil

import joblib
import numpy as np
import wandb
import pandas as pd
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import torch

from scream.config.schema import StreamConfig, TrainConfig
from scream.data.datamodules import EM_CATHODELinearDataModule
from scream.models.lit_em_mlp import EM_LitLinearModel
from scream.utils.hpc import get_scratch_dir
from evaluate import _run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="K-fold cross-validation for EM-CATHODE MLP")
    parser.add_argument("--stream", default="configs/streams/gd1.yaml", help="Path to stream YAML config")
    parser.add_argument("--experiment", default="configs/experiments/em_cathode_mlp.yaml", help="Path to experiment YAML config")
    parser.add_argument("--run_name", required=True, help="Base run name; folds become {run_name}_{i}")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--fold", type=int, default=None, help="Run a single fold index only (0-indexed)")
    parser.add_argument("--wandb_project", default=None, help="WandB project override")
    return parser.parse_args()


def train_fold(fold_idx, run_name, n_folds, stream_cfg, train_cfg, wandb_project):
    """Train one fold and return (source_ids, probs) for the real stars in the test fold."""
    fold_run_name = f"{run_name}_{fold_idx}"
    scratch_dir = get_scratch_dir(stream_cfg.name)

    L.seed_everything(train_cfg.seed, workers=True)

    # --- Data ---
    data_module = EM_CATHODELinearDataModule(
        name=fold_run_name,
        stream=stream_cfg.name.upper(),
        load_data_dir=stream_cfg.generated_data_path,
        batch_size=train_cfg.batch_size,
        subsample_generated_seed=12345,
        n_folds=n_folds,
        fold_idx=fold_idx,
    )
    data_module.setup("fit")
    steps_per_epoch = len(data_module.train_dataloader())

    scaler_mean  = data_module.scaler.mean_.astype(np.float32)
    scaler_scale = data_module.scaler.scale_.astype(np.float32)

    loaders_dir = scratch_dir / "loaders"
    loaders_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(data_module.scaler, loaders_dir / f"scaler_{fold_run_name}.pkl")

    # --- Model ---
    model = EM_LitLinearModel(
        input_dim=len(stream_cfg.features),
        lr=train_cfg.lr,
        EPOCHS=train_cfg.max_epochs,
        steps_per_epoch=steps_per_epoch,
        pos_weight=None,
        num_layers=train_cfg.num_layers,
        hidden_units=train_cfg.hidden_units,
        dropout=train_cfg.dropout,
        pct_start=train_cfg.pct_start,
        num_mc_samples=train_cfg.num_mc_samples,
        weight_decay=train_cfg.weight_decay,
        layer_norm=train_cfg.use_layer_norm,
        activation=train_cfg.activation,
        residual=train_cfg.use_residual,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        n_extinction_iter=stream_cfg.n_extinction_iter,
    )

    # --- Callbacks ---
    ckpt_dir = scratch_dir / "checkpoints" / fold_run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}",
        every_n_epochs=1,
        save_top_k=-1,
    )
    best_ckpt_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",
        monitor="True validation f1 score (0.8 thresh)",
        mode="max",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor="True validation f1 score (0.8 thresh)",
        min_delta=0.0,
        patience=train_cfg.early_stopping_patience,
        verbose=False,
        mode="max",
        strict=True,
    )

    # --- Logger ---
    wandb.finish()
    wandb_logger = WandbLogger(
        log_model="all",
        name=fold_run_name,
        group=run_name,
        project=wandb_project,
    )

    # --- Trainer ---
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=train_cfg.max_epochs,
        log_every_n_steps=1,
        precision="32-true", # I used to use 16-mixed but I'll try 32-true for stability
        logger=wandb_logger,
        callbacks=[checkpoint_callback, best_ckpt_callback, early_stop_callback],
    )

    trainer.fit(model=model, datamodule=data_module)

    # --- Inference on test fold ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs, _, is_real, source_ids = _run_inference(
        data_module.test_dataloader(), model, device
    )

    return source_ids[is_real], probs[is_real]


def main():
    args = parse_args()

    stream_cfg = StreamConfig(**yaml.safe_load(open(args.stream)))
    train_cfg  = TrainConfig(**yaml.safe_load(open(args.experiment)))

    wandb_project = args.wandb_project if args.wandb_project is not None else train_cfg.wandb_project

    folds_to_run = [args.fold] if args.fold is not None else range(args.n_folds)

    all_source_ids = []
    all_probs = []

    for i in folds_to_run:
        print(f"\n{'='*60}")
        print(f"  Fold {i} / {args.n_folds - 1}  —  run: {args.run_name}_{i}")
        print(f"{'='*60}\n")
        source_ids, probs = train_fold(
            fold_idx=i,
            run_name=args.run_name,
            n_folds=args.n_folds,
            stream_cfg=stream_cfg,
            train_cfg=train_cfg,
            wandb_project=wandb_project,
        )
        all_source_ids.append(source_ids)
        all_probs.append(probs)

    # --- Write combined predictions CSV ---
    pred_df = pd.DataFrame({
        "source_id": np.concatenate(all_source_ids),
        "model_prob": np.concatenate(all_probs),
    })

    results_dir = get_scratch_dir(stream_cfg.name) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.run_name}_cv_predictions.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"\nCV predictions written to {out_path}  ({len(pred_df)} real stars)")


if __name__ == "__main__":
    main()
