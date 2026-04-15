"""WandB sweep script for EM-CATHODE MLP.

Usage — initialize a new sweep (run once on a login node):
    python scripts/sweep_train.py

Usage — run as an agent (called automatically by the SLURM script):
    python scripts/sweep_train.py --sweep-id pratsosa/SCREAM_GD1_SWEEP/<id> --count 50

Multi-GPU usage (4 GPUs, 50 total runs split across agents):
    CUDA_VISIBLE_DEVICES=0 python scripts/sweep_train.py --sweep-id ... --count 13 --seed-offset 0  &
    CUDA_VISIBLE_DEVICES=1 python scripts/sweep_train.py --sweep-id ... --count 13 --seed-offset 13 &
    CUDA_VISIBLE_DEVICES=2 python scripts/sweep_train.py --sweep-id ... --count 12 --seed-offset 26 &
    CUDA_VISIBLE_DEVICES=3 python scripts/sweep_train.py --sweep-id ... --count 12 --seed-offset 38 &
"""
import argparse

import joblib
import numpy as np
import torch
import wandb
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from scream.config.schema import StreamConfig, TrainConfig
from scream.data.datamodules import EM_CATHODELinearDataModule
from scream.models.lit_em_mlp import EM_LitLinearModel
from scream.utils.hpc import get_scratch_dir


STREAM_CONFIG_PATH = "configs/streams/gd1.yaml"

sweep_config = {
    "name": "em-cathode-sweep",
    "method": "bayes",
    "metric": {
        "goal": "maximize",
        "name": "val/auc_mean",
    },
    "run_cap": 50,
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 10,
        "eta": 2,
    },
    "parameters": {
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1,
        },
        "hidden_units": {
            "values": [
                [512, 512, 256, 64],
                [512, 256, 256, 64],
                [512, 256, 128, 64],
                [256, 256, 128, 64],
                [256, 128, 64, 632],
                [128, 64, 64, 32],
                [64, 32, 32, 16],
                [512, 256, 64],
                [256, 128, 32],
                [128, 64, 32],
                [64, 32],
            ]
        },
        "activation": {
            "values": ["relu", "gelu", "silu"],
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2,
        },
        "batch_size": {
            "values": [256, 512, 1024, 2048, 16384],
        },
        "num_mc_samples": {
            "values": [1, 5, 10, 50],
        },
    },
}


def make_train_fn(seed_offset: int):
    """Return a train function that assigns seeds seed_offset+1, seed_offset+2, ... per run."""
    counter = [0]

    def train():
        counter[0] += 1
        seed = seed_offset + counter[0]

        try:
            with wandb.init() as run:
                cfg = run.config

                stream_cfg = StreamConfig(**yaml.safe_load(open(STREAM_CONFIG_PATH)))

                train_cfg = TrainConfig(
                    lr=cfg.lr,
                    hidden_units=cfg.hidden_units,
                    num_layers=len(cfg.hidden_units),
                    activation=cfg.activation,
                    weight_decay=cfg.weight_decay,
                    batch_size=cfg.batch_size,
                    num_mc_samples=cfg.num_mc_samples,
                    # --- fixed ---
                    max_epochs=100,
                    early_stopping_patience=20,
                    pct_start=0.1,
                    dropout=0.0,
                    use_layer_norm=False,
                    use_residual=False,
                    seed=seed,
                )

                run_name = run.name
                L.seed_everything(train_cfg.seed, workers=True)

                data_module = EM_CATHODELinearDataModule(
                    name=run_name,
                    stream=stream_cfg.name.upper(),
                    load_data_dir=stream_cfg.generated_data_path,
                    batch_size=train_cfg.batch_size,
                    subsample_generated_seed=12345,
                )
                data_module.setup("fit")
                steps_per_epoch = len(data_module.train_dataloader())

                scaler_mean = data_module.scaler.mean_.astype(np.float32)
                scaler_scale = data_module.scaler.scale_.astype(np.float32)

                loaders_dir = get_scratch_dir(stream_cfg.name) / "loaders"
                loaders_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(data_module.scaler, loaders_dir / f"scaler_{run_name}.pkl")

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

                ckpt_dir = get_scratch_dir(stream_cfg.name) / "checkpoints" / "sweep" / run_name
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                best_ckpt_callback = ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    filename="best",
                    monitor="val/auc_mean",
                    mode="max",
                    save_top_k=1,
                )
                early_stop_callback = EarlyStopping(
                    monitor="val/auc_mean",
                    min_delta=0.0,
                    patience=train_cfg.early_stopping_patience,
                    verbose=False,
                    mode="max",
                    strict=True,
                )

                wandb_logger = WandbLogger(log_model="all", name=run_name)

                trainer = L.Trainer(
                    accelerator="gpu",
                    devices=1,
                    max_epochs=train_cfg.max_epochs,
                    log_every_n_steps=1,
                    precision="16-mixed",
                    logger=wandb_logger,
                    callbacks=[best_ckpt_callback, early_stop_callback],
                )

                trainer.fit(model=model, datamodule=data_module)
        except Exception:
            raise
        finally:
            torch.cuda.empty_cache()

    return train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-id",
        default=None,
        help="Full sweep ID (entity/project/id). If omitted, initializes a new sweep.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Max runs this agent will attempt.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help=(
            "Seeds assigned to runs are seed_offset+1, seed_offset+2, ... "
            "Use 0 for single-GPU. For multi-GPU, split the range across agents "
            "(e.g. 0/13/26/38 for four agents covering 50 runs)."
        ),
    )
    args = parser.parse_args()

    if args.sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_config, project="SCREAM_GD1_SWEEP")
        print(f"SWEEP_ID: {sweep_id}")
        print("Re-run with --sweep-id to start agents, or submit the SLURM job.")
    else:
        wandb.agent(args.sweep_id, function=make_train_fn(args.seed_offset), count=args.count)
