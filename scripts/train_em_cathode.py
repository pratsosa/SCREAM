"""Train the EM-CATHODE MLP for a given stream and experiment configuration.

Usage:
    python scripts/train_em_cathode.py \
        --stream configs/streams/gd1.yaml \
        --experiment configs/experiments/em_cathode_mlp.yaml \
        [--seed 42] \
        [--run-name my_run]
"""
import argparse
import os

import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from scream.config.schema import StreamConfig, TrainConfig
from scream.data.datamodules import EM_CATHODELinearDataModule
from scream.models.lit_em_mlp import EM_LitLinearModel
from scream.utils.hpc import date_string, get_scratch_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train EM-CATHODE MLP")
    parser.add_argument("--stream", required=True, help="Path to stream YAML config")
    parser.add_argument("--experiment", required=True, help="Path to experiment YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides experiment YAML)")
    parser.add_argument("--run-name", default=None, help="WandB run name and checkpoint directory name")
    return parser.parse_args()


def main():
    args = parse_args()

    stream_cfg = StreamConfig(**yaml.safe_load(open(args.stream)))
    train_cfg = TrainConfig(**yaml.safe_load(open(args.experiment)))

    if args.seed is not None:
        train_cfg.seed = args.seed

    run_name = args.run_name if args.run_name is not None else date_string()

    L.seed_everything(train_cfg.seed, workers=True)

    # Data
    data_module = EM_CATHODELinearDataModule(
        name=run_name,
        stream=stream_cfg.name.upper(),
        load_data_dir=stream_cfg.generated_data_path,
        batch_size=train_cfg.batch_size,
        p_wiggle=train_cfg.p_wiggle,
    )
    data_module.setup("fit")
    steps_per_epoch = len(data_module.train_dataloader())

    # Model
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
        anneal_noise=False,
    )

    # Callbacks
    ckpt_dir = get_scratch_dir(stream_cfg.name) / "checkpoints" / run_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}",
        every_n_epochs=1,
        save_top_k=-1,
    )
    early_stop_callback = EarlyStopping(
        monitor="True validation f1 score (0.8 thresh)",
        min_delta=0.0,
        patience=train_cfg.early_stopping_patience,
        verbose=False,
        mode="max",
        strict=True,
    )

    # Logger
    wandb_logger = WandbLogger(
        log_model="all",
        name=run_name,
        project=train_cfg.wandb_project,
    )

    # Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=train_cfg.max_epochs,
        log_every_n_steps=1,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
