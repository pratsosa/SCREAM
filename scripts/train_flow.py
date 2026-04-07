"""CLI script: train normalizing flow and generate background samples.

Usage
-----
python scripts/train_flow.py --stream configs/streams/gd1.yaml

The script:
  1. Trains a pzflow RollingSplineCoupling flow on the sideband region.
  2. Generates background samples conditioned on KDE-sampled pm_ra values.
  3. Creates a timestamped run directory at $PSCRATCH/<STREAM>_SCREAM/generated/<run_id>/
     containing:
       <stream>.yaml    — copy of the stream config used
       run_params.yaml  — all CLI args passed to this script
       <stream>_generated.csv — combined signal + generated background
       plots/           — loss curves and feature histograms

The trained flow object is NOT saved to disk — sampling runs immediately
after training using the in-memory flow, avoiding serialization issues with
the custom bijector.
"""

import argparse
import shutil
from pathlib import Path

import yaml
from lightning.pytorch import seed_everything

from scream.config.schema import StreamConfig
from scream.flow.trainer import train_flow
from scream.flow.sampler import generate_samples, save_samples
from scream.flow.diagnostics import plot_loss_curves, plot_feature_histograms
from scream.utils.hpc import get_scratch_dir, date_string


def int_or_none(value):
    if value is None or value.lower() == 'none':
        return None
    return int(value)

def parse_args():
    parser = argparse.ArgumentParser(description="Train pzflow and generate CATHODE samples")
    parser.add_argument("--stream", required=True,
                        help="Path to stream YAML config (e.g. configs/streams/gd1.yaml)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Mini-batch size (default: 512)")
    parser.add_argument("--max-lr", type=float, default=3e-4,
                        help="Peak learning rate for one-cycle schedule (default: 3e-4)")
    parser.add_argument("--patience", type=int_or_none, default=None,
                        help="Patience for early stopping in epochs (default: None, i.e. no early stopping)")
    parser.add_argument("--n-multiplier", type=int, default=4,
                        help="Generated samples per signal-region star (default: 4)")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed (default: 12345)")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    cfg = StreamConfig(**yaml.safe_load(open(args.stream)))

    run_id = date_string()
    scratch = get_scratch_dir(cfg.name)
    run_dir = scratch / "generated" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stream:   {cfg.name}")
    print(f"Run ID:   {run_id}")
    print(f"Run dir:  {run_dir}")
    print(f"Raw data: {cfg.raw_data_path}")

    # Save stream config and CLI args for reproducibility
    shutil.copy(args.stream, run_dir / Path(args.stream).name)
    run_params = {
        "stream": args.stream,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_lr": args.max_lr,
        "patience": args.patience,
        "n_multiplier": args.n_multiplier,
        "seed": args.seed,
    }
    with open(run_dir / "run_params.yaml", "w") as f:
        yaml.dump(run_params, f, default_flow_style=False, sort_keys=False)

    # Train flow
    print(f"\n--- Training flow ({args.epochs} epochs) ---")
    flow, scaler, signal_mask, full_embeddings, ebv, source_id, stream, col_names, \
        train_losses, test_losses = train_flow(
        cfg,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_lr=args.max_lr,
        seed=args.seed,
    )

    # Loss curve
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(train_losses, test_losses, plots_dir / "loss_curves.png")

    # Generate samples
    print(f"\n--- Generating background samples (x{args.n_multiplier} signal region) ---")
    df = generate_samples(
        flow, scaler, signal_mask, full_embeddings, ebv,
        source_id, stream, col_names, cfg,
        n_multiplier=args.n_multiplier,
        seed=args.seed,
    )
    print(f"Generated dataframe shape: {df.shape}")
    print(f"  Signal-region stars (CWoLa_Label=1): {(df['CWoLa_Label'] == 1).sum()}")
    print(f"  Generated background (CWoLa_Label=0): {(df['CWoLa_Label'] == 0).sum()}")

    # Feature histograms
    plot_feature_histograms(
        full_embeddings, signal_mask, col_names,
        df, cfg.flow_data_columns,
        plots_dir / "histograms",
    )

    # Write CSV
    csv_path = run_dir / f"{cfg.name}_generated.csv"
    out_path = save_samples(df, csv_path)
    print(f"\nRun directory: {run_dir}")
    print(f"Generated CSV: {out_path}")
    print(f"\nTo use this run with train_em_cathode.py, update generated_data_path in")
    print(f"  {args.stream}")
    print(f"to: {out_path}")


if __name__ == "__main__":
    main()
