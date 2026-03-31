"""CLI script: train normalizing flow and generate background samples.

Usage
-----
python scripts/train_flow.py --stream configs/streams/gd1.yaml

The script:
  1. Trains a pzflow RollingSplineCoupling flow on the sideband region.
  2. Saves the trained flow to $PSCRATCH/<STREAM>_SCREAM/generated/<name>.pkl.
  3. Generates background samples conditioned on KDE-sampled pm_ra values.
  4. Writes the combined signal + generated CSV to cfg.generated_data_path.
  5. Saves diagnostic plots to $PSCRATCH/<STREAM>_SCREAM/plots/<run_id>/.
"""

import argparse
import pickle
from pathlib import Path

import yaml
from lightning.pytorch import seed_everything

from scream.config.schema import StreamConfig
from scream.flow.trainer import train_flow
from scream.flow.sampler import generate_samples, save_samples
from scream.flow.diagnostics import plot_loss_curves, plot_feature_histograms
from scream.utils.hpc import get_scratch_dir, date_string


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
    parser.add_argument("--n-multiplier", type=int, default=4,
                        help="Generated samples per signal-region star (default: 4)")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed (default: 12345)")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    cfg = StreamConfig(**yaml.safe_load(open(args.stream)))
    print(f"Stream: {cfg.name}")
    print(f"Raw data: {cfg.raw_data_path}")
    print(f"Output: {cfg.generated_data_path}")

    run_id = date_string()
    scratch = get_scratch_dir(cfg.name)
    plots_dir = scratch / "plots" / run_id

    # Train flow
    print(f"\n--- Training flow ({args.epochs} epochs) ---")
    flow, scaler, signal_mask, full_embeddings, source_id, stream, col_names, \
        train_losses, test_losses = train_flow(
        cfg,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_lr=args.max_lr,
        seed=args.seed,
    )

    # Save trained flow
    out_dir = scratch / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    flow_path = out_dir / f"{cfg.name}_flow_{run_id}.pkl"
    with open(flow_path, "wb") as f:
        pickle.dump(flow, f)
    print(f"\nFlow saved to: {flow_path}")

    # Loss curve
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(train_losses, test_losses, plots_dir / "loss_curves.png")

    # Generate samples
    print(f"\n--- Generating background samples (x{args.n_multiplier} signal region) ---")
    df = generate_samples(
        flow, scaler, signal_mask, full_embeddings,
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
    out_path = save_samples(df, cfg)
    print(f"\nGenerated CSV written to: {out_path}")


if __name__ == "__main__":
    main()
