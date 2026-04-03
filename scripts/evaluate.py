"""Run inference with a trained EM-CATHODE MLP and write plots + predictions.

Usage:
    python scripts/evaluate.py \\
        --stream configs/streams/gd1.yaml \\
        --run-name <name> \\
        [--checkpoint /path/to/epoch=N.ckpt]   # default: {ckpt_dir}/best.ckpt
        [--threshold 0.96]                      # default: PR-curve optimal on val set
        [--output-dir /path/to/plots/]          # default: $PSCRATCH/GD1_SCREAM/plots/<run_name>/
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch import nn

from scream.config.schema import StreamConfig
from scream.models.lit_em_mlp import EM_LitLinearModel
from scream.plotting import (
    plot_cmd_decals_gr,
    plot_cmd_decals_rz,
    plot_cmd_gaia,
    plot_confusion_matrix,
    plot_phi1_phi2_preds,
    plot_phi1_pm_tracks,
)
from scream.utils.hpc import get_scratch_dir

# ---------------------------------------------------------------------------
# Error-column indices for MC perturbation during inference
# ---------------------------------------------------------------------------
# The errors tensor in the dataloader has 9 columns in the same order as
# stream_cfg.features: [ra, dec, pm_ra, pm_dec, gmag, color, rmag0, g_r, r_z].
# Only pm_ra (index 2) and pm_dec (index 3) carry real Gaia measurement errors;
# all other columns are wiggle-augmented synthetic values that should not be
# perturbed during inference.
#
# TODO: derive these dynamically by stripping the '_error' suffix from each entry
# in stream_cfg.error_features and looking up the result in stream_cfg.features.
# Hard-coded for now because the GD-1 feature list is fixed (pm_ra is always
# index 2, pm_dec always index 3).
_PM_ERROR_COLS = [2, 3]

_N_MC_INFERENCE = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained EM-CATHODE MLP")
    parser.add_argument("--stream", required=True, help="Path to stream YAML config")
    parser.add_argument("--run-name", required=True, help="Run name used during training")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint file (default: {ckpt_dir}/best.ckpt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (default: optimal from val PR curve)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots and eval_config.yaml (default: $PSCRATCH/.../plots/<run_name>/)",
    )
    return parser.parse_args()


def _run_inference(loader, model, device):
    """Return (probs, true_labels, is_real_star) arrays over the full loader.

    is_real_star is True for rows where sampled_data == 0 (real Gaia stars,
    not flow-generated background).
    """
    all_probs = []
    all_true = []
    all_real = []

    sig = nn.Sigmoid()
    model.eval()
    model.to(torch.float32)

    with torch.inference_mode():
        for x, y, errs_in, sampled_data in loader:
            B, D = x.shape

            # Build errors tensor: zeros everywhere except pm_ra / pm_dec columns.
            # See _PM_ERROR_COLS comment at module top for why indices 2 and 3.
            errors = torch.zeros_like(x)
            for col in _PM_ERROR_COLS:
                errors[:, col] = errs_in[:, col]

            x_samples = (
                x.unsqueeze(1)
                + torch.randn(B, _N_MC_INFERENCE, D).to(x.device) * errors.unsqueeze(1)
            )
            logits = model.model.to(device)(x_samples.to(device)).squeeze(-1)
            # logits shape: (B, N_MC) — average over MC samples in probability space
            probs_batch = sig(logits).mean(dim=1).cpu().numpy()

            true_batch = y[:, 1].numpy()  # column 1 = true SF label
            real_batch = ~sampled_data.numpy().astype(bool)  # True = real star

            all_probs.append(probs_batch)
            all_true.append(true_batch)
            all_real.append(real_batch)

    probs = np.concatenate(all_probs)
    true_labels = np.concatenate(all_true)
    is_real = np.concatenate(all_real)
    return probs, true_labels, is_real


def main():
    args = parse_args()

    stream_cfg = StreamConfig(**yaml.safe_load(open(args.stream)))
    scratch_dir = get_scratch_dir(stream_cfg.name)
    run_name = args.run_name

    # --- Resolve paths ---
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        scratch_dir / "checkpoints" / run_name / "best.ckpt"
    )
    scaler_path = scratch_dir / "loaders" / f"scaler_{run_name}.pkl"
    val_loader_path = scratch_dir / "loaders" / f"linear_val_loader_{run_name}.pth"
    test_loader_path = scratch_dir / "loaders" / f"linear_test_loader_{run_name}.pth"
    results_dir = scratch_dir / "results"
    output_dir = Path(args.output_dir) if args.output_dir else (
        scratch_dir / "plots" / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load artifacts ---
    scaler = joblib.load(scaler_path)
    val_loader = torch.load(val_loader_path, weights_only=False)
    test_loader = torch.load(test_loader_path, weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EM_LitLinearModel.load_from_checkpoint(str(ckpt_path))
    model.eval()

    # --- Val inference ---
    probs_val_full, true_val_full, real_val = _run_inference(val_loader, model, device)
    probs_val = probs_val_full[real_val]
    true_val = true_val_full[real_val]

    # --- Threshold resolution ---
    if args.threshold is not None:
        threshold = args.threshold
        threshold_source = "manual"
    else:
        precision_arr, recall_arr, thresh_arr = precision_recall_curve(true_val, probs_val)
        # Closest point to (precision=1, recall=1) on the PR curve
        distances = np.sqrt((1 - precision_arr[:-1]) ** 2 + (1 - recall_arr[:-1]) ** 2)
        optimal_idx = np.argmin(distances)
        threshold = float(thresh_arr[optimal_idx])
        threshold_source = "pr_curve"

    print(f"Threshold ({threshold_source}): {threshold:.4f}")

    # --- Test inference ---
    probs_test_full, true_test_full, real_test = _run_inference(test_loader, model, device)

    # Collect raw scaled features and real-star mask from test loader
    scaled_data_list = []
    for x, y, errs_in, sampled_data in test_loader:
        scaled_data_list.append(x.numpy())
    scaled_data_full = np.concatenate(scaled_data_list, axis=0)

    # Filter to real stars only
    probs_test = probs_test_full[real_test]
    true_test = true_test_full[real_test]
    scaled_data = scaled_data_full[real_test]

    # Inverse-transform features; column names from YAML feature order
    features_orig = scaler.inverse_transform(scaled_data)
    feature_names = stream_cfg.features  # e.g. [ra, dec, pm_ra, pm_dec, gmag, color, rmag0, g_r, r_z]

    # --- Metrics ---
    preds_test = (probs_test >= threshold).astype(int)
    rec = recall_score(true_test, preds_test)
    prec = precision_score(true_test, preds_test)
    f1 = f1_score(true_test, preds_test)

    print(f"Recall    : {rec:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"F1        : {f1:.4f}")

    # --- Write eval_config.yaml before any plotting ---
    eval_config = {
        "run_name": run_name,
        "stream_config": str(args.stream),
        "checkpoint": str(ckpt_path.resolve()),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "n_mc_samples_inference": _N_MC_INFERENCE,
        "val_set_size": int(real_val.sum()),
        "test_set_size": int(real_test.sum()),
        "metrics": {
            "recall": float(rec),
            "precision": float(prec),
            "f1": float(f1),
        },
        "evaluated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(output_dir / "eval_config.yaml", "w") as f:
        yaml.dump(eval_config, f, default_flow_style=False, sort_keys=False)

    # --- Write predictions CSV ---
    pred_df = pd.DataFrame(features_orig, columns=feature_names)
    pred_df["model_prob"] = probs_test
    pred_df["true_label"] = true_test.astype(int)
    pred_df.to_csv(results_dir / f"{run_name}_predictions.csv", index=False)
    print(f"Predictions written to {results_dir / f'{run_name}_predictions.csv'}")

    # --- Unpack features by name (no hard-coded positional indexing) ---
    def _col(name):
        return features_orig[:, feature_names.index(name)]

    phi1 = _col("ra")
    phi2 = _col("dec")
    pm_mu1 = _col("pm_ra")
    pm_mu2 = _col("pm_dec")
    gmag = _col("gmag")
    color = _col("color")
    rmag0 = _col("rmag0")
    g_r = _col("g_r")
    r_z = _col("r_z")

    true_mask = true_test.astype(bool)
    preds_bool = preds_test.astype(bool)
    tp = true_mask & preds_bool
    fp = ~true_mask & preds_bool
    fn = true_mask & ~preds_bool

    # --- Plots ---
    plot_confusion_matrix(true_test, preds_test, output_dir / "confusion_matrix")

    plot_phi1_phi2_preds(phi1, phi2, true_mask, tp, fp, fn,
                         output_dir / "phi1_phi2_preds.png")

    plot_cmd_gaia(color, gmag, true_mask, tp, fp, fn,
                  output_dir / "Gmag_BP_RP_preds.png")

    plot_cmd_decals_gr(g_r, rmag0, true_mask, tp, fp, fn,
                       output_dir / "rmag_g_r_preds.png")

    plot_cmd_decals_rz(r_z, rmag0, true_mask, tp, fp, fn,
                       output_dir / "rmag_r_z_preds.png")

    plot_phi1_pm_tracks(phi1, pm_mu1, pm_mu2, true_mask, preds_bool, probs_test,
                        output_dir)

    print(f"Plots written to {output_dir}")


if __name__ == "__main__":
    main()
