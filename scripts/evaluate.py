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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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

from scream.config.schema import StreamConfig
from scream.data.datamodules import _gaia_extinction_numpy
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
    """Return (probs, true_labels, is_real_star, source_ids) arrays over the full loader.

    is_real_star is True for rows where id_plus_sample == 0 (real Gaia stars,
    not flow-generated background).

    Inference goes through model.shared_step, which applies the same flux-space
    MC perturbation and extinction correction as training.
    """
    all_probs = []
    all_true = []
    all_real = []
    all_source_ids = []

    model.eval()
    model.to(device)
    model.to(torch.float32)

    with torch.inference_mode():
        for x_raw, y, errors, id_plus_sample, source_id in loader:
            batch_on_device = (
                x_raw.to(device),
                y.to(device),
                errors.to(device),
                id_plus_sample.to(device),
                source_id.to(device),
            )
            _, p_marginal, _, y_true = model.shared_step(batch_on_device, stage='eval')

            all_probs.append(p_marginal.numpy())
            all_true.append(y_true.numpy())
            all_real.append(~id_plus_sample.numpy().astype(bool))
            all_source_ids.append(source_id.numpy())

    probs = np.concatenate(all_probs)
    true_labels = np.concatenate(all_true)
    is_real = np.concatenate(all_real)
    source_ids = np.concatenate(all_source_ids)
    return probs, true_labels, is_real, source_ids


def plot_threshold_table(probs_val, true_val, threshold, output_path):
    """Render a table of Recall / Precision / F1 for thresholds near *threshold*.

    Covers up to 10 steps of 0.01 above (capped at 1.00) and below (capped at
    0.80).  The selected threshold row is highlighted in amber.
    """
    # If threshold is below 0.80 the lower-bound cap doesn't apply — show only
    # thresholds at or above the selected value.
    if threshold < 0.80:
        steps_below = 0
    else:
        steps_below = min(10, int(round((threshold - 0.80) / 0.01)))
    steps_above = min(10, int(round((1.00 - threshold) / 0.01)))
    thresholds = [
        round(threshold + i * 0.01, 2)
        for i in range(-steps_below, steps_above + 1)
    ]

    rows = []
    for t in thresholds:
        preds = (probs_val >= t).astype(int)
        rec  = recall_score(true_val, preds, zero_division=0)
        prec = precision_score(true_val, preds, zero_division=0)
        f1   = f1_score(true_val, preds, zero_division=0)
        rows.append([f"{t:.2f}", f"{rec:.4f}", f"{prec:.4f}", f"{f1:.4f}"])

    col_labels = ["Threshold", "Recall", "Precision", "F1"]
    n_rows = len(rows)
    highlight_idx = steps_below  # position of the selected threshold in the list

    HEADER_BG   = "#2c3e50"   # dark navy
    HIGHLIGHT   = "#f39c12"   # amber
    ROW_EVEN    = "#f4f6f7"   # very light grey
    ROW_ODD     = "#ffffff"   # white

    cell_colors = []
    for i in range(n_rows):
        if i == highlight_idx:
            cell_colors.append([HIGHLIGHT] * len(col_labels))
        else:
            cell_colors.append([ROW_EVEN if i % 2 == 0 else ROW_ODD] * len(col_labels))

    fig_height = max(3.5, 0.45 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.2, 2.0)

    # Style the header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color="white", fontweight="bold", fontsize=13)

    # Bold text for the highlighted (selected) threshold row
    for j in range(len(col_labels)):
        table[highlight_idx + 1, j].set_text_props(fontweight="bold", fontsize=13)

    ax.set_title(
        f"Val-set metrics by threshold  (selected: {threshold:.2f})",
        fontsize=14, fontweight="bold", pad=16,
    )

    legend_patch = mpatches.Patch(color=HIGHLIGHT, label=f"Selected threshold ({threshold:.2f})")
    ax.legend(
        handles=[legend_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        fontsize=11,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Threshold table written to {output_path}")


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
    model = EM_LitLinearModel.load_from_checkpoint(
        str(ckpt_path),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )
    model.eval()

    # --- Val inference ---
    probs_val_full, true_val_full, real_val, _ = _run_inference(val_loader, model, device)
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
    probs_test_full, true_test_full, real_test, source_ids_full = _run_inference(test_loader, model, device)

    # Collect raw features and errors from test loader for plotting
    raw_data_list = []
    errors_list = []
    for x_raw, y, errors, id_plus_sample, source_id in test_loader:
        raw_data_list.append(x_raw.numpy())
        errors_list.append(errors.numpy())
    raw_data_full   = np.concatenate(raw_data_list,  axis=0)
    errors_full     = np.concatenate(errors_list,    axis=0)

    # Filter to real stars only
    probs_test   = probs_test_full[real_test]
    true_test    = true_test_full[real_test]
    raw_data     = raw_data_full[real_test]
    errors_arr   = errors_full[real_test]
    source_ids   = source_ids_full[real_test]

    # Compute extinction-corrected MLP features for visualisation
    # raw_data cols: phi1, phi2, pm_phi1, pm_phi2, G_mag, Bp_mag, Rp_mag, g_mag, r_mag, z_mag
    # errors_arr col 10: ebv
    phi1_raw = raw_data[:, 0];  phi2_raw = raw_data[:, 1]
    pm_phi1  = raw_data[:, 2];  pm_phi2  = raw_data[:, 3]
    G_mag    = raw_data[:, 4];  Bp_mag   = raw_data[:, 5];  Rp_mag = raw_data[:, 6]
    g_mag    = raw_data[:, 7];  r_mag    = raw_data[:, 8];  z_mag  = raw_data[:, 9]
    ebv      = errors_arr[:, 10]

    AG, ABp, ARp = _gaia_extinction_numpy(G_mag, Bp_mag, Rp_mag, ebv)
    G0    = G_mag  - AG
    Bp0   = Bp_mag - ABp
    Rp0   = Rp_mag - ARp
    g0    = g_mag  - 3.214 * ebv
    r0    = r_mag  - 2.165 * ebv
    z0    = z_mag  - 1.211 * ebv
    BpRp0 = Bp0 - Rp0
    gr0   = g0  - r0
    rz0   = r0  - z0

    features_orig = np.column_stack([phi1_raw, phi2_raw, pm_phi1, pm_phi2,
                                      G0, BpRp0, r0, gr0, rz0])
    feature_names = stream_cfg.features  # phi1, phi2, pm_phi1, pm_phi2, G0, Bp0_Rp0, rmag0, g0_r0, r0_z0

    # --- Metrics ---
    preds_test = (probs_test >= threshold).astype(int)
    rec = recall_score(true_test, preds_test)
    prec = precision_score(true_test, preds_test)
    f1 = f1_score(true_test, preds_test)

    print(f"Recall    : {rec:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"F1        : {f1:.4f}")

    # --- Threshold sensitivity table (val set) ---
    plot_threshold_table(
        probs_val, true_val, threshold,
        output_dir / "threshold_table.png",
    )

    # --- Write eval_config.yaml before any plotting ---
    eval_config = {
        "run_name": run_name,
        "stream_config": str(args.stream),
        "checkpoint": str(ckpt_path.resolve()),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "n_mc_samples_inference": model.num_mc_samples,
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
    pred_df["source_id"] = source_ids
    pred_df["model_prob"] = probs_test
    pred_df["true_label"] = true_test.astype(int)
    pred_df.to_csv(results_dir / f"{run_name}_predictions.csv", index=False)
    print(f"Predictions written to {results_dir / f'{run_name}_predictions.csv'}")

    # --- Unpack features by name (no hard-coded positional indexing) ---
    def _col(name):
        return features_orig[:, feature_names.index(name)]

    phi1   = _col("phi1")
    phi2   = _col("phi2")
    pm_mu1 = _col("pm_phi1")
    pm_mu2 = _col("pm_phi2")
    gmag   = _col("G0")
    color  = _col("Bp0_Rp0")
    rmag0  = _col("rmag0")
    g_r    = _col("g0_r0")
    r_z    = _col("r0_z0")

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
