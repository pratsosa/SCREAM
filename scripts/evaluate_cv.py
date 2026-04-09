"""Visualise cross-validation predictions produced by cross_validate.py.

Loads the CV predictions CSV (source_id, model_prob), cross-matches with the
original data CSV on source_id to recover feature columns and true labels, then
produces the same diagnostic plots as evaluate.py.  No model checkpoint or
dataloader is required — inference has already been done.

Usage:
    python scripts/evaluate_cv.py \\
        --stream configs/streams/gd1.yaml \\
        --run_name cv_run \\
        [--threshold 0.96]           # default: PR-curve optimal on CV predictions
        [--output-dir /path/to/out]  # default: $PSCRATCH/.../plots/<run_name>_cv/
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from scream.config.schema import StreamConfig
from scream.data.datamodules import _gaia_extinction_numpy
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
    parser = argparse.ArgumentParser(
        description="Visualise CV predictions from cross_validate.py"
    )
    parser.add_argument("--stream", required=True, help="Path to stream YAML config")
    parser.add_argument(
        "--run_name", required=True, help="Base run name used in cross_validate.py"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (default: optimal from CV PR curve)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots and eval_config.yaml (default: $PSCRATCH/.../plots/<run_name>_cv/)",
    )
    return parser.parse_args()


def plot_threshold_table(probs, true_labels, threshold, output_path):
    """Render a table of Recall / Precision / F1 for thresholds near *threshold*.

    Covers up to 10 steps of 0.01 above (capped at 1.00) and below (capped at
    0.80).  The selected threshold row is highlighted in amber.
    """
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
        preds = (probs >= t).astype(int)
        rec  = recall_score(true_labels, preds, zero_division=0)
        prec = precision_score(true_labels, preds, zero_division=0)
        f1   = f1_score(true_labels, preds, zero_division=0)
        rows.append([f"{t:.2f}", f"{rec:.4f}", f"{prec:.4f}", f"{f1:.4f}"])

    col_labels = ["Threshold", "Recall", "Precision", "F1"]
    n_rows = len(rows)
    highlight_idx = steps_below

    HEADER_BG = "#2c3e50"
    HIGHLIGHT  = "#f39c12"
    ROW_EVEN   = "#f4f6f7"
    ROW_ODD    = "#ffffff"

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

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color="white", fontweight="bold", fontsize=13)

    for j in range(len(col_labels)):
        table[highlight_idx + 1, j].set_text_props(fontweight="bold", fontsize=13)

    ax.set_title(
        f"CV metrics by threshold  (selected: {threshold:.2f})",
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

    # --- Paths ---
    cv_predictions_path = scratch_dir / "results" / f"{run_name}_cv_predictions.csv"
    output_dir = Path(args.output_dir) if args.output_dir else (
        scratch_dir / "plots" / f"{run_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load CV predictions ---
    print(f"Loading CV predictions from {cv_predictions_path}")
    pred_df = pd.read_csv(cv_predictions_path)
    print(f"  {len(pred_df)} real stars in CV predictions")

    # --- Load original data CSV and filter to real stars ---
    data_path = stream_cfg.generated_data_path
    print(f"Loading original data from {data_path}")
    if data_path.endswith(".csv"):
        data_df = pd.read_csv(data_path)
    else:
        from astropy.table import Table
        data_df = Table.read(data_path).to_pandas()

    real_mask = data_df["source_id"] != -1
    data_df = data_df[real_mask].reset_index(drop=True)
    print(f"  {len(data_df)} real stars in original data")

    # --- Cross-match on source_id ---
    merged = pred_df.merge(data_df, on="source_id", how="inner")
    n_matched = len(merged)
    n_unmatched = len(pred_df) - n_matched
    if n_unmatched > 0:
        print(f"  WARNING: {n_unmatched} CV predictions had no match in original data")
    print(f"  {n_matched} stars after cross-match")

    # --- Extract raw feature columns ---
    phi1_raw = merged["phi1"].values.astype("float64")
    phi2_raw = merged["phi2"].values.astype("float64")
    pm_phi1  = merged["pm_phi1"].values.astype("float64")
    pm_phi2  = merged["pm_phi2"].values.astype("float64")
    G_mag    = merged["G_mag"].values.astype("float64")
    Bp_mag   = merged["Bp_mag"].values.astype("float64")
    Rp_mag   = merged["Rp_mag"].values.astype("float64")
    g_mag    = merged["g_mag"].values.astype("float64")
    r_mag    = merged["r_mag"].values.astype("float64")
    z_mag    = merged["z_mag"].values.astype("float64")
    ebv      = merged["ebv"].values.astype("float64")

    # --- Compute extinction-corrected MLP features ---
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

    feature_names = stream_cfg.features  # phi1, phi2, pm_phi1, pm_phi2, G0, Bp0_Rp0, rmag0, g0_r0, r0_z0
    features_orig = np.column_stack([phi1_raw, phi2_raw, pm_phi1, pm_phi2,
                                     G0, BpRp0, r0, gr0, rz0])

    # --- True labels: stream column (1=stream, 0=background) ---
    stream_col = merged["stream"].values.copy()
    stream_col[stream_col == 2] = 0   # generated stars mapped to 0 (shouldn't occur for real stars)
    true_labels = stream_col.astype(int)

    probs = merged["model_prob"].values

    # --- Threshold resolution from CV predictions (fully held-out) ---
    if args.threshold is not None:
        threshold = args.threshold
        threshold_source = "manual"
    else:
        precision_arr, recall_arr, thresh_arr = precision_recall_curve(true_labels, probs)
        distances = np.sqrt((1 - precision_arr[:-1]) ** 2 + (1 - recall_arr[:-1]) ** 2)
        optimal_idx = np.argmin(distances)
        threshold = float(thresh_arr[optimal_idx])
        threshold_source = "pr_curve_cv"

    print(f"Threshold ({threshold_source}): {threshold:.4f}")

    # --- Metrics ---
    preds = (probs >= threshold).astype(int)
    rec  = recall_score(true_labels, preds)
    prec = precision_score(true_labels, preds)
    f1   = f1_score(true_labels, preds)

    print(f"Recall    : {rec:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"F1        : {f1:.4f}")

    # --- Threshold sensitivity table (uses CV predictions as the scoring set) ---
    plot_threshold_table(
        probs, true_labels, threshold,
        output_dir / "threshold_table.png",
    )

    # --- Write eval_config.yaml ---
    eval_config = {
        "run_name": run_name,
        "stream_config": str(args.stream),
        "cv_predictions": str(cv_predictions_path),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "n_cv_stars": n_matched,
        "metrics": {
            "recall": float(rec),
            "precision": float(prec),
            "f1": float(f1),
        },
        "evaluated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(output_dir / "eval_config.yaml", "w") as f:
        yaml.dump(eval_config, f, default_flow_style=False, sort_keys=False)

    # --- Unpack features ---
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

    true_mask  = true_labels.astype(bool)
    preds_bool = preds.astype(bool)
    tp = true_mask & preds_bool
    fp = ~true_mask & preds_bool
    fn = true_mask & ~preds_bool

    # --- Plots ---
    plot_confusion_matrix(true_labels, preds, output_dir / "confusion_matrix")

    plot_phi1_phi2_preds(phi1, phi2, true_mask, tp, fp, fn,
                         output_dir / "phi1_phi2_preds.png")

    plot_cmd_gaia(color, gmag, true_mask, tp, fp, fn,
                  output_dir / "Gmag_BP_RP_preds.png")

    plot_cmd_decals_gr(g_r, rmag0, true_mask, tp, fp, fn,
                       output_dir / "rmag_g_r_preds.png")

    plot_cmd_decals_rz(r_z, rmag0, true_mask, tp, fp, fn,
                       output_dir / "rmag_r_z_preds.png")

    plot_phi1_pm_tracks(phi1, pm_mu1, pm_mu2, true_mask, preds_bool, probs,
                        output_dir)

    print(f"Plots written to {output_dir}")


if __name__ == "__main__":
    main()
