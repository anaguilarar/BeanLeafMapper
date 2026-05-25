"""Field-vs-lab calibration and validation.

The lab photos provide ground-truth per-leaf areas for a small, hand-selected
subset of each plant (the H1, H2... leaves on the lab sheet). We use them in two
roles:

  1. **Calibration** — for each plot, match the K largest field detections to the
     K lab leaves (rank-paired) and compute a per-plot correction factor:

         factor = median(lab_area) / median(matched_field_area)

     This factor folds together leaf foreshortening, calibration drift, and any
     systematic under-segmentation by SAM3.

  2. **Validation** — apply the factor to all field detections of the same plot
     to get corrected areas, then report uncalibrated and calibrated totals plus
     the calibration factor itself (so anomalies are visible).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_validation_report(
    field_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Aggregate field and lab DataFrames by plot_key, compute calibration, write outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    field_summary = _aggregate_field(field_df)
    lab_summary = _aggregate_lab(lab_df)
    calibration = _calibration_factors(field_df, lab_df)

    merged = field_summary.merge(lab_summary, on="plot_key", how="outer")
    merged = merged.merge(calibration, on="plot_key", how="left")
    merged["field_total_area_cm2_calibrated"] = (
        merged["field_total_area_cm2"] * merged["calibration_factor"]
    )
    merged["field_mean_area_cm2_calibrated"] = (
        merged["field_mean_area_cm2"] * merged["calibration_factor"]
    )
    merged = merged.sort_values("plot_key").reset_index(drop=True)

    merged.to_csv(output_dir / "validation_report.csv", index=False)
    _save_calibration_plot(merged, output_dir / "validation_scatter.png")
    return merged


def _aggregate_field(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "area_cm2" not in df.columns:
        return pd.DataFrame(columns=["plot_key", "field_n_leaves", "field_total_area_cm2",
                                     "field_mean_area_cm2", "field_median_area_cm2"])
    valid = df.dropna(subset=["area_cm2"])
    return (
        valid.groupby("plot_key")["area_cm2"]
        .agg(field_n_leaves="count", field_total_area_cm2="sum",
             field_mean_area_cm2="mean", field_median_area_cm2="median")
        .reset_index()
    )


def _aggregate_lab(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "area_cm2" not in df.columns:
        return pd.DataFrame(columns=["plot_key", "lab_n_leaves", "lab_total_area_cm2",
                                     "lab_mean_area_cm2", "lab_median_area_cm2"])
    valid = df.dropna(subset=["area_cm2"])
    return (
        valid.groupby("plot_key")["area_cm2"]
        .agg(lab_n_leaves="count", lab_total_area_cm2="sum",
             lab_mean_area_cm2="mean", lab_median_area_cm2="median")
        .reset_index()
    )


def _calibration_factors(field_df: pd.DataFrame, lab_df: pd.DataFrame) -> pd.DataFrame:
    """Per-plot calibration factor = median(lab_area) / median(top-K field areas).

    K = number of lab leaves available for that plot. Lab leaves are deliberately
    chosen (large, mature), so matching them to the K largest field detections is
    the closest available pairing without per-leaf identity."""
    if field_df.empty or lab_df.empty:
        return pd.DataFrame(columns=["plot_key", "calibration_factor", "calibration_k"])

    rows = []
    field_valid = field_df.dropna(subset=["area_cm2"])
    lab_valid = lab_df.dropna(subset=["area_cm2"])
    for plot_key in lab_valid["plot_key"].unique():
        lab_leaves = lab_valid[lab_valid["plot_key"] == plot_key]["area_cm2"].sort_values(ascending=False)
        field_leaves = field_valid[field_valid["plot_key"] == plot_key]["area_cm2"].sort_values(ascending=False)
        if lab_leaves.empty or field_leaves.empty:
            continue
        k = min(len(lab_leaves), len(field_leaves))
        matched_field = field_leaves.head(k)
        factor = float(np.median(lab_leaves.head(k))) / float(np.median(matched_field))
        rows.append(
            {
                "plot_key": plot_key,
                "calibration_factor": factor,
                "calibration_k": int(k),
                "lab_top_k_median": float(np.median(lab_leaves.head(k))),
                "field_top_k_median": float(np.median(matched_field)),
            }
        )
    return pd.DataFrame(rows)


def _save_calibration_plot(merged: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    valid = merged.dropna(subset=["field_top_k_median", "lab_top_k_median"])
    if not valid.empty:
        ax.scatter(valid["field_top_k_median"], valid["lab_top_k_median"], alpha=0.8)
        for _, row in valid.iterrows():
            ax.annotate(row["plot_key"], (row["field_top_k_median"], row["lab_top_k_median"]),
                        fontsize=8, alpha=0.7)
        lim = max(valid["field_top_k_median"].max(), valid["lab_top_k_median"].max()) * 1.1
        ax.plot([0, lim], [0, lim], linestyle="--", color="grey", label="1:1")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend()
    ax.set_xlabel("Field median of top-K leaves (cm²)")
    ax.set_ylabel("Lab median of top-K leaves (cm²)")
    ax.set_title("Calibration pairing — field vs lab")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
