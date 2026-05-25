"""Plot helpers for saving annotated images and leaf-area histograms."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_annotated(image: np.ndarray, out_path: str | Path, title: str | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(image)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_leaf_area_histogram(df: pd.DataFrame, out_path: str | Path, title: str | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if df.empty:
        ax.text(0.5, 0.5, "no leaves detected", ha="center", va="center")
    else:
        sns.histplot(data=df, x="area_cm2", ax=ax, binwidth=3, color="green")
        ax.axvline(df["area_cm2"].median(), linestyle="--", color="black", label="median")
        ax.legend()
    ax.set_xlabel("Leaf area (cm²)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
