"""Sampling and time-window truncation routines for telemetry RCA sensitivity."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def read_instance_metrics(instance_dir: Path) -> pd.DataFrame:
    """Reads telemetry metrics CSV from instance directory."""
    data_csv: Path = instance_dir / "data.csv"
    simple_metrics_csv: Path = instance_dir / "simple_metrics.csv"

    if data_csv.exists():
        return pd.read_csv(data_csv)
    if simple_metrics_csv.exists():
        return pd.read_csv(simple_metrics_csv)

    raise FileNotFoundError(f"No data.csv or simple_metrics.csv found in {instance_dir}")


def read_inject_time(instance_dir: Path) -> int:
    """Reads integer timestamp of fault injection from inject_time.txt."""
    inject_time_path: Path = instance_dir / "inject_time.txt"
    if not inject_time_path.exists():
        raise FileNotFoundError(f"Missing inject_time.txt in {instance_dir}")
    return int(inject_time_path.read_text().strip())


def prepare_truncated_baro_windows(
    instance_dir: Path,
    window_minutes: int,
    tdelta: int,
) -> Tuple[pd.DataFrame, int]:
    """Extracts a symmetric telemetry window of specified duration around fault injection."""
    if window_minutes <= 0:
        raise ValueError(f"Window minutes must be positive, got {window_minutes}")

    df: pd.DataFrame = read_instance_metrics(instance_dir)
    inject_time: int = read_inject_time(instance_dir)

    # Filter out 50th percentile latency columns
    df = df.loc[:, ~df.columns.str.endswith("_latency-50")]

    if "time" not in df.columns:
        raise ValueError(f"Telemetry metrics missing 'time' column in {instance_dir}")

    window_seconds: int = window_minutes * 60
    start_time: int = inject_time - window_seconds
    end_time: int = inject_time + window_seconds + tdelta

    windowed_df: pd.DataFrame = df[(df["time"] >= start_time) & (df["time"] <= end_time)].copy()

    normal_count: int = int((windowed_df["time"] < inject_time).sum())
    anomal_count: int = int((windowed_df["time"] >= inject_time).sum())

    if normal_count == 0:
        raise ValueError(
            f"Zero pre-fault baseline observations for window {window_minutes}m in {instance_dir}"
        )
    if anomal_count == 0:
        raise ValueError(
            f"Zero post-fault anomalous observations for window {window_minutes}m in {instance_dir}"
        )

    return (windowed_df, inject_time)


def subsample_telemetry_metrics(
    data_df: pd.DataFrame,
    metric_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Subsamples telemetry metric columns without replacement to simulate metric sparsity."""
    if metric_fraction <= 0.0 or metric_fraction > 1.0:
        raise ValueError(f"Metric fraction must be in range (0.0, 1.0], got {metric_fraction}")

    if metric_fraction == 1.0:
        return data_df.copy()

    has_time: bool = "time" in data_df.columns
    metric_cols: List[str] = [c for c in data_df.columns if c != "time"]

    if len(metric_cols) == 0:
        raise ValueError("Data DataFrame contains zero metric columns.")

    n_sample: int = max(1, int(round(len(metric_cols) * metric_fraction)))
    chosen_metrics: np.ndarray = rng.choice(metric_cols, size=n_sample, replace=False)

    cols_to_keep: List[str] = []
    if has_time:
        cols_to_keep.append("time")
    cols_to_keep.extend(list(chosen_metrics))

    return data_df[cols_to_keep].copy()
