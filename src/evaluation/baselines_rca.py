import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RCA_EVAL_REPO = ROOT_DIR / "data" / "rca_eval_repo"


def _ensure_rca_eval_importable() -> None:
    if str(RCA_EVAL_REPO) not in sys.path:
        sys.path.insert(0, str(RCA_EVAL_REPO))


def _read_metrics_path(instance_dir: Path) -> Path:
    if (instance_dir / "data.csv").exists():
        return instance_dir / "data.csv"
    if (instance_dir / "simple_metrics.csv").exists():
        return instance_dir / "simple_metrics.csv"
    raise FileNotFoundError("No data.csv or simple_metrics.csv found")


def _infer_dataset(instance_dir: Path) -> str:
    parts = {part.lower() for part in instance_dir.parts}
    for suite in ("re1", "re2", "re3"):
        for system in ("ob", "ss", "tt"):
            if f"{suite}-{system}" in parts:
                return f"{suite}-{system}"
    raise ValueError(f"Could not infer RCA-Eval dataset from path: {instance_dir}")


def _prepare_baro_input(
    instance_dir: Path,
    length_minutes: int = 20,
    tdelta: int = 0,
) -> tuple[pd.DataFrame, int, str]:
    data_path = _read_metrics_path(instance_dir)
    dataset = _infer_dataset(instance_dir)

    data = pd.read_csv(data_path)
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]

    if "RE2-TT" in str(data_path) or "RE3-TT" in str(data_path):
        time_col = data["time"]
        data = data.loc[:, data.columns.str.startswith("ts-")]
        data["time"] = time_col

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.ffill().fillna(0)

    inject_time = int((instance_dir / "inject_time.txt").read_text().strip()) + tdelta
    window_size = length_minutes * 60 // 2
    normal_df = data[data["time"] < inject_time].tail(window_size)
    anomal_df = data[data["time"] >= inject_time].head(window_size)
    data = pd.concat([normal_df, anomal_df], ignore_index=True)

    data = data.rename(
        columns={
            column: column.replace("_latency-90", "_latency")
            for column in data.columns
            if column.endswith("_latency-90")
        }
    )

    return data, inject_time, dataset


def service_ranks_from_metric_ranks(metric_ranks: List[str]) -> List[str]:
    service_ranks = []
    for metric in metric_ranks:
        service = metric.split("_")[0].replace("-db", "")
        if service not in service_ranks:
            service_ranks.append(service)
    return service_ranks


def compute_baro_baseline(
    instance_dir: Path,
    length_minutes: int = 20,
    tdelta: int = 0,
) -> dict[str, List[str]]:
    _ensure_rca_eval_importable()
    from RCAEval.e2e.baro import baro

    data, inject_time, dataset = _prepare_baro_input(instance_dir, length_minutes, tdelta)
    output = baro(
        data,
        inject_time,
        dataset=dataset,
        anomalies=None,
        dk_select_useful=False,
        sli=None,
        verbose=False,
        n_iter=len(data.columns) - 1,
    )

    metric_ranks = output["ranks"]
    return {
        "metric_ranks": metric_ranks,
        "service_ranks": service_ranks_from_metric_ranks(metric_ranks),
    }


def compute_zscore_baseline(
    instance_dir: Path,
    length_minutes: int = 20,
    tdelta: int = 0,
) -> dict[str, List[str]]:
    """Simple baseline that ranks metrics by their max absolute Z-score after injection."""
    data, inject_time, _ = _prepare_baro_input(instance_dir, length_minutes, tdelta)
    
    normal_df = data[data["time"] < inject_time].drop(columns=["time"], errors="ignore")
    anomal_df = data[data["time"] >= inject_time].drop(columns=["time"], errors="ignore")
    
    if normal_df.empty or anomal_df.empty:
        return {"metric_ranks": [], "service_ranks": []}
    
    mean = normal_df.mean()
    std = normal_df.std().replace(0.0, 1e-6)
    
    zscores = ((anomal_df - mean) / std).abs().max()
    ranked_metrics = zscores.sort_values(ascending=False).index.tolist()
    
    return {
        "metric_ranks": ranked_metrics,
        "service_ranks": service_ranks_from_metric_ranks(ranked_metrics),
    }
