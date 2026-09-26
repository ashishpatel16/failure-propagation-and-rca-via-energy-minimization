"""Statistical aggregator and comparative verdict generator for microservice RCA sensitivity."""

from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from rca_models import (
    RCASampleTrialResult,
    RCASensitivitySummaryRow,
    RCASensitivityVerdict,
    VerdictType,
)


class RCASensitivityAnalyzer:
    """Aggregates trial results and computes comparative metrics for RCA sensitivity."""

    def __init__(self) -> None:
        pass

    def trial_results_to_dataframe(
        self, trials: List[RCASampleTrialResult]
    ) -> pd.DataFrame:
        """Converts a list of RCASampleTrialResult dataclasses into a pandas DataFrame."""
        if len(trials) == 0:
            raise ValueError("Cannot convert empty RCA trial list to DataFrame.")
        records: List[Dict[str, object]] = [asdict(t) for t in trials]
        return pd.DataFrame(records)

    def summary_rows_to_dataframe(
        self, summaries: List[RCASensitivitySummaryRow]
    ) -> pd.DataFrame:
        """Converts a list of RCASensitivitySummaryRow dataclasses into a pandas DataFrame."""
        if len(summaries) == 0:
            raise ValueError("Cannot convert empty RCA summary list to DataFrame.")
        records: List[Dict[str, object]] = [asdict(s) for s in summaries]
        return pd.DataFrame(records)

    def verdicts_to_dataframe(
        self, verdicts: List[RCASensitivityVerdict]
    ) -> pd.DataFrame:
        """Converts a list of RCASensitivityVerdict dataclasses into a pandas DataFrame."""
        if len(verdicts) == 0:
            raise ValueError("Cannot convert empty RCA verdict list to DataFrame.")
        records: List[Dict[str, object]] = [asdict(v) for v in verdicts]
        return pd.DataFrame(records)

    def compute_window_verdicts(
        self, trials_df: pd.DataFrame
    ) -> List[RCASensitivityVerdict]:
        """Computes head-to-head performance comparisons across observation time windows."""
        win_df: pd.DataFrame = trials_df[trials_df["dimension_type"] == "time_window"].copy()
        if win_df.empty:
            return []

        verdicts: List[RCASensitivityVerdict] = []
        for (inst, win_min), group in win_df.groupby(["instance_name", "dimension_value"]):
            base_rows: pd.DataFrame = group[group["lambda_val"] == 0.0]
            if base_rows.empty:
                continue

            base_row: pd.Series = base_rows.iloc[0]
            b_rank: float = float(base_row["predicted_rank"])
            b_top1: float = 1.0 if b_rank == 1.0 else 0.0
            b_mrr: float = float(base_row["mrr"])

            for _, pw_row in group[group["lambda_val"] > 0.0].iterrows():
                lambd: float = float(pw_row["lambda_val"])
                p_rank: float = float(pw_row["predicted_rank"])
                p_top1: float = 1.0 if p_rank == 1.0 else 0.0
                p_mrr: float = float(pw_row["mrr"])

                if p_rank < b_rank:
                    v_str: str = "improves"
                elif p_rank > b_rank:
                    v_str = "worsens"
                else:
                    v_str = "equivalent"

                verdicts.append(
                    RCASensitivityVerdict(
                        instance_name=str(inst),
                        dimension_type="time_window",
                        dimension_value=float(win_min),
                        pairwise_lambda=lambd,
                        baseline_std=0.0,
                        pairwise_std=0.0,
                        std_ratio_base_over_pairwise=1.0,
                        baseline_top_1=b_top1,
                        pairwise_top_1=p_top1,
                        baseline_mrr=b_mrr,
                        pairwise_mrr=p_mrr,
                        verdict=v_str,
                    )
                )

        return verdicts

    def build_window_aggregate_report(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates Top-1, Top-3, and MRR metrics across observation window budgets."""
        win_df: pd.DataFrame = trials_df[trials_df["dimension_type"] == "time_window"].copy()
        if win_df.empty:
            raise ValueError("No time-window trials found in dataset.")

        report_rows: List[Dict[str, object]] = []

        for (win_min, lambd), group in win_df.groupby(
            ["dimension_value", "lambda_val"]
        ):
            n_total: int = len(group)
            top1_acc: float = float((group["is_top_1"]).mean()) * 100.0
            top3_acc: float = float((group["is_top_3"]).mean()) * 100.0
            top5_acc: float = float((group["is_top_5"]).mean()) * 100.0
            mean_mrr: float = float(group["mrr"].mean())
            mean_rank: float = float(group["predicted_rank"].mean())

            arm_label: str = "BARO (Baseline)" if lambd == 0.0 else f"BARO-GC (λ={lambd})"

            report_rows.append(
                {
                    "window_minutes": int(win_min),
                    "arm": arm_label,
                    "lambda_val": lambd,
                    "total_instances": n_total,
                    "top_1_accuracy": round(top1_acc, 2),
                    "top_3_accuracy": round(top3_acc, 2),
                    "top_5_accuracy": round(top5_acc, 2),
                    "mean_mrr": round(mean_mrr, 4),
                    "mean_rank": round(mean_rank, 2),
                }
            )

        return pd.DataFrame(report_rows).sort_values(["window_minutes", "lambda_val"])
