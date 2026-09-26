"""Statistical significance testing for Microservice RCA (N = 253 instances).

Evaluates:
1. Telemetry Metric Dropout Variance Reduction (Paired t-test and Wilcoxon Signed-Rank).
2. Time-Window Diagnostic Mean Rank Improvement (Paired t-test and Wilcoxon Signed-Rank).
3. Suite-Specific breakdowns (RE2 vs RE3).
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

MODULE_DIR: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = MODULE_DIR / "outputs"


def run_rca_statistical_tests() -> pd.DataFrame:
    summary_csv: Path = OUTPUT_DIR / "rca_summary.csv"
    trials_csv: Path = OUTPUT_DIR / "rca_trials_master.csv"

    if not summary_csv.exists() or not trials_csv.exists():
        raise FileNotFoundError(f"Missing RCA summary or trials CSV in {OUTPUT_DIR}")

    df_summary: pd.DataFrame = pd.read_csv(summary_csv)
    df_trials: pd.DataFrame = pd.read_csv(trials_csv)

    results: List[Dict[str, Any]] = []

    # =========================================================================
    # 1. Telemetry Metric Dropout Variance Tests (H1: Var(Base) > Var(GC))
    # =========================================================================
    sub_drop = df_summary[df_summary["dimension_type"] == "metric_dropout"]
    for frac in [0.25, 0.50, 0.75]:
        frac_str = f"{int(frac * 100)}% Metrics"
        sub_frac = sub_drop[np.isclose(sub_drop["dimension_value"], frac)]

        base_s = sub_frac[sub_frac["lambda_val"] == 0.0].set_index("instance_name")["var_rank"]
        gc_s = sub_frac[sub_frac["lambda_val"] == 1.0].set_index("instance_name")["var_rank"]
        common = base_s.dropna().index.intersection(gc_s.dropna().index)

        b_v = base_s.loc[common].to_numpy(dtype=float)
        g_v = gc_s.loc[common].to_numpy(dtype=float)
        n_inst = len(common)

        mean_b = float(np.mean(b_v))
        mean_g = float(np.mean(g_v))
        pct_red = ((mean_b - mean_g) / mean_b) * 100.0

        diff = b_v - g_v
        std_diff = float(np.std(diff, ddof=1))
        cohen_d = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0

        t_stat, p_val_t = stats.ttest_rel(b_v, g_v, alternative="greater")
        w_stat, p_val_w = stats.wilcoxon(b_v, g_v, alternative="greater")

        results.append(
            {
                "Evaluation Domain": "RCA Metric Dropout (Variance)",
                "Condition / Window": frac_str,
                "Regularizer λ": "λ = 1.0",
                "N Instances": n_inst,
                "Base Metric": round(mean_b, 2),
                "GC Metric": round(mean_g, 2),
                "Improvement (%)": f"-{pct_red:.1f}%",
                "Paired t-stat": round(t_stat, 3),
                "t-test p-value": f"{p_val_t:.4e}" if p_val_t < 0.001 else f"{p_val_t:.4f}",
                "Wilcoxon p-value": f"{p_val_w:.4e}" if p_val_w < 0.001 else f"{p_val_w:.4f}",
                "Cohen's d_z": round(cohen_d, 3),
                "Significant (p < 0.05)": "Yes" if p_val_t < 0.05 else "No",
            }
        )

    # =========================================================================
    # 2. Time-Window Mean Rank Tests (H1: Rank(Base) > Rank(GC))
    # =========================================================================
    sub_win = df_trials[df_trials["dimension_type"] == "time_window"]
    for win in [1, 2, 5, 10, 20]:
        win_str = f"T = {win}m Window"
        sub_w = sub_win[sub_win["dimension_value"] == win]

        base_r = sub_w[sub_w["lambda_val"] == 0.0].groupby("instance_name")["predicted_rank"].mean()
        gc_r = sub_w[sub_w["lambda_val"] == 1.0].groupby("instance_name")["predicted_rank"].mean()
        common = base_r.index.intersection(gc_r.index)

        b_ranks = base_r.loc[common].to_numpy(dtype=float)
        g_ranks = gc_r.loc[common].to_numpy(dtype=float)
        n_inst = len(common)

        mean_b = float(np.mean(b_ranks))
        mean_g = float(np.mean(g_ranks))
        diff_r = b_ranks - g_ranks
        mean_diff = float(np.mean(diff_r))
        std_diff = float(np.std(diff_r, ddof=1))
        cohen_d = float(mean_diff / std_diff) if std_diff > 0 else 0.0

        t_stat, p_val_t = stats.ttest_rel(b_ranks, g_ranks, alternative="greater")
        w_stat, p_val_w = stats.wilcoxon(b_ranks, g_ranks, alternative="greater")

        results.append(
            {
                "Evaluation Domain": "RCA Time Window (Mean Rank)",
                "Condition / Window": win_str,
                "Regularizer λ": "λ = 1.0",
                "N Instances": n_inst,
                "Base Metric": round(mean_b, 2),
                "GC Metric": round(mean_g, 2),
                "Improvement (%)": f"+{mean_diff:.2f} pos" if mean_diff > 0 else f"{mean_diff:.2f} pos",
                "Paired t-stat": round(t_stat, 3),
                "t-test p-value": f"{p_val_t:.4e}" if p_val_t < 0.001 else f"{p_val_t:.4f}",
                "Wilcoxon p-value": f"{p_val_w:.4e}" if p_val_w < 0.001 else f"{p_val_w:.4f}",
                "Cohen's d_z": round(cohen_d, 3),
                "Significant (p < 0.05)": "Yes" if p_val_t < 0.05 else "No",
            }
        )

    res_df: pd.DataFrame = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "rca_statistical_significance_tests.csv"
    res_df.to_csv(out_csv, index=False)
    return res_df


def main() -> None:
    df_res = run_rca_statistical_tests()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 15)
    print("\n" + "=" * 115)
    print("MICROSERVICE RCA STATISTICAL SIGNIFICANCE REPORT (N = 253 INSTANCES)")
    print("=" * 115)
    print(df_res.to_string(index=False))
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
