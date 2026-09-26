"""Visualization generator for Microservice RCA sensitivity and time-window budgets."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE_DIR: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = MODULE_DIR / "outputs"


class RCASensitivityPlotter:
    """Generates publication-quality charts for microservice RCA sensitivity."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_time_to_accuracy(
        self,
        aggregate_df: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        """Plots Top-1 Accuracy and MRR as a function of observation window budget."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        lambdas: List[float] = sorted(aggregate_df["lambda_val"].unique())
        colors: dict = {
            0.0: ("#E53E3E", "o-", "BARO (Baseline λ=0.0)"),
            0.1: ("#3182CE", "s--", "BARO-GC (λ=0.1)"),
            1.0: ("#805AD5", "^-.", "BARO-GC (λ=1.0)"),
            10.0: ("#38A169", "d-", "BARO-GC (λ=10.0)"),
        }

        for lambd in lambdas:
            arm_df: pd.DataFrame = aggregate_df[
                aggregate_df["lambda_val"] == lambd
            ].sort_values("window_minutes")
            if arm_df.empty:
                continue

            windows: np.ndarray = arm_df["window_minutes"].to_numpy()
            top1: np.ndarray = arm_df["top_1_accuracy"].to_numpy()
            mrr: np.ndarray = arm_df["mean_mrr"].to_numpy()

            color, style, label = colors.get(
                lambd, ("#718096", "o-", f"BARO-GC (λ={lambd})")
            )

            ax1.plot(
                windows,
                top1,
                style,
                color=color,
                linewidth=2.2,
                markersize=7,
                label=label,
            )

            ax2.plot(
                windows,
                mrr,
                style,
                color=color,
                linewidth=2.2,
                markersize=7,
                label=label,
            )

        ax1.set_title("Top-1 Accuracy vs Observation Window Budget", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Observation Window Length (Minutes)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
        ax1.set_xticks([2, 5, 10, 20])
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=9, loc="lower right", framealpha=0.9)

        ax2.set_title("Mean Reciprocal Rank (MRR) vs Observation Window Budget", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Observation Window Length (Minutes)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=11, fontweight="bold")
        ax2.set_xticks([2, 5, 10, 20])
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(fontsize=9, loc="lower right", framealpha=0.9)

        plt.suptitle(
            "Microservice Root Cause Localization Under Shortened Incident Observation Windows",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        out_path: Path = self.output_dir / output_filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_window_mean_rank(
        self,
        aggregate_df: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        """Plots Mean Root Cause Rank across observation windows (lower is better)."""
        fig, ax = plt.subplots(figsize=(8, 5))

        lambdas: List[float] = sorted(aggregate_df["lambda_val"].unique())
        colors: dict = {
            0.0: ("#E53E3E", "o-", "BARO Baseline (λ=0.0)"),
            0.1: ("#3182CE", "s--", "BARO-GC (λ=0.1)"),
            1.0: ("#805AD5", "^-.", "BARO-GC (λ=1.0)"),
            10.0: ("#38A169", "d-", "BARO-GC (λ=10.0)"),
        }

        for lambd in lambdas:
            arm_df: pd.DataFrame = aggregate_df[
                aggregate_df["lambda_val"] == lambd
            ].sort_values("window_minutes")
            if arm_df.empty:
                continue

            windows: np.ndarray = arm_df["window_minutes"].to_numpy()
            ranks: np.ndarray = arm_df["mean_rank"].to_numpy()

            color, style, label = colors.get(
                lambd, ("#718096", "o-", f"BARO-GC (λ={lambd})")
            )

            ax.plot(
                windows,
                ranks,
                style,
                color=color,
                linewidth=2.2,
                markersize=7,
                label=label,
            )

        ax.set_title("Mean Root Cause Rank vs Incident Observation Time (Lower is Better)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Observation Window Length (Minutes)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Root Cause Rank", fontsize=11, fontweight="bold")
        ax.set_xticks([2, 5, 10, 20])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

        plt.tight_layout()
        out_path: Path = self.output_dir / output_filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path


def main() -> None:
    """Generates visual charts from RCA sensitivity aggregate CSVs."""
    agg_csv: Path = OUTPUT_DIR / "rca_window_aggregate_report.csv"
    if not agg_csv.exists():
        raise FileNotFoundError(
            f"Missing aggregate report in {OUTPUT_DIR}. Run run_rca_sensitivity.py first."
        )

    agg_df: pd.DataFrame = pd.read_csv(agg_csv)
    plotter: RCASensitivityPlotter = RCASensitivityPlotter(output_dir=OUTPUT_DIR)

    p1: Path = plotter.plot_time_to_accuracy(
        aggregate_df=agg_df,
        output_filename="rca_time_to_accuracy.png",
    )
    p2: Path = plotter.plot_window_mean_rank(
        aggregate_df=agg_df,
        output_filename="rca_window_mean_rank.png",
    )
    print(f"Generated RCA time-to-accuracy plot: {p1}")
    print(f"Generated RCA mean rank degradation plot: {p2}")


if __name__ == "__main__":
    main()
