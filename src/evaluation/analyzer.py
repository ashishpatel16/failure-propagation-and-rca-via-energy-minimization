"""Evaluate fault-localization rankings against curated Defects4J ground truth."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.benchmark_reader import BenchmarkReader, InstanceResult
from evaluation.config import BENCHMARK_TARGET, GROUND_TRUTH_CSV, OUTPUT_DIR

logger = logging.getLogger(__name__)

METHOD_COLUMN = "Method"
NODE_KEY_COLUMN = "_node_key"
REQUIRED_GROUND_TRUTH_COLUMNS = {
    "project",
    "bug_id",
    "buggy_methods",
    "buggy_nodes",
    "trace_status",
}


class TraceStatus(str, Enum):
    EXACT_MATCH = "exact_match"
    PARTIAL_MATCH = "partial_match"
    TRACE_GAP = "trace_gap"
    REPRESENTATION_GAP = "representation_gap"


class MetricFamily(str, Enum):
    TARANTULA = "tarantula"
    TARANTULA_GC = "tarantula_gc"
    OCHIAI = "ochiai"
    OCHIAI_GC = "ochiai_gc"
    DSTAR = "dstar"
    DSTAR_GC = "dstar_gc"


@dataclass(frozen=True)
class InstanceMetric:
    project: str
    bug_id: int
    lambd: float
    family: MetricFamily
    first_fault_expected_rank: float
    first_fault_min_rank: float
    first_fault_exam_score: float


@dataclass(frozen=True)
class AggregatedMetric:
    lambd: float
    family: MetricFamily
    mean_exam_score: float
    median_exam_score: float
    mean_expected_rank: float
    median_expected_rank: float
    top_1_rate: float
    top_3_rate: float
    top_5_rate: float
    total_instances: int


@dataclass(frozen=True)
class GroundTruthEntry:
    project: str
    bug_id: int
    methods: tuple[str, ...]
    nodes: tuple[str, ...]
    trace_status: TraceStatus


def normalize_node_name(name: object) -> str:
    """Normalize a call-graph node while retaining its argument signature."""
    return str(name).strip().replace("$", ".")


def method_key(name: object) -> str:
    """Return a signature-free method name for display and source-level matching."""
    return normalize_node_name(name).split("(", maxsplit=1)[0]


class GroundTruthManager:
    """Load and validate the curated ground-truth CSV."""

    def __init__(self, ground_truth_path: Path) -> None:
        self.ground_truth_path = ground_truth_path
        self._entries = self._load()

    def _load_json_list(self, raw_value: object, field: str, instance_id: str) -> list[str]:
        try:
            values = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid {field} JSON for {instance_id}") from exc

        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"Invalid {field} format for {instance_id}")
        return values

    def _load(self) -> dict[tuple[str, int], GroundTruthEntry]:
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {self.ground_truth_path}")

        frame = pd.read_csv(self.ground_truth_path, dtype={"project": str, "bug_id": int})
        missing_columns = REQUIRED_GROUND_TRUTH_COLUMNS.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Ground truth is missing columns: {sorted(missing_columns)}")

        entries: dict[tuple[str, int], GroundTruthEntry] = {}
        for row in frame.itertuples(index=False):
            instance_id = f"{row.project}_{row.bug_id}"
            methods = self._load_json_list(row.buggy_methods, "buggy_methods", instance_id)
            nodes = self._load_json_list(row.buggy_nodes, "buggy_nodes", instance_id)
            key = (row.project, int(row.bug_id))
            if key in entries:
                raise ValueError(f"Duplicate ground-truth row: {instance_id}")

            try:
                trace_status = TraceStatus(row.trace_status)
            except ValueError as exc:
                raise ValueError(f"Invalid trace_status for {instance_id}: {row.trace_status}") from exc

            entries[key] = GroundTruthEntry(
                project=row.project,
                bug_id=int(row.bug_id),
                methods=tuple(sorted({method_key(method) for method in methods})),
                nodes=tuple(sorted({normalize_node_name(node) for node in nodes})),
                trace_status=trace_status,
            )
        return entries

    def get_entry(self, project: str, bug_id: int) -> GroundTruthEntry:
        try:
            return self._entries[(project, bug_id)]
        except KeyError as exc:
            raise KeyError(f"No ground truth found for {project}_{bug_id}") from exc


class BenchmarkAnalyzer:
    """Calculate first-fault localization metrics for benchmark instances."""

    def __init__(self, ground_truth_manager: GroundTruthManager) -> None:
        self.gt_manager = ground_truth_manager

    def _matching_fault_rows(self, instance: InstanceResult, *, warn_missing_ground_truth: bool) -> pd.DataFrame:
        if METHOD_COLUMN not in instance.metrics_df.columns:
            return pd.DataFrame()

        try:
            ground_truth = self.gt_manager.get_entry(instance.project, instance.bug_id)
        except KeyError as exc:
            if warn_missing_ground_truth:
                logger.warning("Skipping %s_%s: %s", instance.project, instance.bug_id, exc)
            return pd.DataFrame()

        if not ground_truth.nodes:
            return pd.DataFrame()

        rows = instance.metrics_df.copy()
        rows[NODE_KEY_COLUMN] = rows[METHOD_COLUMN].map(normalize_node_name)
        return rows[rows[NODE_KEY_COLUMN].isin(ground_truth.nodes)]

    def analyze_instance(self, instance: InstanceResult) -> bool:
        """Return whether an exact ground-truth faulty node occurs in the ranking."""
        return not self._matching_fault_rows(instance, warn_missing_ground_truth=True).empty

    def calculate_instance_metrics(self, instance: InstanceResult) -> list[InstanceMetric]:
        """Return each metric family's best rank for the first traceable fault."""
        fault_rows = self._matching_fault_rows(instance, warn_missing_ground_truth=False)
        if fault_rows.empty:
            return []

        metrics: list[InstanceMetric] = []
        for family in MetricFamily:
            expected_column = f"{family.value}_expected_inspections"
            rank_column = f"{family.value}_rank"
            exam_column = f"{family.value}_exam_score"
            required_columns = (expected_column, rank_column, exam_column)
            if not set(required_columns).issubset(fault_rows.columns):
                continue

            metrics.append(
                InstanceMetric(
                    project=instance.project,
                    bug_id=instance.bug_id,
                    lambd=instance.lambd,
                    family=family,
                    first_fault_expected_rank=float(fault_rows[expected_column].min()),
                    first_fault_min_rank=float(fault_rows[rank_column].min()),
                    first_fault_exam_score=float(fault_rows[exam_column].min()),
                )
            )
        return metrics

    @staticmethod
    def aggregate_lambda_metrics(lambd: float, instance_metrics: Iterable[InstanceMetric]) -> list[AggregatedMetric]:
        """Aggregate first-fault metrics for one regularization value."""
        metrics_for_lambda = [metric for metric in instance_metrics if metric.lambd == lambd]
        aggregated: list[AggregatedMetric] = []

        for family in MetricFamily:
            family_metrics = [metric for metric in metrics_for_lambda if metric.family == family]
            if not family_metrics:
                continue

            exam_scores = pd.Series([metric.first_fault_exam_score for metric in family_metrics])
            expected_ranks = pd.Series([metric.first_fault_expected_rank for metric in family_metrics])
            total_instances = len(family_metrics)

            aggregated.append(
                AggregatedMetric(
                    lambd=lambd,
                    family=family,
                    mean_exam_score=float(exam_scores.mean()),
                    median_exam_score=float(exam_scores.median()),
                    mean_expected_rank=float(expected_ranks.mean()),
                    median_expected_rank=float(expected_ranks.median()),
                    top_1_rate=float((expected_ranks <= 1).mean()),
                    top_3_rate=float((expected_ranks <= 3).mean()),
                    top_5_rate=float((expected_ranks <= 5).mean()),
                    total_instances=total_instances,
                )
            )
        return aggregated


class BenchmarkVisualizer:
    """Create comparison plots from per-instance metrics."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    @staticmethod
    def _metrics_to_frame(metrics: Iterable[InstanceMetric]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Project": metric.project,
                "Bug ID": metric.bug_id,
                "Lambda": metric.lambd,
                "Family": metric.family.value,
                "Expected Rank": metric.first_fault_expected_rank,
                "EXAM Score": metric.first_fault_exam_score,
                "Top-1": int(metric.first_fault_expected_rank <= 1),
                "Top-3": int(metric.first_fault_expected_rank <= 3),
                "Top-5": int(metric.first_fault_expected_rank <= 5),
            }
            for metric in metrics
        )

    @staticmethod
    def _add_variant_columns(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["Algorithm Type"] = result["Family"].map(
            lambda family: "Energy Min (GC)" if family.endswith("_gc") else "Baseline SBFL"
        )
        result["Base Family"] = result["Family"].str.removesuffix("_gc").str.capitalize()
        return result

    def plot_project_performance(self, lambd: float, metrics: Iterable[InstanceMetric]) -> None:
        """Plot Top-1 rate, expected rank, and Tarantula EXAM score by project."""
        frame = self._metrics_to_frame(metric for metric in metrics if metric.lambd == lambd)
        if frame.empty:
            logger.warning("No data available for project plots at lambda %s", lambd)
            return
        frame = self._add_variant_columns(frame)

        top1_frame = frame.groupby(["Project", "Base Family", "Algorithm Type"], as_index=False)["Top-1"].mean()
        top1_frame["Top-1 Rate (%)"] = top1_frame["Top-1"] * 100
        grid = sns.catplot(
            data=top1_frame,
            kind="bar",
            x="Project",
            y="Top-1 Rate (%)",
            hue="Algorithm Type",
            col="Base Family",
            height=5,
            aspect=1.2,
            palette="Set2",
        )
        grid.figure.suptitle(f"Top-1 Root Cause Identification Rate by Project (λ={lambd})", y=1.05)
        grid.set_axis_labels("Project", "Top-1 Accuracy (%)")
        grid.figure.savefig(self.output_dir / "top1_rate_by_project.png", bbox_inches="tight", dpi=300)
        plt.close(grid.figure)

        rank_frame = frame.groupby(["Project", "Base Family", "Algorithm Type"], as_index=False)["Expected Rank"].mean()
        grid = sns.catplot(
            data=rank_frame,
            kind="bar",
            x="Project",
            y="Expected Rank",
            hue="Algorithm Type",
            col="Base Family",
            height=5,
            aspect=1.2,
            palette="Set2",
        )
        grid.figure.suptitle(f"Mean Expected Rank by Project (λ={lambd}; lower is better)", y=1.05)
        grid.set_axis_labels("Project", "Mean Expected Rank")
        grid.figure.savefig(self.output_dir / "expected_rank_by_project.png", bbox_inches="tight", dpi=300)
        plt.close(grid.figure)

        tarantula_frame = frame[frame["Base Family"] == "Tarantula"]
        figure, axis = plt.subplots(figsize=(16, 6))
        sns.boxplot(
            data=tarantula_frame,
            x="Project",
            y="EXAM Score",
            hue="Algorithm Type",
            palette="Set3",
            showfliers=False,
            ax=axis,
        )
        axis.set_title(f"Tarantula EXAM Score Distribution (λ={lambd})")
        axis.set_ylabel("EXAM Score (lower is better)")
        figure.savefig(self.output_dir / "exam_score_boxplot_tarantula.png", bbox_inches="tight", dpi=300)
        plt.close(figure)

    def plot_lambda_comparison(self, metrics: Iterable[InstanceMetric], lambdas: list[float]) -> None:
        """Compare baseline and graph-cut variants across regularization values."""
        if not lambdas:
            logger.warning("No lambda values supplied for lambda comparison")
            return

        frame = self._metrics_to_frame(metrics)
        if frame.empty:
            logger.warning("No data available for lambda comparison")
            return

        baseline = frame[(frame["Lambda"] == lambdas[0]) & ~frame["Family"].str.endswith("_gc")].copy()
        baseline["Variant"] = "Baseline"
        baseline["Base Family"] = baseline["Family"].str.capitalize()

        variants = [baseline]
        for lambd in lambdas:
            graph_cut = frame[(frame["Lambda"] == lambd) & frame["Family"].str.endswith("_gc")].copy()
            graph_cut["Variant"] = f"GC (λ={lambd})"
            graph_cut["Base Family"] = graph_cut["Family"].str.removesuffix("_gc").str.capitalize()
            variants.append(graph_cut)

        comparison_frame = pd.concat(variants, ignore_index=True)
        if comparison_frame.empty:
            logger.warning("No baseline or graph-cut data available for lambda comparison")
            return

        aggregate = comparison_frame.groupby(["Base Family", "Variant"], as_index=False).agg(
            {"Top-1": "mean", "Top-5": "mean", "Expected Rank": "mean", "EXAM Score": "mean"}
        )
        aggregate["Top-1 Rate (%)"] = aggregate["Top-1"] * 100
        aggregate["Top-5 Rate (%)"] = aggregate["Top-5"] * 100
        hue_order = ["Baseline", *(f"GC (λ={lambd})" for lambd in lambdas)]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#937860"]
        palette = {variant: colors[index % len(colors)] for index, variant in enumerate(hue_order)}

        self._plot_lambda_pair(
            aggregate,
            "Top-1 Rate (%)",
            "Top-5 Rate (%)",
            "Top-1 Localization Rate",
            "Top-5 Localization Rate",
            "Impact of λ on Localization Accuracy",
            "lambda_comparison_rates.png",
            hue_order,
            palette,
        )
        self._plot_lambda_pair(
            aggregate,
            "Expected Rank",
            "EXAM Score",
            "Mean Expected Rank (lower is better)",
            "Mean EXAM Score (lower is better)",
            "Impact of λ on Average Effort",
            "lambda_comparison_effort.png",
            hue_order,
            palette,
        )

    def _plot_lambda_pair(
        self,
        frame: pd.DataFrame,
        left_metric: str,
        right_metric: str,
        left_title: str,
        right_title: str,
        title: str,
        filename: str,
        hue_order: list[str],
        palette: dict[str, str],
    ) -> None:
        figure, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(16, 6))
        for axis, metric, axis_title in (
            (left_axis, left_metric, left_title),
            (right_axis, right_metric, right_title),
        ):
            sns.barplot(
                data=frame,
                x="Base Family",
                y=metric,
                hue="Variant",
                hue_order=hue_order,
                palette=palette,
                ax=axis,
            )
            axis.set_title(axis_title)
        figure.suptitle(title, y=1.02, fontsize=14)
        figure.tight_layout()
        figure.savefig(self.output_dir / filename, bbox_inches="tight", dpi=300)
        plt.close(figure)


def _lambda_directories(target_dir: Path) -> Iterable[tuple[float, Path]]:
    for directory in sorted(target_dir.iterdir()):
        if not directory.is_dir() or not directory.name.startswith("eval_lambd_"):
            continue
        try:
            yield float(directory.name.removeprefix("eval_lambd_")), directory
        except ValueError:
            logger.warning("Skipping directory with invalid lambda: %s", directory.name)


def _log_aggregates(aggregates: Iterable[AggregatedMetric]) -> None:
    for aggregate in aggregates:
        logger.info("\n  [Family: %s]", aggregate.family.value)
        logger.info("    Total Traceable Instances: %s", aggregate.total_instances)
        logger.info("    EXAM Score (Mean / Median): %.4f / %.4f", aggregate.mean_exam_score, aggregate.median_exam_score)
        logger.info(
            "    Expected Rank (Mean / Median): %.2f / %.2f",
            aggregate.mean_expected_rank,
            aggregate.median_expected_rank,
        )
        logger.info(
            "    Top-K Rates: Top-1=%.2f%%, Top-3=%.2f%%, Top-5=%.2f%%",
            aggregate.top_1_rate * 100,
            aggregate.top_3_rate * 100,
            aggregate.top_5_rate * 100,
        )


def run() -> int:
    """Run the command-line benchmark analysis."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target_dir = OUTPUT_DIR / BENCHMARK_TARGET
    if not target_dir.exists():
        logger.error("Target directory %s does not exist.", target_dir)
        return 1

    try:
        analyzer = BenchmarkAnalyzer(GroundTruthManager(GROUND_TRUTH_CSV))
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to initialize ground truth: %s", exc)
        return 1

    logger.info("Scanning target directory: %s", target_dir)
    for lambd, directory in _lambda_directories(target_dir):
        results = BenchmarkReader(directory).read_all_instances()
        traceable_results = [result for result in results if analyzer.analyze_instance(result)]
        instance_metrics = [
            metric
            for result in traceable_results
            for metric in analyzer.calculate_instance_metrics(result)
        ]
        missing_fault_count = len(results) - len(traceable_results)

        logger.info("\n%s", "=" * 40)
        logger.info("--- Lambda: %s ---", lambd)
        logger.info("Total Instances Read: %s", len(results))
        logger.info("Instances LACKING exact buggy node in call graph: %s", missing_fault_count)
        _log_aggregates(analyzer.aggregate_lambda_metrics(lambd, instance_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
