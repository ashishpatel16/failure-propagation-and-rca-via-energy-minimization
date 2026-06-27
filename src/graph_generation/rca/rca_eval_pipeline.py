"""Download, extract, and validate RCA-Eval RE2/RE3 trace graphs.

Configure ``COMMAND`` and ``SELECTED_SUITES`` below, then run from the
repository root:

    python -m src.graph_generation.rca.rca_eval_pipeline

The canonical graph is constructed from all observed runtime RPC interactions
and retains raw invocation frequency for the graph-cut coupling. A pre-injection
graph is emitted only as a sensitivity artifact. Existing ``call_graph.json``
files are intentionally never modified.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterator, TypeAlias

import networkx as nx
import pandas as pd
from sklearn.preprocessing import RobustScaler


LOGGER = logging.getLogger("rca_eval_pipeline")


class Suite(Enum):
    RE2 = "re2"
    RE3 = "re3"


class PipelineCommand(Enum):
    DOWNLOAD = "download"
    EXTRACT_GRAPHS = "extract_graphs"
    EXTRACT = "extract"
    EVALUATE_BARO = "evaluate_baro"
    VERIFY = "verify"
    PREPARE = "prepare"
    SELF_TEST = "self_test"


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT: Path = REPO_ROOT / "data"
EXTRACTION_MANIFEST_PATH: Path = DATA_ROOT / "rca_eval_graph_extraction.csv"
GRAPH_EXTRACTION_MANIFEST_PATH: Path = DATA_ROOT / "rca_eval_trace_graph_extraction.csv"
VALIDATION_MANIFEST_PATH: Path = DATA_ROOT / "rca_eval_graph_validation.csv"
BARO_BENCHMARK_SUMMARY_PATH: Path = DATA_ROOT / "rca_baro_benchmark_summary.csv"
BARO_EVALUATION_MANIFEST_PATH: Path = DATA_ROOT / "rca_baro_evaluation_manifest.csv"
BARO_COMPARISON_SUMMARY_PATH: Path = DATA_ROOT / "rca_baro_graphcut_comparison.csv"
COMMAND: PipelineCommand = PipelineCommand.SELF_TEST
SELECTED_SUITES: tuple[Suite, ...] = (Suite.RE2, Suite.RE3)
OVERWRITE_ARTIFACTS: bool = False
BARO_WINDOW_SIZE: int = 600
LAMBDA_VALUES: tuple[float, ...] = (0.1, 1.0)

SUITE_DIRS: dict[Suite, str] = {Suite.RE2: "RE2", Suite.RE3: "RE3"}
EXPECTED_CASES: dict[Suite, int] = {Suite.RE2: 270, Suite.RE3: 90}
REQUIRED_TRACE_COLUMNS = {
    "traceID",
    "spanID",
    "serviceName",
    "startTime",
    "duration",
    "parentSpanID",
}
ROOT_PARENT_VALUES = {"", "0", "none", "null", "nan", "<na>"}
TIMESTAMP_FACTORS = (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0)
RCAEvalDownloader: TypeAlias = Callable[[str], None]


@dataclass
class Inspection:
    record: dict[str, Any]
    pre_graph: dict[str, Any] | None
    runtime_graph: dict[str, Any] | None


@dataclass
class BaroEvaluation:
    service_scores: list[dict[str, Any]]
    summary: list[dict[str, Any]]


@dataclass
class ServiceScoreAggregation:
    scores: dict[str, float]
    excluded_metric_count: int


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def normalized_text(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("A required identifier is null")
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(numeric_value))
    text = str(value).strip()
    if not text:
        raise ValueError("A required identifier is empty")
    return text


def normalized_parent_identifier(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(numeric_value))
    text = str(value).strip()
    if text.casefold() in ROOT_PARENT_VALUES:
        return ""
    return text


def load_rca_downloaders() -> dict[Suite, tuple[RCAEvalDownloader, ...]]:
    """Import individual upstream downloaders so partial suite downloads resume."""
    try:
        from RCAEval.utility import (
            download_re2ob_dataset,
            download_re2ss_dataset,
            download_re2tt_dataset,
            download_re3ob_dataset,
            download_re3ss_dataset,
            download_re3tt_dataset,
        )
    except ImportError:
        local_package = REPO_ROOT / "data" / "rca_eval_repo"
        if local_package.is_dir():
            sys.path.insert(0, str(local_package))
            try:
                from RCAEval.utility import (
                    download_re2ob_dataset,
                    download_re2ss_dataset,
                    download_re2tt_dataset,
                    download_re3ob_dataset,
                    download_re3ss_dataset,
                    download_re3tt_dataset,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "RCAEval could not be imported. Install the local package with "
                    "`uv pip install -e './data/rca_eval_repo[default]'`."
                ) from exc
        else:
            raise RuntimeError(
                "RCAEval could not be imported. Install the local package with "
                "`uv pip install -e './data/rca_eval_repo[default]'`."
            )

    return {
        Suite.RE2: (download_re2ob_dataset, download_re2ss_dataset, download_re2tt_dataset),
        Suite.RE3: (download_re3ob_dataset, download_re3ss_dataset, download_re3tt_dataset),
    }


@contextmanager
def temporary_cwd(path: Path) -> Iterator[None]:
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def download_suites(data_root: Path, suites: tuple[Suite, ...]) -> None:
    downloaders = load_rca_downloaders()
    data_root.mkdir(parents=True, exist_ok=True)

    # Upstream functions place transient zip files in the current directory.
    # Keep those files outside the repository while extracting to an absolute path.
    with tempfile.TemporaryDirectory(prefix="rca-eval-download-") as temp_dir:
        with temporary_cwd(Path(temp_dir)):
            for suite in suites:
                suite_root = data_root / SUITE_DIRS[suite]
                for downloader in downloaders[suite]:
                    LOGGER.info("Ensuring %s dataset component is available", suite.value.upper())
                    downloader(local_path=str(suite_root))


def discover_instances(data_root: Path, suites: tuple[Suite, ...]) -> tuple[list[tuple[Suite, Path]], list[str]]:
    instances: list[tuple[Suite, Path]] = []
    errors: list[str] = []
    for suite in suites:
        suite_root = data_root / SUITE_DIRS[suite]
        if not suite_root.is_dir():
            errors.append(f"Missing suite directory: {repo_relative(suite_root)}")
            continue

        trace_files = sorted(suite_root.rglob("traces.csv"))
        if len(trace_files) != EXPECTED_CASES[suite]:
            errors.append(
                f"{suite.value.upper()} has {len(trace_files)} traces.csv files; expected {EXPECTED_CASES[suite]}."
            )
        instances.extend((suite, trace_file.parent) for trace_file in trace_files)
    return instances, errors


def read_injection_time(instance_dir: Path) -> float:
    path = instance_dir / "inject_time.txt"
    if not path.is_file():
        raise ValueError("Missing inject_time.txt")
    try:
        value = float(path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ValueError("inject_time.txt does not contain a numeric timestamp") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("inject_time.txt must contain a positive, finite timestamp")
    return value


def infer_trace_timestamp_factor(start_times: pd.Series, injection_time: float) -> float:
    median_start = float(start_times.abs().median())
    if not math.isfinite(median_start) or median_start <= 0:
        raise ValueError("Trace startTime values do not permit timestamp unit inference")

    candidates = []
    for factor in TIMESTAMP_FACTORS:
        converted_injection = injection_time * factor
        distance = abs(math.log10(median_start / converted_injection))
        candidates.append((distance, factor))
    candidates.sort()
    best_distance, best_factor = candidates[0]

    if best_distance > 0.25:
        raise ValueError(
            "Unable to align trace startTime units with inject_time.txt; "
            f"best logarithmic distance is {best_distance:.3f}."
        )
    if len(candidates) > 1 and candidates[1][0] - best_distance < 0.10:
        raise ValueError("Trace timestamp unit inference is ambiguous")
    return best_factor


def make_graph(df: pd.DataFrame, selected_trace_ids: set[str]) -> tuple[dict[str, Any], dict[str, int]]:
    selected = df[df["trace_id"].isin(selected_trace_ids)]
    if selected.empty:
        graph = {"metadata": {"total_nodes": 0, "total_edges": 0}, "nodes": [], "edges": []}
        return graph, {"root_spans": 0, "unmatched_parents": 0, "self_loops_omitted": 0}

    span_services = {
        (trace_id, span_id): service_id
        for trace_id, span_id, service_id in zip(
            selected["trace_id"], selected["span_id"], selected["service_id"], strict=True
        )
    }
    edge_counts: Counter[tuple[str, str]] = Counter()
    root_spans = 0
    unmatched_parents = 0
    self_loops_omitted = 0

    for trace_id, parent_id, service_id in zip(
        selected["trace_id"], selected["parent_id"], selected["service_id"], strict=True
    ):
        if not parent_id:
            root_spans += 1
            continue
        parent_key = (trace_id, parent_id)
        if parent_key not in span_services:
            unmatched_parents += 1
            continue
        parent_service = span_services[parent_key]
        if parent_service == service_id:
            self_loops_omitted += 1
            continue
        edge_counts[(parent_service, service_id)] += 1

    nodes = sorted(selected["service_id"].unique().tolist())
    edges = [
        {"caller": caller, "callee": callee, "frequency": frequency}
        for (caller, callee), frequency in sorted(edge_counts.items())
    ]
    return (
        {
            "metadata": {"total_nodes": len(nodes), "total_edges": len(edges)},
            "nodes": nodes,
            "edges": edges,
        },
        {
            "root_spans": root_spans,
            "unmatched_parents": unmatched_parents,
            "self_loops_omitted": self_loops_omitted,
        },
    )


def inspect_instance(suite: Suite, instance_dir: Path) -> Inspection:
    trace_path = instance_dir / "traces.csv"
    record: dict[str, Any] = {
        "suite": suite.value.upper(),
        "instance": repo_relative(instance_dir),
        "status": "ok",
        "error": "",
    }
    try:
        if not trace_path.is_file():
            raise ValueError("Missing traces.csv")
        df = pd.read_csv(trace_path)
        missing_columns = sorted(REQUIRED_TRACE_COLUMNS - set(df.columns))
        if missing_columns:
            raise ValueError(f"Missing required trace columns: {', '.join(missing_columns)}")
        if df.empty:
            raise ValueError("traces.csv contains no spans")

        df = df.copy()
        df["trace_id"] = df["traceID"].map(normalized_text)
        df["span_id"] = df["spanID"].map(normalized_text)
        df["parent_id"] = df["parentSpanID"].map(normalized_parent_identifier)
        df["service_id"] = df["serviceName"].map(normalized_text)
        if (df["trace_id"] == "").any() or (df["span_id"] == "").any() or (df["service_id"] == "").any():
            raise ValueError("Trace, span, and service identifiers must be non-empty")

        df["start_value"] = pd.to_numeric(df["startTime"], errors="coerce")
        df["duration_value"] = pd.to_numeric(df["duration"], errors="coerce")
        if df[["start_value", "duration_value"]].isna().any().any():
            raise ValueError("startTime and duration must be numeric")
        if not df["start_value"].map(math.isfinite).all() or not df["duration_value"].map(math.isfinite).all():
            raise ValueError("startTime and duration must be finite")
        if (df["duration_value"] < 0).any():
            raise ValueError("duration must be non-negative")

        duplicate_count = int(df.duplicated(["trace_id", "span_id"]).sum())
        if duplicate_count:
            raise ValueError(f"Found {duplicate_count} duplicate (traceID, spanID) pairs")

        injection_seconds = read_injection_time(instance_dir)
        timestamp_factor = infer_trace_timestamp_factor(df["start_value"], injection_seconds)
        injection_trace_time = injection_seconds * timestamp_factor
        df["end_value"] = df["start_value"] + df["duration_value"]
        trace_end = df.groupby("trace_id")["end_value"].max()
        pre_trace_ids = set(trace_end[trace_end < injection_trace_time].index.tolist())

        pre_graph, pre_stats = make_graph(df, pre_trace_ids)
        runtime_graph, runtime_stats = make_graph(df, set(trace_end.index.tolist()))
        if not runtime_graph["nodes"]:
            raise ValueError("No service nodes are available for the runtime graph")
        record.update(
            {
                "spans": len(df),
                "traces": len(trace_end),
                "services": df["service_id"].nunique(),
                "pre_injection_traces": len(pre_trace_ids),
                "post_or_cross_injection_traces": len(trace_end) - len(pre_trace_ids),
                "timestamp_factor": int(timestamp_factor) if timestamp_factor.is_integer() else timestamp_factor,
                "duplicate_span_keys": duplicate_count,
                "pre_nodes": pre_graph["metadata"]["total_nodes"],
                "pre_edges": pre_graph["metadata"]["total_edges"],
                "pre_root_spans": pre_stats["root_spans"],
                "pre_unmatched_parents": pre_stats["unmatched_parents"],
                "runtime_nodes": runtime_graph["metadata"]["total_nodes"],
                "runtime_edges": runtime_graph["metadata"]["total_edges"],
                "runtime_root_spans": runtime_stats["root_spans"],
                "runtime_unmatched_parents": runtime_stats["unmatched_parents"],
            }
        )
        return Inspection(record=record, pre_graph=pre_graph, runtime_graph=runtime_graph)
    except Exception as exc:
        record.update({"status": "error", "error": str(exc)})
        return Inspection(record=record, pre_graph=None, runtime_graph=None)


def load_baro_dependencies() -> tuple[Callable[..., dict[str, Any]], Callable[..., pd.DataFrame]]:
    try:
        from RCAEval.e2e.baro import baro
        from RCAEval.io.time_series import preprocess
    except ImportError:
        local_package = REPO_ROOT / "data" / "rca_eval_repo"
        if not local_package.is_dir():
            raise RuntimeError("RCAEval source is unavailable for BARO evaluation")
        sys.path.insert(0, str(local_package))
        from RCAEval.e2e.baro import baro
        from RCAEval.io.time_series import preprocess
    return baro, preprocess


def read_metric_data(instance_dir: Path) -> pd.DataFrame:
    data_path = instance_dir / "data.csv"
    simple_data_path = instance_dir / "simple_metrics.csv"
    if data_path.is_file():
        return pd.read_csv(data_path)
    if simple_data_path.is_file():
        return pd.read_csv(simple_data_path)
    raise FileNotFoundError(f"Neither data.csv nor simple_metrics.csv exists in {repo_relative(instance_dir)}")


def dataset_name(instance_dir: Path) -> str:
    system_dir = instance_dir.parents[1].name
    if not system_dir.startswith("RE") or "-" not in system_dir:
        raise ValueError(f"Cannot derive RCA-Eval dataset name from {repo_relative(instance_dir)}")
    return system_dir.casefold()


def select_baro_window(metric_data: pd.DataFrame, injection_time: float) -> pd.DataFrame:
    if "time" not in metric_data.columns:
        raise ValueError("BARO metric data is missing the time column")

    numeric_data = metric_data.apply(pd.to_numeric, errors="coerce")
    non_numeric_values = numeric_data.isna() & ~metric_data.isna()
    if non_numeric_values.any().any():
        raise ValueError("BARO metric data contains non-numeric values")
    # This is RCA-Eval's canonical baseline preprocessing for sparse telemetry.
    numeric_data = numeric_data.ffill().fillna(0.0)
    if not numeric_data.map(math.isfinite).all().all():
        raise ValueError("BARO metric data contains non-finite values")

    filtered_data = numeric_data.loc[:, ~numeric_data.columns.str.endswith("_latency-50")]
    filtered_data = filtered_data.rename(
        columns={
            column: column.replace("_latency-90", "_latency")
            for column in filtered_data.columns
            if column.endswith("_latency-90")
        }
    )
    normal_data = filtered_data[filtered_data["time"] < injection_time].tail(BARO_WINDOW_SIZE)
    anomalous_data = filtered_data[filtered_data["time"] >= injection_time].head(BARO_WINDOW_SIZE)
    if normal_data.empty or anomalous_data.empty:
        raise ValueError("BARO requires both normal and anomalous metric windows")
    return pd.concat([normal_data, anomalous_data], ignore_index=True)


def compute_baro_metric_scores(metric_data: pd.DataFrame, injection_time: float, dataset: str) -> dict[str, float]:
    baro, preprocess = load_baro_dependencies()
    windowed_data = select_baro_window(metric_data, injection_time)
    normal_data = windowed_data[windowed_data["time"] < injection_time]
    anomalous_data = windowed_data[windowed_data["time"] >= injection_time]
    normal_data = preprocess(data=normal_data, dataset=dataset, dk_select_useful=False)
    anomalous_data = preprocess(data=anomalous_data, dataset=dataset, dk_select_useful=False)
    common_columns = [column for column in normal_data.columns if column in anomalous_data.columns]
    if not common_columns:
        raise ValueError("BARO preprocessing produced no common metric columns")
    normal_data = normal_data[common_columns]
    anomalous_data = anomalous_data[common_columns]

    metric_scores: dict[str, float] = {}
    for column in common_columns:
        normal_values = normal_data[column].to_numpy()
        anomalous_values = anomalous_data[column].to_numpy()
        robust_scaler = RobustScaler().fit(normal_values.reshape(-1, 1))
        score = float(max(robust_scaler.transform(anomalous_values.reshape(-1, 1))[:, 0]))
        if not math.isfinite(score):
            raise ValueError(f"BARO produced a non-finite score for metric {column}")
        metric_scores[column] = score

    baro_result = baro(data=windowed_data, inject_time=injection_time, dataset=dataset, dk_select_useful=False)
    if set(baro_result["ranks"]) != set(metric_scores):
        raise ValueError("BARO rank output does not match the reconstructed metric score set")
    return metric_scores


def metric_service(metric_name: str, services: set[str]) -> str:
    matching_services = [service for service in services if metric_name.startswith(f"{service}_")]
    if len(matching_services) != 1:
        raise ValueError(f"Cannot map BARO metric {metric_name} to exactly one traced service")
    return matching_services[0]


def aggregate_baro_service_scores(metric_scores: dict[str, float], services: set[str]) -> ServiceScoreAggregation:
    grouped_scores: dict[str, list[float]] = {service: [] for service in services}
    excluded_metric_count = 0
    for metric_name, score in metric_scores.items():
        if not any(metric_name.startswith(f"{service}_") for service in services):
            excluded_metric_count += 1
            continue
        service = metric_service(metric_name, services)
        grouped_scores[service].append(score)

    service_scores: dict[str, float] = {}
    for service, scores in grouped_scores.items():
        if not scores:
            raise ValueError(f"Traced service {service} has no BARO metric score")
        service_scores[service] = max(scores)
    return ServiceScoreAggregation(
        scores=service_scores,
        excluded_metric_count=excluded_metric_count,
    )


def graph_from_json(graph_data: dict[str, Any]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in graph_data["nodes"]:
        graph.add_node(node)
    for edge in graph_data["edges"]:
        graph.add_edge(edge["caller"], edge["callee"], weight=float(edge["frequency"]))
    return graph


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    finite_scores = [score for score in scores.values() if math.isfinite(score)]
    if not finite_scores:
        raise ValueError("Cannot normalize an empty set of finite BARO scores")
    lower_bound = min(finite_scores)
    upper_bound = max(finite_scores)
    if lower_bound == upper_bound:
        return {service: 0.5 for service in scores}
    return {service: (score - lower_bound) / (upper_bound - lower_bound) for service, score in scores.items()}


def compute_gc_baro_scores(graph: nx.DiGraph, service_scores: dict[str, float], lambd: float) -> dict[str, float]:
    from src.cut_algorithms.boykov_jolly import BoykovJollyCut

    if set(graph.nodes) != set(service_scores):
        raise ValueError("BARO services and runtime graph nodes differ")
    cut_algorithm = BoykovJollyCut(graph, normalize_scores(service_scores), lambd)
    return {
        service: energy_normal - energy_faulty
        for service, (energy_normal, energy_faulty) in (
            (service, cut_algorithm.compute_min_marginals(service)) for service in cut_algorithm.nodes
        )
    }


def ground_truth_service(instance_dir: Path) -> str:
    fault_directory = instance_dir.parent.name
    if "_f" in fault_directory:
        service, separator, fault = fault_directory.partition("_f")
        if separator != "_f" or not service or not fault:
            raise ValueError(f"Cannot derive root-cause service from {repo_relative(instance_dir)}")
        return service

    service, separator, fault = fault_directory.rpartition("_")
    if separator != "_" or not service or not fault:
        raise ValueError(f"Cannot derive root-cause service from {repo_relative(instance_dir)}")
    return service


def evaluation_row(variant: str, lambd: float, scores: dict[str, float], root_cause: str) -> dict[str, Any]:
    if root_cause not in scores:
        raise ValueError(f"Ground-truth service {root_cause} is absent from the candidate services")
    rank_series = pd.Series(scores, dtype=float).rank(ascending=False, method="average")
    expected_rank = float(rank_series[root_cause])
    candidate_count = len(scores)
    return {
        "variant": variant,
        "lambda": lambd,
        "root_cause_service": root_cause,
        "candidate_services": candidate_count,
        "expected_rank": expected_rank,
        "exam_score": expected_rank / candidate_count,
        "top_1": int(expected_rank <= 1.0),
        "top_3": int(expected_rank <= 3.0),
        "top_5": int(expected_rank <= 5.0),
    }


def evaluate_baro(instance_dir: Path, runtime_graph_data: dict[str, Any]) -> BaroEvaluation:
    metric_data = read_metric_data(instance_dir)
    injection_time = read_injection_time(instance_dir)
    graph = graph_from_json(runtime_graph_data)
    services = set(graph.nodes)
    root_cause = ground_truth_service(instance_dir)
    if root_cause not in services:
        raise ValueError(f"Ground-truth service {root_cause} is absent from the runtime graph")
    metric_scores = compute_baro_metric_scores(metric_data, injection_time, dataset_name(instance_dir))
    aggregation = aggregate_baro_service_scores(metric_scores, services)
    service_scores = aggregation.scores

    service_rows: list[dict[str, Any]] = []
    gc_scores_by_lambda: dict[float, dict[str, float]] = {}
    for lambd in LAMBDA_VALUES:
        gc_scores_by_lambda[lambd] = compute_gc_baro_scores(graph, service_scores, lambd)
    for service in sorted(services):
        row: dict[str, Any] = {
            "service": service,
            "baro_score": service_scores[service],
            "excluded_metric_count": aggregation.excluded_metric_count,
        }
        for lambd in LAMBDA_VALUES:
            row[f"gc_baro_lambda_{lambd:g}"] = gc_scores_by_lambda[lambd][service]
        service_rows.append(row)

    summary = [evaluation_row("baro", 0.0, service_scores, root_cause)]
    for lambd in LAMBDA_VALUES:
        summary.append(evaluation_row("gc_baro", lambd, gc_scores_by_lambda[lambd], root_cause))
    return BaroEvaluation(service_scores=service_rows, summary=summary)


def graph_paths(instance_dir: Path) -> dict[str, Path]:
    return {
        "pre": instance_dir / "call_graph_pre_injection.json",
        "runtime": instance_dir / "call_graph_runtime.json",
        "baro_scores": instance_dir / "baro_service_scores.csv",
        "baro_evaluation": instance_dir / "baro_evaluation.csv",
    }


def write_json(path: Path, data: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {repo_relative(path)}")
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {repo_relative(path)}")
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV artifact: {repo_relative(path)}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_graph_file(path: Path, expected: dict[str, Any]) -> str | None:
    if not path.is_file():
        return f"Missing graph artifact: {repo_relative(path)}"
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"Cannot parse {repo_relative(path)}: {exc}"
    if actual != expected:
        return f"Graph contents differ from deterministic extraction: {repo_relative(path)}"
    try:
        nodes = actual["nodes"]
        edges = actual["edges"]
        metadata = actual["metadata"]
        total_nodes = metadata["total_nodes"]
        total_edges = metadata["total_edges"]
    except (KeyError, TypeError):
        return f"Graph schema is incomplete: {repo_relative(path)}"
    if total_nodes != len(nodes) or total_edges != len(edges):
        return f"Graph metadata count mismatch: {repo_relative(path)}"
    node_set = set(nodes)
    for edge in edges:
        try:
            frequency = edge["frequency"]
            caller = edge["caller"]
            callee = edge["callee"]
        except KeyError:
            return f"Graph edge schema is incomplete: {repo_relative(path)}"
        if not isinstance(frequency, int) or frequency <= 0:
            return f"Invalid edge frequency in {repo_relative(path)}"
        if caller not in node_set or callee not in node_set:
            return f"Graph edge references an unknown node: {repo_relative(path)}"
    return None


def manifest_fields(records: list[dict[str, Any]]) -> list[str]:
    first_fields = ["suite", "instance", "status", "error"]
    other_fields = sorted({key for record in records for key in record} - set(first_fields))
    return first_fields + other_fields


def write_manifest(path: Path, records: list[dict[str, Any]], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {repo_relative(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = manifest_fields(records)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def aggregate_baro_results(data_root: Path, suites: tuple[Suite, ...], force: bool) -> None:
    instances, discovery_errors = discover_instances(data_root, suites)
    if discovery_errors:
        raise ValueError("Cannot aggregate BARO results for incomplete dataset suites")

    instance_rows: list[pd.DataFrame] = []
    for suite, instance_dir in instances:
        evaluation_path = graph_paths(instance_dir)["baro_evaluation"]
        if not evaluation_path.is_file():
            raise FileNotFoundError(f"Missing BARO evaluation artifact: {repo_relative(evaluation_path)}")
        evaluation_data = pd.read_csv(evaluation_path)
        if evaluation_data.empty:
            raise ValueError(f"BARO evaluation artifact is empty: {repo_relative(evaluation_path)}")
        evaluation_data["suite"] = suite.value.upper()
        evaluation_data["instance"] = repo_relative(instance_dir)
        instance_rows.append(evaluation_data)

    combined_data = pd.concat(instance_rows, ignore_index=True)
    summary_data = (
        combined_data.groupby(["variant", "lambda"], as_index=False)
        .agg(
            instances=("instance", "count"),
            mean_expected_rank=("expected_rank", "mean"),
            mean_exam_score=("exam_score", "mean"),
            top_1=("top_1", "mean"),
            top_3=("top_3", "mean"),
            top_5=("top_5", "mean"),
        )
        .sort_values(["variant", "lambda"])
    )
    write_csv(BARO_BENCHMARK_SUMMARY_PATH, summary_data.to_dict(orient="records"), force)


def extract_all(data_root: Path, suites: tuple[Suite, ...], force: bool) -> tuple[list[dict[str, Any]], bool]:
    instances, discovery_errors = discover_instances(data_root, suites)
    records = [
        {"suite": "DISCOVERY", "instance": "", "status": "error", "error": message}
        for message in discovery_errors
    ]
    success = not discovery_errors

    for suite, instance_dir in instances:
        inspection = inspect_instance(suite, instance_dir)
        if inspection.record["status"] != "ok":
            success = False
            records.append(inspection.record)
            LOGGER.error(f"{inspection.record['instance']}: {inspection.record['error']}")
            continue
        try:
            paths = graph_paths(instance_dir)
            if inspection.pre_graph is None or inspection.runtime_graph is None:
                raise RuntimeError("Successful inspection did not produce all graph artifacts")
            baro_evaluation = evaluate_baro(instance_dir, inspection.runtime_graph)
            write_json(paths["pre"], inspection.pre_graph, force)
            write_json(paths["runtime"], inspection.runtime_graph, force)
            write_csv(paths["baro_scores"], baro_evaluation.service_scores, force)
            write_csv(paths["baro_evaluation"], baro_evaluation.summary, force)
            records.append(inspection.record)
        except OSError as exc:
            success = False
            inspection.record.update({"status": "error", "error": str(exc)})
            records.append(inspection.record)
            LOGGER.error(f"{inspection.record['instance']}: {inspection.record['error']}")
    return records, success


def extract_graphs(data_root: Path, suites: tuple[Suite, ...], force: bool) -> tuple[list[dict[str, Any]], bool]:
    """Extract trace-derived graph artifacts without running BARO evaluation."""
    instances, discovery_errors = discover_instances(data_root, suites)
    records = [
        {"suite": "DISCOVERY", "instance": "", "status": "error", "error": message}
        for message in discovery_errors
    ]
    success = not discovery_errors

    for suite, instance_dir in instances:
        inspection = inspect_instance(suite, instance_dir)
        if inspection.record["status"] != "ok":
            success = False
            records.append(inspection.record)
            LOGGER.error(f"{inspection.record['instance']}: {inspection.record['error']}")
            continue
        try:
            if inspection.pre_graph is None or inspection.runtime_graph is None:
                raise RuntimeError("Successful inspection did not produce all graph artifacts")
            paths = graph_paths(instance_dir)
            existing_graphs = paths["pre"].exists() or paths["runtime"].exists()
            if existing_graphs:
                errors = [
                    validate_graph_file(paths["pre"], inspection.pre_graph),
                    validate_graph_file(paths["runtime"], inspection.runtime_graph),
                ]
                errors = [error for error in errors if error]
                if errors:
                    raise FileExistsError(" | ".join(errors))
                records.append(inspection.record)
                continue
            write_json(paths["pre"], inspection.pre_graph, force)
            write_json(paths["runtime"], inspection.runtime_graph, force)
            records.append(inspection.record)
        except OSError as exc:
            success = False
            inspection.record.update({"status": "error", "error": str(exc)})
            records.append(inspection.record)
            LOGGER.error(f"{inspection.record['instance']}: {inspection.record['error']}")
    return records, success


def evaluate_baro_all(data_root: Path, suites: tuple[Suite, ...], force: bool) -> tuple[list[dict[str, Any]], bool]:
    """Evaluate BARO and GC-BARO for instances with traceable root-cause services."""
    instances, discovery_errors = discover_instances(data_root, suites)
    records = [
        {"suite": "DISCOVERY", "instance": "", "status": "error", "error": message}
        for message in discovery_errors
    ]
    success = not discovery_errors

    for suite, instance_dir in instances:
        inspection = inspect_instance(suite, instance_dir)
        if inspection.record["status"] != "ok":
            success = False
            records.append(inspection.record)
            LOGGER.error(f"{inspection.record['instance']}: {inspection.record['error']}")
            continue
        try:
            if inspection.runtime_graph is None:
                raise RuntimeError("Successful inspection did not produce a runtime graph")
            runtime_path = graph_paths(instance_dir)["runtime"]
            graph_error = validate_graph_file(runtime_path, inspection.runtime_graph)
            if graph_error:
                raise ValueError(graph_error)

            paths = graph_paths(instance_dir)
            baro_evaluation = evaluate_baro(instance_dir, inspection.runtime_graph)
            write_csv(paths["baro_scores"], baro_evaluation.service_scores, force)
            write_csv(paths["baro_evaluation"], baro_evaluation.summary, force)
            records.append(inspection.record)
        except (FileNotFoundError, OSError, ValueError) as exc:
            success = False
            inspection.record.update({"status": "error", "error": str(exc)})
            records.append(inspection.record)
            LOGGER.error(f"{inspection.record['instance']}: {inspection.record['error']}")
    return records, success


def aggregate_baro_comparison(records: list[dict[str, Any]], force: bool) -> None:
    """Aggregate the paired BARO and GC-BARO results from successful evaluations."""
    evaluation_frames: list[pd.DataFrame] = []
    for record in records:
        if record["status"] != "ok":
            continue
        instance_dir = REPO_ROOT / record["instance"]
        evaluation_path = graph_paths(instance_dir)["baro_evaluation"]
        if not evaluation_path.is_file():
            raise FileNotFoundError(f"Missing BARO evaluation artifact: {repo_relative(evaluation_path)}")
        evaluation_data = pd.read_csv(evaluation_path)
        if evaluation_data.empty:
            raise ValueError(f"BARO evaluation artifact is empty: {repo_relative(evaluation_path)}")
        evaluation_data["system"] = instance_dir.parents[1].name
        evaluation_frames.append(evaluation_data)

    if not evaluation_frames:
        raise ValueError("No successful BARO evaluations are available for aggregation")

    combined_data = pd.concat(evaluation_frames, ignore_index=True)
    evaluated_suites = sorted({str(record["suite"]) for record in records if record["status"] == "ok"})
    overall_scope = f"{'/'.join(evaluated_suites)} traceable"
    metric_columns = {
        "instances": ("root_cause_service", "count"),
        "mean_expected_rank": ("expected_rank", "mean"),
        "median_expected_rank": ("expected_rank", "median"),
        "mean_exam_score": ("exam_score", "mean"),
        "median_exam_score": ("exam_score", "median"),
        "top_1_rate": ("top_1", "mean"),
        "top_3_rate": ("top_3", "mean"),
        "top_5_rate": ("top_5", "mean"),
    }
    overall_summary = combined_data.groupby(["variant", "lambda"], as_index=False).agg(**metric_columns)
    overall_summary.insert(0, "scope", overall_scope)
    system_summary = combined_data.groupby(["system", "variant", "lambda"], as_index=False).agg(**metric_columns)
    system_summary = system_summary.rename(columns={"system": "scope"})
    summary = pd.concat([overall_summary, system_summary], ignore_index=True)
    write_csv(BARO_COMPARISON_SUMMARY_PATH, summary.to_dict(orient="records"), force)


def verify_all(data_root: Path, suites: tuple[Suite, ...]) -> tuple[list[dict[str, Any]], bool]:
    instances, discovery_errors = discover_instances(data_root, suites)
    records = [
        {"suite": "DISCOVERY", "instance": "", "status": "error", "error": message}
        for message in discovery_errors
    ]
    success = not discovery_errors

    for suite, instance_dir in instances:
        inspection = inspect_instance(suite, instance_dir)
        if inspection.record["status"] != "ok":
            success = False
            records.append(inspection.record)
            continue
        paths = graph_paths(instance_dir)
        if inspection.pre_graph is None or inspection.runtime_graph is None:
            raise RuntimeError("Successful inspection did not produce both graph artifacts")
        errors = [
            validate_graph_file(paths["pre"], inspection.pre_graph),
            validate_graph_file(paths["runtime"], inspection.runtime_graph),
        ]
        if not paths["baro_scores"].is_file():
            errors.append(f"Missing BARO service-score artifact: {repo_relative(paths['baro_scores'])}")
        if not paths["baro_evaluation"].is_file():
            errors.append(f"Missing BARO evaluation artifact: {repo_relative(paths['baro_evaluation'])}")
        errors = [error for error in errors if error]
        if errors:
            success = False
            inspection.record.update({"status": "error", "error": " | ".join(errors)})
        records.append(inspection.record)
    return records, success


def run_self_test() -> None:
    """Exercise trace-key isolation and deterministic graph serialization without repository writes."""
    with tempfile.TemporaryDirectory(prefix="rca-eval-self-test-") as temp_dir:
        instance = Path(temp_dir) / "RE2" / "RE2-OB" / "cartservice_cpu" / "1"
        instance.mkdir(parents=True)
        (instance / "inject_time.txt").write_text("100\n", encoding="utf-8")
        pd.DataFrame(
            [
                ["normal", "1", "frontend", "GET /", "root", 80, 5, ""],
                ["normal", "2", "cartservice", "GET /cart", "cart", 82, 2, "1"],
                ["fault", "1", "frontend", "GET /", "root", 120, 12, ""],
                ["fault", "2", "paymentservice", "POST /pay", "pay", 124, 3, "1"],
            ],
            columns=[
                "traceID",
                "spanID",
                "serviceName",
                "operationName",
                "methodName",
                "startTime",
                "duration",
                "parentSpanID",
            ],
        ).to_csv(instance / "traces.csv", index=False)
        pd.DataFrame(
            {
                "time": [80, 82, 84, 86, 88, 90, 110, 112, 114, 116, 118, 120],
                "frontend_cpu": [1, 2, 1, 2, 1, 2, 10, 11, 12, 11, 13, 12],
                "cartservice_cpu": [2, 3, 2, 3, 2, 3, 8, 9, 8, 10, 9, 11],
                "paymentservice_cpu": [3, 4, 3, 4, 3, 4, 7, 8, 9, 8, 10, 9],
            }
        ).to_csv(instance / "data.csv", index=False)

        inspection = inspect_instance(Suite.RE2, instance)
        if inspection.record["status"] != "ok":
            raise AssertionError(inspection.record["error"])
        assert inspection.pre_graph is not None
        assert inspection.runtime_graph is not None
        assert inspection.pre_graph["nodes"] == ["cartservice", "frontend"]
        assert inspection.pre_graph["edges"] == [
            {"caller": "frontend", "callee": "cartservice", "frequency": 1}
        ]
        assert inspection.runtime_graph["edges"] == [
            {"caller": "frontend", "callee": "cartservice", "frequency": 1},
            {"caller": "frontend", "callee": "paymentservice", "frequency": 1},
        ]
        baro_evaluation = evaluate_baro(instance, inspection.runtime_graph)
        assert len(baro_evaluation.service_scores) == 3
        assert len(baro_evaluation.summary) == len(LAMBDA_VALUES) + 1
        assert baro_evaluation.summary[0]["root_cause_service"] == "cartservice"
        assert inspection.record["runtime_unmatched_parents"] == 0
    LOGGER.info("Self-test passed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if COMMAND == PipelineCommand.SELF_TEST:
        run_self_test()
        return

    if COMMAND in {PipelineCommand.DOWNLOAD, PipelineCommand.PREPARE}:
        download_suites(DATA_ROOT, SELECTED_SUITES)
        if COMMAND == PipelineCommand.DOWNLOAD:
            return

    if COMMAND == PipelineCommand.EXTRACT_GRAPHS:
        records, success = extract_graphs(DATA_ROOT, SELECTED_SUITES, OVERWRITE_ARTIFACTS)
        manifest_path = GRAPH_EXTRACTION_MANIFEST_PATH
    elif COMMAND == PipelineCommand.EVALUATE_BARO:
        records, success = evaluate_baro_all(DATA_ROOT, SELECTED_SUITES, OVERWRITE_ARTIFACTS)
        manifest_path = BARO_EVALUATION_MANIFEST_PATH
        aggregate_baro_comparison(records, OVERWRITE_ARTIFACTS)
    elif COMMAND == PipelineCommand.VERIFY:
        records, success = verify_all(DATA_ROOT, SELECTED_SUITES)
        manifest_path = VALIDATION_MANIFEST_PATH
    else:
        records, success = extract_all(DATA_ROOT, SELECTED_SUITES, OVERWRITE_ARTIFACTS)
        manifest_path = EXTRACTION_MANIFEST_PATH
        if success:
            aggregate_baro_results(DATA_ROOT, SELECTED_SUITES, OVERWRITE_ARTIFACTS)
        if success and COMMAND == PipelineCommand.PREPARE:
            records, success = verify_all(DATA_ROOT, SELECTED_SUITES)
            manifest_path = VALIDATION_MANIFEST_PATH

    write_manifest(manifest_path, records, OVERWRITE_ARTIFACTS)
    LOGGER.info("Wrote pipeline manifest to %s", repo_relative(manifest_path))
    if not success:
        raise RuntimeError("RCA-Eval graph pipeline failed; inspect the validation manifest")


if __name__ == "__main__":
    main()
