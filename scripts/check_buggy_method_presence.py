#!/usr/bin/env python3
"""Check exact buggy-node presence and Top-k benchmark performance."""

# %% Imports and defaults
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TRUTH = ROOT_DIR / "ground_truth.csv"
EXPECTED_RANK_SUFFIX = "_expected_inspections"
TOP_KS = (1, 3, 5)


# %% Name normalization and input loading
def method_key(name: str) -> str:
    """Normalize inner-class spelling while preserving the full signature."""
    return str(name).strip().replace("$", ".")


def load_ground_truth(path: Path) -> dict[tuple[str, int], dict]:
    frame = pd.read_csv(path, dtype={"project": str, "bug_id": int})
    required = {
        "project", "bug_id", "buggy_methods", "buggy_nodes", "trace_status"
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Ground truth is missing columns: {sorted(missing)}")

    truth: dict[tuple[str, int], dict] = {}
    for row in frame.itertuples(index=False):
        try:
            methods = json.loads(row.buggy_methods)
            nodes = json.loads(row.buggy_nodes)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid buggy_methods JSON for {row.project}_{row.bug_id}"
            ) from exc

        if (
            not isinstance(methods, list)
            or not all(isinstance(method, str) for method in methods)
            or not isinstance(nodes, list)
            or not all(isinstance(node, str) for node in nodes)
        ):
            raise ValueError(
                f"buggy_methods must be a JSON string list for "
                f"{row.project}_{row.bug_id}"
            )

        instance = (row.project, int(row.bug_id))
        if instance in truth:
            raise ValueError(f"Duplicate ground-truth instance: {instance}")
        truth[instance] = {
            "methods": sorted(set(methods)),
            "nodes": sorted({str(node).strip().replace("$", ".") for node in nodes}),
            "trace_status": row.trace_status,
        }

    return truth


def identify_instance(
    result_path: Path,
    truth: dict[tuple[str, int], dict],
) -> tuple[str, int]:
    """Identify Project_BugID while allowing an optional filename suffix."""
    stem = result_path.stem
    candidates = [
        instance
        for instance in truth
        if stem == f"{instance[0]}_{instance[1]}"
        or stem.startswith(f"{instance[0]}_{instance[1]}_")
    ]
    if not candidates:
        raise ValueError("filename does not match a ground-truth Project_BugID")

    # Prefer the longest exact Project_BugID prefix if project names overlap.
    candidates.sort(key=lambda item: len(f"{item[0]}_{item[1]}"), reverse=True)
    return candidates[0]


# %% Per-file check
def check_result(
    result_path: Path,
    truth: dict[tuple[str, int], dict],
) -> tuple[dict, list[dict]]:
    project, bug_id = identify_instance(result_path, truth)
    frame = pd.read_csv(result_path)
    if "Method" not in frame or frame.empty:
        raise ValueError("missing Method column or empty result")

    frame = frame.copy()
    frame["_node"] = frame["Method"].map(method_key)

    result_nodes = set(frame["_node"].dropna())
    annotation = truth[(project, bug_id)]
    buggy_nodes = annotation["nodes"]
    matched = [node for node in buggy_nodes if node in result_nodes]
    missing = [node for node in buggy_nodes if node not in result_nodes]

    if annotation["trace_status"] == "representation_gap":
        status = "representation_gap"
    elif annotation["trace_status"] == "trace_gap":
        status = "trace_gap"
    elif not missing:
        status = "conforming"
    elif matched:
        status = "partial"
    else:
        status = "missing"

    presence = {
        "project": project,
        "bug_id": bug_id,
        "result_file": str(result_path),
        "status": status,
        "result_methods": len(frame),
        "source_buggy_methods": len(annotation["methods"]),
        "buggy_nodes": len(buggy_nodes),
        "matched_buggy_methods": len(matched),
        "matched": json.dumps(matched),
        "missing": json.dumps(missing),
    }

    metric_rows: list[dict] = []
    if status == "conforming":
        buggy_rows = frame[frame["_node"].isin(buggy_nodes)]
        rank_columns = sorted(
            column
            for column in frame.columns
            if column.endswith(EXPECTED_RANK_SUFFIX)
        )
        for rank_column in rank_columns:
            metric = rank_column[: -len(EXPECTED_RANK_SUFFIX)]
            if metric not in frame.columns:
                continue
            ranks = pd.to_numeric(buggy_rows[rank_column], errors="coerce").dropna()
            if ranks.empty:
                continue
            first_fault_rank = float(ranks.min())
            metric_row = {
                "result_set": result_path.parent.name,
                "project": project,
                "bug_id": bug_id,
                "result_file": str(result_path),
                "metric": metric,
                "first_fault_expected_rank": first_fault_rank,
                "total_methods": len(frame),
                "first_fault_exam": first_fault_rank / len(frame),
            }
            for k in TOP_KS:
                metric_row[f"top_{k}"] = first_fault_rank <= k
            metric_rows.append(metric_row)

    return presence, metric_rows


# %% Folder-level report
def build_report(
    results_folder: Path,
    ground_truth_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    truth = load_ground_truth(ground_truth_path)
    result_paths = sorted(results_folder.rglob("*.csv"))
    if not result_paths:
        raise FileNotFoundError(f"No CSV files found under {results_folder}")

    rows: list[dict] = []
    metric_rows: list[dict] = []
    skipped: list[dict] = []
    for result_path in result_paths:
        try:
            presence, file_metrics = check_result(result_path, truth)
            rows.append(presence)
            metric_rows.extend(file_metrics)
        except Exception as exc:
            skipped.append({"result_file": str(result_path), "reason": str(exc)})

    report = pd.DataFrame(rows)
    metrics = pd.DataFrame(metric_rows)
    if not report.empty:
        report = report.sort_values(["project", "bug_id", "result_file"])
    if not metrics.empty:
        metrics = metrics.sort_values(["result_set", "metric", "project", "bug_id"])
    return report, metrics, skipped


def aggregate_top_k(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate first-fault Top-k rates over fully traceable instances."""
    if metrics.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (result_set, metric), group in metrics.groupby(
        ["result_set", "metric"], sort=True
    ):
        row = {
            "result_set": result_set,
            "metric": metric,
            "instances": len(group),
            "mean_expected_rank": group["first_fault_expected_rank"].mean(),
            "median_expected_rank": group["first_fault_expected_rank"].median(),
            "mean_exam": group["first_fault_exam"].mean(),
        }
        for k in TOP_KS:
            count = int(group[f"top_{k}"].sum())
            row[f"top_{k}_count"] = count
            row[f"top_{k}_rate"] = count / len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def print_report(
    report: pd.DataFrame,
    metrics: pd.DataFrame,
    skipped: list[dict],
) -> None:
    total = len(report)
    representation_gaps = (
        int((report["status"] == "representation_gap").sum()) if total else 0
    )
    trace_gaps = int((report["status"] == "trace_gap").sum()) if total else 0
    evaluable = total - representation_gaps - trace_gaps
    conforming = int((report["status"] == "conforming").sum()) if total else 0
    partial = int((report["status"] == "partial").sum()) if total else 0
    missing = int((report["status"] == "missing").sum()) if total else 0
    at_least_one = conforming + partial

    print("Buggy-method presence report")
    print(f"Checked result CSVs:       {total}")
    print(f"Evaluable instances:       {evaluable}")
    print(
        f"Conforming (all present):  {conforming} "
        f"({conforming / evaluable:.1%})"
        if evaluable
        else "Conforming (all present):  0"
    )
    print(
        f"At least one present:      {at_least_one} "
        f"({at_least_one / evaluable:.1%})"
        if evaluable
        else "At least one present:      0"
    )
    print(f"Partial matches:           {partial}")
    print(f"No buggy method present:   {missing}")
    print(f"Source/call-graph gaps:    {trace_gaps}")
    print(f"Empty ground-truth lists:  {representation_gaps}")
    print(f"Skipped CSVs:              {len(skipped)}")

    nonconforming = report[report["status"] != "conforming"]
    if not nonconforming.empty:
        print("\nNonconforming instances:")
        print(
            nonconforming[
                ["project", "bug_id", "status", "matched", "missing", "result_file"]
            ].to_string(index=False)
        )

    if skipped:
        print("\nSkipped CSVs:")
        print(pd.DataFrame(skipped).to_string(index=False))

    top_k = aggregate_top_k(metrics)
    if not top_k.empty:
        display = top_k.copy()
        for column in ["top_1_rate", "top_3_rate", "top_5_rate", "mean_exam"]:
            display[column] = display[column].map(lambda value: f"{value:.1%}")
        for column in ["mean_expected_rank", "median_expected_rank"]:
            display[column] = display[column].map(lambda value: f"{value:.2f}")
        print("\nTop-k comparison (first exact buggy node; average tie rank):")
        print(display.to_string(index=False))


# %% Command-line entry point
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether ground-truth buggy methods occur in result CSVs."
    )
    parser.add_argument(
        "results_folder",
        type=Path,
        help="Folder containing result CSVs; subfolders are searched recursively.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help=f"Ground-truth CSV (default: {DEFAULT_GROUND_TRUTH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path at which to save the per-result report as CSV.",
    )
    parser.add_argument(
        "--top-k-output",
        type=Path,
        help="Optional CSV path for per-instance, per-metric Top-k results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_report, metric_report, skipped_results = build_report(
        args.results_folder.resolve(), args.ground_truth.resolve()
    )
    print_report(result_report, metric_report, skipped_results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result_report.to_csv(args.output, index=False)
        print(f"\nSaved detailed report to {args.output}")
    if args.top_k_output:
        args.top_k_output.parent.mkdir(parents=True, exist_ok=True)
        metric_report.to_csv(args.top_k_output, index=False)
        print(f"Saved per-instance Top-k report to {args.top_k_output}")
