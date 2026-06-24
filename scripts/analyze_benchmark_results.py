#!/usr/bin/env python3
"""Analyze benchmark rankings against the curated buggy-method ground truth.

The script expects result CSVs produced by ``run_benchmark_pregen.py`` under
directories named ``eval_lambd_<value>``. It never uses the legacy
``buggy_methods.txt`` files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SUFFIX = "_expected_inspections"


def method_key(name: str) -> str:
    return str(name).strip().replace("$", ".").split("(", 1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare benchmark rankings using the curated ground_truth.csv."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "first_10",
        help="Directory containing eval_lambd_* result folders.",
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=ROOT / "ground_truth.csv",
        help="CSV with project, bug_id and JSON-list buggy_methods columns.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: <results-dir>/analysis).",
    )
    parser.add_argument(
        "--top-k", default="1,3,5,10",
        help="Comma-separated Top-k thresholds based on expected tie rank.",
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=2000,
        help="Bootstrap resamples used for 95%% confidence intervals; 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ground_truth(path: Path) -> dict[tuple[str, int], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth not found: {path}")
    frame = pd.read_csv(path, dtype={"project": str, "bug_id": int})
    required = {"project", "bug_id", "buggy_methods"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Ground truth must contain {sorted(required)}")

    result: dict[tuple[str, int], list[str]] = {}
    for row in frame.itertuples(index=False):
        try:
            methods = json.loads(row.buggy_methods)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid buggy_methods JSON for {row.project}_{row.bug_id}"
            ) from exc
        if not isinstance(methods, list) or not all(isinstance(m, str) for m in methods):
            raise ValueError(f"buggy_methods must be a JSON list for {row.project}_{row.bug_id}")
        key = (row.project, int(row.bug_id))
        if key in result:
            raise ValueError(f"Duplicate ground-truth row: {row.project}_{row.bug_id}")
        result[key] = sorted(set(method_key(m) for m in methods))
    return result


def identify_result(path: Path) -> tuple[str, int, str, float]:
    match = re.fullmatch(r"eval_lambd_(.+)", path.parent.name)
    if not match:
        raise ValueError(f"Result must be inside eval_lambd_<value>: {path}")
    lambda_label = match.group(1)
    try:
        lambda_value = float(lambda_label)
    except ValueError as exc:
        raise ValueError(f"Invalid lambda in directory {path.parent.name}") from exc

    suffix = f"_{lambda_label}"
    stem = path.stem
    if not stem.endswith(suffix):
        raise ValueError(f"Filename does not end with {suffix}: {path.name}")
    bug_name = stem[: -len(suffix)]
    try:
        project, bug_id_text = bug_name.rsplit("_", 1)
        bug_id = int(bug_id_text)
    except ValueError as exc:
        raise ValueError(f"Cannot parse project and bug ID from {path.name}") from exc
    return project, bug_id, lambda_label, lambda_value


def discover_results(results_dir: Path, output_dir: Path) -> list[Path]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    paths = []
    for path in results_dir.glob("eval_lambd_*/*.csv"):
        if output_dir in path.parents:
            continue
        paths.append(path)
    return sorted(paths)


def score_families(frame: pd.DataFrame) -> list[str]:
    families = []
    for column in frame.columns:
        if not column.endswith(EXPECTED_SUFFIX):
            continue
        family = column[: -len(EXPECTED_SUFFIX)]
        required = {family, f"{family}_rank", f"{family}_exam_score"}
        if required.issubset(frame.columns):
            families.append(family)
    return families


def analyze_file(
    path: Path,
    truth: dict[tuple[str, int], list[str]],
    top_ks: list[int],
) -> tuple[dict, list[dict], list[dict]]:
    project, bug_id, lambda_label, lambda_value = identify_result(path)
    frame = pd.read_csv(path)
    if "Method" not in frame.columns or frame.empty:
        raise ValueError(f"Missing Method column or empty result: {path}")
    families = score_families(frame)
    if not families:
        raise ValueError(f"No complete score/rank column families in {path}")

    gt_methods = truth.get((project, bug_id))
    if gt_methods is None:
        status = "missing_ground_truth"
        gt_methods = []
    elif not gt_methods:
        status = "representation_gap"
    else:
        status = "pending"

    frame = frame.copy()
    frame["_method_key"] = frame["Method"].map(method_key)
    matched_keys = [key for key in gt_methods if (frame["_method_key"] == key).any()]
    if status == "pending":
        if not matched_keys:
            status = "fault_not_in_results"
        elif len(matched_keys) < len(gt_methods):
            status = "partially_traceable"
        else:
            status = "fully_traceable"

    ambiguous = sum((frame["_method_key"] == key).sum() > 1 for key in matched_keys)
    instance = {
        "project": project,
        "bug_id": bug_id,
        "lambda": lambda_value,
        "lambda_label": lambda_label,
        "result_file": str(path.resolve()),
        "status": status,
        "total_methods": len(frame),
        "ground_truth_methods": len(gt_methods),
        "matched_ground_truth_methods": len(matched_keys),
        "ambiguous_overload_keys": ambiguous,
    }

    metric_rows = []
    if matched_keys:
        for family in families:
            per_fault = []
            for key in matched_keys:
                candidates = frame.loc[frame["_method_key"] == key]
                # Argument-free ground truth can match overloads. The best overload
                # rank is used and the ambiguity is explicitly counted above.
                per_fault.append({
                    "method_key": key,
                    "expected_rank": float(candidates[f"{family}_expected_inspections"].min()),
                    "min_rank": float(candidates[f"{family}_rank"].min()),
                    "exam": float(candidates[f"{family}_exam_score"].min()),
                })
            expected = np.array([item["expected_rank"] for item in per_fault])
            min_ranks = np.array([item["min_rank"] for item in per_fault])
            exams = np.array([item["exam"] for item in per_fault])
            row = {
                **{k: instance[k] for k in (
                    "project", "bug_id", "lambda", "lambda_label", "status",
                    "total_methods", "ground_truth_methods", "matched_ground_truth_methods",
                    "ambiguous_overload_keys",
                )},
                "metric": family,
                "first_fault_expected_rank": float(expected.min()),
                "first_fault_min_rank": float(min_ranks.min()),
                "mean_fault_expected_rank": float(expected.mean()),
                "worst_fault_expected_rank": float(expected.max()),
                "first_fault_exam": float(exams.min()),
                "mean_fault_exam": float(exams.mean()),
                "worst_fault_exam": float(exams.max()),
                "mrr": float(1.0 / expected.min()),
            }
            for k in top_ks:
                row[f"top_{k}"] = bool(expected.min() <= k)
            metric_rows.append(row)

    invariance_rows = []
    if math.isclose(lambda_value, 0.0, abs_tol=1e-15):
        for baseline in ("tarantula", "ochiai", "dstar"):
            gc = f"{baseline}_gc"
            left = f"{baseline}_expected_inspections"
            right = f"{gc}_expected_inspections"
            if left not in frame or right not in frame:
                continue
            delta = (frame[left] - frame[right]).abs()
            invariance_rows.append({
                "project": project,
                "bug_id": bug_id,
                "metric": baseline,
                "methods": len(frame),
                "unequal_expected_ranks": int((delta > 1e-12).sum()),
                "max_absolute_rank_difference": float(delta.max()),
                "lambda_zero_invariant": bool((delta <= 1e-12).all()),
            })
    return instance, metric_rows, invariance_rows


def bootstrap_mean_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0 or samples <= 0:
        return math.nan, math.nan
    if len(values) == 1:
        value = float(values[0])
        return value, value
    means = np.empty(samples)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def cohort_views(metrics: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    any_traceable = metrics[metrics["matched_ground_truth_methods"] > 0]
    fully_traceable = metrics[metrics["status"] == "fully_traceable"]
    return [("any_traceable", any_traceable), ("fully_traceable", fully_traceable)]


def aggregate_metrics(
    metrics: pd.DataFrame,
    top_ks: list[int],
    bootstrap_samples: int,
    rng: np.random.Generator,
    by_project: bool = False,
) -> pd.DataFrame:
    output = []
    for cohort, subset in cohort_views(metrics):
        group_columns = ["project", "lambda", "metric"] if by_project else ["lambda", "metric"]
        for group_key, group in subset.groupby(group_columns, sort=True):
            if by_project:
                project, lambda_value, metric = group_key
            else:
                lambda_value, metric = group_key
                project = None
            exams = group["first_fault_exam"].to_numpy(float)
            ci_low, ci_high = bootstrap_mean_ci(exams, bootstrap_samples, rng)
            row = {
                "cohort": cohort,
                "lambda": lambda_value,
                "metric": metric,
                "instances": len(group),
                "mean_first_fault_rank": group["first_fault_expected_rank"].mean(),
                "median_first_fault_rank": group["first_fault_expected_rank"].median(),
                "mrr": group["mrr"].mean(),
                "mean_exam": group["first_fault_exam"].mean(),
                "median_exam": group["first_fault_exam"].median(),
                "mean_exam_ci95_low": ci_low,
                "mean_exam_ci95_high": ci_high,
                "mean_all_fault_exam": group["mean_fault_exam"].mean(),
                "mean_worst_fault_exam": group["worst_fault_exam"].mean(),
            }
            if by_project:
                row["project"] = project
            for k in top_ks:
                row[f"top_{k}_rate"] = group[f"top_{k}"].mean()
            output.append(row)
    columns = (["project"] if by_project else []) + [
        "cohort", "lambda", "metric", "instances", "mean_first_fault_rank",
        "median_first_fault_rank", "mrr", "mean_exam", "median_exam",
        "mean_exam_ci95_low", "mean_exam_ci95_high", "mean_all_fault_exam",
        "mean_worst_fault_exam", *[f"top_{k}_rate" for k in top_ks],
    ]
    return pd.DataFrame(output).reindex(columns=columns)


def rank_biserial_for_improvement(deltas: np.ndarray) -> float:
    nonzero = deltas[~np.isclose(deltas, 0.0, atol=1e-12)]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    worse = ranks[nonzero > 0].sum()
    better = ranks[nonzero < 0].sum()
    return float((better - worse) / ranks.sum())


def paired_comparisons(
    metrics: pd.DataFrame,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    output = []
    key_columns = ["project", "bug_id", "lambda"]
    for cohort, subset in cohort_views(metrics):
        for baseline in ("tarantula", "ochiai", "dstar"):
            gc = f"{baseline}_gc"
            left = subset[subset["metric"] == baseline]
            right = subset[subset["metric"] == gc]
            paired = left.merge(right, on=key_columns, suffixes=("_baseline", "_gc"))
            for lambda_value, group in paired.groupby("lambda", sort=True):
                delta = (
                    group["first_fault_exam_gc"] - group["first_fault_exam_baseline"]
                ).to_numpy(float)
                if len(delta) == 0:
                    continue
                nonzero = delta[~np.isclose(delta, 0.0, atol=1e-12)]
                if len(nonzero):
                    try:
                        p_value = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                    except ValueError:
                        p_value = 1.0
                else:
                    p_value = 1.0
                ci_low, ci_high = bootstrap_mean_ci(delta, bootstrap_samples, rng)
                output.append({
                    "cohort": cohort,
                    "lambda": lambda_value,
                    "baseline": baseline,
                    "instances": len(delta),
                    "gc_wins": int((delta < -1e-12).sum()),
                    "ties": int(np.isclose(delta, 0.0, atol=1e-12).sum()),
                    "gc_losses": int((delta > 1e-12).sum()),
                    "mean_exam_delta_gc_minus_baseline": float(delta.mean()),
                    "median_exam_delta_gc_minus_baseline": float(np.median(delta)),
                    "mean_delta_ci95_low": ci_low,
                    "mean_delta_ci95_high": ci_high,
                    "wilcoxon_p": p_value,
                    "rank_biserial_effect_favoring_gc": rank_biserial_for_improvement(delta),
                })
    frame = pd.DataFrame(output)
    if frame.empty:
        return frame
    # Holm correction across the reported paired tests.
    order = np.argsort(frame["wilcoxon_p"].to_numpy())
    adjusted = np.empty(len(frame))
    running = 0.0
    total = len(frame)
    for rank, index in enumerate(order):
        candidate = min(1.0, frame.iloc[index]["wilcoxon_p"] * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    frame["wilcoxon_p_holm"] = adjusted
    return frame


def write_outputs(
    output_dir: Path,
    instances: pd.DataFrame,
    metrics: pd.DataFrame,
    invariance: pd.DataFrame,
    top_ks: list[int],
    bootstrap_samples: int,
    seed: int,
    source_files: list[Path],
    ground_truth: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    aggregate = aggregate_metrics(metrics, top_ks, bootstrap_samples, rng)
    projects = aggregate_metrics(metrics, top_ks, bootstrap_samples, rng, by_project=True)
    comparisons = paired_comparisons(metrics, bootstrap_samples, rng)
    cohort = (
        instances.groupby(["lambda", "status"], dropna=False)
        .size().rename("instances").reset_index()
    )

    instances.to_csv(output_dir / "instance_coverage.csv", index=False)
    metrics.to_csv(output_dir / "per_bug_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)
    projects.to_csv(output_dir / "project_summary.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    cohort.to_csv(output_dir / "cohort_summary.csv", index=False)
    invariance.to_csv(output_dir / "lambda_zero_invariance.csv", index=False)
    manifest = {
        "ground_truth": str(ground_truth.resolve()),
        "result_files": len(source_files),
        "top_k": top_ks,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "outputs": [
            "instance_coverage.csv", "per_bug_metrics.csv", "aggregate_summary.csv",
            "project_summary.csv", "paired_comparisons.csv", "cohort_summary.csv",
            "lambda_zero_invariance.csv",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir / "analysis"
    top_ks = sorted({int(item) for item in args.top_k.split(",") if item.strip()})
    if not top_ks or min(top_ks) <= 0:
        raise ValueError("--top-k values must be positive integers")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative")

    truth = load_ground_truth(args.ground_truth)
    result_files = discover_results(args.results_dir, output_dir)
    if not result_files:
        raise FileNotFoundError(f"No eval_lambd_*/*.csv files under {args.results_dir}")

    instances, metrics, invariance = [], [], []
    errors = []
    for path in result_files:
        try:
            instance, file_metrics, file_invariance = analyze_file(path, truth, top_ks)
            instances.append(instance)
            metrics.extend(file_metrics)
            invariance.extend(file_invariance)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise RuntimeError("Failed to analyze result files:\n" + "\n".join(errors))

    instance_frame = pd.DataFrame(instances).sort_values(["lambda", "project", "bug_id"])
    metric_frame = pd.DataFrame(metrics).sort_values(["lambda", "project", "bug_id", "metric"])
    invariance_frame = pd.DataFrame(invariance)
    write_outputs(
        output_dir, instance_frame, metric_frame, invariance_frame, top_ks,
        args.bootstrap_samples, args.seed, result_files, args.ground_truth,
    )

    statuses = instance_frame.groupby("status").size().to_dict()
    print(f"Analyzed {len(result_files)} result files using curated ground truth.")
    print("Instance statuses:", ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    print(f"Wrote analysis to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
