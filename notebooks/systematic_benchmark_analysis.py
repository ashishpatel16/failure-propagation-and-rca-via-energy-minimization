# %% [markdown]
# # Systematic benchmark analysis
#
# This percent-format notebook evaluates the benchmark results against the
# curated `ground_truth.csv`. Run it as a normal Python script or cell-by-cell
# in VS Code/Jupyter-compatible editors.
#
# The analysis distinguishes:
#
# - **fully traceable** bugs: every curated buggy method occurs in the ranking;
# - **partially traceable** bugs: only some curated methods occur;
# - **fault not in results**: no curated method can be ranked;
# - **representation gaps**: the patch changes only field/static/class state.
#
# Primary comparisons use paired baseline/GraphCut EXAM scores on the same bugs.

# %%
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd()
    if ROOT.name == "notebooks":
        ROOT = ROOT.parent

RESULTS_DIR = Path(os.environ.get("BENCHMARK_RESULTS_DIR", ROOT / "first_10"))
GROUND_TRUTH = Path(os.environ.get("GROUND_TRUTH_CSV", ROOT / "ground_truth.csv"))
OUTPUT_DIR = Path(os.environ.get(
    "SYSTEMATIC_ANALYSIS_DIR", RESULTS_DIR / "systematic_analysis"
))
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Keep plotting caches inside the writable analysis directory.
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

sys.path.insert(0, str(ROOT / "scripts"))
import analyze_benchmark_results as benchmark_analysis

SELECTED_LAMBDA = float(os.environ.get("SELECTED_LAMBDA", "1.0"))
TOP_K = [1, 3, 5, 10]
BOOTSTRAP_SAMPLES = int(os.environ.get("BOOTSTRAP_SAMPLES", "2000"))
RANDOM_SEED = 42

sns.set_theme(style="whitegrid", context="talk")
BASELINES = ["tarantula", "ochiai", "dstar"]
BASELINE_LABELS = {"tarantula": "Tarantula", "ochiai": "Ochiai", "dstar": "D*"}
STATUS_LABELS = {
    "fully_traceable": "Fully traceable",
    "partially_traceable": "Partially traceable",
    "fault_not_in_results": "Fault absent",
    "representation_gap": "Representation gap",
    "missing_ground_truth": "Missing ground truth",
}

print(f"Results: {RESULTS_DIR}")
print(f"Ground truth: {GROUND_TRUTH}")
print(f"Outputs: {OUTPUT_DIR}")

# %% [markdown]
# ## 1. Rebuild analysis tables from curated ground truth
#
# This cell deliberately bypasses every legacy `buggy_methods.txt` file.

# %%
truth = benchmark_analysis.load_ground_truth(GROUND_TRUTH)
result_files = benchmark_analysis.discover_results(RESULTS_DIR, OUTPUT_DIR)
if not result_files:
    raise FileNotFoundError(f"No eval_lambd_*/*.csv files under {RESULTS_DIR}")

instance_records: list[dict] = []
metric_records: list[dict] = []
invariance_records: list[dict] = []
for result_file in result_files:
    instance, metrics_for_file, invariance_for_file = benchmark_analysis.analyze_file(
        result_file, truth, TOP_K
    )
    instance_records.append(instance)
    metric_records.extend(metrics_for_file)
    invariance_records.extend(invariance_for_file)

instances = pd.DataFrame(instance_records).sort_values(["lambda", "project", "bug_id"])
metrics = pd.DataFrame(metric_records).sort_values(
    ["lambda", "project", "bug_id", "metric"]
)
lambda_zero = pd.DataFrame(invariance_records)

benchmark_analysis.write_outputs(
    TABLE_DIR,
    instances,
    metrics,
    lambda_zero,
    TOP_K,
    BOOTSTRAP_SAMPLES,
    RANDOM_SEED,
    result_files,
    GROUND_TRUTH,
)

aggregate = pd.read_csv(TABLE_DIR / "aggregate_summary.csv")
project_summary = pd.read_csv(TABLE_DIR / "project_summary.csv")
paired_summary = pd.read_csv(TABLE_DIR / "paired_comparisons.csv")
cohort_summary = pd.read_csv(TABLE_DIR / "cohort_summary.csv")

print(f"Analyzed {len(result_files)} result files.")
display(instances.groupby(["lambda", "status"]).size().rename("instances").reset_index())

# %% [markdown]
# ## 2. Build paired per-bug comparisons
#
# Negative `exam_delta` means GraphCut requires less inspection effort than its
# corresponding baseline. Every comparison is paired within the same bug.

# %%
def build_paired_detail(metric_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    traceable = metric_frame[metric_frame["matched_ground_truth_methods"] > 0]
    keys = ["project", "bug_id", "lambda", "status", "total_methods"]
    for baseline in BASELINES:
        base = traceable[traceable["metric"] == baseline]
        graphcut = traceable[traceable["metric"] == f"{baseline}_gc"]
        merged = base.merge(graphcut, on=keys, suffixes=("_baseline", "_gc"))
        for _, row in merged.iterrows():
            base_exam = row["first_fault_exam_baseline"]
            gc_exam = row["first_fault_exam_gc"]
            rows.append({
                "project": row["project"],
                "bug_id": row["bug_id"],
                "bug": f"{row['project']}_{row['bug_id']}",
                "lambda": row["lambda"],
                "status": row["status"],
                "total_methods": row["total_methods"],
                "baseline": baseline,
                "baseline_label": BASELINE_LABELS[baseline],
                "baseline_exam": base_exam,
                "gc_exam": gc_exam,
                "exam_delta": gc_exam - base_exam,
                "relative_exam_change_pct": (
                    100.0 * (gc_exam - base_exam) / base_exam if base_exam > 0 else np.nan
                ),
                "baseline_rank": row["first_fault_expected_rank_baseline"],
                "gc_rank": row["first_fault_expected_rank_gc"],
            })
    return pd.DataFrame(rows)


paired_detail = build_paired_detail(metrics)
paired_detail.to_csv(TABLE_DIR / "paired_per_bug.csv", index=False)

selected_pairs = paired_detail[np.isclose(paired_detail["lambda"], SELECTED_LAMBDA)]
print(f"Paired comparisons at lambda={SELECTED_LAMBDA:g}")
display(
    selected_pairs.groupby("baseline_label").agg(
        bugs=("bug", "nunique"),
        wins=("exam_delta", lambda s: int((s < -1e-12).sum())),
        ties=("exam_delta", lambda s: int(np.isclose(s, 0.0, atol=1e-12).sum())),
        losses=("exam_delta", lambda s: int((s > 1e-12).sum())),
        mean_delta=("exam_delta", "mean"),
        median_delta=("exam_delta", "median"),
    )
)

# %%
def save_figure(name: str) -> None:
    plt.tight_layout()
    for extension in ("png", "svg"):
        plt.savefig(FIGURE_DIR / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close()


def annotate_bars(axis: plt.Axes, fmt: str = "{:.0f}") -> None:
    for container in axis.containers:
        axis.bar_label(container, fmt=fmt, padding=3, fontsize=10)

# %% [markdown]
# ## 3. Dataset validity: which bugs are actually evaluable?
#
# A ranking metric is undefined when the curated fault is absent. Those bugs
# must be counted and reported, not silently assigned a worst rank.

# %%
unique_instances = instances.sort_values("lambda").drop_duplicates(["project", "bug_id"])
status_counts = (
    unique_instances.assign(status_label=lambda d: d["status"].map(STATUS_LABELS))
    .groupby("status_label").size().sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))
axis = sns.barplot(x=status_counts.index, y=status_counts.values, color="#4C78A8")
annotate_bars(axis)
axis.set(title="Ground-truth traceability of evaluated bugs", xlabel="", ylabel="Bugs")
axis.tick_params(axis="x", rotation=20)
save_figure("01_traceability_counts")

display(status_counts.rename("bugs").to_frame())

# %% [markdown]
# ## 4. Absolute performance across lambda
#
# Lower EXAM is better. Confidence intervals are bug-level bootstrap intervals.
# A useful graph model should improve its paired baseline across multiple SBFL
# metrics, rather than only under one favorable metric.

# %%
primary = aggregate[aggregate["cohort"] == "fully_traceable"].copy()
available_lambdas = sorted(primary["lambda"].unique())

figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
for axis, baseline in zip(axes, BASELINES):
    for metric, label, color, marker in (
        (baseline, BASELINE_LABELS[baseline], "#6B7280", "o"),
        (f"{baseline}_gc", f"{BASELINE_LABELS[baseline]} + GraphCut", "#D62728", "s"),
    ):
        subset = primary[primary["metric"] == metric].sort_values("lambda")
        if subset.empty:
            continue
        lower = subset["mean_exam"] - subset["mean_exam_ci95_low"]
        upper = subset["mean_exam_ci95_high"] - subset["mean_exam"]
        axis.errorbar(
            subset["lambda"].astype(str), subset["mean_exam"],
            yerr=np.vstack([lower.clip(lower=0), upper.clip(lower=0)]),
            label=label, color=color, marker=marker, linewidth=2, capsize=4,
        )
    axis.set_title(BASELINE_LABELS[baseline])
    axis.set_xlabel("λ")
    axis.legend(fontsize=9)
axes[0].set_ylabel("Mean first-fault EXAM (lower is better)")
figure.suptitle("Baseline versus GraphCut across λ", y=1.03)
save_figure("02_mean_exam_by_lambda")

# %% [markdown]
# ## 5. Relative merit: paired improvement heatmap
#
# This plot isolates GraphCut's contribution. Negative percentages are
# improvements; positive percentages mean GraphCut made inspection more costly.

# %%
relative_change = (
    paired_detail.groupby(["baseline_label", "lambda"])["relative_exam_change_pct"]
    .mean().unstack("lambda")
    .reindex([BASELINE_LABELS[b] for b in BASELINES])
)

plt.figure(figsize=(9, 4.5))
axis = sns.heatmap(
    relative_change, annot=True, fmt=".1f", center=0, cmap="RdYlGn_r",
    cbar_kws={"label": "Mean paired EXAM change (%)"},
)
axis.set(title="GraphCut change relative to baseline", xlabel="λ", ylabel="")
axis.tick_params(axis="y", rotation=0)
save_figure("03_relative_exam_change_heatmap")

display(relative_change)

# %% [markdown]
# ## 6. Per-bug behavior at the selected lambda
#
# Aggregate means can hide regressions. Dumbbell plots show whether improvement
# is consistent or driven by a small number of bugs.

# %%
selected_bugs = sorted(selected_pairs["bug"].unique())
figure, axes = plt.subplots(1, 3, figsize=(19, max(6, 0.55 * len(selected_bugs))), sharey=True)
for axis, baseline in zip(axes, BASELINES):
    subset = (
        selected_pairs[selected_pairs["baseline"] == baseline]
        .set_index("bug").reindex(selected_bugs).dropna(subset=["baseline_exam", "gc_exam"])
    )
    y = np.arange(len(subset))
    for pos, (_, row) in enumerate(subset.iterrows()):
        color = "#2CA02C" if row.gc_exam < row.baseline_exam else (
            "#D62728" if row.gc_exam > row.baseline_exam else "#9CA3AF"
        )
        axis.plot([row.baseline_exam, row.gc_exam], [pos, pos], color=color, linewidth=2)
    axis.scatter(subset["baseline_exam"], y, color="#4C78A8", label="Baseline", zorder=3)
    axis.scatter(subset["gc_exam"], y, color="#F58518", marker="s", label="GraphCut", zorder=3)
    axis.set(title=BASELINE_LABELS[baseline], xlabel="First-fault EXAM")
    axis.set_yticks(y, subset.index)
    axis.invert_yaxis()
    axis.legend(fontsize=9)
axes[0].set_ylabel("Bug")
figure.suptitle(f"Per-bug effect at λ={SELECTED_LAMBDA:g}", y=1.01)
save_figure("04_per_bug_dumbbell")

# %% [markdown]
# ## 7. Top-k localization rates
#
# Expected tie ranks are used consistently. This avoids optimistic Top-k scores
# caused by assigning every member of a tie the minimum rank.

# %%
selected_aggregate = primary[np.isclose(primary["lambda"], SELECTED_LAMBDA)]
top_rows = []
for baseline in BASELINES:
    for metric, variant in ((baseline, "Baseline"), (f"{baseline}_gc", "GraphCut")):
        row = selected_aggregate[selected_aggregate["metric"] == metric]
        if row.empty:
            continue
        for k in TOP_K:
            top_rows.append({
                "baseline": BASELINE_LABELS[baseline], "variant": variant,
                "k": f"Top-{k}", "rate": float(row.iloc[0][f"top_{k}_rate"]),
            })
top_frame = pd.DataFrame(top_rows)

figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for axis, baseline in zip(axes, [BASELINE_LABELS[b] for b in BASELINES]):
    sns.barplot(
        data=top_frame[top_frame["baseline"] == baseline],
        x="k", y="rate", hue="variant", palette=["#4C78A8", "#F58518"], ax=axis,
    )
    axis.set(title=baseline, xlabel="", ylabel="Localization rate" if axis is axes[0] else "")
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=9)
figure.suptitle(f"Top-k rates at λ={SELECTED_LAMBDA:g}", y=1.03)
save_figure("05_top_k_rates")

# %% [markdown]
# ## 8. λ=0 sanity check
#
# With no pairwise term, GraphCut should preserve baseline ordering. Any changed
# expected ranks indicate numerical tie-breaking or normalization artifacts,
# not graph-derived information.

# %%
if not lambda_zero.empty:
    zero_summary = (
        lambda_zero.groupby("metric")[["unequal_expected_ranks", "methods"]].sum()
        .assign(unequal_fraction=lambda d: d["unequal_expected_ranks"] / d["methods"])
        .reset_index()
    )
    zero_summary["metric"] = zero_summary["metric"].map(BASELINE_LABELS)
    plt.figure(figsize=(8, 5))
    axis = sns.barplot(data=zero_summary, x="metric", y="unequal_fraction", color="#E45756")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f%%", labels=[f"{v * 100:.1f}%" for v in container.datavalues], padding=3, fontsize=11)
    axis.set(
        title="λ=0 ranking-invariance violations",
        xlabel="", ylabel="Changed-rank fraction", ylim=(0, 1),
    )
    save_figure("06_lambda_zero_invariance")
    display(zero_summary)

# %% [markdown]
# ## 9. Does graph size predict regressions?
#
# A relationship between graph size and EXAM degradation would indicate that
# smoothing strength is not calibrated consistently across instances.

# %%
plt.figure(figsize=(10, 6))
axis = sns.scatterplot(
    data=selected_pairs, x="total_methods", y="exam_delta", hue="baseline_label",
    style="baseline_label", s=100,
)
axis.axhline(0, color="black", linestyle="--", linewidth=1)
axis.set_xscale("log")
axis.set(
    title=f"Graph size versus GraphCut effect at λ={SELECTED_LAMBDA:g}",
    xlabel="Ranked methods (log scale)",
    ylabel="EXAM delta (GraphCut − baseline)",
)
save_figure("07_graph_size_vs_exam_delta")

# %% [markdown]
# ## 10. Systematic interpretation
#
# Evidence for merit requires paired improvements, effect sizes, confidence
# intervals, and robustness across baselines. Evidence of shortcomings includes
# traceability exclusions, λ sensitivity, λ=0 invariance failures, and results
# dominated by one project or a few bugs.

# %%
fully_traceable = unique_instances[unique_instances["status"] == "fully_traceable"]
absent = unique_instances[unique_instances["status"] == "fault_not_in_results"]
project_counts = unique_instances.groupby("project").size().to_dict()

selected_stats = paired_summary[
    (paired_summary["cohort"] == "fully_traceable")
    & np.isclose(paired_summary["lambda"], SELECTED_LAMBDA)
].copy()

finding_lines = [
    "# Systematic benchmark findings",
    "",
    f"- Result files analyzed: **{len(result_files)}**.",
    f"- Distinct bugs evaluated: **{len(unique_instances)}** ({project_counts}).",
    f"- Fully traceable bugs: **{len(fully_traceable)}**; curated fault absent: **{len(absent)}**.",
    f"- Primary paired analysis therefore uses **{len(fully_traceable)}** bugs.",
    "",
    f"## Paired results at λ={SELECTED_LAMBDA:g}",
    "",
]
for row in selected_stats.itertuples(index=False):
    direction = "improvement" if row.mean_exam_delta_gc_minus_baseline < 0 else "degradation"
    finding_lines.append(
        f"- {BASELINE_LABELS[row.baseline]}: {row.gc_wins} wins, {row.ties} ties, "
        f"{row.gc_losses} losses; mean EXAM delta {row.mean_exam_delta_gc_minus_baseline:+.4f} "
        f"({direction}); Holm-adjusted p={row.wilcoxon_p_holm:.4g}."
    )

if not lambda_zero.empty:
    violating_files = int((~lambda_zero["lambda_zero_invariant"]).sum())
    finding_lines.extend([
        "",
        "## Shortcomings and validity threats",
        "",
        f"- λ=0 failed exact ranking invariance for **{violating_files}/{len(lambda_zero)}** "
        "bug/metric checks. λ=0 changes must not be interpreted as graph merit.",
    ])
else:
    finding_lines.extend(["", "## Shortcomings and validity threats", ""])

finding_lines.extend([
    f"- **{len(absent)}** evaluated bugs cannot be scored because their curated fault is absent.",
    f"- The current sample covers only **{len(project_counts)}** project(s), so project-level "
    "generalization cannot be claimed.",
    "- Argument-free ground truth may match overloaded methods; ambiguity counts are available "
    "in `instance_coverage.csv`.",
    "- Large changes across λ indicate sensitivity to graph size/frequency calibration.",
    "- Statistical power is low for small cohorts; confidence intervals and effect sizes matter "
    "more than isolated mean improvements.",
    "",
    "## What would establish merit",
    "",
    "- Regenerate call graphs after fixing multi-trigger graph overwriting.",
    "- Evaluate all fully traceable bugs and report macro averages by project.",
    "- Tune λ on held-out projects, then evaluate once on untouched projects.",
    "- Include failing-trace-filtered SBFL and shuffled/unweighted graph controls to isolate "
    "the contribution of graph structure and edge frequency.",
])

findings_path = OUTPUT_DIR / "systematic_findings.md"
findings_path.write_text("\n".join(finding_lines) + "\n")
print(findings_path.read_text())

# %% [markdown]
# ## 11. Output inventory

# %%
inventory = sorted(path.relative_to(OUTPUT_DIR) for path in OUTPUT_DIR.rglob("*") if path.is_file())
print("\n".join(map(str, inventory)))
