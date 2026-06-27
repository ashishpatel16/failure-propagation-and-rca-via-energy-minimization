import argparse
import json
import logging
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "defects4j"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def json_to_digraph(graph_json: Path) -> nx.DiGraph:
    with open(graph_json, 'r') as f:
        graph_data = json.load(f)

    G = nx.DiGraph()
    for edge in graph_data.get('edges', []):
        caller = edge['caller'].replace('$', '.')
        callee = edge['callee'].replace('$', '.')
        G.add_edge(caller, callee, weight=float(edge['frequency']))
    
    for node in graph_data.get('nodes', []):
        n_clean = node.replace('$', '.')
        if n_clean not in G:
            G.add_node(n_clean)
    return G

def load_ground_truth(gt_path: Path) -> dict:
    df = pd.read_csv(gt_path)
    gt_map = {}
    for _, row in df.iterrows():
        try:
            nodes = json.loads(row["buggy_nodes"])
            # Normalize to match the evaluation (replace $ with . and strip)
            nodes = [n.replace('$', '.').strip() for n in nodes]
            gt_map[(row["project"], int(row["bug_id"]))] = nodes
        except Exception:
            continue
    return gt_map

def main():
    parser = argparse.ArgumentParser(description="Analyze correlation between graph metrics and EXAM delta.")
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "evals_lambd0_1")
    parser.add_argument("--ground-truth", type=Path, default=ROOT_DIR / "ground_truth.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "evals_lambd0_1" / "systematic_analysis")
    args = parser.parse_args()

    if not args.results_dir.exists():
        logger.error(f"Results dir {args.results_dir} not found.")
        sys.exit(1)
        
    gt_map = load_ground_truth(args.ground_truth)
    logger.info(f"Loaded ground truth for {len(gt_map)} instances.")

    records = []
    
    csv_files = list(args.results_dir.glob("eval_lambd_*/*.csv"))
    logger.info(f"Found {len(csv_files)} evaluation files. Computing metrics...")
    
    baselines = ["tarantula", "ochiai", "dstar"]
    
    # Track unique bugs processed to compute graph metrics once per bug
    bug_metrics_cache = {}

    for csv_path in tqdm(csv_files):
        # Format: Project_BugId_Lambda.csv
        parts = csv_path.stem.split('_')
        if len(parts) != 3:
            continue
        project = parts[0]
        bug_id = int(parts[1])
        lambd = float(parts[2])
        
        bug_key = f"{project}_{bug_id}"
        gt_nodes = gt_map.get((project, bug_id), [])
        if not gt_nodes:
            continue
            
        df = pd.read_csv(csv_path)
        if "Method" not in df.columns:
            continue
            
        # Normalize methods to match nodes
        df["normalized_method"] = df["Method"].apply(lambda m: str(m).replace('$', '.').strip())
        
        fault_rows = df[df["normalized_method"].isin(gt_nodes)]
        if fault_rows.empty:
            continue # Fault not found in ranking
            
        # Graph metrics calculation (only once per bug)
        if bug_key not in bug_metrics_cache:
            bug_dir = DATA_DIR / bug_key
            graph_json = bug_dir / "call_graph.json"
            if not graph_json.exists():
                continue
                
            G = json_to_digraph(graph_json)
            density = nx.density(G)
            
            try:
                assortativity = nx.degree_assortativity_coefficient(G)
                if np.isnan(assortativity):
                    assortativity = 0.0
            except Exception:
                assortativity = 0.0
                
            in_degrees = []
            out_degrees = []
            for node in gt_nodes:
                if node in G:
                    in_degrees.append(G.in_degree(node))
                    out_degrees.append(G.out_degree(node))
                    
            avg_in = np.mean(in_degrees) if in_degrees else 0.0
            avg_out = np.mean(out_degrees) if out_degrees else 0.0
            
            bug_metrics_cache[bug_key] = {
                "density": density,
                "assortativity": assortativity,
                "buggy_in_degree": avg_in,
                "buggy_out_degree": avg_out,
                "total_nodes": len(G)
            }
            
        g_metrics = bug_metrics_cache[bug_key]
        
        # Extract exam scores for each baseline
        for base in baselines:
            exam_col = f"{base}_exam_score"
            gc_exam_col = f"{base}_gc_exam_score"
            rank_col = f"{base}_rank"
            gc_rank_col = f"{base}_gc_rank"
            
            if exam_col in fault_rows.columns and gc_exam_col in fault_rows.columns:
                base_exam = float(fault_rows[exam_col].min())
                gc_exam = float(fault_rows[gc_exam_col].min())
                exam_delta = gc_exam - base_exam
                
                # Fetch ranks if available
                base_rank = float(fault_rows[rank_col].min()) if rank_col in fault_rows.columns else np.nan
                gc_rank = float(fault_rows[gc_rank_col].min()) if gc_rank_col in fault_rows.columns else np.nan
                
                records.append({
                    "project": project,
                    "bug_id": bug_id,
                    "bug": bug_key,
                    "lambda": lambd,
                    "baseline_label": base.capitalize(),
                    "base_exam": base_exam,
                    "gc_exam": gc_exam,
                    "exam_delta": exam_delta,
                    "base_rank": base_rank,
                    "gc_rank": gc_rank,
                    **g_metrics
                })

    if not records:
        logger.error("No valid traceable records found to analyze.")
        sys.exit(1)
        
    merged = pd.DataFrame(records)
    
    out_dir = args.output_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the dataset
    (args.output_dir / "tables").mkdir(exist_ok=True)
    merged.to_csv(args.output_dir / "tables" / "graph_metrics_correlation.csv", index=False)
    
    # Analyze correlations
    logger.info("\n--- Correlations with EXAM Delta ---")
    logger.info("(Negative EXAM delta means Graph Cut improved the score)")
    
    metrics_to_plot = ["density", "assortativity", "buggy_in_degree", "buggy_out_degree"]
    
    for baseline in merged["baseline_label"].unique():
        for lambd in sorted(merged["lambda"].unique()):
            subset = merged[(merged["baseline_label"] == baseline) & (merged["lambda"] == lambd)]
            if len(subset) < 2:
                continue
                
            logger.info(f"\nBaseline: {baseline} | Lambda: {lambd}")
            for m in metrics_to_plot:
                spearman, p_s = spearmanr(subset[m], subset["exam_delta"])
                pearson, p_p = pearsonr(subset[m], subset["exam_delta"])
                logger.info(f"  {m:16s} -> Spearman: {spearman:+.4f} (p={p_s:.4f}) | Pearson: {pearson:+.4f} (p={p_p:.4f})")
                
    # Plotting
    sns.set_theme(style="whitegrid", context="talk")
    for m in metrics_to_plot:
        logger.info(f"Plotting {m}...")
        
        g = sns.lmplot(
            data=merged,
            x=m,
            y="exam_delta",
            hue="lambda",
            col="baseline_label",
            height=5,
            aspect=1.0,
            palette="Set1",
            scatter_kws={'alpha': 0.6},
            sharex=False
        )
        
        # Add horizontal line at 0 (no improvement)
        for ax in g.axes.flatten():
            ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            
        g.set_axis_labels(m.replace('_', ' ').title(), "EXAM Delta (GraphCut - Baseline)")
        g.fig.suptitle(f"Effect of {m.replace('_', ' ').title()} on Graph Cut Performance", y=1.05)
                
        g.savefig(out_dir / f"correlation_{m}.png", bbox_inches="tight", dpi=200)
        plt.close(g.fig)

    # Lambda Sweep Plots
    logger.info("Generating Lambda Sweep plots...")
    
    # We want to plot Lambda on X-axis, and (Base EXAM, GC EXAM) on Y-axis
    melted_exam = merged.melt(
        id_vars=["bug", "lambda", "baseline_label"],
        value_vars=["base_exam", "gc_exam"],
        var_name="Variant",
        value_name="EXAM Score"
    )
    melted_exam["Variant"] = melted_exam["Variant"].map({"base_exam": "Baseline", "gc_exam": "Graph Cut"})
    
    g_exam = sns.catplot(
        data=melted_exam,
        x="lambda",
        y="EXAM Score",
        hue="Variant",
        col="baseline_label",
        kind="point",
        height=5,
        aspect=1.0,
        palette=["#e74c3c", "#2ecc71"], # Red for Baseline, Green for GC
        sharey=False
    )
    g_exam.fig.suptitle("Impact of Lambda on Mean EXAM Score", y=1.05)
    g_exam.savefig(out_dir / "lambda_sweep_exam.png", bbox_inches="tight", dpi=200)
    plt.close(g_exam.fig)

    melted_rank = merged.melt(
        id_vars=["bug", "lambda", "baseline_label"],
        value_vars=["base_rank", "gc_rank"],
        var_name="Variant",
        value_name="Rank"
    )
    melted_rank["Variant"] = melted_rank["Variant"].map({"base_rank": "Baseline", "gc_rank": "Graph Cut"})
    
    g_rank = sns.catplot(
        data=melted_rank,
        x="lambda",
        y="Rank",
        hue="Variant",
        col="baseline_label",
        kind="point",
        height=5,
        aspect=1.0,
        palette=["#e74c3c", "#2ecc71"],
        sharey=False
    )
    # Log scale for ranks often makes more sense
    for ax in g_rank.axes.flatten():
        ax.set_yscale("log")
        
    g_rank.fig.suptitle("Impact of Lambda on Mean Buggy Node Rank (Log Scale)", y=1.05)
    g_rank.savefig(out_dir / "lambda_sweep_rank.png", bbox_inches="tight", dpi=200)
    plt.close(g_rank.fig)

    # Individual instance trajectory (exam_delta)
    logger.info("Generating Individual Instance Sweep plots...")
    g_delta = sns.relplot(
        data=merged,
        x="lambda",
        y="exam_delta",
        hue="bug",
        col="baseline_label",
        kind="line",
        estimator=None,
        units="bug",
        height=5,
        aspect=1.0,
        alpha=0.2,
        legend=False
    )
    # Add horizontal line at 0
    for ax in g_delta.axes.flatten():
        ax.axhline(0, color="black", linestyle="--", linewidth=1.5, zorder=10)
        
    g_delta.fig.suptitle("Impact of Lambda on EXAM Delta (Individual Instances)", y=1.05)
    g_delta.savefig(out_dir / "lambda_sweep_individual_delta.png", bbox_inches="tight", dpi=200)
    plt.close(g_delta.fig)

    # Individual instance trajectory for Rank
    g_ind_rank = sns.relplot(
        data=merged,
        x="lambda",
        y="gc_rank",
        hue="bug",
        col="baseline_label",
        kind="line",
        estimator=None,
        units="bug",
        height=5,
        aspect=1.0,
        alpha=0.2,
        legend=False
    )
    for ax in g_ind_rank.axes.flatten():
        ax.set_yscale("log")
    g_ind_rank.fig.suptitle("Impact of Lambda on GC Rank (Individual Instances)", y=1.05)
    g_ind_rank.savefig(out_dir / "lambda_sweep_individual_rank.png", bbox_inches="tight", dpi=200)
    plt.close(g_ind_rank.fig)

    logger.info("Done! All plots saved to figures directory.")

if __name__ == "__main__":
    main()
