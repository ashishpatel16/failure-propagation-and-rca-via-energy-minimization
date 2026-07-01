# %%
# Setup & Function Definitions
import os
import sys
from pathlib import Path
import logging
import pandas as pd
from dotenv import load_dotenv
import networkx as nx
from typing import Tuple, Any, List, Dict
from IPython.display import IFrame, display, HTML

# Reduce logging noise
logging.getLogger().setLevel(logging.WARNING)

ROOT_DIR = Path(os.getcwd())
if ROOT_DIR.name == "notebooks":
    ROOT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from graph_generation.sbfl.extractor import ExtractionResult
from visualization.graph_plots import plot_json_graph
from cut_algorithms.boykov_jolly import BoykovJollyCut
from cut_algorithms.data_parsers import read_call_graph

def load_pregen_data(project: str, bug_id: str, quiet: bool = False) -> Tuple[ExtractionResult, Path]:
    """Loads pre-generated extraction data instead of running D4J in real-time."""
    data_dir = ROOT_DIR / "data" / "defects4j" / f"{project}_{bug_id}"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Pre-generated data not found at {data_dir}")
        
    metrics_csv = data_dir / "sbfl_metrics.csv"
    graph_json = data_dir / "call_graph.json"
    buggy_txt = data_dir / "buggy_methods.txt"
    
    sbfl_metrics = pd.read_csv(metrics_csv).to_dict(orient="records")
    
    with open(buggy_txt, 'r') as f:
        buggy_methods = [line.strip() for line in f if line.strip()]
        
    res = ExtractionResult(
        graph_json=str(graph_json),
        sbfl_metrics=sbfl_metrics,
        buggy_methods=buggy_methods
    )
    
    if not quiet:
        plot_title = f"{project}-{bug_id} Call Graph"
        plot_json_graph(res.graph_json, plot_title, save_prefix=None, buggy_methods=res.buggy_methods)
    
    return res, data_dir

def analyze_baseline_sbfl(extraction_res: ExtractionResult, quiet: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(extraction_res.sbfl_metrics)
    buggy_set = set(extraction_res.buggy_methods)
    df["Is_Buggy"] = df["Method"].apply(lambda m: m.split('(')[0] in buggy_set)
    df['Ochiai_Rank'] = df['Ochiai'].rank(method='min', ascending=False)
    df['Tarantula_Rank'] = df['Tarantula'].rank(method='min', ascending=False)
    
    if not quiet:
        bug_df = df[df["Is_Buggy"]][["Method", "Tarantula", "Tarantula_Rank", "Ochiai", "Ochiai_Rank"]]
        display(HTML("<b>Baseline Bug Rankings:</b>"))
        display(bug_df)
    return df

def setup_energy_model(res: ExtractionResult, sbfl_df: pd.DataFrame, lambda_weight: float) -> Tuple[nx.DiGraph, BoykovJollyCut]:
    tarantula_scores = dict(zip(sbfl_df["Method"], sbfl_df["Tarantula"]))
    G = read_call_graph(res.graph_json)
    cut_algo = BoykovJollyCut(G, tarantula_scores, lambda_weight)
    return G, cut_algo

def compute_min_cut(G: nx.DiGraph, cut_algo: BoykovJollyCut, quiet: bool = False) -> None:
    min_energy, optimal_labeling = cut_algo.compute_min_cut()
    if not quiet:
        display(HTML(f"<b>Global Min Energy:</b> {min_energy:.2f}"))

def compute_marginal_scores(cut_algo: BoykovJollyCut, sbfl_df: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for node in cut_algo.nodes:
        e0, e1 = cut_algo.compute_min_marginals(node)
        scores.append({"Method": node, "GraphCut_Score": e0 - e1})
        
    merged_df = pd.merge(sbfl_df, pd.DataFrame(scores), on="Method", how="left")
    merged_df['GraphCut_Rank'] = merged_df['GraphCut_Score'].rank(method='min', ascending=False)
    return merged_df

def show_final_rankings(merged_df: pd.DataFrame) -> None:
    display(HTML("<b>Final Rankings vs Baseline:</b>"))
    display(merged_df[merged_df["Is_Buggy"]][["Method", "Tarantula_Rank", "Ochiai_Rank", "GraphCut_Rank"]])

def trace_subgraph(res: ExtractionResult, G: nx.DiGraph, merged_df: pd.DataFrame) -> None:
    for buggy_method in res.buggy_methods:
        if buggy_method in G:
            bug_row = merged_df[merged_df["Method"] == buggy_method].iloc[0]
            bug_gc = bug_row['GraphCut_Score']
            
            pulling_up = []
            pulling_down = []
            
            for n in G.successors(buggy_method):
                n_score = merged_df[merged_df["Method"] == n].iloc[0]['GraphCut_Score']
                if n_score > bug_gc:
                    pulling_up.append((n, n_score))
                else:
                    pulling_down.append((n, n_score))
                    
            print(f"Buggy Node: {buggy_method} (Score: {bug_gc:.4f})")
            print(f"  ↑ {len(pulling_up)} neighbors pulling score UP.")
            print(f"  ↓ {len(pulling_down)} neighbors pulling score DOWN.")

def show_top_results(df: pd.DataFrame, top_n: int = 5) -> None:
    display(HTML(f"<b>Top {top_n} by Ochiai:</b>"))
    display(df.sort_values("Ochiai", ascending=False).head(top_n)[["Method", "Ochiai", "Is_Buggy"]])
    
    display(HTML(f"<b>Top {top_n} by GraphCut:</b>"))
    display(df.sort_values("GraphCut_Score", ascending=False).head(top_n)[["Method", "GraphCut_Score", "Is_Buggy"]])


# %%
# 1. Load Data for Chart 1
project = "Chart"
bug_id = "1"
res_c1, out_dir_c1 = load_pregen_data(project, bug_id)

# %%
# 2. Display Baseline SBFL
sbfl_df_c1 = analyze_baseline_sbfl(res_c1)

# %%
# 3. Setup Energy Minimization Model
LAMBDA_C1 = 1.0
G_c1, cut_algo_c1 = setup_energy_model(res_c1, sbfl_df_c1, LAMBDA_C1)

# %%
# 4. Compute Min-Cut
compute_min_cut(G_c1, cut_algo_c1)

# %%
# 5. Compute Min-Marginal Confidence Scores
merged_df_c1 = compute_marginal_scores(cut_algo_c1, sbfl_df_c1)

# %%
# 6. Compare Final Rankings
show_final_rankings(merged_df_c1)

# %%
# 7. Trace Subgraph Context
trace_subgraph(res_c1, G_c1, merged_df_c1)

# %%
# 8. Show Top Predicted Methods
show_top_results(merged_df_c1)

# %%
# 9. Batch Evaluation Loop
def compute_metrics(df: pd.DataFrame, rank_col: str, prefix: str) -> Dict[str, Any]:
    total_methods = len(df)
    buggy_ranks = df[df["Is_Buggy"]][rank_col]
    if buggy_ranks.empty:
        return {f"{prefix}_buggy_rank": None, f"{prefix}_top1": False, f"{prefix}_top3": False, f"{prefix}_top5": False, f"{prefix}_exam_score": None}
    
    best_rank = int(buggy_ranks.min())
    return {
        f"{prefix}_buggy_rank": best_rank,
        f"{prefix}_top1": best_rank <= 1,
        f"{prefix}_top3": best_rank <= 3,
        f"{prefix}_top5": best_rank <= 5,
        f"{prefix}_exam_score": best_rank / total_methods
    }

results = []
defects4j_dir = ROOT_DIR / "data" / "defects4j"
chart_dirs = [d for d in defects4j_dir.iterdir() if d.is_dir() and d.name.startswith("Chart_")]

# Sort by bug id numerically
chart_dirs.sort(key=lambda d: int(d.name.split('_')[1]))

for d in chart_dirs:
    bug_id = d.name.split('_')[1]
    
    try:
        res, out_dir = load_pregen_data("Chart", bug_id, quiet=True)
        sbfl_df = analyze_baseline_sbfl(res, quiet=True)
        
        LAMBDA_W = 1.0
        G, cut_algo = setup_energy_model(res, sbfl_df, LAMBDA_W)
        compute_min_cut(G, cut_algo, quiet=True)
        merged_df = compute_marginal_scores(cut_algo, sbfl_df)
        
        row = {"Project": "Chart", "BugId": bug_id}
        row.update(compute_metrics(merged_df, "Tarantula_Rank", "Tarantula"))
        row.update(compute_metrics(merged_df, "Ochiai_Rank", "Ochiai"))
        row.update(compute_metrics(merged_df, "GraphCut_Rank", "GraphCut"))
        results.append(row)
    except Exception as e:
        # Some Chart bugs might not have extracted correctly or were omitted
        logging.warning(f"Failed to process Chart {bug_id}: {e}")

if results:
    results_df = pd.DataFrame(results)
    display(HTML("<b>Batch Results on Chart Dataset:</b>"))
    display(results_df)

    summary = {
        "Algorithm": ["Tarantula", "Ochiai", "GraphCut"],
        "Top-1": [results_df["Tarantula_top1"].sum(), results_df["Ochiai_top1"].sum(), results_df["GraphCut_top1"].sum()],
        "Top-3": [results_df["Tarantula_top3"].sum(), results_df["Ochiai_top3"].sum(), results_df["GraphCut_top3"].sum()],
        "Top-5": [results_df["Tarantula_top5"].sum(), results_df["Ochiai_top5"].sum(), results_df["GraphCut_top5"].sum()],
        "Mean EXAM": [results_df["Tarantula_exam_score"].mean(), results_df["Ochiai_exam_score"].mean(), results_df["GraphCut_exam_score"].mean()]
    }
    display(HTML("<b>Summary:</b>"))
    display(pd.DataFrame(summary))
else:
    print("No valid results found.")
