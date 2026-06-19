import os
import sys
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(os.getcwd())
sys.path.insert(0, str(ROOT_DIR / "src"))

from benchmarking.sbfl_metrics import compute_sbfl_scores_from_csv
from visualization.graph_plots import plot_json_graph
from cut_algorithms.boykov_jolly import BoykovJollyCut
from cut_algorithms.data_parsers import read_call_graph

def get_best_rank(df: pd.DataFrame, score_col: str) -> int:
    sorted_df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    sorted_df['Rank'] = sorted_df[score_col].rank(method='min', ascending=False)
    buggy_ranks = sorted_df[sorted_df['Is_Buggy']]['Rank']
    return int(buggy_ranks.min())

def main() -> None:
    folder_path = ROOT_DIR / "data" / "defects4j" / "Lang_1"
    if not folder_path.exists() or not folder_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {folder_path}")

    call_graph_json = folder_path / "call_graph.json"
    coverage_csv = folder_path / "coverage.csv"
    buggy_txt = folder_path / "buggy_methods.txt"

    if not call_graph_json.exists():
        raise FileNotFoundError(f"Missing file: {call_graph_json}")
    if not coverage_csv.exists():
        raise FileNotFoundError(f"Missing file: {coverage_csv}")
    if not buggy_txt.exists():
        raise FileNotFoundError(f"Missing file: {buggy_txt}")

    logging.info(f"Processing folder: {folder_path.name}")

    # Read buggy methods
    with open(buggy_txt, 'r') as f:
        buggy_set = set([line.strip() for line in f if line.strip()])

    # Compute SBFL scores
    logging.info("Computing SBFL scores from coverage.csv...")
    sbfl_df = compute_sbfl_scores_from_csv(str(coverage_csv))
    
    tarantula_scores = dict(zip(sbfl_df["Method"], sbfl_df["Tarantula"]))

    # Plot JSON Graph
    plot_title = f"{folder_path.name} Call Graph"
    save_prefix = str(folder_path / "call_graph_pregenerated")
    logging.info(f"Visualizing {plot_title}...")
    plot_json_graph(str(call_graph_json), plot_title, save_prefix)

    # Compute Graph Cuts
    logging.info("Loading graph and computing cuts...")
    G = read_call_graph(str(call_graph_json))
    cut_algo = BoykovJollyCut(G, tarantula_scores, 1.0)
    
    cut_scores = []
    for node in cut_algo.nodes:
        e0, e1 = cut_algo.compute_min_marginals(node)
        confidence = e0 - e1 # Higher is more buggy
        cut_scores.append({"Method": node, "GraphCut_Score": confidence})
        
    cut_df = pd.DataFrame(cut_scores)
    
    # Merge
    merged_df = pd.merge(sbfl_df, cut_df, on="Method", how="left")
    merged_df["Is_Buggy"] = merged_df["Method"].apply(lambda m: m.split('(')[0] in buggy_set)
    
    # Save detailed rankings
    rankings_path = folder_path / "rankings_pregenerated.csv"
    merged_df.to_csv(rankings_path, index=False)
    logging.info(f"Saved rankings to {rankings_path}")
    
    best_ochiai = get_best_rank(merged_df, "Ochiai")
    best_tarantula = get_best_rank(merged_df, "Tarantula")
    best_graphcut = get_best_rank(merged_df, "GraphCut_Score")

    print("\n" + "="*80)
    print(f"{'PREGENERATED FOLDER RESULTS':^80}")
    print("="*80)
    print(f"Folder:         {folder_path.name}")
    print(f"Buggy Methods:  {len(buggy_set)}")
    print(f"Best Ochiai:    {best_ochiai}")
    print(f"Best Tarantula: {best_tarantula}")
    print(f"Best GraphCut:  {best_graphcut}")
    print("="*80)

if __name__ == "__main__":
    main()
