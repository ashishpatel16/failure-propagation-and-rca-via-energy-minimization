import os
import sys
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(os.getcwd())
sys.path.insert(0, str(ROOT_DIR / "src"))

from cut_algorithms.data_parsers import read_call_graph
from cut_algorithms.sampling_cuts import SamplingCuts

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

    with open(buggy_txt, 'r') as f:
        buggy_set = set([line.strip() for line in f if line.strip()])

    logging.info("Loading graph...")
    G = read_call_graph(str(call_graph_json))
    
    logging.info("Reading coverage data...")
    coverage_df = pd.read_csv(coverage_csv)

    logging.info("Running Sampling Cuts (Two-way perturbation)...")
    sampling_cuts = SamplingCuts(coverage_df, G, lambd=1.0, num_iterations=100)
    cut_df = sampling_cuts.run()
    
    # Read existing rankings to merge and compare
    rankings_pregenerated = folder_path / "rankings_pregenerated.csv"
    if rankings_pregenerated.exists():
        merged_df = pd.read_csv(rankings_pregenerated)
        merged_df = pd.merge(merged_df, cut_df, on="Method", how="left")
    else:
        merged_df = cut_df
        merged_df["Is_Buggy"] = merged_df["Method"].apply(lambda m: m.split('(')[0] in buggy_set)
    
    rankings_path = folder_path / "rankings_sampled.csv"
    merged_df.to_csv(rankings_path, index=False)
    logging.info(f"Saved sampled rankings to {rankings_path}")
    
    if "Ochiai" in merged_df.columns:
        best_ochiai = get_best_rank(merged_df, "Ochiai")
        best_tarantula = get_best_rank(merged_df, "Tarantula")
    else:
        best_ochiai, best_tarantula = -1, -1
        
    best_expected_min = get_best_rank(merged_df, "Expected_MinMarginal")
    best_selection_freq = get_best_rank(merged_df, "Selection_Frequency")

    print("\n" + "="*80)
    print(f"{'SAMPLING CUTS RESULTS':^80}")
    print("="*80)
    print(f"Folder:                    {folder_path.name}")
    print(f"Buggy Methods:             {len(buggy_set)}")
    print(f"Best Ochiai:               {best_ochiai}")
    print(f"Best Tarantula:            {best_tarantula}")
    print(f"Best Expected MinMarginal: {best_expected_min}")
    print(f"Best Selection Frequency:  {best_selection_freq}")
    print("="*80)

if __name__ == "__main__":
    main()
