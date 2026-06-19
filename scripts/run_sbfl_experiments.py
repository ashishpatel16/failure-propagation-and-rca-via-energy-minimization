import os
import sys
import shutil
from pathlib import Path
import logging
import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(os.getcwd())
sys.path.insert(0, str(ROOT_DIR / "src"))

from graph_generation.sbfl.extractor import SBFLGraphExtractor
from visualization.graph_plots import plot_json_graph
from cut_algorithms.boykov_jolly import BoykovJollyCut
from cut_algorithms.data_parsers import read_call_graph


def get_best_rank(df: pd.DataFrame, score_col: str) -> int:
    sorted_df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    sorted_df['Rank'] = sorted_df[score_col].rank(method='min', ascending=False)
    buggy_ranks = sorted_df[sorted_df['Is_Buggy']]['Rank']
    return int(buggy_ranks.min())


def main() -> None:
    load_dotenv()
    
    JAVA_HOME = os.environ["JAVA_HOME_PATH"]
    D4J_PATH = os.environ["D4J_PATH"]
    
    if not Path(JAVA_HOME).exists():
        raise FileNotFoundError(f"JAVA_HOME path not found: {JAVA_HOME}")
        
    if not Path(D4J_PATH).exists():
        raise FileNotFoundError(f"Defects4J installation not found at {D4J_PATH}")
        
    extractor = SBFLGraphExtractor(
        d4j_path=D4J_PATH,
        java_home=JAVA_HOME
    )
    
    targets = [
        ("Lang", "6", "org.apache.commons"), 
    ]
    
    results = []
    
    for project, bug_id, package_prefix in targets:
        output_dir = ROOT_DIR / "data" / "defects4j" / f"{project}_{bug_id}"
        
        extraction_res = extractor.extract(project, bug_id, str(output_dir), package_prefix)
        
        plot_title = f"{project}-{bug_id} Call Graph"
        logging.info(f"Visualizing {plot_title}...")
        plot_json_graph(extraction_res.graph_json, plot_title, None)
            
        sbfl_df = pd.DataFrame(extraction_res.sbfl_metrics)
        
        tarantula_scores = dict(zip(sbfl_df["Method"], sbfl_df["Tarantula"]))
        
        G = read_call_graph(extraction_res.graph_json)
        logging.info(f"Computing Graph Cuts rankings for {project}-{bug_id}...")
        cut_algo = BoykovJollyCut(G, tarantula_scores, 1.0)
        
        cut_scores = []
        for node in cut_algo.nodes:
            e0, e1 = cut_algo.compute_min_marginals(node)
            confidence = e0 - e1 # Higher is more buggy
            cut_scores.append({"Method": node, "GraphCut_Score": confidence})
            
        cut_df = pd.DataFrame(cut_scores)
        
        merged_df = pd.merge(sbfl_df, cut_df, on="Method", how="left")
        
        buggy_set = set(extraction_res.buggy_methods)
        merged_df["Is_Buggy"] = merged_df["Method"].apply(lambda m: m.split('(')[0] in buggy_set)
        
        rankings_path = output_dir / "rankings.csv"
        merged_df.to_csv(rankings_path, index=False)
        logging.info(f"Saved rankings to {rankings_path}")
        
        best_ochiai = get_best_rank(merged_df, "Ochiai")
        best_tarantula = get_best_rank(merged_df, "Tarantula")
        best_graphcut = get_best_rank(merged_df, "GraphCut_Score")

        results.append({
            "Project": project,
            "BugId": bug_id,
            "Status": "Success",
            "Buggy_Methods": len(extraction_res.buggy_methods),
            "Best_Ochiai": best_ochiai,
            "Best_Tarantula": best_tarantula,
            "Best_GraphCut": best_graphcut
        })
        
        work_dir = output_dir / "workspace"
        if work_dir.exists():
            logging.info(f"Cleaning up workspace: {work_dir}")
            shutil.rmtree(work_dir)
            
        gzoltar_out_dir = output_dir / "gzoltar-out"
        if gzoltar_out_dir.exists():
            logging.info(f"Cleaning up raw gzoltar output: {gzoltar_out_dir}")
            shutil.rmtree(gzoltar_out_dir)
            
    print("\n" + "="*80)
    print(f"{'D4J BATCH EXTRACTION SUMMARY':^80}")
    print("="*80)
    if results:
        print(pd.DataFrame(results).to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
