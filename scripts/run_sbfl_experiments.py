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
from visualization.energy_plots import plot_3d_energy_landscape


TARGETS: list[tuple[str, str, str]] = [
    # ("Lang", "6", "org.apache.commons"),
    ("Lang", "1", "org.apache.commons"),
    ("Lang", "3", "org.apache.commons"),
    ("Lang", "5", "org.apache.commons"),
    ("Lang", "7", "org.apache.commons"),
    ("Lang", "8", "org.apache.commons"),
    ("Lang", "10", "org.apache.commons"),
]


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
    
    results = []
    
    for project, bug_id, package_prefix in TARGETS:
        output_dir = ROOT_DIR / "data" / "defects4j" / f"{project}_{bug_id}"
        
        extraction_res = extractor.extract(project, bug_id, str(output_dir), package_prefix)
        
        buggy_nodes_path = output_dir / "buggy_nodes.txt"
        with open(buggy_nodes_path, "w") as f:
            for method in extraction_res.buggy_methods:
                f.write(f"{method}\n")
        logging.info(f"Saved buggy nodes to {buggy_nodes_path}")
        
        plot_title = f"{project}-{bug_id} Call Graph"
        plot_prefix = str(output_dir / "call_graph_plot")
        logging.info(f"Visualizing and saving {plot_title} to {plot_prefix}.png/svg...")
        plot_json_graph(extraction_res.graph_json, plot_title, plot_prefix, buggy_methods=extraction_res.buggy_methods)
            
        sbfl_df = pd.DataFrame(extraction_res.sbfl_metrics)
        
        tarantula_scores = dict(zip(sbfl_df["Method"], sbfl_df["Tarantula"]))
        
        G = read_call_graph(extraction_res.graph_json)
        logging.info(f"Computing Graph Cuts rankings for {project}-{bug_id}...")
        cut_algo = BoykovJollyCut(G, tarantula_scores, 1.0)
        
        logging.info(f"Generating 3D Energy Landscape for {project}-{bug_id}...")
        _, optimal_labeling = cut_algo.compute_min_cut()
        plot_3d_energy_landscape(G, cut_algo, optimal_labeling, output_dir / "3d_energy_landscape.html", layout_type="hierarchical", buggy_methods=extraction_res.buggy_methods)
        plot_3d_energy_landscape(G, cut_algo, optimal_labeling, output_dir / "3d_energy_landscape_kamada.html", layout_type="kamada_kawai", buggy_methods=extraction_res.buggy_methods)
        
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
        
        # work_dir = output_dir / "workspace"
        # if work_dir.exists():
        #     logging.info(f"Cleaning up workspace: {work_dir}")
        #     shutil.rmtree(work_dir)
            
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
