import os
import sys
from pathlib import Path
import logging
import pandas as pd
import networkx as nx
from typing import Tuple
import json
import math
# Configure logging to see output from processes
logging.basicConfig(level=logging.INFO, format='%(message)s')

ROOT_DIR = Path(os.getcwd())
if ROOT_DIR.name == "notebooks":
    ROOT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from graph_generation.sbfl.extractor import ExtractionResult
from cut_algorithms.boykov_jolly import BoykovJollyCut


def load_pregen_data(project: str, bug_id: str) -> ExtractionResult:
    """Loads pre-generated extraction data instead of running D4J in real-time."""
    data_dir = ROOT_DIR / "data" / "defects4j" / f"{project}_{bug_id}"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Pre-generated data not found at {data_dir}")
        
    metrics_csv = data_dir / "sbfl_metrics.csv"
    graph_json = data_dir / "call_graph.json"
    buggy_txt = data_dir / "buggy_methods.txt"
    
    sbfl_metrics = pd.read_csv(metrics_csv).to_dict(orient="records")
    
    with open(buggy_txt, 'r') as f:
        buggy_methods = [line.strip().replace('$', '.') for line in f if line.strip()]
        
    res = ExtractionResult(
        graph_json=str(graph_json),
        sbfl_metrics=sbfl_metrics,
        buggy_methods=buggy_methods
    )
    
    return res

def json_to_digraph(graph_json: str) -> nx.DiGraph:
    with open(graph_json, 'r') as f:
        graph_data = json.load(f)

    G = nx.DiGraph()
    for edge in graph_data['edges']:
        caller = edge['caller'].replace('$', '.')
        callee = edge['callee'].replace('$', '.')
        G.add_edge(caller, callee, weight=float(edge['frequency']))
    
    for node in graph_data.get('nodes', []):
        n_clean = node.replace('$', '.')
        if n_clean not in G:
            G.add_node(n_clean)
    return G

def compute_method_tarantula(coverage_df: pd.DataFrame) -> dict[str, float]:
    df = coverage_df
    
    # Aggregate line level to method level
    method_data = {}
    for col in df.columns:
        if col == 'Result':
            continue
        method = map_line_to_method(col)
        if method not in method_data:
            method_data[method] = df[col].copy()
        else:
            method_data[method] = method_data[method] | df[col]
            
    method_df = pd.DataFrame(method_data)
    method_df['Result'] = df['Result']
    
    total_fail = len(method_df[method_df['Result'] == 'Fail'])
    total_pass = len(method_df[method_df['Result'] == 'Pass'])
    
    tarantula_scores = {}
    for method in method_df.columns:
        if method == 'Result':
            continue
        cf = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Fail')])
        cp = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Pass')])
        
        if total_fail == 0 or (cf == 0 and cp == 0):
            tarantula = 0.0
        else:
            fail_ratio = cf / total_fail
            pass_ratio = cp / total_pass if total_pass > 0 else 0.0
            if fail_ratio + pass_ratio == 0:
                tarantula = 0.0
            else:
                tarantula = fail_ratio / (fail_ratio + pass_ratio)
                
        tarantula_scores[method] = float(tarantula)
        
    return tarantula_scores

def map_line_to_method(line_name: str) -> str:
    """Maps a coverage column like 'com.example$App#process(int):30' to 'com.example.App#process(int)'."""
    if "#" not in line_name:
        return line_name.replace('$', '.')
    return line_name.split(':')[0].replace('$', '.')

def compute_method_ochiai(coverage_df: pd.DataFrame) -> dict[str, float]:
    """
    Computes Ochiai suspiciousness score for each method based on the coverage matrix.
    Aggregates lines into methods using OR logic before computing the score.
    """
    df = coverage_df
    
    # 1. Aggregate line level to method level (Same as your Tarantula code)
    method_data = {}
    for col in df.columns:
        if col == 'Result':
            continue
        method = map_line_to_method(col)
        if method not in method_data:
            method_data[method] = df[col].copy()
        else:
            method_data[method] = method_data[method] | df[col]
            
    method_df = pd.DataFrame(method_data)
    method_df['Result'] = df['Result']
    
    # 2. Global metric: Ochiai only needs total_fail, not total_pass
    total_fail = len(method_df[method_df['Result'] == 'Fail'])
    
    ochiai_scores = {}
    for method in method_df.columns:
        if method == 'Result':
            continue
            
        # 3. Local metrics
        cf = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Fail')])
        cp = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Pass')])
        
        # 4. Ochiai Math & Edge Case Handling
        # Prevent division by zero if there are no fails or the method is never covered
        if total_fail == 0 or (cf + cp) == 0:
            ochiai = 0.0
        else:
            # The Ochiai Equation
            ochiai = cf / math.sqrt(total_fail * (cf + cp))
                
        ochiai_scores[method] = float(ochiai)
        
    return ochiai_scores


def compute_method_dstar(coverage_df: pd.DataFrame, star: int = 2) -> dict[str, float]:
    """
    Computes D* (D-Star) suspiciousness score for each method based on the coverage matrix.
    Aggregates lines into methods using OR logic before computing the score.
    The 'star' parameter controls the exponent weight (default is 2).
    """
    df = coverage_df
    
    # 1. Aggregate line level to method level
    method_data = {}
    for col in df.columns:
        if col == 'Result':
            continue
        method = map_line_to_method(col)
        if method not in method_data:
            method_data[method] = df[col].copy()
        else:
            method_data[method] = method_data[method] | df[col]
            
    method_df = pd.DataFrame(method_data)
    method_df['Result'] = df['Result']
    
    # 2. Global metric: D* needs total_fail
    total_fail = len(method_df[method_df['Result'] == 'Fail'])
    
    dstar_scores = {}
    for method in method_df.columns:
        if method == 'Result':
            continue
            
        # 3. Local metrics
        cf = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Fail')])
        cp = len(method_df[(method_df[method] == 1) & (method_df['Result'] == 'Pass')])
        
        # nf is the number of failing tests that DID NOT execute this method
        nf = total_fail - cf
        denominator = cp + nf
        
        # 4. D* Math & Edge Case Handling
        if denominator == 0:
            if cf > 0:
                dstar = float('inf')
            else:
                dstar = 0.0
        else:
            dstar = (cf ** star) / denominator
                
        dstar_scores[method] = float(dstar)
        
    return dstar_scores


def normalize_scores(scores: dict) -> dict:
    finite = [v for v in scores.values() if math.isfinite(v)]
    if not finite:
        return {k: 0.0 for k in scores}
    lo, hi = min(finite), max(finite)
    if hi == lo:
        return {k: 0.5 for k in scores}
    out = {}
    for k, v in scores.items():
        out[k] = 1.0 if not math.isfinite(v) else (v - lo) / (hi - lo)
    return out

def compute_graphcut_scores(G: nx.DiGraph, node_scores: dict, lambd: float) -> dict:
    cut_algo = BoykovJollyCut(G, node_scores, lambd=lambd)
    return {node: (lambda em: em[0] - em[1])(cut_algo.compute_min_marginals(node))
            for node in cut_algo.nodes}


def evaluate(project, bug_id, lambd=1.0):
    extraction_res = load_pregen_data(project, bug_id)

    G = json_to_digraph(extraction_res.graph_json)
    nodes = list(G.nodes())

    graph_nodes_no_args = {n.split('(')[0] for n in nodes}
    for buggy_method in extraction_res.buggy_methods:
        bm = buggy_method.replace('$', '.').split('(')[0]
        if bm not in graph_nodes_no_args:
            logging.warning(
                f"[{project}:{bug_id}] buggy method {buggy_method} not in call graph "
                f"(ground truth only; rankings still exported)"
            )

    data_dir = ROOT_DIR / "data" / "defects4j" / f"{project}_{bug_id}"
    coverage_df = pd.read_csv(data_dir / "coverage.csv")

    baselines = {
        "tarantula": compute_method_tarantula(coverage_df),
        "ochiai": compute_method_ochiai(coverage_df),
        "dstar": compute_method_dstar(coverage_df),
    }

    nodes_coverage = set(baselines["tarantula"].keys())
    missing_in_cov = set(nodes) - nodes_coverage
    if missing_in_cov:
        logging.warning(
            f"[{project}:{bug_id}] {len(missing_in_cov)} graph nodes missing from "
            f"coverage (assigned score 0.0)"
        )

    # Project each baseline onto the graph nodes (uncovered nodes -> 0.0).
    metric_on_nodes = {
        name: {node: scores.get(node, 0.0) for node in nodes}
        for name, scores in baselines.items()
    }

    results = pd.DataFrame(metric_on_nodes)

    for name, node_scores in metric_on_nodes.items():
        gc_scores = compute_graphcut_scores(G, node_scores, lambd)
        results[f"{name}_gc"] = results.index.map(gc_scores)

    results = results.reset_index().rename(columns={"index": "Method"})

    # Ranks + EXAM (expected inspection cost) for every baseline and its GC pair.
    score_cols = ["tarantula", "tarantula_gc",
                  "ochiai", "ochiai_gc",
                  "dstar", "dstar_gc"]
    total_methods = len(results)
    for col in score_cols:
        results[f"{col}_rank"] = results[col].rank(ascending=False, method="min")
        expected_rank = results[col].rank(ascending=False, method="average")
        results[f"{col}_expected_inspections"] = expected_rank
        results[f"{col}_exam_score"] = expected_rank / total_methods

    return results

def get_single_bugs(bugs):
    working_instances = {}
    for bug in bugs:
        project, bug_id = str(bug).split("/")[-1].split("_")
        graph_json = bug / "call_graph.json"
        buggy_methods = bug / "buggy_methods.txt"
        try:
            with open(graph_json, 'r') as f:
                graph_data = json.load(f)

            buggy_methods = bug / "buggy_methods.txt"
            with open(buggy_methods, 'r', encoding='utf-8') as file:
                buggy_nodes = [line.strip().replace('$', '.') for line in file if line.strip()]
            
            if graph_data["metadata"]["total_nodes"] > 0:
                graph_data["metadata"]["buggy_nodes"] = buggy_nodes
                graph_data["metadata"]["num_buggy_nodes"] = len(buggy_nodes)
                working_instances[str(bug).split("/")[-1]] = graph_data["metadata"]
        except Exception as e:
            continue
    df_active = pd.DataFrame(working_instances).T
    split_index = df_active.index.str.split('_', expand=True)
    df_active['project'] = split_index.get_level_values(0)
    df_active['bug_id'] = split_index.get_level_values(1)

    df_active['bug_id'] = pd.to_numeric(df_active['bug_id'])
    single_bug_instances = df_active[df_active['num_buggy_nodes']==1]

    return single_bug_instances.sort_values(by="total_nodes")

def _init_worker():
    """Make workers ignore SIGINT so Ctrl-C is handled only by the parent,
    which can then terminate the whole pool cleanly in one place."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _run_instance(task):
    """Worker body. Self-contained and never raises: returns a status dict so
    the parent can collect failures. Must stay at module level to be picklable."""
    project, bug_id, lambd, output_dir = task
    filepath = os.path.join(output_dir, f"{project}_{bug_id}_{lambd}.csv")

    logging.info(f"[{project}:{bug_id}] Starting evaluation...")

    if os.path.exists(filepath):
        logging.info(f"[{project}:{bug_id}] Skipped (already exists)")
        return {"project": project, "bug_id": bug_id, "status": "skipped"}

    try:
        result_df = evaluate(project, bug_id, lambd=lambd)
        result_df.to_csv(filepath, index=False)
        logging.info(f"[{project}:{bug_id}] Completed successfully")
        return {"project": project, "bug_id": bug_id, "status": "ok"}
    except Exception as e:
        logging.error(f"[{project}:{bug_id}] Failed: {e}")
        return {"project": project, "bug_id": bug_id, "status": "failed", "error": str(e)}

if __name__ == "__main__":
    from multiprocessing import Pool
    from tqdm import tqdm
    eval_parent = "evals_no_norm"

    lambdas_to_ablate = [0.1]
    for lambd_value in lambdas_to_ablate:
        output_dir = f"{eval_parent}/eval_lambd_{lambd_value}"
        os.makedirs(output_dir, exist_ok=True)
        lambd_value = lambd_value
        max_workers = 4

        print(f"Starting benchmark with lambda = {lambd_value}")

        base_path = ROOT_DIR / "data" / "defects4j"
        bugs = sorted([item for item in base_path.iterdir() if item.is_dir()])

        single_bug_instances = get_single_bugs(bugs)[:10]
        print(single_bug_instances)

        # # TARGET_PROJECTS allows you to filter which projects to run. 
        # # Leave as an empty list [] to run all projects.
        # # Example: TARGET_PROJECTS = ["Math", "Lang", "Chart"]
        # TARGET_PROJECTS = ["Collections", "JacksonDatabind"]
        
        # Flatten to one task per instance; the project grouping was only cosmetic.
        tasks = [
            (row["project"], int(row["bug_id"]), lambd_value, output_dir)
            for _, row in single_bug_instances.iterrows()
            # if not TARGET_PROJECTS or row["project"] in TARGET_PROJECTS
        ]

        print(f"Starting benchmark: {len(tasks)} instances on {max_workers} workers...\n")
        print("Press Ctrl-C to stop all workers.\n")

        failed_runs = []
        pool = Pool(processes=max_workers, initializer=_init_worker)
        try:
            for res in tqdm(pool.imap_unordered(_run_instance, tasks),
                            total=len(tasks), desc="Benchmark"):
                if res["status"] == "failed":
                    failed_runs.append(res)
            pool.close()
        except KeyboardInterrupt:
            print("\nInterrupted — terminating workers...")
            pool.terminate()
        finally:
            pool.join()

        # Final summary
        print("Benchmark stopped for lambda = {lambd_value}")
        if failed_runs:
            print(f"Encountered {len(failed_runs)} failed instances:")
            for r in failed_runs:
                print(f"  {r['project']}_{r['bug_id']}: {r['error']}")