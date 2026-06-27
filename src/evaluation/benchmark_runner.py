import os
import json
import logging
from typing import List, Dict, Tuple, Any
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import signal

from evaluation.d4j_eval import evaluate_instance, compute_and_save_graph_properties
from evaluation.config import ROOT_DIR

def get_single_bugs(data_dir: Path) -> pd.DataFrame:
    bugs: List[Path] = sorted([item for item in data_dir.iterdir() if item.is_dir()])
    working_instances: Dict[str, Any] = {}
    
    for bug in bugs:
        project: str
        bug_id_str: str
        project, bug_id_str = bug.name.split("_")
        
        graph_json: Path = bug / "call_graph.json"
        buggy_methods_path: Path = bug / "buggy_methods.txt"
        
        try:
            with open(graph_json, 'r') as f:
                graph_data: Dict = json.load(f)

            with open(buggy_methods_path, 'r', encoding='utf-8') as file:
                buggy_nodes: List[str] = [line.strip().replace('$', '.') for line in file if line.strip()]
            
            if graph_data["metadata"]["total_nodes"] > 0:
                    graph_data["metadata"]["buggy_nodes"] = buggy_nodes
                    graph_data["metadata"]["num_buggy_nodes"] = len(buggy_nodes)
                    working_instances[bug.name] = graph_data["metadata"]
        except Exception as e:
            continue
            
    df_active: pd.DataFrame = pd.DataFrame(working_instances).T
    if df_active.empty:
        return df_active
        
    split_index = df_active.index.str.split('_', expand=True)
    df_active['project'] = split_index.get_level_values(0)
    df_active['bug_id'] = split_index.get_level_values(1)

    df_active['bug_id'] = pd.to_numeric(df_active['bug_id'])
    single_bug_instances: pd.DataFrame = df_active[df_active['num_buggy_nodes'] == 1]

    return single_bug_instances.sort_values(by="total_nodes")

def _init_worker() -> None:
    """Make workers ignore SIGINT so Ctrl-C is handled only by the parent."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _run_instance(task: Tuple[str, int, float, Path, Path, Path]) -> Dict[str, Any]:
    """Worker body."""
    project, bug_id, lambd, lambd_output_dir, data_dir, base_output_dir = task
    filepath: Path = lambd_output_dir / f"{project}_{bug_id}_{lambd}.csv"

    logging.info(f"[{project}:{bug_id}] Starting evaluation...")

    if filepath.exists():
        logging.info(f"[{project}:{bug_id}] Skipped (already exists)")
        return {"project": project, "bug_id": bug_id, "status": "skipped"}

    try:
        # Save graph properties to a shared directory across all lambdas to prevent re-computing
        shared_graph_dir = base_output_dir / "graph_properties"
        shared_graph_dir.mkdir(exist_ok=True)
        compute_and_save_graph_properties(project, bug_id, data_dir, shared_graph_dir)
        
        result_df: pd.DataFrame = evaluate_instance(project, bug_id, lambd, data_dir)
        result_df.to_csv(filepath, index=False)
        logging.info(f"[{project}:{bug_id}] Completed successfully")
        return {"project": project, "bug_id": bug_id, "status": "ok"}
    except Exception as e:
        logging.error(f"[{project}:{bug_id}] Failed: {e}")
        return {"project": project, "bug_id": bug_id, "status": "failed", "error": str(e)}

def run_evaluation_batch(target: str, lambdas: List[float], max_workers: int, sample_size: float, random_seed: int, output_dir: Path) -> None:
    dataset_folder: str = "defects4j" if target == "d4j" else target
    base_path: Path = ROOT_DIR / "data" / dataset_folder
    if not base_path.exists():
        raise FileNotFoundError(f"Target data path does not exist: {base_path}")
        
    single_bug_instances: pd.DataFrame = get_single_bugs(base_path)
    if single_bug_instances.empty:
        print(f"No valid single-bug instances found in {base_path}")
        return
        
    if sample_size < 1.0:
        n_samples: int = int(len(single_bug_instances) * sample_size)
        single_bug_instances = single_bug_instances.sample(n=n_samples, random_state=random_seed)
        print(f"Sampled {n_samples} instances ({sample_size*100}%) using seed {random_seed}.")
    
    for lambd_value in lambdas:
        lambd_output_dir: Path = output_dir / target / f"eval_lambd_{lambd_value}"
        lambd_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting benchmark with lambda = {lambd_value}")
        print(single_bug_instances)

        print(single_bug_instances)

        base_output_dir = output_dir / target
        tasks: List[Tuple[str, int, float, Path, Path, Path]] = [
            (row["project"], int(row["bug_id"]), lambd_value, lambd_output_dir, base_path / f"{row['project']}_{row['bug_id']}", base_output_dir)
            for _, row in single_bug_instances.iterrows()
        ]

        print(f"Starting benchmark: {len(tasks)} instances on {max_workers} workers...\n")
        print("Press Ctrl-C to stop all workers.\n")

        failed_runs: List[Dict[str, Any]] = []
        pool: Pool = Pool(processes=max_workers, initializer=_init_worker)
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

        print(f"Benchmark stopped for lambda = {lambd_value}")
        if failed_runs:
            print(f"Encountered {len(failed_runs)} failed instances:")
            for r in failed_runs:
                print(f"  {r['project']}_{r['bug_id']}: {r['error']}")
