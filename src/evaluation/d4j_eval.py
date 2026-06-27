import pandas as pd
import networkx as nx
from typing import Dict, List, Set, Any
from pathlib import Path
import json
import logging
import math
import random
import ast

from graph_generation.sbfl.extractor import ExtractionResult
from cut_algorithms.boykov_jolly import BoykovJollyCut
from evaluation.config import GROUND_TRUTH_CSV
from evaluation.sbfl_baselines import (
    aggregate_coverage,
    compute_tarantula,
    compute_ochiai,
    compute_dstar
)

def get_buggy_nodes(proj: str, bugid: int) -> List[str]:
    df_gt = pd.read_csv(GROUND_TRUTH_CSV)
    buggy_nodes = df_gt[(df_gt['project'] == proj) & (df_gt['bug_id'] == bugid)]['buggy_nodes'].values[0]
    return ast.literal_eval(buggy_nodes)

def load_pregen_data(project: str, bug_id: int, data_dir: Path) -> ExtractionResult:
    """Loads pre-generated extraction data instead of running D4J in real-time."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Pre-generated data not found at {data_dir}")
        
    metrics_csv: Path = data_dir / "sbfl_metrics.csv"
    graph_json: Path = data_dir / "call_graph.json"
    buggy_txt: Path = data_dir / "buggy_methods.txt"
    
    sbfl_metrics: List[Dict] = pd.read_csv(metrics_csv).to_dict(orient="records")
    
    with open(buggy_txt, 'r') as f:
        buggy_methods: List[str] = [line.strip().replace('$', '.') for line in f if line.strip()]
        
    res: ExtractionResult = ExtractionResult(
        graph_json=str(graph_json),
        sbfl_metrics=sbfl_metrics,
        buggy_methods=buggy_methods
    )
    
    return res

def json_to_digraph(graph_json: str) -> nx.DiGraph:
    with open(graph_json, 'r') as f:
        graph_data: Dict = json.load(f)

    G: nx.DiGraph = nx.DiGraph()
    for edge in graph_data['edges']:
        caller: str = edge['caller'].replace('$', '.')
        callee: str = edge['callee'].replace('$', '.')
        G.add_edge(caller, callee, weight=float(edge['frequency']))
    
    for node in graph_data.get('nodes', []):
        n_clean: str = node.replace('$', '.')
        if n_clean not in G:
            G.add_node(n_clean)
    return G


def compute_graphcut_scores(G: nx.DiGraph, node_scores: Dict[str, float], lambd: float) -> Dict[str, float]:
    cut_algo: BoykovJollyCut = BoykovJollyCut(G, node_scores, lambd=lambd)
    return {node: (lambda em: em[0] - em[1])(cut_algo.compute_min_marginals(node))
            for node in cut_algo.nodes}

def evaluate_instance(project: str, bug_id: int, lambd: float, data_dir: Path) -> pd.DataFrame:
    extraction_res: ExtractionResult = load_pregen_data(project, bug_id, data_dir)

    G: nx.DiGraph = json_to_digraph(extraction_res.graph_json)
    nodes: List[str] = list(G.nodes())
    
    buggy_methods: List[str] = get_buggy_nodes(project, bug_id)

    graph_nodes_no_args: Set[str] = {n.split('(')[0] for n in nodes}
    for buggy_method in buggy_methods:
        bm: str = buggy_method.replace('$', '.').split('(')[0]
        if bm not in graph_nodes_no_args:
            raise ValueError(f"buggy method {buggy_method} not in call graph")

    coverage_df: pd.DataFrame = pd.read_csv(data_dir / "coverage.csv")
    method_df: pd.DataFrame = aggregate_coverage(coverage_df)

    baselines: Dict[str, Dict[str, float]] = {
        "tarantula": compute_tarantula(method_df),
        "ochiai": compute_ochiai(method_df),
        "dstar": compute_dstar(method_df),
    }

    nodes_coverage: Set[str] = set(baselines["tarantula"].keys())
    missing_in_cov: Set[str] = set(nodes) - nodes_coverage
    if missing_in_cov:
        logging.warning(
            f"[{project}:{bug_id}] {len(missing_in_cov)} graph nodes missing from "
            f"coverage (assigned score 0.0)"
        )

    # Project each baseline onto the graph nodes (uncovered nodes -> 0.0).
    metric_on_nodes: Dict[str, Dict[str, float]] = {
        name: {node: scores.get(node, 0.0) for node in nodes}
        for name, scores in baselines.items()
    }

    results: pd.DataFrame = pd.DataFrame(metric_on_nodes)

    for name, node_scores in metric_on_nodes.items():
        gc_scores: Dict[str, float] = compute_graphcut_scores(G, node_scores, lambd)
        results[f"{name}_gc"] = results.index.map(gc_scores)

    results = results.reset_index().rename(columns={"index": "Method"})

    # Ranks + EXAM (expected inspection cost) for every baseline and its GC pair.
    score_cols: List[str] = ["tarantula", "tarantula_gc",
                  "ochiai", "ochiai_gc",
                  "dstar", "dstar_gc"]
    total_methods: int = len(results)
    for col in score_cols:
        expected_rank: pd.Series = results[col].rank(ascending=False, method="average")
        results[f"{col}_rank"] = expected_rank
        results[f"{col}_exam_score"] = expected_rank / total_methods

    return results


def get_node_properties_df(G: nx.DiGraph) -> pd.DataFrame:
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    in_centrality = nx.in_degree_centrality(G)
    out_centrality = nx.out_degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)
    
    clustering = nx.clustering(G)

    df = pd.DataFrame({
        'In_Degree': in_degree,
        'Out_Degree': out_degree,
        'In_Degree_Centrality': in_centrality,
        'Out_Degree_Centrality': out_centrality,
        'Betweenness_Centrality': betweenness,
        'Closeness_Centrality': closeness,
        'PageRank': pagerank,
        'Clustering_Coefficient': clustering
    })
    
    df.index.name = 'Method'
    df = df.reset_index()
    return df

def get_graph_properties_dict(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Computes global topological properties for a NetworkX DiGraph
    and returns them as a dictionary.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    is_dag = nx.is_directed_acyclic_graph(G)
    reciprocity = nx.overall_reciprocity(G)
    
    num_scc = nx.number_strongly_connected_components(G)
    num_wcc = nx.number_weakly_connected_components(G)
    
    try:
        degree_assortativity = nx.degree_assortativity_coefficient(G)
    except Exception:
        degree_assortativity = None
        
    if num_nodes > 0 and num_scc > 0:
        largest_scc_nodes = max(nx.strongly_connected_components(G), key=len)
        largest_scc = G.subgraph(largest_scc_nodes)
        
        if len(largest_scc) > 1:
            avg_path_length = nx.average_shortest_path_length(largest_scc)
            diameter = nx.diameter(largest_scc)
        else:
            avg_path_length = 0
            diameter = 0
    else:
        avg_path_length = None
        diameter = None

    data = {
        'Num_Nodes': num_nodes,
        'Num_Edges': num_edges,
        'Density': density,
        'Is_DAG': is_dag,
        'Reciprocity': reciprocity,
        'Num_SCC': num_scc,
        'Num_WCC': num_wcc,
        'Degree_Assortativity': degree_assortativity,
        'Largest_SCC_Avg_Path_Length': avg_path_length,
        'Largest_SCC_Diameter': diameter
    }
    
    return data

def compute_and_save_graph_properties(project: str, bug_id: int, data_dir: Path, output_dir: Path) -> None:
    topo_filepath: Path = output_dir / f"{project}_{bug_id}_topological.csv"
    network_filepath: Path = output_dir / f"{project}_{bug_id}_network.json"

    if topo_filepath.exists() and network_filepath.exists():
        return

    extraction_res = load_pregen_data(project, bug_id, data_dir)
    G = json_to_digraph(extraction_res.graph_json)

    if not topo_filepath.exists():
        topo_df = get_node_properties_df(G)
        topo_df.to_csv(topo_filepath, index=False)

    if not network_filepath.exists():
        network_dict = get_graph_properties_dict(G)
        with open(network_filepath, 'w') as f:
            json.dump(network_dict, f, indent=4)
