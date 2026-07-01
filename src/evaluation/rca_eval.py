import json
import math
from pathlib import Path
import pandas as pd
import sys
from typing import Dict, Iterable, Optional, Tuple
import time

# Ensure the root project directory is in the PYTHONPATH so we can import from `src.`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.graph_generation.rca.rca_eval_pipeline import list_instances, to_networkx
from src.evaluation.baselines_rca import compute_baro_baseline, compute_zscore_baseline, _prepare_baro_input
from src.visualization.graph_plots import plot_rca_json_graph

# Global Configuration
DATA_DIR = Path("/Users/ashish/master-thesis/Master-Thesis/failure-propagation-and-rca-via-energy-minimization/notebooks/data")
OUT_PATH = Path("rca_energy_eval_results.csv")
BENCHMARK_OUTPUT_DIR = Path("outputs_rca")
SOURCE = "SOURCE"
TERMINAL = "TERMINAL"
SAFE_INF = 1e9


def _service_sort_key(node: str) -> str:
    return str(node)


def _service_aliases(node: str) -> list[str]:
    node = str(node)
    aliases = [node]
    suffix = "service"
    if node.endswith(suffix) and len(node) > len(suffix):
        aliases.append(node[: -len(suffix)])
    return aliases


def _lookup_by_service_alias(mapping: Dict[str, object], node: str):
    for alias in _service_aliases(node):
        if alias in mapping:
            return mapping[alias]
    return None


def _is_root_node(node: str, root_cause: str) -> bool:
    return root_cause in _service_aliases(node)


def _bounded_baro_priors(
    graph_nodes: Iterable[str],
    service_ranks: list[str],
    eps: float = 1e-3,
) -> Dict[str, float]:
    """Convert a BARO service ranking into finite anomaly priors."""
    n_ranked = len(service_ranks)
    rank_by_service = {service: rank for rank, service in enumerate(service_ranks)}
    priors: Dict[str, float] = {}

    for node in graph_nodes:
        rank = _lookup_by_service_alias(rank_by_service, node)
        if rank is not None and n_ranked > 0:
            q = (n_ranked - rank) / (n_ranked + 1)
            priors[node] = eps + (1.0 - 2.0 * eps) * q
        else:
            priors[node] = eps

    return priors


def _print_baro_baseline(
    priors: Dict[str, float],
    service_ranks: list[str],
    root_cause: str,
) -> None:
    rank_lookup = {service: rank + 1 for rank, service in enumerate(service_ranks)}
    rows = []
    for node, prior in priors.items():
        rows.append((node, _lookup_by_service_alias(rank_lookup, node), prior))
    rows.sort(key=lambda row: (-row[2], _service_sort_key(row[0])))

    print("\n[BARO BASELINE]")
    print(f"  root_cause={root_cause}")
    print(f"  ranked_services={len(service_ranks)}")
    for idx, (node, raw_rank, prior) in enumerate(rows, start=1):
        marker = "  <-- root cause" if _is_root_node(node, root_cause) else ""
        raw_rank_text = str(raw_rank) if raw_rank is not None else "unranked"
        print(
            f"  #{idx:02d} node={node} baro_rank={raw_rank_text} "
            f"S_i={prior:.6f}{marker}"
        )


def _safe_corr(corr_matrix: pd.DataFrame, col_u: str, col_v: str) -> float:
    try:
        corr = corr_matrix.loc[col_u, col_v]
    except Exception:
        return 0.0
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _add_latency_correlations(G, service_to_col: Dict[str, str], corr_matrix: pd.DataFrame) -> None:
    for u, v in G.edges():
        col_u = _lookup_by_service_alias(service_to_col, u)
        col_v = _lookup_by_service_alias(service_to_col, v)
        if col_u is not None and col_v is not None:
            corr = _safe_corr(corr_matrix, col_u, col_v)
        else:
            corr = 0.0
        G[u][v]["correlation"] = corr


class RCADirectedSTCut:
    """
    RCA-specific s-t graph construction.

    Original trace edges are caller -> callee. The main n-link is inserted in
    the reverse direction, callee -> caller, so the cut pays when a callee is
    Buggy while its caller remains Normal.
    """

    def __init__(
        self,
        G,
        anomaly_priors: Dict[str, float],
        lambd: float,
        *,
        rho: float = 0.0,
        eps: float = 1e-3,
    ) -> None:
        self.original_graph = G
        self.anomaly_priors = anomaly_priors
        self.lambd = float(lambd)
        self.rho = float(rho)
        self.eps = float(eps)
        self.nodes = sorted(
            [n for n in G.nodes() if n not in {SOURCE, TERMINAL}],
            key=_service_sort_key,
        )
        self.D0: Dict[str, float] = {}
        self.D1: Dict[str, float] = {}
        self.edge_terms: list[dict] = []

        self._compute_unary_potentials()
        self._compute_pairwise_terms()

    def _compute_unary_potentials(self) -> None:
        for node in self.nodes:
            prior = float(self.anomaly_priors.get(node, self.eps))
            prior = min(max(prior, self.eps), 1.0 - self.eps)
            self.D0[node] = -math.log(1.0 - prior)
            self.D1[node] = -math.log(prior)

    def _compute_pairwise_terms(self) -> None:
        self.edge_terms = []
        for u, v, data in sorted(
            self.original_graph.edges(data=True),
            key=lambda item: (_service_sort_key(item[0]), _service_sort_key(item[1])),
        ):
            if u == v:
                continue

            freq = float(data.get("weight", 0.0))
            degree_u = max(float(self.original_graph.degree[u]), 1.0)
            degree_v = max(float(self.original_graph.degree[v]), 1.0)
            trace_strength = math.log1p(freq) / math.sqrt(degree_u * degree_v)

            corr_raw = data.get("correlation", 0.0)
            corr = 0.0 if pd.isna(corr_raw) else float(corr_raw)
            corr_positive = max(0.0, corr)
            weight = trace_strength * corr_positive

            self.edge_terms.append(
                {
                    "caller": u,
                    "callee": v,
                    "frequency": freq,
                    "degree_caller": degree_u,
                    "degree_callee": degree_v,
                    "trace_strength": trace_strength,
                    "correlation": corr,
                    "correlation_positive": corr_positive,
                    "weight": weight,
                    "reverse_capacity": self.lambd * weight,
                    "forward_capacity": self.rho * self.lambd * weight,
                }
            )

    def build_st_graph(
        self,
        constrained_node: Optional[str] = None,
        constrained_label: Optional[int] = None,
        *,
        verbose: bool = False,
    ):
        G_cut = type(self.original_graph)()
        G_cut.add_node(SOURCE)
        G_cut.add_node(TERMINAL)

        if verbose:
            constraint = (
                "none"
                if constrained_node is None
                else f"{constrained_node}={constrained_label}"
            )
            print("\n[ST GRAPH BUILD]")
            print(
                f"  lambda={self.lambd:.6f} rho={self.rho:.6f} "
                f"constraint={constraint}"
            )

        for node in self.nodes:
            cap_s_to_node = self.D0[node]
            cap_node_to_t = self.D1[node]
            constraint_note = ""

            if node == constrained_node and constrained_label is not None:
                if constrained_label == 0:
                    cap_node_to_t = SAFE_INF
                    constraint_note = " force_Normal"
                elif constrained_label == 1:
                    cap_s_to_node = SAFE_INF
                    constraint_note = " force_Buggy"

            G_cut.add_edge(SOURCE, node, capacity=cap_s_to_node)
            G_cut.add_edge(node, TERMINAL, capacity=cap_node_to_t)

            if verbose:
                score = self.anomaly_priors.get(node, self.eps)
                print(
                    f"  T-LINK node={node} S_i={score:.6f} "
                    f"D_i(0 Normal)={self.D0[node]:.6f} "
                    f"D_i(1 Buggy)={self.D1[node]:.6f} "
                    f"S->node={cap_s_to_node:.6f} node->T={cap_node_to_t:.6f}"
                    f"{constraint_note}"
                )

        for term in self.edge_terms:
            caller = term["caller"]
            callee = term["callee"]
            reverse_capacity = term["reverse_capacity"]
            forward_capacity = term["forward_capacity"]

            if reverse_capacity > 0.0:
                G_cut.add_edge(callee, caller, capacity=reverse_capacity)
            if forward_capacity > 0.0:
                G_cut.add_edge(caller, callee, capacity=forward_capacity)

            if verbose:
                print(
                    f"  N-LINK trace_edge={caller}->{callee} "
                    f"freq={term['frequency']:.0f} "
                    f"deg=({term['degree_caller']:.0f},{term['degree_callee']:.0f}) "
                    f"trace_strength={term['trace_strength']:.6f} "
                    f"correlation={term['correlation']:.6f} "
                    f"correlation_pos={term['correlation_positive']:.6f} "
                    f"C_ij={term['weight']:.6f} "
                    f"reverse n-link {callee}->{caller} cap={reverse_capacity:.6f} "
                    f"forward n-link {caller}->{callee} cap={forward_capacity:.6f}"
                )

        if verbose:
            print(
                f"  built_nodes={G_cut.number_of_nodes()} "
                f"built_edges={G_cut.number_of_edges()}"
            )

        return G_cut

    def compute_min_cut(self, *, verbose: bool = False) -> Tuple[float, Dict[str, int]]:
        t0 = time.perf_counter()
        G_cut = self.build_st_graph(verbose=verbose)
        if verbose:
            print(f"  [Profile] build_st_graph took {time.perf_counter() - t0:.6f}s")
        
        t1 = time.perf_counter()
        cut_value, partition = pd_nx_minimum_cut(G_cut)
        if verbose:
            print(f"  [Profile] minimum_cut took {time.perf_counter() - t1:.6f}s")
            
        source_partition, _ = partition
        labeling = {
            node: 1 if node in source_partition else 0
            for node in self.nodes
        }
        return cut_value, labeling

    def compute_min_marginals(self, node: str) -> Tuple[float, float]:
        G_cut_0 = self.build_st_graph(constrained_node=node, constrained_label=0)
        energy_0, _ = pd_nx_minimum_cut(G_cut_0)

        G_cut_1 = self.build_st_graph(constrained_node=node, constrained_label=1)
        energy_1, _ = pd_nx_minimum_cut(G_cut_1)

        return energy_0, energy_1


def pd_nx_minimum_cut(G_cut):
    """Keep the NetworkX import local to make the cut dependency explicit."""
    import networkx as nx

    return nx.minimum_cut(
        G_cut,
        SOURCE,
        TERMINAL,
        capacity="capacity",
        flow_func=nx.algorithms.flow.boykov_kolmogorov,
    )


def _rank_graphcut_nodes(cut_algo: RCADirectedSTCut) -> pd.DataFrame:
    rows = []
    for node in cut_algo.nodes:
        energy_0, energy_1 = cut_algo.compute_min_marginals(node)
        score = energy_0 - energy_1
        rows.append(
            {
                "node": node,
                "prior": cut_algo.anomaly_priors.get(node, cut_algo.eps),
                "D0": cut_algo.D0[node],
                "D1": cut_algo.D1[node],
                "energy_if_normal": energy_0,
                "energy_if_buggy": energy_1,
                "score": score,
            }
        )

    ranked = pd.DataFrame(rows).sort_values(
        ["score", "prior", "node"],
        ascending=[False, False, True],
    )
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def _print_graphcut_ranking(
    ranked: pd.DataFrame,
    root_cause: str,
    *,
    lambd: float,
    energy: float,
    labeling: Dict[str, int],
) -> None:
    suspicious_nodes = [node for node, label in labeling.items() if label == 1]
    print("\n[MIN-MARGINAL RCA RANKING]")
    print(
        f"  lambda={lambd:.6f} global_energy={energy:.6f} "
        f"source_side_size={len(suspicious_nodes)} "
        f"source_side={suspicious_nodes}"
    )

    for row in ranked.itertuples(index=False):
        marker = "  <-- root cause" if _is_root_node(row.node, root_cause) else ""
        label = labeling.get(row.node, 0)
        print(
            f"  #{int(row.rank):02d} node={row.node} label={label} "
            f"S_i={row.prior:.6f} D_i(0)={row.D0:.6f} D_i(1)={row.D1:.6f} "
            f"E_min(L=0)={row.energy_if_normal:.6f} "
            f"E_min(L=1)={row.energy_if_buggy:.6f} "
            f"GraphCut Score={row.score:.6f}{marker}"
        )


def _ranking_frame_for_output(
    priors: Dict[str, float],
    graphcut_ranked: pd.DataFrame,
) -> pd.DataFrame:
    """Return the same per-node ranking shape used by scripts/run_evaluation.py."""
    total_nodes = len(priors)

    results = pd.DataFrame(
        [
            {
                "Method": node,
                "baro": prior,
            }
            for node, prior in priors.items()
        ]
    )

    gc_scores = graphcut_ranked.set_index("node")["score"].to_dict()
    results["baro_gc"] = results["Method"].map(gc_scores)

    score_cols = ["baro", "baro_gc"]
    for col in score_cols:
        expected_rank = results[col].rank(ascending=False, method="average")
        min_rank = results[col].rank(ascending=False, method="min")
        max_rank = results[col].rank(ascending=False, method="max")

        results[f"{col}_rank"] = expected_rank
        results[f"{col}_exam_score"] = expected_rank / total_nodes
        results[f"{col}_rank_min"] = min_rank
        results[f"{col}_exam_score_min"] = min_rank / total_nodes
        results[f"{col}_rank_max"] = max_rank
        results[f"{col}_exam_score_max"] = max_rank / total_nodes

    return results.sort_values(
        ["baro_gc", "baro", "Method"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _safe_instance_stem(instance_dir: Path, data_dir: Path, window: str, lambd: float) -> str:
    relative = instance_dir.relative_to(data_dir)
    safe = "_".join(relative.parts)
    return f"{safe}_{window}_{lambd}"


def _summary_for_ranking(
    ranked: pd.DataFrame,
    root_cause: str,
    labeling: Dict[str, int],
) -> dict:
    root_rows = ranked[ranked["node"].map(lambda node: _is_root_node(node, root_cause))]
    if root_rows.empty:
        return {
            "root_rank": None,
            "root_score": None,
            "hit_at_1": 0,
            "hit_at_3": 0,
            "hit_at_5": 0,
            "cut_size": sum(1 for label in labeling.values() if label == 1),
        }

    root_rank = int(root_rows.iloc[0]["rank"])
    root_score = float(root_rows.iloc[0]["score"])
    return {
        "root_rank": root_rank,
        "root_score": root_score,
        "hit_at_1": int(root_rank <= 1),
        "hit_at_3": int(root_rank <= 3),
        "hit_at_5": int(root_rank <= 5),
        "cut_size": sum(1 for label in labeling.values() if label == 1),
    }

def evaluate_rca_instances(
    limit: int | None = 5,
    *,
    lambdas: list[float] | None = None,
    eps: float = 1e-3,
    rho: float = 0.0,
    verbose_st_graph: bool = False,
    verbose_baro: bool = False,
    save_ranking_csvs: bool = False,
    benchmark_output_dir: Path = BENCHMARK_OUTPUT_DIR,
    plot_graphs: bool = False,
):
    if lambdas is None:
        lambdas = [0.0, 0.01, 0.1, 1.0]

    instances = list_instances(DATA_DIR)
    print(f"Found {len(instances)} RCA instances")
    print(f"Writing summary CSV to {OUT_PATH.resolve()}")
    if save_ranking_csvs:
        print(f"Writing benchmark-format ranking CSVs under {(benchmark_output_dir / 'rca').resolve()}")
    print(f"Using BARO eps={eps} directed_rho={rho}")
    
    results = []
    
    selected_instances = instances if limit is None else instances[:limit]
    for idx, instance_dir in enumerate(selected_instances):
        print("\n" + "=" * 100)
        print(f"Processing {idx+1}/{len(selected_instances)}: {instance_dir.relative_to(DATA_DIR)}")
        
        # Infer ground truth from parent directory name (e.g. emailservice_f1 -> emailservice)
        try:
            fault_name = instance_dir.parent.name
            root_cause = fault_name.split('_')[0]
        except Exception as e:
            root_cause = "unknown"
            print(f"Warning: {e}")

        print(f"fault={fault_name} root_cause={root_cause}")
            
        # Compute BARO baseline
        try:
            baro_out = compute_baro_baseline(instance_dir)
            service_ranks = baro_out.get("service_ranks", [])
            data, inject_time, dataset = _prepare_baro_input(instance_dir)

            if verbose_baro:
                print(f"BARO metric ranks: {baro_out.get('metric_ranks', [])}")
                print(f"BARO service ranks: {service_ranks}")
            print(f"BARO input dataset={dataset} inject_time={inject_time}")
            
            # Create mapping from service name to its latency metric column
            service_to_col = {}
            for col in data.columns:
                if col.endswith('_latency'):
                    service = col.split('_')[0].replace('-db', '')
                    if service not in service_to_col:
                        service_to_col[service] = col
                        
            # Compute correlation matrix for latency metrics
            latency_cols = list(service_to_col.values())
            corr_matrix = data[latency_cols].corr() if latency_cols else pd.DataFrame()
            if verbose_baro:
                print(f"Latency metric service map: {service_to_col}")
        except Exception as e:
            print(f"Failed to compute BARO/Correlation for {instance_dir}: {e}")
            service_ranks = []
            service_to_col = {}
            corr_matrix = pd.DataFrame()

        # Compute Z-score baseline
        try:
            zscore_out = compute_zscore_baseline(instance_dir)
            zscore_service_ranks = zscore_out.get("service_ranks", [])
            if verbose_baro:
                print(f"Z-SCORE service ranks: {zscore_service_ranks}")
        except Exception as e:
            print(f"Failed to compute Z-score for {instance_dir}: {e}")
            zscore_service_ranks = []
            
        for window in ["fault", "runtime"]:
            print("\n" + "-" * 100)
            print(f"Window={window}")
            graph_path = instance_dir / f"trace_service_graph_{window}.json"
            if not graph_path.exists():
                print(f"Graph {graph_path} not found. Skipping.")
                continue
                
            try:
                # Export the system graph visualisations to the instance directory
                if plot_graphs:
                    save_prefix = str(instance_dir / f"trace_service_graph_{window}")
                    plot_rca_json_graph(str(graph_path), title=f"RCA System Graph ({window.capitalize()})", save_prefix=save_prefix, root_cause=root_cause)
                
                t0 = time.perf_counter()
                with open(graph_path, 'r') as f:
                    graph_data = json.load(f)
                G = to_networkx(graph_data)
                graph_gen_time = time.perf_counter() - t0
                print(
                    f"Loaded graph nodes={G.number_of_nodes()} "
                    f"edges={G.number_of_edges()} from {graph_path} "
                    f"(Graph Gen Time: {graph_gen_time:.4f}s)"
                )
            except Exception as e:
                print(f"Failed to load graph {graph_path}: {e}")
                continue
                
            priors = _bounded_baro_priors(G.nodes, service_ranks, eps=eps)
            if verbose_baro:
                _print_baro_baseline(priors, service_ranks, root_cause)
            _add_latency_correlations(G, service_to_col, corr_matrix)

            baro_ranked = (
                pd.DataFrame(
                    [
                        {"node": node, "prior": prior}
                        for node, prior in priors.items()
                    ]
                )
                .sort_values(["prior", "node"], ascending=[False, True])
                .reset_index(drop=True)
            )
            baro_ranked["rank"] = baro_ranked.index + 1
            baro_labeling = {node: 0 for node in G.nodes}
            baro_summary = _summary_for_ranking(
                baro_ranked.assign(score=baro_ranked["prior"]),
                root_cause,
                baro_labeling,
            )

            try:
                suite_name = instance_dir.parents[1].name
            except Exception:
                suite_name = "unknown"

            results.append({
                "instance": str(instance_dir.relative_to(DATA_DIR)),
                "suite": suite_name,
                "fault": fault_name,
                "root_cause": root_cause,
                "window": window,
                "method": "baro",
                "lambda": None,
                "rho": None,
                "total_nodes": len(G.nodes),
                "total_edges": len(G.edges),
                "baro_nodes": len(service_ranks),
                "energy": None,
                "status": "ok",
                **baro_summary,
            })

            # Z-score ranking summary
            zscore_priors = _bounded_baro_priors(G.nodes, zscore_service_ranks, eps=eps)
            zscore_ranked = (
                pd.DataFrame(
                    [
                        {"node": node, "prior": prior}
                        for node, prior in zscore_priors.items()
                    ]
                )
                .sort_values(["prior", "node"], ascending=[False, True])
                .reset_index(drop=True)
            )
            zscore_ranked["rank"] = zscore_ranked.index + 1
            zscore_summary = _summary_for_ranking(
                zscore_ranked.assign(score=zscore_ranked["prior"]),
                root_cause,
                baro_labeling,
            )

            results.append({
                "instance": str(instance_dir.relative_to(DATA_DIR)),
                "suite": suite_name,
                "fault": fault_name,
                "root_cause": root_cause,
                "window": window,
                "method": "zscore",
                "lambda": None,
                "rho": None,
                "total_nodes": len(G.nodes),
                "total_edges": len(G.edges),
                "baro_nodes": len(zscore_service_ranks),
                "energy": None,
                "status": "ok",
                **zscore_summary,
            })

            for lambd in lambdas:
                try:
                    cut_algo = RCADirectedSTCut(
                        G,
                        priors,
                        lambd,
                        rho=rho,
                        eps=eps,
                    )
                    energy, labeling = cut_algo.compute_min_cut(verbose=verbose_st_graph)
                    ranked = _rank_graphcut_nodes(cut_algo)
                    if verbose_st_graph:
                        _print_graphcut_ranking(
                            ranked,
                            root_cause,
                            lambd=lambd,
                            energy=energy,
                            labeling=labeling,
                        )
                    summary = _summary_for_ranking(ranked, root_cause, labeling)

                    if save_ranking_csvs:
                        lambd_output_dir = benchmark_output_dir / "rca" / f"eval_lambd_{lambd}"
                        lambd_output_dir.mkdir(parents=True, exist_ok=True)
                        out_name = _safe_instance_stem(instance_dir, DATA_DIR, window, lambd)
                        out_path = lambd_output_dir / f"{out_name}.csv"
                        ranking_output = _ranking_frame_for_output(priors, ranked)
                        ranking_output.to_csv(out_path, index=False)
                    
                    results.append({
                        "instance": str(instance_dir.relative_to(DATA_DIR)),
                        "suite": suite_name,
                        "fault": fault_name,
                        "root_cause": root_cause,
                        "window": window,
                        "method": "baro_directed_trace_corr_cut",
                        "lambda": lambd,
                        "rho": rho,
                        "total_nodes": len(G.nodes),
                        "total_edges": len(G.edges),
                        "baro_nodes": len(service_ranks),
                        "energy": energy,
                        "status": "ok",
                        **summary,
                    })
                except Exception as e:
                    print(f"Cut failed for {instance_dir} {window} lambd={lambd}: {e}")
                    results.append({
                        "instance": str(instance_dir.relative_to(DATA_DIR)),
                        "suite": suite_name,
                        "fault": fault_name,
                        "root_cause": root_cause,
                        "window": window,
                        "method": "baro_directed_trace_corr_cut",
                        "lambda": lambd,
                        "rho": rho,
                        "status": f"error: {str(e)}",
                    })
                    
    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved evaluation results to {OUT_PATH}")

if __name__ == '__main__':
    evaluate_rca_instances(limit=None, save_ranking_csvs=True)
