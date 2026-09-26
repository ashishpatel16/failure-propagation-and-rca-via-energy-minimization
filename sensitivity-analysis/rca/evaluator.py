import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
RCA_EVAL_DIR: Path = ROOT_DIR / "rca-eval"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(RCA_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RCA_EVAL_DIR))

from graph_builder import (
    annotate_graph_with_telemetry,
    build_call_graph,
    compute_latency_correlation_matrix,
    load_traces_dataframe,
)
from rca_models import (
    RCAInstanceMetadata,
    RCASampleTrialResult,
    RCASamplingDimension,
    RCASensitivityConfig,
    RCASensitivitySummaryRow,
    UnaryPriorStrategy,
)
from sampler import prepare_truncated_baro_windows, subsample_telemetry_metrics


class RCASensitivityEvaluator:
    def get_service_aliases(self, service_name: str) -> List[str]:
        """Returns canonical name and normalized aliases for a microservice identifier."""
        clean_name: str = str(service_name)
        if "::" in clean_name:
            clean_name = clean_name.split("::")[0]

        aliases: List[str] = [
            clean_name,
            clean_name.replace("-", ""),
            clean_name.replace("_", ""),
            clean_name.lower(),
        ]
        suffix: str = "service"
        if clean_name.endswith(suffix) and len(clean_name) > len(suffix):
            base: str = clean_name[: -len(suffix)]
            aliases.extend([base, base.rstrip("-"), base.rstrip("_"), base.lower()])
        return list(set(aliases))

    def find_root_cause_rank(
        self,
        ranked_services: List[str],
        root_cause: str,
        all_graph_nodes: List[str],
    ) -> int:
        """Finds 1-based rank of the root cause service; assigns bottom rank if unmeasured."""
        clean_rc: str = (
            root_cause.lower()
            .replace("-", "")
            .replace("_", "")
            .replace("service", "")
        )

        for idx, service_name in enumerate(ranked_services, start=1):
            aliases: List[str] = self.get_service_aliases(service_name)
            for alias in aliases:
                alias_clean: str = (
                    alias.lower()
                    .replace("-", "")
                    .replace("_", "")
                    .replace("service", "")
                )
                if clean_rc == alias_clean:
                    return idx

        # If not found in ranked services, rank is placed at the bottom of the graph nodes
        return max(len(ranked_services) + 1, len(all_graph_nodes))

    def extract_service_ranks(
        self, metric_ranks: List[str], all_graph_nodes: List[str]
    ) -> List[str]:
        """Maps ranked metric names to unique microservice names and appends remaining graph nodes."""
        service_ranks: List[str] = []
        for metric in metric_ranks:
            service_name: str = metric.split("_")[0].replace("-db", "")
            if service_name not in service_ranks:
                service_ranks.append(service_name)

        # Append remaining nodes that had zero anomaly score
        for node in all_graph_nodes:
            if node not in service_ranks:
                service_ranks.append(node)

        return service_ranks

    def compute_raw_service_scores(
        self,
        normal_df: pd.DataFrame,
        anomal_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Fits RobustScaler on pre-fault baseline and returns max peak z-score per service."""
        varying_cols: List[str] = [
            col
            for col in normal_df.columns
            if (normal_df[col] != normal_df[col].iloc[0]).any()
        ]
        if len(varying_cols) == 0:
            raise ValueError("No varying metric columns found in pre-fault baseline window.")

        filtered_normal: pd.DataFrame = normal_df[varying_cols]
        filtered_anomal: pd.DataFrame = anomal_df[varying_cols]

        service_scores: Dict[str, float] = {}
        for col in filtered_normal.columns:
            a: np.ndarray = filtered_normal[col].to_numpy().reshape(-1, 1)
            b: np.ndarray = filtered_anomal[col].to_numpy().reshape(-1, 1)

            scaler: RobustScaler = RobustScaler().fit(a)
            zscores: np.ndarray = scaler.transform(b)[:, 0]
            max_score: float = float(np.max(zscores))

            service: str = col.split("_")[0].replace("-db", "")
            if service not in service_scores or max_score > service_scores[service]:
                service_scores[service] = max_score

        return service_scores

    def compute_unary_priors(
        self,
        service_scores: Dict[str, float],
        graph_nodes: List[str],
        strategy: UnaryPriorStrategy,
        eps: float,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Calculates bounded anomaly priors S_i in [eps, 1-eps] and unary costs D0, D1."""
        if len(service_scores) == 0:
            raise ValueError("Cannot compute unary priors on empty service scores.")

        raw_values: List[float] = list(service_scores.values())
        min_z: float = min(raw_values)
        max_z: float = max(raw_values)
        range_z: float = max_z - min_z

        priors: Dict[str, float] = {}
        d0: Dict[str, float] = {}
        d1: Dict[str, float] = {}

        if strategy == UnaryPriorStrategy.SIGMOID:
            for node in graph_nodes:
                z: float = service_scores[node] if node in service_scores else min_z
                norm_z: float = (z - min_z) / range_z if range_z > 0 else 0.5
                prob: float = 1.0 / (1.0 + np.exp(-6.0 * (norm_z - 0.5)))
                priors[node] = float(min(max(prob, eps), 1.0 - eps))

        elif strategy == UnaryPriorStrategy.MINMAX:
            for node in graph_nodes:
                z = service_scores[node] if node in service_scores else min_z
                norm_z = (z - min_z) / range_z if range_z > 0 else 0.5
                bounded_prob: float = eps + (1.0 - 2.0 * eps) * norm_z
                priors[node] = float(bounded_prob)

        elif strategy == UnaryPriorStrategy.RANK_BASED:
            sorted_ranks = sorted(service_scores.items(), key=lambda x: x[1], reverse=True)
            rank_lookup: Dict[str, int] = {svc: r for r, (svc, _) in enumerate(sorted_ranks)}
            n_ranked: int = len(rank_lookup)
            for node in graph_nodes:
                rank: int = rank_lookup[node] if node in rank_lookup else n_ranked
                q: float = (n_ranked - rank) / float(n_ranked + 1)
                bounded_prob = eps + (1.0 - 2.0 * eps) * q
                priors[node] = float(bounded_prob)
        else:
            raise ValueError(f"Unknown UnaryPriorStrategy: {strategy}")

        for node, s in priors.items():
            d0[node] = float(-np.log(1.0 - s))
            d1[node] = float(-np.log(s))

        return priors, d0, d1

    def compute_graphcut_min_marginals(
        self,
        G: nx.DiGraph,
        d0: Dict[str, float],
        d1: Dict[str, float],
        lambd: float,
        rho: float,
    ) -> Dict[str, float]:
        """Computes continuous min-marginal energy differences using fast in-place clamping."""
        nodes: List[str] = list(G.nodes())

        # Construct base flow graph
        G_cut: nx.DiGraph = nx.DiGraph()
        G_cut.add_node("SOURCE")
        G_cut.add_node("TERMINAL")

        for node in nodes:
            G_cut.add_edge("SOURCE", node, capacity=d0[node])
            G_cut.add_edge(node, "TERMINAL", capacity=d1[node])

        # Pairwise n-links
        for u, v in G.edges():
            if u == v:
                continue
            trace_freq: float = float(G[u][v].get("weight", 1.0))
            corr: float = float(G[u][v].get("corr", 0.0))
            combined_capacity: float = lambd * (trace_freq + rho * corr)
            if combined_capacity > 0:
                G_cut.add_edge(u, v, capacity=combined_capacity)

        safe_inf: float = 1e9
        scores: Dict[str, float] = {}

        for node in nodes:
            orig_s: float = float(G_cut["SOURCE"][node]["capacity"])
            orig_t: float = float(G_cut[node]["TERMINAL"]["capacity"])

            # 1. Clamped to Normal (Label 0 -> Terminal)
            G_cut[node]["TERMINAL"]["capacity"] = safe_inf
            e0, _ = nx.minimum_cut(
                G_cut,
                "SOURCE",
                "TERMINAL",
                capacity="capacity",
                flow_func=nx.algorithms.flow.boykov_kolmogorov,
            )
            G_cut[node]["TERMINAL"]["capacity"] = orig_t

            # 2. Clamped to Buggy (Label 1 -> Source)
            G_cut["SOURCE"][node]["capacity"] = safe_inf
            e1, _ = nx.minimum_cut(
                G_cut,
                "SOURCE",
                "TERMINAL",
                capacity="capacity",
                flow_func=nx.algorithms.flow.boykov_kolmogorov,
            )
            G_cut["SOURCE"][node]["capacity"] = orig_s

            scores[node] = float(round(e0 - e1, 6))

        return scores

    def evaluate_instance_sensitivity(
        self,
        instance: RCAInstanceMetadata,
        config: RCASensitivityConfig,
    ) -> Tuple[List[RCASampleTrialResult], List[RCASensitivitySummaryRow]]:
        """Executes RCA sensitivity evaluation across observation windows and metric dropouts."""
        instance_name: str = f"{instance.suite.value}_{instance.dataset}_{instance.fault}_{instance.instance_id}"
        full_window_data, full_inject_time = prepare_truncated_baro_windows(
            instance_dir=instance.instance_dir,
            window_minutes=20,
            tdelta=config.tdelta,
        )
        full_normal: pd.DataFrame = full_window_data[
            full_window_data["time"] < full_inject_time
        ].drop(columns=["time"], errors="ignore")
        full_anomal: pd.DataFrame = full_window_data[
            full_window_data["time"] >= full_inject_time
        ].drop(columns=["time"], errors="ignore")

        full_raw_scores: Dict[str, float] = self.compute_raw_service_scores(
            full_normal, full_anomal
        )
        full_metric_ranks: List[str] = [
            k
            for k, _ in sorted(
                full_raw_scores.items(), key=lambda item: item[1], reverse=True
            )
        ]
        all_metric_services: List[str] = list(full_raw_scores.keys())
        full_service_ranks: List[str] = self.extract_service_ranks(
            full_metric_ranks, all_metric_services
        )
        ref_baseline_rank: int = self.find_root_cause_rank(
            full_service_ranks, instance.root_cause, all_metric_services
        )

        traces_csv: Path = instance.instance_dir / "traces.csv"
        traces_df: pd.DataFrame = load_traces_dataframe(traces_csv)
        G_base: nx.DiGraph = build_call_graph(traces_df, all_metric_services)
        corr_matrix: pd.DataFrame = compute_latency_correlation_matrix(full_window_data)
        G: nx.DiGraph = annotate_graph_with_telemetry(G_base, corr_matrix)
        all_nodes: List[str] = list(G.nodes())
        total_services: int = len(all_nodes)

        trial_results: List[RCASampleTrialResult] = []
        summary_rows: List[RCASensitivitySummaryRow] = []

        for window_min in config.window_minutes_list:
            win_data, win_inject = prepare_truncated_baro_windows(
                instance_dir=instance.instance_dir,
                window_minutes=window_min,
                tdelta=config.tdelta,
            )
            win_normal: pd.DataFrame = win_data[win_data["time"] < win_inject].drop(
                columns=["time"], errors="ignore"
            )
            win_anomal: pd.DataFrame = win_data[win_data["time"] >= win_inject].drop(
                columns=["time"], errors="ignore"
            )

            win_corr: pd.DataFrame = compute_latency_correlation_matrix(win_data)
            G_win: nx.DiGraph = annotate_graph_with_telemetry(G_base, win_corr)

            raw_scores: Dict[str, float] = self.compute_raw_service_scores(
                win_normal, win_anomal
            )
            _, d0, d1 = self.compute_unary_priors(
                service_scores=raw_scores,
                graph_nodes=all_nodes,
                strategy=config.strategy,
                eps=config.eps,
            )

            baro_metric_ranks: List[str] = [
                k
                for k, _ in sorted(
                    raw_scores.items(), key=lambda item: item[1], reverse=True
                )
            ]
            baro_services: List[str] = self.extract_service_ranks(
                baro_metric_ranks, all_nodes
            )
            baro_rank: int = self.find_root_cause_rank(
                baro_services, instance.root_cause, all_nodes
            )

            res_baro: RCASampleTrialResult = RCASampleTrialResult(
                instance_name=instance_name,
                suite=instance.suite.value,
                dataset=instance.dataset,
                fault=instance.fault,
                root_cause=instance.root_cause,
                dimension_type=RCASamplingDimension.TIME_WINDOW.value,
                dimension_value=float(window_min),
                trial_idx=0,
                lambda_val=0.0,
                rho_val=0.0,
                strategy=config.strategy.value,
                arm_name="BARO_baseline",
                predicted_rank=baro_rank,
                is_top_1=(baro_rank == 1),
                is_top_3=(baro_rank <= 3),
                is_top_5=(baro_rank <= 5),
                mrr=float(1.0 / baro_rank),
                full_window_baseline_rank=ref_baseline_rank,
                rank_deviation=float(abs(baro_rank - ref_baseline_rank)),
                total_services=total_services,
            )
            trial_results.append(res_baro)

            # Graph Cuts for each Lambda > 0 using window-specific graph G_win
            for lambd in config.lambdas:
                if lambd == 0.0:
                    continue
                gc_scores: Dict[str, float] = self.compute_graphcut_min_marginals(
                    G=G_win,
                    d0=d0,
                    d1=d1,
                    lambd=lambd,
                    rho=config.rho,
                )
                gc_metric_ranks: List[str] = [
                    k
                    for k, _ in sorted(
                        gc_scores.items(), key=lambda item: item[1], reverse=True
                    )
                ]
                gc_services: List[str] = self.extract_service_ranks(
                    gc_metric_ranks, all_nodes
                )
                gc_rank: int = self.find_root_cause_rank(
                    gc_services, instance.root_cause, all_nodes
                )

                res_gc: RCASampleTrialResult = RCASampleTrialResult(
                    instance_name=instance_name,
                    suite=instance.suite.value,
                    dataset=instance.dataset,
                    fault=instance.fault,
                    root_cause=instance.root_cause,
                    dimension_type=RCASamplingDimension.TIME_WINDOW.value,
                    dimension_value=float(window_min),
                    trial_idx=0,
                    lambda_val=lambd,
                    rho_val=config.rho,
                    strategy=config.strategy.value,
                    arm_name=f"BARO_GC_lambd_{lambd}",
                    predicted_rank=gc_rank,
                    is_top_1=(gc_rank == 1),
                    is_top_3=(gc_rank <= 3),
                    is_top_5=(gc_rank <= 5),
                    mrr=float(1.0 / gc_rank),
                    full_window_baseline_rank=ref_baseline_rank,
                    rank_deviation=float(abs(gc_rank - ref_baseline_rank)),
                    total_services=total_services,
                )
                trial_results.append(res_gc)
        rng: np.random.Generator = np.random.default_rng(config.random_seed)

        for frac in config.metric_fractions:
            metric_trials_by_lambda: Dict[float, List[RCASampleTrialResult]] = {
                l: [] for l in [0.0] + [l for l in config.lambdas if l > 0.0]
            }

            for trial_i in range(config.n_metric_trials):
                sub_data: pd.DataFrame = subsample_telemetry_metrics(
                    data_df=full_window_data,
                    metric_fraction=frac,
                    rng=rng,
                )
                sub_normal: pd.DataFrame = sub_data[
                    sub_data["time"] < full_inject_time
                ].drop(columns=["time"], errors="ignore")
                sub_anomal: pd.DataFrame = sub_data[
                    sub_data["time"] >= full_inject_time
                ].drop(columns=["time"], errors="ignore")

                sub_raw: Dict[str, float] = self.compute_raw_service_scores(
                    sub_normal, sub_anomal
                )
                _, d0_sub, d1_sub = self.compute_unary_priors(
                    service_scores=sub_raw,
                    graph_nodes=list(G.nodes()),
                    strategy=config.strategy,
                    eps=config.eps,
                )

                b_metrics: List[str] = [
                    k
                    for k, _ in sorted(
                        sub_raw.items(), key=lambda item: item[1], reverse=True
                    )
                ]
                b_svcs: List[str] = self.extract_service_ranks(b_metrics, all_nodes)
                b_rank: int = self.find_root_cause_rank(
                    b_svcs, instance.root_cause, all_nodes
                )

                t_baro: RCASampleTrialResult = RCASampleTrialResult(
                    instance_name=instance_name,
                    suite=instance.suite.value,
                    dataset=instance.dataset,
                    fault=instance.fault,
                    root_cause=instance.root_cause,
                    dimension_type=RCASamplingDimension.METRIC_DROPOUT.value,
                    dimension_value=float(frac),
                    trial_idx=trial_i,
                    lambda_val=0.0,
                    rho_val=0.0,
                    strategy=config.strategy.value,
                    arm_name="BARO_baseline",
                    predicted_rank=b_rank,
                    is_top_1=(b_rank == 1),
                    is_top_3=(b_rank <= 3),
                    is_top_5=(b_rank <= 5),
                    mrr=float(1.0 / b_rank),
                    full_window_baseline_rank=ref_baseline_rank,
                    rank_deviation=float(abs(b_rank - ref_baseline_rank)),
                    total_services=total_services,
                )
                trial_results.append(t_baro)
                metric_trials_by_lambda[0.0].append(t_baro)

                sub_corr: pd.DataFrame = compute_latency_correlation_matrix(sub_data)
                G_sub: nx.DiGraph = annotate_graph_with_telemetry(G_base, sub_corr)

                for lambd in config.lambdas:
                    if lambd == 0.0:
                        continue
                    gc_s: Dict[str, float] = self.compute_graphcut_min_marginals(
                        G=G_sub,
                        d0=d0_sub,
                        d1=d1_sub,
                        lambd=lambd,
                        rho=config.rho,
                    )
                    gc_m_ranks: List[str] = [
                        k
                        for k, _ in sorted(
                            gc_s.items(), key=lambda item: item[1], reverse=True
                        )
                    ]
                    gc_svcs: List[str] = self.extract_service_ranks(
                        gc_m_ranks, all_nodes
                    )
                    g_rank: int = self.find_root_cause_rank(
                        gc_svcs, instance.root_cause, all_nodes
                    )

                    t_gc: RCASampleTrialResult = RCASampleTrialResult(
                        instance_name=instance_name,
                        suite=instance.suite.value,
                        dataset=instance.dataset,
                        fault=instance.fault,
                        root_cause=instance.root_cause,
                        dimension_type=RCASamplingDimension.METRIC_DROPOUT.value,
                        dimension_value=float(frac),
                        trial_idx=trial_i,
                        lambda_val=lambd,
                        rho_val=config.rho,
                        strategy=config.strategy.value,
                        arm_name=f"BARO_GC_lambd_{lambd}",
                        predicted_rank=g_rank,
                        is_top_1=(g_rank == 1),
                        is_top_3=(g_rank <= 3),
                        is_top_5=(g_rank <= 5),
                        mrr=float(1.0 / g_rank),
                        full_window_baseline_rank=ref_baseline_rank,
                        rank_deviation=float(abs(g_rank - ref_baseline_rank)),
                        total_services=total_services,
                    )
                    trial_results.append(t_gc)
                    metric_trials_by_lambda[lambd].append(t_gc)

            # Summarize Metric Dropout trials
            for lambd, list_trials in metric_trials_by_lambda.items():
                ranks_arr: np.ndarray = np.array(
                    [r.predicted_rank for r in list_trials], dtype=float
                )
                mrrs_arr: np.ndarray = np.array(
                    [r.mrr for r in list_trials], dtype=float
                )
                devs_arr: np.ndarray = np.array(
                    [r.rank_deviation for r in list_trials], dtype=float
                )

                arm_label: str = (
                    "BARO_baseline" if lambd == 0.0 else f"BARO_GC_lambd_{lambd}"
                )
                sum_row: RCASensitivitySummaryRow = RCASensitivitySummaryRow(
                    instance_name=instance_name,
                    dimension_type=RCASamplingDimension.METRIC_DROPOUT.value,
                    dimension_value=float(frac),
                    lambda_val=lambd,
                    arm_name=arm_label,
                    n_trials=len(list_trials),
                    full_window_baseline_rank=ref_baseline_rank,
                    mean_rank=float(np.mean(ranks_arr)),
                    median_rank=float(np.median(ranks_arr)),
                    std_rank=(
                        float(np.std(ranks_arr, ddof=1))
                        if len(ranks_arr) > 1
                        else 0.0
                    ),
                    var_rank=(
                        float(np.var(ranks_arr, ddof=1))
                        if len(ranks_arr) > 1
                        else 0.0
                    ),
                    mean_deviation_from_reference=float(np.mean(devs_arr)),
                    top_1_rate=float(np.mean(ranks_arr == 1.0)),
                    top_3_rate=float(np.mean(ranks_arr <= 3.0)),
                    top_5_rate=float(np.mean(ranks_arr <= 5.0)),
                    mean_mrr=float(np.mean(mrrs_arr)),
                    total_services=total_services,
                )
                summary_rows.append(sum_row)

        return (trial_results, summary_rows)
