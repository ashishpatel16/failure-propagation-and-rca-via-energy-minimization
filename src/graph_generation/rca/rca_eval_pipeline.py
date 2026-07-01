import json
import math
from pathlib import Path
from collections import Counter

import pandas as pd
import networkx as nx


ROOT_PARENT_VALUES = {"", "0", "none", "null", "nan", "<na>"}


def norm_id(x):
    if pd.isna(x):
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x).strip()


def norm_parent(x):
    x = norm_id(x)
    return "" if x.casefold() in ROOT_PARENT_VALUES else x

def load_metric_services(instance_dir: Path) -> set[str]:
    path = instance_dir / "simple_metrics.csv"
    if not path.exists():
        path = instance_dir / "metrics.csv"
    if not path.exists():
        return set()

    columns = pd.read_csv(path, nrows=0).columns
    services = set()
    for column in columns:
        if column == "time" or "_" not in column:
            continue

        if path.name == "simple_metrics.csv":
            service = column.rsplit("_", 1)[0]
        else:
            service = column.split("_", 1)[0]

        if service:
            services.add(service)

    return services


def load_traces(instance_dir: Path) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(instance_dir / "traces.csv")

    required = {"traceID", "spanID", "parentSpanID", "serviceName", "startTime", "duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing trace columns: {missing}")

    df = df.copy()
    df["trace_id"] = df["traceID"].map(norm_id)
    df["span_id"] = df["spanID"].map(norm_id)
    df["parent_id"] = df["parentSpanID"].map(norm_parent)
    df["node"] = df["serviceName"].map(norm_id)

    df["start"] = pd.to_numeric(df["startTime"])
    df["duration"] = pd.to_numeric(df["duration"])
    df["end"] = df["start"] + df["duration"]

    # Drop duplicate spans just in case the tracing backend exported duplicates
    df = df.drop_duplicates(subset=["trace_id", "span_id"])
    
    # Drop rows where service name could not be resolved
    df = df[df["node"] != ""]

    injection_time = float((instance_dir / "inject_time.txt").read_text().strip())

    # Simple unit alignment: if traces look like ms/us/ns while inject_time is seconds.
    median_start = df["start"].abs().median()
    if injection_time != 0 and median_start != 0:
        for factor in [1, 1_000, 1_000_000, 1_000_000_000]:
            if abs(math.log10(median_start / (injection_time * factor))) < 0.25:
                injection_time *= factor
                break

    return df, injection_time

def split_traces(df: pd.DataFrame, injection_time: float):
    # Find by trace ids, what was their exact duration
    trace_bounds = df.groupby("trace_id").agg(
        start=("start", "min"),
        end=("end", "max"),
    )

    return {
        "pre": set(trace_bounds[trace_bounds["end"] < injection_time].index),
        "fault": set(trace_bounds[trace_bounds["end"] >= injection_time].index),
        "runtime": set(trace_bounds.index),
    }

def build_service_graph(df: pd.DataFrame, trace_ids: set[str], metric_services: set[str] | None = None) -> dict:
    selected = df[df["trace_id"].isin(trace_ids)]

    span_to_service = {
        (r.trace_id, r.span_id): r.node
        for r in selected.itertuples()
    }

    edges = Counter()
    root_spans = 0
    unmatched_parents = 0
    self_loops = 0

    for r in selected.itertuples():
        if not r.parent_id:
            root_spans += 1
            continue

        parent_key = (r.trace_id, r.parent_id)

        if parent_key not in span_to_service:
            unmatched_parents += 1
            continue

        parent_service = span_to_service[parent_key]
        child_service = r.node

        if parent_service == child_service:
            self_loops += 1
            continue

        edges[(parent_service, child_service)] += 1

    trace_nodes = set(selected["node"].unique())
    metric_services = metric_services or set()
    nodes = sorted(trace_nodes | metric_services)

    return {
        "metadata": {
            "traces": selected["trace_id"].nunique(),
            "spans": len(selected),
            "nodes": len(nodes),
            "trace_nodes": len(trace_nodes),
            "metric_nodes": len(metric_services),
            "edges": len(edges),
            "root_spans": root_spans,
            "unmatched_parents": unmatched_parents,
            "self_loops_omitted": self_loops,
            "nodes_without_trace_spans": sorted(metric_services - trace_nodes),
        },
        "nodes": nodes,
        "edges": [
            {
                "source": u,
                "target": v,
                "weight": int(w),
            }
            for (u, v), w in sorted(edges.items())
        ],
    }

def extract_graphs(instance_dir: Path) -> dict[str, dict]:
    df, injection_time = load_traces(instance_dir)
    windows = split_traces(df, injection_time)
    metric_services = load_metric_services(instance_dir)

    return {
        name: build_service_graph(df, trace_ids, metric_services)
        for name, trace_ids in windows.items()
    }

def save_graphs(instance_dir: Path, graphs: dict[str, dict], overwrite: bool = False):
    for name, graph in graphs.items():
        path = instance_dir / f"trace_service_graph_{name}.json"

        if path.exists() and not overwrite:
            raise FileExistsError(path)

        path.write_text(json.dumps(graph, indent=2) + "\n")

def to_networkx(graph_json: dict) -> nx.DiGraph:
    G = nx.DiGraph()

    for node in graph_json["nodes"]:
        G.add_node(node)

    for edge in graph_json["edges"]:
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])

    return G

def list_instances(data_dir: Path, suites=("RE2", "RE3")) -> list[Path]:
    """
    Find all RCAEval instance directories under data_dir.

    An instance is any directory containing both:
    - traces.csv
    - inject_time.txt
    """
    data_dir = Path(data_dir)
    instances = []

    for suite in suites:
        suite_dir = data_dir / suite

        if not suite_dir.exists():
            print(f"Skipping missing suite directory: {suite_dir}")
            continue

        for trace_file in suite_dir.rglob("traces.csv"):
            instance_dir = trace_file.parent

            if (instance_dir / "inject_time.txt").exists():
                instances.append(instance_dir)

    return sorted(instances)


def extract_graphs_for_all_instances(
    data_dir: Path,
    suites=("RE2", "RE3"),
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Extract pre/fault/runtime service graphs for all available RCAEval instances.

    Returns a manifest dataframe with status and graph sizes.
    """
    instances = list_instances(data_dir, suites=suites)
    records = []

    for instance_dir in instances:
        record = {
            "instance": str(instance_dir),
            "status": "ok",
            "error": "",
        }

        try:
            graphs = extract_graphs(instance_dir)
            save_graphs(instance_dir, graphs, overwrite=overwrite)

            for name, graph in graphs.items():
                record[f"{name}_nodes"] = graph["metadata"]["nodes"]
                record[f"{name}_edges"] = graph["metadata"]["edges"]
                record[f"{name}_traces"] = graph["metadata"]["traces"]
                record[f"{name}_spans"] = graph["metadata"]["spans"]

        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)

        records.append(record)

    return pd.DataFrame(records)


if __name__ == '__main__':
    from pathlib import Path
    data_dir = Path("/Users/ashish/master-thesis/Master-Thesis/failure-propagation-and-rca-via-energy-minimization/notebooks/data")
    instances = list_instances(data_dir)
    print(f"Found {len(instances)} instances")

    manifest = extract_graphs_for_all_instances(
        data_dir=data_dir,
        suites=("RE2", "RE3"),
        overwrite=True,
    )

    manifest.to_csv("graph_extraction_manifest.csv", index=False)

    manifest.head()
