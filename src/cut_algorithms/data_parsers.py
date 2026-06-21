import json
import networkx as nx
import pandas as pd
import re
from enum import Enum

class Granularity(Enum):
    SERVICE = "service"
    OPERATION = "operation"

def read_call_graph(filepath: str) -> nx.DiGraph:
    """
    Reads the call graph from a JSON file and returns a NetworkX DiGraph.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)
        
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for node in data["nodes"]:
        G.add_node(node)
        
    # Add edges with frequency as weight
    for edge in data["edges"]:
        caller = edge["caller"]
        callee = edge["callee"]
        if not caller or not callee:
            continue
        weight = edge["frequency"]
        G.add_edge(caller, callee, weight=weight)
        
    # Remove isolated nodes (nodes with 0 incoming and 0 outgoing edges)
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
        
    return G

def map_line_to_method(line_name: str) -> str:
    """Maps a coverage column like 'com.example$App#process(int):30' to 'com.example.App#process(int)'."""
    if "#" not in line_name:
        return line_name.replace('$', '.')
    return line_name.split(':')[0].replace('$', '.')

def compute_method_tarantula(coverage_csv: str) -> dict[str, float]:
    """
    Computes Tarantula suspiciousness score for each method based on the coverage matrix.
    Aggregates lines into methods using OR logic before computing the score.
    """
    df = pd.read_csv(coverage_csv)
    
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

class RCAEvalTraceParser:
    """
    Parses trace data from the RCAEval benchmark and reconstructs a NetworkX DiGraph.
    """
    def __init__(self, trace_csv_path: str) -> None:
        self.trace_csv_path = trace_csv_path

    def build_call_graph(self, granularity: Granularity) -> nx.DiGraph:
        """
        Builds a call graph from the traces.
        """
        df = pd.read_csv(self.trace_csv_path)
        
        # Determine node names
        if granularity == Granularity.OPERATION:
            if "methodName" in df.columns:
                df["methodName"] = df["methodName"].fillna(df["operationName"])
                df["node_name"] = df["serviceName"] + "_" + df["methodName"]
            else:
                df["node_name"] = df["serviceName"] + "_" + df["operationName"]
        elif granularity == Granularity.SERVICE:
            df["node_name"] = df["serviceName"]
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

        op_dict: dict[str, str] = {}  # spanID -> node_name
        child_dict: dict[str, list[str]] = {}  # parentSpanID -> list of child spanIDs
        
        for _, row in df.iterrows():
            span_id = row["spanID"]
            parent_span_id = row.get("parentSpanID", None)
            node_name = row["node_name"]
            
            op_dict[span_id] = node_name
            
            if pd.notna(parent_span_id) and parent_span_id != "":
                if parent_span_id not in child_dict:
                    child_dict[parent_span_id] = []
                child_dict[parent_span_id].append(span_id)
                
        G = nx.DiGraph()
        
        # Build edges based on parent-child relationships
        edge_frequencies: dict[tuple[str, str], int] = {}
        for parent_id, children in child_dict.items():
            if parent_id in op_dict:
                parent_node = op_dict[parent_id]
                for child_id in children:
                    if child_id in op_dict:
                        child_node = op_dict[child_id]
                        # Track frequencies for all distinct spans
                        edge = (parent_node, child_node)
                        edge_frequencies[edge] = edge_frequencies.get(edge, 0) + 1
                            
        for (u, v), freq in edge_frequencies.items():
            G.add_edge(u, v, weight=freq)
            
        # Ensure all seen nodes are added, even if they have no inter-service edges
        for node in op_dict.values():
            if not G.has_node(node):
                G.add_node(node)
                
        return G
