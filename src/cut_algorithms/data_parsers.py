import json
import networkx as nx
import pandas as pd
import re

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
