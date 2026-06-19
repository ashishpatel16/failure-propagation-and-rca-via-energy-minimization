import json
import pandas as pd
import sys
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from graph_generation.sbfl.call_graph_parser import parse_raw_call_graph
from benchmarking.sbfl_metrics import compute_sbfl_scores_from_csv
from graph_generation.sbfl.call_graph_parser import save_to_json

@pytest.fixture
def example_1_paths():
    example_dir = ROOT_DIR / "data" / "synthetic_examples" / "example_1"
    return {
        "gt": example_dir / "ground_truth.json",
        "call_graph": example_dir / "call_graph.txt",
        "coverage": example_dir / "coverage_matrix.csv"
    }

@pytest.fixture
def ground_truth(example_1_paths):
    with open(example_1_paths["gt"], 'r') as f:
        return json.load(f)

def test_example_1_files_exist(example_1_paths):
    """Ensure all required input files for the synthetic example exist."""
    assert example_1_paths["gt"].exists(), "Ground truth missing"
    assert example_1_paths["call_graph"].exists(), "Call graph text missing"
    assert example_1_paths["coverage"].exists(), "Coverage matrix missing"

def test_example_1_call_graph_parsing(example_1_paths, ground_truth):
    """Test that the graph parser extracts and filters nodes correctly."""
    parsed_graph = parse_raw_call_graph(str(example_1_paths["call_graph"]))
    
    # Filter test nodes as the extractor does
    nodes = [n for n in parsed_graph.nodes if "Test" not in n]
    edges = [e for e in parsed_graph.edges if "Test" not in e.caller and "Test" not in e.callee]
    
    gt_nodes = set(ground_truth['nodes'])
    parsed_nodes_set = set(nodes)
    
    assert gt_nodes.issubset(parsed_nodes_set), "Parsed graph is missing expected ground truth nodes"

def test_example_1_sbfl_computation_and_ranking(example_1_paths, ground_truth):
    """Test that SBFL metrics are computed accurately and buggy methods are ranked highest."""
    method_df = compute_sbfl_scores_from_csv(str(example_1_paths["coverage"]))
    
    assert not method_df.empty, "SBFL metrics dataframe should not be empty"
    assert 'Method' in method_df.columns
    assert 'Ochiai' in method_df.columns
    
    buggy_methods = ground_truth['buggy_methods']
    
    # Sort by Ochiai descending
    sorted_df = method_df.sort_values(by="Ochiai", ascending=False)
    top_method = sorted_df.iloc[0]['Method']
    top_score = sorted_df.iloc[0]['Ochiai']
    
    # Strip parameter signatures for comparison
    top_method_base = top_method.split('(')[0]
    assert top_method_base in buggy_methods, f"Expected buggy method {buggy_methods} to be ranked 1st, but got {top_method_base} with score {top_score}"
    
    buggy_row = method_df[method_df['Method'].apply(lambda m: m.split('(')[0]) == buggy_methods[0]]
    assert not buggy_row.empty, "Buggy method not found in SBFL output"
    assert buggy_row.iloc[0]['Ochiai'] > 0, "Ochiai score of buggy method should be strictly positive"

def test_call_graph_parser_validation(tmp_path):
    """Test that the parser enforces strict format validation and throws ValueError on malformed lines."""
    malformed_txt = tmp_path / "malformed.txt"
    
    # 1. Missing frequency
    malformed_txt.write_text("com.example.App#A -> com.example.App#B\n")
    with pytest.raises(ValueError, match="Malformed edge"):
        parse_raw_call_graph(str(malformed_txt))
        
    # 2. Missing arrow
    malformed_txt.write_text("com.example.App#A com.example.App#B : 5\n")
    with pytest.raises(ValueError, match="Malformed edge"):
        parse_raw_call_graph(str(malformed_txt))
        
    # 3. Invalid frequency
    malformed_txt.write_text("com.example.App#A -> com.example.App#B : five\n")
    with pytest.raises(ValueError, match="Invalid frequency count"):
        parse_raw_call_graph(str(malformed_txt))
        
    # 4. Empty caller/callee
    malformed_txt.write_text(" -> com.example.App#B : 5\n")
    with pytest.raises(ValueError, match="Empty caller or callee"):
        parse_raw_call_graph(str(malformed_txt))

def test_call_graph_serialization(example_1_paths, tmp_path):
    """Test that the CallGraph dataclass correctly serializes to dict and JSON."""
    parsed_graph = parse_raw_call_graph(str(example_1_paths["call_graph"]))
    
    graph_dict = parsed_graph.to_dict()
    assert "metadata" in graph_dict
    assert "nodes" in graph_dict
    assert "edges" in graph_dict
    assert isinstance(graph_dict["edges"], list)
    assert len(graph_dict["edges"]) > 0
    assert "caller" in graph_dict["edges"][0]
    
    out_json = tmp_path / "out_graph.json"
    save_to_json(parsed_graph, str(out_json))
    
    assert out_json.exists()
    with open(out_json, "r") as f:
        loaded = json.load(f)
        assert loaded["metadata"]["total_nodes"] == parsed_graph.metadata.total_nodes

def test_example_1_explicit_graph_structure(example_1_paths):
    """
    Manually verifies that every specific node and edge (with exact frequency) 
    that should logically exist based on App.java and AppTest.java is present.
    """
    parsed_graph = parse_raw_call_graph(str(example_1_paths["call_graph"]))
    
    # Expected Nodes (Test nodes are now filtered by the parser itself)
    expected_nodes = {
        "com.example.App#mainProcess",
        "com.example.App#step1",
        "com.example.App#step2",
        "com.example.App#utilA",
        "com.example.App#utilB"
    }
    
    # Expected Edges: (caller, callee) -> frequency (Test edges are now filtered)
    expected_edges = {
        ("com.example.App#mainProcess", "com.example.App#step1"): 2,
        ("com.example.App#mainProcess", "com.example.App#step2"): 2,
        ("com.example.App#step1", "com.example.App#utilA"): 3,
        ("com.example.App#step2", "com.example.App#utilB"): 2
    }
    
    parsed_nodes = set(parsed_graph.nodes)
    assert parsed_nodes == expected_nodes, f"Parsed nodes do not match expected. Missing: {expected_nodes - parsed_nodes}"
    
    parsed_edges_dict = {(e.caller, e.callee): e.frequency for e in parsed_graph.edges}
    
    assert len(parsed_edges_dict) == len(expected_edges), "Number of edges does not match expected."
    
    for (caller, callee), expected_freq in expected_edges.items():
        assert (caller, callee) in parsed_edges_dict, f"Missing edge: {caller} -> {callee}"
        actual_freq = parsed_edges_dict[(caller, callee)]
        assert actual_freq == expected_freq, f"Edge {caller} -> {callee} expected frequency {expected_freq}, got {actual_freq}"

