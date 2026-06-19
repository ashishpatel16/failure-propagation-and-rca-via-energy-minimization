import pytest
import networkx as nx
import sys
import os
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(os.getcwd()) / "src"))

from cut_algorithms.boykov_jolly import BoykovJollyCut
from visualization.energy_plots import plot_energy_vs_probability_lambdas
from visualization.graph_plots import visualize_graph, plot_json_graph
from benchmarking.d4j_manager import D4JManager
from benchmarking.gzoltar_manager import GZoltarRunner

def test_boykov_jolly_init():
    # BoykovJollyCut requires lambd to be passed explicitly now
    G = nx.DiGraph()
    G.add_node("SOURCE")
    G.add_node("TERMINAL")
    scores = {"SOURCE": 0.5, "TERMINAL": 0.5}
    
    # Should work without exception if we provide lambd explicitly
    algo = BoykovJollyCut(G, scores, 1.0)
    
    # Check that building the graph requires explicit args
    with pytest.raises(TypeError):
        # Should raise TypeError because positional arguments constrained_node and constrained_label are missing
        algo.build_st_graph()
        
    G_cut = algo.build_st_graph(None, None)
    assert isinstance(G_cut, nx.DiGraph)

def test_graph_plots_signatures():
    G = nx.DiGraph()
    
    with pytest.raises(TypeError):
        # Should raise TypeError because save_prefix is missing
        visualize_graph(G, "Test")

def test_energy_plots_signatures():
    with pytest.raises(TypeError):
        # Should raise TypeError because temperature is missing
        plot_energy_vs_probability_lambdas([1.0], {1.0: [1, 2]})

def test_managers_init():
    # Just verify they can be instantiated without errors
    d4j = D4JManager("/fake/path", "/fake/java")
    gzoltar = GZoltarRunner("/fake/gzoltar", "/fake/java")
    
    assert d4j is not None
    assert gzoltar is not None
