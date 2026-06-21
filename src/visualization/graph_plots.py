import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

COLORS = {
    "normal": "#FFFFFF",      # Sleek academic white
    "terminal": "#E5E5E5",    # Neutral terminal gray
    "edge": "#444444", 
}

def setup_academic_plot_params():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

def get_hierarchical_pos(G):
    """
    Computes a hierarchical layout:
    - Normal nodes are ordered left-to-right (callers on left, callees on right).
    - SOURCE is placed at the extreme top.
    - TERMINAL is placed at the extreme bottom.
    """
    normal_nodes = [n for n in G.nodes if n not in ["SOURCE", "TERMINAL"]]
    H = G.subgraph(normal_nodes)
    
    layer = {n: 0 for n in H.nodes()}
    
    # Try topological sort to find depth, fallback to shortest paths if cyclic
    try:
        for node in nx.topological_sort(H):
            for succ in H.successors(node):
                layer[succ] = max(layer[succ], layer[node] + 1)
    except nx.NetworkXUnfeasible:
        in_degree = dict(H.in_degree())
        roots = [n for n, d in in_degree.items() if d == 0]
        if not roots and H.nodes:
            roots = [list(H.nodes())[0]]
        lengths = nx.multi_source_dijkstra_path_length(H, roots)
        for n, d in lengths.items():
            layer[n] = d

    layers = {}
    for n, l in layer.items():
        layers.setdefault(l, []).append(n)
        
    pos = {}
    max_layer = max(layers.keys()) if layers else 0
    
    for l, nodes in layers.items():
        x_spacing = 3.0
        y_spacing = 1.0
        x = l * x_spacing
        y_start = (len(nodes) - 1) * y_spacing / 2.0
        for i, node in enumerate(nodes):
            y = y_start - i * y_spacing
            jitter = 0.0
            if len(nodes) > 1:
                jitter = (i % 3 - 1) * 0.5
            pos[node] = np.array([x + jitter, y])
            
    # Position SOURCE and TERMINAL
    if "SOURCE" in G.nodes or "TERMINAL" in G.nodes:
        all_y = [p[1] for p in pos.values()] if pos else [0]
        max_y = max(all_y) if all_y else 0
        min_y = min(all_y) if all_y else 0
        mid_x = (max_layer * 3.0) / 2.0 if max_layer else 0
        
        if "SOURCE" in G.nodes:
            pos["SOURCE"] = np.array([mid_x, max_y + 2.0])
        if "TERMINAL" in G.nodes:
            pos["TERMINAL"] = np.array([mid_x, min_y - 2.0])
            
    return pos

def visualize_graph(G: nx.DiGraph, title: str, save_prefix: Optional[str] = None, layout_type: str = "hierarchical", buggy_methods: Optional[list] = None) -> None:
    """
    Plots a simple NetworkX graph showing topology and nodes.
    If save_prefix is provided, saves as .png and .svg instead of showing interactively.
    """
    if not isinstance(G, nx.DiGraph):
        raise ValueError("G must be a nx.DiGraph")
    if not isinstance(title, str):
        raise ValueError("title must be a string")
        
    setup_academic_plot_params()
    fig_width = max(16, len(G.nodes) // 5)
    fig_height = max(12, len(G.nodes) // 5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")
    
    if layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
        for node in pos:
            pos[node] += np.random.uniform(-0.15, 0.15, 2)
    else:
        pos = get_hierarchical_pos(G)
    
    terminal_nodes = [n for n in G.nodes if n in ["SOURCE", "TERMINAL"]]
    normal_nodes = [n for n in G.nodes if n not in ["SOURCE", "TERMINAL"]]
    
    buggy_nodes = []
    if buggy_methods:
        buggy_set = set(buggy_methods)
        new_normal = []
        for n in normal_nodes:
            node_base = n.split('(')[0] if isinstance(n, str) else str(n)
            if node_base in buggy_set:
                buggy_nodes.append(n)
            else:
                new_normal.append(n)
        normal_nodes = new_normal
    
    if normal_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=normal_nodes, 
            node_color=COLORS["normal"], node_shape='o', 
            node_size=300, edgecolors='#333333', linewidths=1.0, ax=ax
        )
        
    if buggy_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=buggy_nodes, 
            node_color="#ff9999", node_shape='o', 
            node_size=350, edgecolors='#cc0000', linewidths=2.0, ax=ax
        )
        
    if terminal_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=terminal_nodes, 
            node_color=COLORS["terminal"], node_shape='o', 
            node_size=500, edgecolors='#444444', linewidths=1.5, ax=ax
        )
    
    labels = {}
    for n in G.nodes():
        if n in ["SOURCE", "TERMINAL"]:
            labels[n] = str(n)
        else:
            name = str(n).split('#')[-1]
            name = name.split('(')[0]
            name = name.split('.')[-1]
            
            D0 = G.nodes[n].get('D0')
            D1 = G.nodes[n].get('D1')
            if D1 == float('inf'):
                labels[n] = f"{name}\n(D0:{D0:.2f}, D1:∞)"
            elif D1 is not None and D0 is not None:
                labels[n] = f"{name}\n(D0:{D0:.2f}, D1:{D1:.2f})"
            else:
                labels[n] = name
            
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8.0, font_family="serif", ax=ax)
    
    terminal_edges = [(u, v) for u, v in G.edges() if u in ["SOURCE", "TERMINAL"] or v in ["SOURCE", "TERMINAL"]]
    normal_edges = [(u, v) for u, v in G.edges() if u not in ["SOURCE", "TERMINAL"] and v not in ["SOURCE", "TERMINAL"]]
    
    if normal_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=normal_edges, style='solid', edge_color=COLORS["edge"], 
            width=1.0, arrowstyle='-|>', arrowsize=10, 
            node_size=300, ax=ax, connectionstyle="arc3,rad=0.1"
        )
        
    if terminal_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=terminal_edges, style='dashed', edge_color=COLORS["edge"], 
            width=1.0, arrowstyle='-|>', arrowsize=10, 
            node_size=300, ax=ax, alpha=0.6, connectionstyle="arc3,rad=0.1"
        )
    
    edge_labels = {}
    for u, v in G.edges():
        weight = G[u][v]['weight']
        label_text = f"w:{int(weight)}"
        if 'pairwise' in G[u][v]:
            label_text += f"\np:{int(G[u][v]['pairwise'])}"
        elif 'capacity' in G[u][v]:
            label_text += f"\ncap:{G[u][v]['capacity']:.2f}"
        edge_labels[(u, v)] = label_text
        
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7.0, font_family="serif", alpha=0.8, ax=ax)
    
    plt.title(title, pad=15, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f"{save_prefix}.png", format="png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_prefix}.svg", format="svg", bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_json_graph(json_path: str, title: str, save_prefix: Optional[str] = None, buggy_methods: Optional[list] = None) -> None:
    """Reads a JSON call graph and visualizes it."""
    if not isinstance(json_path, str) or not isinstance(title, str):
        raise ValueError("json_path and title must be strings")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    G = nx.DiGraph()
    if 'nodes' in data:
        for n in data['nodes']:
            G.add_node(n)
            
    if 'edges' in data:
        for e in data['edges']:
            caller = e['caller']
            callee = e['callee']
            if not caller or not callee:
                continue
            G.add_edge(caller, callee, weight=e['frequency'])
            
    visualize_graph(G, title, save_prefix=save_prefix, layout_type="hierarchical", buggy_methods=buggy_methods)
    if save_prefix:
        visualize_graph(G, f"{title} (Kamada-Kawai)", save_prefix=f"{save_prefix}_kamada", layout_type="kamada_kawai", buggy_methods=buggy_methods)