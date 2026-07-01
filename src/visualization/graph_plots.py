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
    
    # Remove cycles to ensure DAG for topological sort
    DAG = H.copy()
    while True:
        try:
            cycle = nx.find_cycle(DAG)
            DAG.remove_edge(cycle[-1][0], cycle[-1][1])
        except nx.NetworkXNoCycle:
            break
            
    for node in nx.topological_sort(DAG):
        for succ in DAG.successors(node):
            layer[succ] = max(layer[succ], layer[node] + 1)

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
            if n in buggy_set:
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
        # plt.savefig(f"{save_prefix}.png", format="png", dpi=300, bbox_inches='tight')
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
def plot_rca_json_graph(json_path: str, title: str, save_prefix: Optional[str] = None, root_cause: Optional[str] = None) -> None:
    """Reads an RCA trace service graph JSON and visualizes it."""
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
            source = e.get('source')
            target = e.get('target')
            if not source or not target:
                continue
            G.add_edge(source, target, weight=e.get('weight', 1))
            
    buggy_methods = [root_cause] if root_cause else None
    visualize_graph(G, title, save_prefix=save_prefix, layout_type="hierarchical", buggy_methods=buggy_methods)
    if save_prefix:
        visualize_graph(G, f"{title} (Kamada-Kawai)", save_prefix=f"{save_prefix}_kamada", layout_type="kamada_kawai", buggy_methods=buggy_methods)

def get_horizontal_hierarchical_pos(G):
    """
    Computes a left-to-right hierarchical layout that is spread out using the entire space.
    - Normal nodes are ordered left-to-right by topological level.
    - Y-coordinates are generated using a force-directed layout to minimize edge crossings and use space optimally.
    - SOURCE is placed at the extreme left.
    - TERMINAL is placed at the extreme right.
    """
    normal_nodes = [n for n in G.nodes if n not in ["SOURCE", "TERMINAL"]]
    H = G.subgraph(normal_nodes)
    
    # 1. Get organic coordinates from Kamada-Kawai for Y-axis spread
    organic_pos = nx.kamada_kawai_layout(H)
    
    # 2. Get hierarchical x-coordinates
    layer = {n: 0 for n in H.nodes()}
    # Remove cycles to ensure DAG for topological sort
    DAG = H.copy()
    while True:
        try:
            cycle = nx.find_cycle(DAG)
            DAG.remove_edge(cycle[-1][0], cycle[-1][1])
        except nx.NetworkXNoCycle:
            break
            
    for node in nx.topological_sort(DAG):
        for succ in DAG.successors(node):
            layer[succ] = max(layer[succ], layer[node] + 1)
            
    max_layer = max(layer.values()) if layer else 0
    
    pos = {}
    x_spacing = 25.0  # Increased horizontal spacing significantly
    y_min_spacing = 4.0  # Minimum vertical distance between nodes in the same layer
    
    # Group nodes by layer
    layers_dict = {}
    for n in normal_nodes:
        layers_dict.setdefault(layer[n], []).append(n)
        
    for l, nodes_in_layer in layers_dict.items():
        # Sort nodes by organic Y coordinate to preserve Kamada-Kawai topology
        nodes_in_layer.sort(key=lambda n: organic_pos[n][1] if organic_pos else 0)
        
        # Start with scaled organic Ys
        if organic_pos:
            y_vals = [organic_pos[n][1] * 15.0 for n in nodes_in_layer]
        else:
            y_vals = [0.0] * len(nodes_in_layer)
            
        # 1D Overlap Removal: force minimum vertical spacing
        for i in range(1, len(nodes_in_layer)):
            if y_vals[i] - y_vals[i-1] < y_min_spacing:
                y_vals[i] = y_vals[i-1] + y_min_spacing
                
        # Center the layer around Y=0
        if y_vals:
            center_y = (max(y_vals) + min(y_vals)) / 2.0
            y_vals = [y - center_y for y in y_vals]
            
        # Assign coordinates with a tiny X jitter
        for i, n in enumerate(nodes_in_layer):
            jitter_x = np.random.uniform(-1.0, 1.0)
            x = l * x_spacing + jitter_x
            y = y_vals[i]
            pos[n] = np.array([x, y])
            
    if "SOURCE" in G.nodes:
        pos["SOURCE"] = np.array([-x_spacing * 1.5, 0])
    if "TERMINAL" in G.nodes:
        pos["TERMINAL"] = np.array([(max_layer + 1.5) * x_spacing, 0])
        
    return pos

import matplotlib.patches as mpatches

def visualize_academic_call_graph(G: nx.DiGraph, title: str, save_prefix: str, buggy_methods: list = None) -> None:
    """
    Very academic left-to-right hierarchical call graph visualization.
    """
    setup_academic_plot_params()
    # Calculate a wide figure width to accommodate the increased horizontal spacing
    pos = get_horizontal_hierarchical_pos(G)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    num_layers = (max(xs) - min(xs)) / 25.0 if xs else 1
    max_height = (max(ys) - min(ys)) if ys else 14
    
    fig_width = max(36, int(num_layers * 10))
    fig_height = max(16, int(max_height / 1.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")
    
    buggy_nodes = []
    new_normal = []
    
    if buggy_methods:
        buggy_set = set(buggy_methods)
        for n in G.nodes:
            if n in buggy_set:
                buggy_nodes.append(n)
            else:
                new_normal.append(n)
    else:
        new_normal = list(G.nodes)
        
    if new_normal:
        nx.draw_networkx_nodes(G, pos, nodelist=new_normal, 
                               node_color="white", node_shape='o', node_size=1000, 
                               edgecolors='black', linewidths=1.5, ax=ax)
    if buggy_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=buggy_nodes, 
                               node_color="#ffcccc", node_shape='o', node_size=1200, 
                               edgecolors='#cc0000', linewidths=2.5, ax=ax)
                               
    labels = {}
    for n in G.nodes():
        name = str(n).split('#')[-1].split('(')[0].split('.')[-1]
        labels[n] = name
        
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family="serif", ax=ax)
    
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.2, arrowstyle='-|>', 
                           arrowsize=15, node_size=1000, ax=ax, connectionstyle="arc3,rad=0.1")
                           
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        if 'weight' in d:
            edge_labels[(u, v)] = f"{int(d['weight'])}"
            
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_family="serif", ax=ax)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='black', label='Normal Method'),
        mpatches.Patch(facecolor='#ffcccc', edgecolor='#cc0000', label='Buggy Method')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True)
    
    plt.title(title, pad=20, fontweight="bold", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.svg", format="svg", bbox_inches='tight')
    plt.close()

def visualize_academic_st_graph(G: nx.DiGraph, title: str, save_prefix: str, buggy_methods: list = None, flow_dict: dict = None) -> None:
    """
    Very academic left-to-right hierarchical s-t graph visualization with flow/capacity format.
    """
    setup_academic_plot_params()
    pos = get_horizontal_hierarchical_pos(G)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    num_layers = (max(xs) - min(xs)) / 25.0 if xs else 1
    max_height = (max(ys) - min(ys)) if ys else 14
    
    fig_width = max(40, int(num_layers * 12))
    fig_height = max(16, int(max_height / 1.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")
    
    terminal_nodes = [n for n in ["SOURCE", "TERMINAL"] if n in G.nodes]
    normal_nodes = [n for n in G.nodes if n not in terminal_nodes]
    
    buggy_nodes = []
    new_normal = []
    
    if buggy_methods:
        buggy_set = set(buggy_methods)
        for n in normal_nodes:
            if n in buggy_set:
                buggy_nodes.append(n)
            else:
                new_normal.append(n)
    else:
        new_normal = normal_nodes
        
    # Draw Terminals
    if terminal_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=terminal_nodes, 
                               node_color="#e0e0e0", node_shape='o', node_size=1200, 
                               edgecolors='#808080', linewidths=2.0, ax=ax)
    # Draw normal
    if new_normal:
        nx.draw_networkx_nodes(G, pos, nodelist=new_normal, 
                               node_color="white", node_shape='o', node_size=1000, 
                               edgecolors='black', linewidths=1.5, ax=ax)
    # Draw buggy
    if buggy_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=buggy_nodes, 
                               node_color="#ffcccc", node_shape='o', node_size=1200, 
                               edgecolors='#cc0000', linewidths=2.5, ax=ax)
                               
    labels = {}
    for n in G.nodes():
        if n in terminal_nodes:
            labels[n] = str(n)
        else:
            name = str(n).split('#')[-1].split('(')[0].split('.')[-1]
            D0 = G.nodes[n].get('D0')
            D1 = G.nodes[n].get('D1')
            lbl = name
            if D0 is not None and D1 is not None:
                lbl += f"\nD0:{D0:.1f}\nD1:{D1:.1f}"
            labels[n] = lbl
            
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_family="serif", ax=ax)
    
    # Draw edges
    terminal_edges = [(u, v) for u, v in G.edges() if u in terminal_nodes or v in terminal_nodes]
    normal_edges = [(u, v) for u, v in G.edges() if u not in terminal_nodes and v not in terminal_nodes]
    
    if normal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, style='solid', edge_color="#666666", 
                               width=1.5, arrowstyle='-|>', arrowsize=15, node_size=1000, ax=ax, connectionstyle="arc3,rad=0.1")
    if terminal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=terminal_edges, style='dashed', edge_color="#999999", 
                               width=1.2, arrowstyle='-|>', arrowsize=15, node_size=1200, ax=ax, connectionstyle="arc3,rad=0.05")
                               
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        cap = d.get('capacity', 0)
        flow = 0
        if flow_dict and u in flow_dict and v in flow_dict[u]:
            flow = flow_dict[u][v]
        
        cap_str = f"{cap:.1f}" if isinstance(cap, float) and cap != float('inf') else str(cap)
        flow_str = f"{flow:.1f}" if isinstance(flow, float) else str(flow)
        edge_labels[(u, v)] = f"{flow_str} / {cap_str}"
        
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_family="serif", ax=ax)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#e0e0e0', edgecolor='#808080', label='Source/Terminal (s-t)'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Normal Method Node'),
        mpatches.Patch(facecolor='#ffcccc', edgecolor='#cc0000', label='Buggy Method Node')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True)
    
    plt.title(title, pad=20, fontweight="bold", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.svg", format="svg", bbox_inches='tight')
    plt.close()
