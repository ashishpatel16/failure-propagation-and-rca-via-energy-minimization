import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def setup_academic_plot_params():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

def plot_energy_vs_probability(energies):
    """
    Plots Probability P(L) vs Energy E(L).
    """
    setup_academic_plot_params()
    
    energies_arr = np.array(energies)
    
    # Filter out the 'infinity' fallbacks (1e9) to keep the plot simple
    valid_energies = energies_arr[energies_arr < 1e8]
    if len(valid_energies) == 0:
        return
        
    # Shift energies to prevent numerical issues
    min_e = np.min(valid_energies)
    shifted = valid_energies - min_e
    
    # Compute Gibbs probabilities
    probs = np.exp(-shifted)
    probs = probs / np.sum(probs)
    
    plt.figure(figsize=(8, 5))
    
    if len(valid_energies) > 1:
        ax2 = plt.gca().twinx()
        sns.kdeplot(
            x=valid_energies, 
            fill=True, 
            color="#3498db", 
            alpha=0.2,
            ax=ax2,
            linewidth=1.5
        )
        ax2.set_ylabel("Density of States (KDE)", fontsize=11, color="#2980b9")
        ax2.tick_params(axis='y', colors='#2980b9')
        ax2.grid(False)  
        
    plt.scatter(valid_energies, probs, color='#2c3e50', alpha=0.8, zorder=3)
    
    plt.title("Gibbs Distribution: Probability vs Energy")
    plt.xlabel("Energy E(L)")
    plt.ylabel("Probability P(L)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def plot_energy_vs_probability_lambdas(lambdas, energy_map, temperature: float):
    """
    Computes energies for different lambda values and plots Probability P(L) vs Energy E(L)
    for each lambda on the same figure.
    
    Args:
        lambdas: List of lambda values.
        energy_map: Dict mapping lambda to a list of energies for all labelings.
        temperature: The temperature for the Gibbs distribution.
    """
    setup_academic_plot_params()
    plt.figure(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e']
    
    for i, lambd in enumerate(lambdas):
        energies = energy_map[lambd]
        energies_arr = np.array(energies)
        valid_energies = energies_arr[energies_arr < 1e8]
        if len(valid_energies) == 0:
            continue
            
        min_e = np.min(valid_energies)
        shifted = valid_energies - min_e
        probs = np.exp(-shifted / temperature)
        probs = probs / np.sum(probs)
        
        color = colors[i % len(colors)]
        plt.scatter(valid_energies, probs, color=color, alpha=0.6, label=f"λ={lambd}", zorder=3)
        
    plt.title(f"Gibbs Distribution: Probability vs Energy (T={temperature})")
    plt.xlabel("Energy E(L)")
    plt.ylabel("Probability P(L)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

import networkx as nx

def plot_3d_energy_landscape(G, cut_algo, optimal_labeling, output_path, layout_type: str = "hierarchical", buggy_methods: list = None, top_pred_node: str = None, metrics: dict = None):
    """
    Plots a 3D interactive energy landscape using Plotly.
    X and Y axes are the Kamada-Kawai layout (atomic lattice).
    Z axis is the individual energy contribution of each node.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Run: pip install plotly")
        return
        
    from visualization.graph_plots import get_hierarchical_pos
    if layout_type == "kamada_kawai":
        import numpy as np
        pos = nx.kamada_kawai_layout(G)
        for node in pos:
            pos[node] += np.random.uniform(-0.15, 0.15, 2)
    else:
        pos = get_hierarchical_pos(G)
    
    # Calculate energy per node
    node_energies = {}
    for node in G.nodes():
        l_i = optimal_labeling.get(node, 0)
        
        # Unary cost
        if l_i == 0:
            unary_cost = cut_algo.D0.get(node, 0)
        else:
            unary_cost = cut_algo.D1.get(node, 0)
            
        # 2. Pairwise cost (Edge Tension)
        pairwise_cost = 0.0
        
        # Use a set of neighbors to avoid double-counting bidirectional edges
        unique_neighbors = set(list(G.successors(node)) + list(G.predecessors(node)))
        
        for neighbor in unique_neighbors:
            l_j = optimal_labeling.get(neighbor, 0)
            
            if hasattr(cut_algo, 'enable_directional') and cut_algo.enable_directional:
                if l_i == 1 and l_j == 0:
                    pairwise_cost += 0.5 * cut_algo.lambd * cut_algo.C_uv.get((node, neighbor), 0.0)
                elif l_i == 0 and l_j == 1:
                    pairwise_cost += 0.5 * cut_algo.lambd * cut_algo.C_uv.get((neighbor, node), 0.0)
            else:
                if l_i != l_j:
                    edge_set = frozenset([node, neighbor])
                    if edge_set in cut_algo.C_ij:
                        # Distribute 50% of the cut penalty to this node
                        pairwise_cost += 0.5 * cut_algo.lambd * cut_algo.C_ij[edge_set]
                    
        # Total Energy = Unary + 50% of incident cuts
        node_energies[node] = unary_cost + pairwise_cost
        
    x_nodes = []
    y_nodes = []
    z_nodes = []
    hover_texts = []
    colors = []
    sizes = []
    line_colors = []
    line_widths = []
    
    max_energy = max(node_energies.values()) if node_energies else 1.0
    
    for node in G.nodes():
        x, y = pos[node]
        z = node_energies[node]
        x_nodes.append(x)
        y_nodes.append(y)
        z_nodes.append(z)
        
        is_buggy_pred = optimal_labeling.get(node, 0) == 1
        
        # Match node name (ignoring arguments) against buggy methods list
        node_base_name = node.split('(')[0] if isinstance(node, str) else str(node)
        is_actual_bug = buggy_methods is not None and node_base_name in buggy_methods
        
        if is_actual_bug:
            label_text = "Actual Bug"
            color = "#ff0000" # Pure bright red
            line_color = "#000000" # Black outline
            line_width = 3.0
            size = 18 + 25 * (z / max_energy) if max_energy > 0 else 22
        elif top_pred_node and node == top_pred_node:
            label_text = "Highest GC Score (Predicted)"
            color = "#f1c40f" # Yellow
            line_color = "#000000"
            line_width = 2.0
            size = 15 + 25 * (z / max_energy) if max_energy > 0 else 20
        elif is_buggy_pred:
            label_text = "Buggy (Predicted)"
            color = "#e74c3c" # Normal red
            line_color = "white"
            line_width = 1.5
            size = 10 + 25 * (z / max_energy) if max_energy > 0 else 15
        else:
            label_text = "Normal"
            color = "#3498db" # Blue
            line_color = "white"
            line_width = 1.5
            size = 10 + 25 * (z / max_energy) if max_energy > 0 else 15
        
        # Format the method signature to be readable
        short_name = node.split('#')[-1] if isinstance(node, str) else str(node)
        
        score_info = ""
        if metrics:
            t = metrics.get('tarantula', {}).get(node, 0.0)
            o = metrics.get('ochiai', {}).get(node, 0.0)
            d = metrics.get('dstar', {}).get(node, 0.0)
            score_info = f"<br>Tarantula: {t:.4f} | Ochiai: {o:.4f} | D*: {d:.4f}"
            
        unary_0 = cut_algo.D0.get(node, 0.0)
        unary_1 = cut_algo.D1.get(node, 0.0)
        reg_info = f"<br>D0 (Normal): {unary_0:.4f} | D1 (Buggy): {unary_1:.4f}"
        
        hover_texts.append(f"<b>{short_name}</b><br>Label: {label_text}<br>Total Energy: {z:.2f}{score_info}{reg_info}")
        colors.append(color)
        line_colors.append(line_color)
        line_widths.append(line_width)
        sizes.append(size)

    # Node trace
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=1.5, color=line_colors),
            opacity=0.95
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Methods (Atoms)'
    )
    
    # Edge trace
    edge_x = []
    edge_y = []
    edge_z = []
    
    edge_mid_x = []
    edge_mid_y = []
    edge_mid_z = []
    edge_hover_texts = []
    
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            z0 = node_energies[u]
            z1 = node_energies[v]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
            # Midpoint for hover text
            u_name = u.split('#')[-1] if isinstance(u, str) else str(u)
            v_name = v.split('#')[-1] if isinstance(v, str) else str(v)
            
            if hasattr(cut_algo, 'enable_directional') and cut_algo.enable_directional:
                c_down = cut_algo.C_uv.get((u, v), 0.0)
                c_up = cut_algo.C_uv.get((v, u), 0.0)
                coupling_text = f"Downstream: {c_down:.4f}<br>Upstream: {c_up:.4f}"
            else:
                edge_set = frozenset([u, v])
                coupling = cut_algo.C_ij.get(edge_set, 0.0)
                coupling_text = f"Coupling: {coupling:.4f}"
            
            edge_mid_x.append((x0 + x1) / 2)
            edge_mid_y.append((y0 + y1) / 2)
            edge_mid_z.append((z0 + z1) / 2)
            edge_hover_texts.append(f"{u_name} &#8596; {v_name}<br>{coupling_text}")
            
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(100, 100, 100, 0.25)', width=1.5),
        hoverinfo='none',
        name='Call Graph Edges'
    )
    
    edge_mid_trace = go.Scatter3d(
        x=edge_mid_x, y=edge_mid_y, z=edge_mid_z,
        mode='markers',
        marker=dict(size=3, color='rgba(200, 200, 200, 0.5)'),
        text=edge_hover_texts,
        hoverinfo='text',
        name='Edge Coupling'
    )
    
    fig = go.Figure(data=[edge_trace, edge_mid_trace, node_trace])
    
    fig.update_layout(
        title="3D Call Graph Energy Landscape",
        scene=dict(
            xaxis_title="Layout X",
            yaxis_title="Layout Y",
            zaxis_title="Energy E_i",
            xaxis=dict(showbackground=True, backgroundcolor='#e8e8e8', showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showbackground=True, backgroundcolor='#e8e8e8', showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showbackground=True, backgroundcolor='#e8e8e8', showgrid=True, gridcolor='white'),
            aspectratio=dict(x=1.2, y=1.2, z=0.8) # Prevent auto-squishing by forcing a comfortable bounding box
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5'
    )
    
    fig.write_html(str(output_path))
    png_path = str(output_path).replace('.html', '.png')
    try:
        fig.write_image(png_path)
        print(f"Saved 3D energy landscape to {output_path} and {png_path}")
    except ValueError as e:
        print(f"Saved 3D energy landscape to {output_path} (PNG export failed: install kaleido)")
