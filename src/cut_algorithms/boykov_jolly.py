import math
import networkx as nx
from typing import Dict, Tuple, List, Optional
import itertools
import pandas as pd

class BoykovJollyCut:
    """
    Boykov-Jolly Energy Minimization via s-t Graph Cuts.
    Optimizes the Gibbs energy E(L) = sum D_i(L_i) + lambda * sum V_ij(L_i, L_j)
    """
    
    def __init__(self, G: nx.DiGraph, sbfl_scores: Dict[str, float], lambd: float):
        """
        Initializes the Boykov-Jolly cut algorithm.
        
        Args:
            G: A NetworkX DiGraph representing the call graph (with 'weight' on edges representing frequencies).
            sbfl_scores: A dictionary mapping method names to their SBFL suspiciousness score [0, 1].
            lambd: The lambda hyperparameter controlling the trade-off between unary and pairwise potentials.
        """
        self.original_graph = G
        self.sbfl_scores = sbfl_scores
        self.lambd = lambd
        
        # Internal state
        self.D0: Dict[str, float] = {}
        self.D1: Dict[str, float] = {}
        self.C_ij: Dict[frozenset, float] = {}
        
        # Valid nodes (excluding any previously inserted SOURCE/TERMINAL nodes if any)
        self.nodes = [n for n in G.nodes() if n not in ["SOURCE", "TERMINAL"]]
        
        self._compute_unary_potentials()
        self._compute_pairwise_potentials()

    def _compute_unary_potentials(self) -> None:
        """Calculates and caches D0 and D1 for all nodes."""
        epsilon = 1e-12
        for node in self.nodes:
            score = self.sbfl_scores.get(node, 0.0)
            
            if score > 0:
                # Clamp score to avoid log(0)
                score_clamped = min(max(score, epsilon), 1.0 - epsilon)
                self.D0[node] = -math.log(1 - score_clamped)
                self.D1[node] = -math.log(score_clamped)
            else:
                self.D0[node] = 0.0
                self.D1[node] = float('inf')

    def _compute_pairwise_potentials(self) -> None:
        """Calculates and caches log-coupling capacities for edges, normalized to [0, 1]."""
        seen_edges = set()
        max_coupling = 0.0
        
        for u, v in self.original_graph.edges():
            if u in ["SOURCE", "TERMINAL"] or v in ["SOURCE", "TERMINAL"]:
                continue

            if u == v:
                continue

            edge_set = frozenset([u, v])
            if edge_set in seen_edges:
                continue
            seen_edges.add(edge_set)
            
            freq_uv = self.original_graph[u][v]['weight']
            freq_vu = self.original_graph[v][u]['weight'] if self.original_graph.has_edge(v, u) else 0
            
            # log(1 + freq) formulation
            coupling = math.log1p(freq_uv + freq_vu)
            self.C_ij[edge_set] = coupling
            
            if coupling > max_coupling:
                max_coupling = coupling
                
        # Normalize to [0, 1]
        if max_coupling > 0:
            for edge_set in self.C_ij:
                self.C_ij[edge_set] /= max_coupling

    def build_st_graph(self, constrained_node: Optional[str], constrained_label: Optional[int]) -> nx.DiGraph:
        """
        Constructs the augmented s-t directed graph.
        
        Args:
            constrained_node: Optional node to force a label onto (for min-marginal computation).
            constrained_label: The label (0 or 1) to force onto the constrained_node.
            
        Returns:
            A NetworkX DiGraph formatted for max_flow computations.
        """
        G_cut = nx.DiGraph()
        
        G_cut.add_node("SOURCE")
        G_cut.add_node("TERMINAL")
        
        for node in self.nodes:
            cap_S_to_node = self.D0[node]
            cap_node_to_T = self.D1[node]
            
            # Apply hard constraints for Min Marginal Energy
            if node == constrained_node and constrained_label is not None:
                if constrained_label == 0:
                    # Force to TERMINAL (0): infinite cost to cut the node->T edge
                    cap_node_to_T = float('inf')
                elif constrained_label == 1:
                    # Force to SOURCE (1): infinite cost to cut the S->node edge
                    cap_S_to_node = float('inf')
            
            # networkx boykov_kolmogorov doesn't handle actual 'inf' correctly for capacities sometimes,
            # so we use a very large finite number for infinity
            safe_inf = 1e9
            if cap_S_to_node == float('inf'): cap_S_to_node = safe_inf
            if cap_node_to_T == float('inf'): cap_node_to_T = safe_inf
            
            G_cut.add_edge("SOURCE", node, capacity=cap_S_to_node)
            G_cut.add_edge(node, "TERMINAL", capacity=cap_node_to_T)
            
        for edge_set, coupling in self.C_ij.items():
            nodes_list = list(edge_set)
            if len(nodes_list) == 1:
                continue # ignore self loops
            u, v = nodes_list[0], nodes_list[1]
            
            pairwise_cost = self.lambd * coupling
            if pairwise_cost > 0:
                # Add bi-directional n-links
                G_cut.add_edge(u, v, capacity=pairwise_cost)
                G_cut.add_edge(v, u, capacity=pairwise_cost)
                
        return G_cut

    def compute_min_cut(self) -> Tuple[float, Dict[str, int]]:
        """
        Computes the global minimum cut representing the optimal Gibbs energy.
        
        Returns:
            Tuple containing the minimum energy value and a dictionary mapping nodes to their labels (0 or 1).
        """
        G_cut = self.build_st_graph(None, None)
        
        # Boykov-Kolmogorov max-flow min-cut
        cut_value, partition = nx.minimum_cut(
            G_cut, "SOURCE", "TERMINAL", 
            capacity="capacity", 
            flow_func=nx.algorithms.flow.boykov_kolmogorov
        )
        
        source_partition, terminal_partition = partition
        
        labeling = {}
        for node in self.nodes:
            if node in source_partition:
                labeling[node] = 1 # BUGGY
            else:
                labeling[node] = 0 # NORMAL
                
        return cut_value, labeling

    def compute_min_marginals(self, node: str) -> Tuple[float, float]:
        """
        Computes the minimum marginal energies for a node being labeled 0 and 1.
        
        Args:
            node: The node to constrain.
            
        Returns:
            Tuple of (E_min(L_node=0), E_min(L_node=1))
        """
        G_cut_0 = self.build_st_graph(constrained_node=node, constrained_label=0)
        energy_0, _ = nx.minimum_cut(
            G_cut_0, "SOURCE", "TERMINAL", 
            capacity="capacity", 
            flow_func=nx.algorithms.flow.boykov_kolmogorov
        )
        
        G_cut_1 = self.build_st_graph(constrained_node=node, constrained_label=1)
        energy_1, _ = nx.minimum_cut(
            G_cut_1, "SOURCE", "TERMINAL", 
            capacity="capacity", 
            flow_func=nx.algorithms.flow.boykov_kolmogorov
        )
        
        return energy_0, energy_1

    def compute_energy_for_labeling(self, labeling: Dict[str, int]) -> float:
        """
        Computes the exact Gibbs energy for an arbitrary labeling configuration.
        
        Args:
            labeling: A dictionary mapping nodes to 0 or 1.
            
        Returns:
            The total Gibbs energy E(L).
        """
        regional_term = 0.0
        for node in self.nodes:
            label = labeling.get(node, 0)
            if label == 0:
                regional_term += self.D0.get(node, 0.0)
            else:
                cost = self.D1.get(node, 0.0)
                regional_term += 1e9 if cost == float('inf') else cost
                
        pairwise_term = 0.0
        for edge_set, coupling in self.C_ij.items():
            nodes_list = list(edge_set)
            if len(nodes_list) == 1:
                continue
            u, v = nodes_list[0], nodes_list[1]
            
            label_u = labeling.get(u, 0)
            label_v = labeling.get(v, 0)
            
            disagreement = 1 if label_u != label_v else 0
            pairwise_term += self.lambd * coupling * disagreement
            
        return regional_term + pairwise_term

    def generate_all_labelings(self) -> List[Dict[str, int]]:
        """
        Utility to generate all 2^|V| possible label assignments.
        WARNING: Only use this for very small graphs (|V| < 20)!
        """
        all_labelings = []
        for combination in itertools.product([0, 1], repeat=len(self.nodes)):
            labeling = {node: label for node, label in zip(self.nodes, combination)}
            all_labelings.append(labeling)
        return all_labelings

    def extract_intermediates(self, optimal_labeling: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts DataFrames for Nodes and Edges based on an optimal labeling for inspection.
        
        Returns:
            (df_nodes, df_edges)
        """
        # Nodes DF
        node_data = []
        for node in self.nodes:
            label = optimal_labeling.get(node, 0)
            D0 = self.D0.get(node, 0.0)
            D1 = self.D1.get(node, 0.0)
            score = self.sbfl_scores.get(node, 0.0)
            
            regional_cost = D1 if label == 1 else D0
            node_data.append({
                'Node': str(node).split('#')[-1],
                'Label': label,
                'SBFL_Score': score,
                'D0': D0,
                'D1': D1,
                'Regional_Cost': regional_cost
            })
        df_nodes = pd.DataFrame(node_data)
        
        # Edges DF
        edge_data = []
        for edge_set, coupling in self.C_ij.items():
            nodes_list = list(edge_set)
            if len(nodes_list) == 1:
                continue
            u, v = nodes_list[0], nodes_list[1]
            
            label_u = optimal_labeling.get(u, 0)
            label_v = optimal_labeling.get(v, 0)
            disagreement = 1 if label_u != label_v else 0
            pairwise = self.lambd * coupling * disagreement
            
            freq_uv = self.original_graph[u][v].get('weight', 0) if self.original_graph.has_edge(u, v) else 0
            freq_vu = self.original_graph[v][u].get('weight', 0) if self.original_graph.has_edge(v, u) else 0
            
            edge_data.append({
                'Node1': str(u).split('#')[-1],
                'Node2': str(v).split('#')[-1],
                'Freq_1_to_2': freq_uv,
                'Freq_2_to_1': freq_vu,
                'Coupling': coupling,
                'Label_1': label_u,
                'Label_2': label_v,
                'Disagreement': disagreement,
                'Pairwise_Cost': pairwise
            })
        df_edges = pd.DataFrame(edge_data)
        
        return df_nodes, df_edges
