# Implementation Details: Boykov-Jolly Energy Minimization

This document outlines the mathematical foundation implemented in the `BoykovJollyCut` algorithm for fault localization using Graph Cuts.

## 1. Energy Minimization Objective

The algorithm optimizes a Gibbs energy function $E(L)$ over a labeling $L$, where each node $i$ (representing a method in the call graph) is assigned a label $L_i \in \{0, 1\}$. 
- $L_i = 0$ corresponds to a **Normal** (non-buggy) method.
- $L_i = 1$ corresponds to a **Buggy** method.

The total energy is defined as:
$$E(L) = \sum_{i \in V} D_i(L_i) + \lambda \sum_{\{i,j\} \in E} V_{ij}(L_i, L_j)$$

Where:
- $\sum D_i(L_i)$ is the **Unary Potential** (Data Term), which measures how much assigning label $L_i$ agrees with the initial SBFL (Spectrum-Based Fault Localization) scores.
- $\sum V_{ij}(L_i, L_j)$ is the **Pairwise Potential** (Smoothness Term), which penalizes neighboring nodes in the call graph for taking different labels.
- $\lambda$ is the hyperparameter that balances the trade-off between the unary and pairwise terms.

---

## 2. Unary Potentials (Data Term)

The unary potentials $D_i(L_i)$ are derived from the SBFL suspiciousness score $S_i \in [0, 1]$ for each method. To avoid $\log(0)$, the score is clamped to the range $[\epsilon, 1-\epsilon]$ where $\epsilon = 10^{-12}$.

The costs are defined as the negative log-likelihoods:
- **Cost of labeling as Normal ($L_i = 0$):** 
  $$D_i(0) = -\log(1 - S_i)$$
  *(If $S_i$ is high, $1 - S_i$ is small, making $D_i(0)$ large, penalizing the normal label.)*

- **Cost of labeling as Buggy ($L_i = 1$):** 
  $$D_i(1) = -\log(S_i)$$
  *(If $S_i$ is high, $D_i(1)$ is small, encouraging the buggy label.)*

If a method has no SBFL score ($S_i = 0$), the implementation forces it to be normal by setting $D_i(0) = 0$ and $D_i(1) = \infty$.

---

## 3. Pairwise Potentials (Smoothness Term)

The pairwise potential $V_{ij}(L_i, L_j)$ acts as a regularizer. It penalizes adjacent nodes $i$ and $j$ in the call graph if they are assigned different labels (i.e., a disagreement where one is marked buggy and the other normal).

$$V_{ij}(L_i, L_j) = C_{ij} \cdot \mathbf{1}(L_i \neq L_j)$$

Where:
- $\mathbf{1}(L_i \neq L_j)$ is an indicator function that equals $1$ if the labels differ, and $0$ if they are the same.
- $C_{ij}$ is the coupling strength (capacity) between nodes $i$ and $j$.

The coupling strength $C_{ij}$ is based on the execution frequency (weights) of the edges between $i$ and $j$. To dampen the effect of extremely high frequencies, a log scale is applied:
$$C_{ij} = \log(1 + f_{i \to j} + f_{j \to i})$$
where $f_{i \to j}$ is the frequency of method $i$ calling method $j$.

---

## 4. Graph Construction for s-t Min-Cut

To minimize this energy using the Boykov-Kolmogorov max-flow/min-cut algorithm, an augmented $s-t$ graph is constructed:
- A virtual **SOURCE** node ($s$) and **TERMINAL** node ($t$) are added.
- **t-links (Unary Constraints):** 
  - Directed edge $(s, i)$ with capacity $D_i(0)$. Cutting this edge costs $D_i(0)$, corresponding to assigning $L_i = 1$.
  - Directed edge $(i, t)$ with capacity $D_i(1)$. Cutting this edge costs $D_i(1)$, corresponding to assigning $L_i = 0$.
- **n-links (Pairwise Constraints):**
  - Bi-directional edges $(i, j)$ and $(j, i)$ with capacity $\lambda \cdot C_{ij}$.
  - Cutting these edges implies $i$ and $j$ are in different partitions (different labels), incurring the pairwise cost.

The global minimum cut partitions the graph into two sets (SOURCE and TERMINAL), yielding the optimal labeling $L^*$ that minimizes the total energy $E(L)$.

---

## 5. Min-Marginal Energies (GraphCut Score)

To compute a "confidence" score for a specific method $k$ being the root cause:
1. $E_{min}(L_k = 0)$ is computed by forcing $L_k = 0$ (setting the capacity of $(k, t)$ to $\infty$) and computing the min-cut.
2. $E_{min}(L_k = 1)$ is computed by forcing $L_k = 1$ (setting the capacity of $(s, k)$ to $\infty$) and computing the min-cut.

The GraphCut confidence score used for ranking is the difference in min-marginals:
$$\text{GraphCut Score} = E_{min}(L_k = 0) - E_{min}(L_k = 1)$$
A higher positive difference indicates that forcing the method to be "Normal" drastically increases the energy compared to forcing it to be "Buggy", meaning the algorithm is highly confident the method is buggy.
