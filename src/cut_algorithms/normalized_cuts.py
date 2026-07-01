"""Calibrated Boykov-Jolly s-t cut for SBFL call-graph smoothing.

This module is an experimental, self-contained alternative to
``cut_algorithms.boykov_jolly.BoykovJollyCut``.  The original class is left
untouched; everything here is additive.

Motivation
----------
The original cut minimises the Gibbs energy

    E(L) = sum_i D_i(L_i) + lambda * sum_{ij} V_ij(L_i, L_j)

with unary terms ``D0 = -log(1-s)``, ``D1 = -log(s)`` and Potts pairwise
coupling ``C_ij = log1p(freq_i_j)``.  The solver itself is provably correct
(its max-flow min-marginals match brute-force exact energies).  The problem is
*calibration*:

1. ``C_ij = log1p(freq)`` is unbounded and uncalibrated against the O(1) unary
   scale, so a high-degree node accumulates a huge smoothness penalty and gets
   demoted purely for its connectivity (a degree bias).
2. ``score == 0`` maps to ``D1 = inf``, turning every uncovered node into an
   immovable NORMAL anchor that drags its neighbours normal through the
   pairwise edges.  Real call graphs are dominated by such nodes.

``NormalizedBoykovJollyCut`` addresses both while preserving the key invariant:

    at lambda == 0 the min-marginal score reduces to ``logit(score)``, i.e.
    the ranking is identical to the raw SBFL baseline.

It subclasses the original purely to reuse the (correct) s-t graph
construction and max-flow min-marginal machinery; only the potential
computation is overridden.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List

import networkx as nx

from cut_algorithms.boykov_jolly import BoykovJollyCut


class CouplingNormalization(str, Enum):
    """How to rescale the raw ``log1p(freq)`` edge coupling."""

    NONE = "none"        # raw log1p(freq) -- matches the original cut
    DEGREE = "degree"    # symmetric: log1p(freq) / sqrt(deg_u * deg_v)
    MAX = "max"          # scale so the largest coupling equals 1


class NormalizedBoykovJollyCut(BoykovJollyCut):
    """Boykov-Jolly cut with a calibrated pairwise term and finite anchors.

    Args:
        G: call graph; edge weights are read from the ``weight`` attribute
            (missing weights default to 0, so the class never KeyErrors).
        sbfl_scores: suspiciousness in [0, 1] per node.
        lambd: smoothness trade-off.
        coupling_norm: pairwise normalization strategy (default DEGREE).
        finite_anchor: if True, clamp every score to ``[eps, 1-eps]`` so
            uncovered nodes get a large but *finite* D1 instead of +inf.
        zero_score_eps: the epsilon used for clamping / finite anchoring.
        restrict_to_scored: if True, build the cut over scored nodes only
            (score > 0) and ignore uncovered nodes entirely.
    """

    def __init__(
        self,
        G: nx.DiGraph,
        sbfl_scores: Dict[str, float],
        lambd: float,
        *,
        coupling_norm: CouplingNormalization = CouplingNormalization.DEGREE,
        finite_anchor: bool = True,
        zero_score_eps: float = 1e-3,
        restrict_to_scored: bool = False,
    ) -> None:
        # Set everything the inherited machinery relies on; deliberately do not
        # call super().__init__ so we control attribute/initialisation order.
        self.original_graph = G
        self.sbfl_scores = sbfl_scores
        self.lambd = lambd
        self.coupling_norm = CouplingNormalization(coupling_norm)
        self.finite_anchor = finite_anchor
        self.zero_score_eps = zero_score_eps
        self.restrict_to_scored = restrict_to_scored

        self.D0: Dict[str, float] = {}
        self.D1: Dict[str, float] = {}
        self.C_ij: Dict[frozenset, float] = {}

        candidate = [n for n in G.nodes() if n not in ("SOURCE", "TERMINAL")]
        if restrict_to_scored:
            candidate = [n for n in candidate if sbfl_scores.get(n, 0.0) > 0.0]
        self.nodes: List[str] = candidate
        self._node_set = set(self.nodes)

        self._compute_unary_potentials()
        self._compute_pairwise_potentials()

    # -- unary -------------------------------------------------------------
    def _compute_unary_potentials(self) -> None:
        eps = self.zero_score_eps
        for node in self.nodes:
            score = self.sbfl_scores.get(node, 0.0)
            if self.finite_anchor:
                # Every node (incl. uncovered) gets a finite D1; uncovered
                # nodes are merely *very unlikely* buggy, not impossible.
                s = min(max(score, eps), 1.0 - eps)
                self.D0[node] = -math.log(1.0 - s)
                self.D1[node] = -math.log(s)
            else:
                if score > 0.0:
                    s = min(max(score, 1e-12), 1.0 - 1e-12)
                    self.D0[node] = -math.log(1.0 - s)
                    self.D1[node] = -math.log(s)
                else:
                    self.D0[node] = 0.0
                    self.D1[node] = float("inf")

    # -- pairwise ----------------------------------------------------------
    def _compute_pairwise_potentials(self) -> None:
        base: Dict[frozenset, float] = {}
        seen: set = set()
        for u, v in self.original_graph.edges():
            if u not in self._node_set or v not in self._node_set:
                continue
            edge_set = frozenset((u, v))
            if len(edge_set) == 1 or edge_set in seen:  # skip self-loops / dups
                continue
            seen.add(edge_set)
            freq_uv = self.original_graph[u][v].get("weight", 0.0)
            freq_vu = (
                self.original_graph[v][u].get("weight", 0.0)
                if self.original_graph.has_edge(v, u)
                else 0.0
            )
            
            # Normalize coupling by the square root of the product of the degrees of the nodes
            base[edge_set] = math.log1p(freq_uv + freq_vu) / math.sqrt(self.original_graph.degree[u] * self.original_graph.degree[v])

        if not base:
            self.C_ij = {}
            return

        if self.coupling_norm == CouplingNormalization.DEGREE:
            # Symmetric normalized-cut style scaling: divide each coupling by
            # sqrt of the endpoints' incident-edge counts, so a hub's total
            # smoothness no longer scales with its raw degree.
            incident: Dict[str, int] = {n: 0 for n in self.nodes}
            for edge_set in base:
                for n in edge_set:
                    incident[n] += 1
            for edge_set, value in base.items():
                u, v = tuple(edge_set)
                denom = math.sqrt(max(incident[u], 1) * max(incident[v], 1))
                self.C_ij[edge_set] = value / denom
        elif self.coupling_norm == CouplingNormalization.MAX:
            scale = max(base.values())
            self.C_ij = {k: (v / scale if scale > 0 else 0.0) for k, v in base.items()}
        else:  # NONE
            self.C_ij = base


def compute_graphcut_scores(
    G: nx.DiGraph,
    node_scores: Dict[str, float],
    lambd: float,
    **kwargs,
) -> Dict[str, float]:
    """Per-node suspiciousness = E_min(node=NORMAL) - E_min(node=BUGGY).

    Higher is more suspicious.  ``node_scores`` should already be normalised to
    [0, 1].  Extra kwargs are forwarded to :class:`NormalizedBoykovJollyCut`.
    """
    cut = NormalizedBoykovJollyCut(G, node_scores, lambd, **kwargs)
    scores: Dict[str, float] = {}
    for node in cut.nodes:
        energy_normal, energy_faulty = cut.compute_min_marginals(node)
        scores[node] = energy_normal - energy_faulty
    return scores
