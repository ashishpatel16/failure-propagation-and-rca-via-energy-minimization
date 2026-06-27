"""Render hierarchical SVGs for extracted RCA-Eval runtime call graphs."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualization.graph_plots import visualize_graph


class Suite(Enum):
    RE3 = "RE3"


@dataclass(frozen=True)
class GraphPlot:
    graph_path: Path
    output_prefix: Path
    title: str


LOGGER = logging.getLogger("rca_eval_graph_renderer")
DATA_DIR: Path = ROOT_DIR / "data"
SELECTED_SUITES: tuple[Suite, ...] = (Suite.RE3,)
RUNTIME_GRAPH_FILENAME = "call_graph_runtime.json"
HIERARCHICAL_PLOT_STEM = "call_graph_runtime_hierarchical"


def load_graph(graph_path: Path) -> nx.DiGraph:
    """Load a runtime graph JSON file into a directed NetworkX graph."""
    graph_data: dict[str, Any] = json.loads(graph_path.read_text(encoding="utf-8"))
    graph = nx.DiGraph()

    for node in graph_data["nodes"]:
        if not isinstance(node, str) or not node:
            raise ValueError(f"Invalid graph node in {graph_path}")
        graph.add_node(node)

    for edge in graph_data["edges"]:
        caller = edge["caller"]
        callee = edge["callee"]
        frequency = edge["frequency"]
        if not isinstance(caller, str) or not caller:
            raise ValueError(f"Invalid edge caller in {graph_path}")
        if not isinstance(callee, str) or not callee:
            raise ValueError(f"Invalid edge callee in {graph_path}")
        if not isinstance(frequency, int) or frequency <= 0:
            raise ValueError(f"Invalid edge frequency in {graph_path}")
        graph.add_edge(caller, callee, weight=frequency)

    if not graph.nodes:
        raise ValueError(f"Runtime graph contains no service nodes: {graph_path}")
    return graph


def discover_plots(data_dir: Path, suites: tuple[Suite, ...]) -> list[GraphPlot]:
    """Discover the runtime graph SVGs to render for the selected benchmark suites."""
    plots: list[GraphPlot] = []
    for suite in suites:
        suite_dir = data_dir / suite.value
        if not suite_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset suite directory: {suite_dir}")

        graph_paths = sorted(suite_dir.rglob(RUNTIME_GRAPH_FILENAME))
        if not graph_paths:
            raise FileNotFoundError(f"No {RUNTIME_GRAPH_FILENAME} files found in {suite_dir}")

        for graph_path in graph_paths:
            instance_dir = graph_path.parent
            instance_name = instance_dir.relative_to(data_dir).as_posix().replace("/", " / ")
            plots.append(
                GraphPlot(
                    graph_path=graph_path,
                    output_prefix=instance_dir / HIERARCHICAL_PLOT_STEM,
                    title=f"{instance_name} Runtime Call Graph",
                )
            )
    return plots


def render_plot(graph_plot: GraphPlot) -> None:
    """Render a single runtime graph with the shared hierarchical visualization logic."""
    graph = load_graph(graph_plot.graph_path)
    visualize_graph(
        graph,
        graph_plot.title,
        save_prefix=str(graph_plot.output_prefix),
        layout_type="hierarchical",
        buggy_methods=None,
    )


def main() -> None:
    """Render hierarchical runtime call-graph plots for the selected suites."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    graph_plots = discover_plots(DATA_DIR, SELECTED_SUITES)
    for graph_plot in graph_plots:
        LOGGER.info(f"Rendering {graph_plot.graph_path.relative_to(ROOT_DIR)}")
        render_plot(graph_plot)
    LOGGER.info(f"Rendered {len(graph_plots)} hierarchical runtime call-graph SVGs")


if __name__ == "__main__":
    main()
