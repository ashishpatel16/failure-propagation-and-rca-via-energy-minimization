import os
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RCAEvalExtractor")

ROOT_DIR = Path(os.getcwd())
sys.path.insert(0, str(ROOT_DIR / "src"))

from cut_algorithms.data_parsers import RCAEvalTraceParser, Granularity
from graph_generation.sbfl.call_graph_parser import CallGraph, CallEdge, GraphMetadata, save_to_json
from visualization.graph_plots import plot_json_graph
import networkx as nx

DATA_DIR: str = "data"
GRANULARITY: Granularity = Granularity.OPERATION
VISUALIZE: bool = True

def convert_nx_to_callgraph(G: nx.DiGraph) -> CallGraph:
    """Converts an nx.DiGraph back into the standardized CallGraph dataclass for saving."""
    edges: list[CallEdge] = []
    for u, v, data in G.edges(data=True):
        freq = data.get("weight", 1)
        edges.append(CallEdge(caller=u, callee=v, frequency=freq))
        
    nodes = list(G.nodes())
    metadata = GraphMetadata(total_nodes=len(nodes), total_edges=len(edges))
    return CallGraph(metadata=metadata, nodes=nodes, edges=edges)

if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist. Run rca_eval_init.py first.")
        sys.exit(1)

    target_datasets = ["RE2", "RE3"]
    trace_files_found = 0
    extracted_graphs = 0

    for dataset in target_datasets:
        dataset_dir = data_dir / dataset
        if not dataset_dir.exists():
            continue

        # Look for traces.csv within subdirectories
        for traces_csv in dataset_dir.rglob("traces.csv"):
            trace_files_found += 1
            folder_path = traces_csv.parent
            logger.info(f"Processing {traces_csv.relative_to(data_dir)}")

            try:
                trace_parser = RCAEvalTraceParser(str(traces_csv))
                G = trace_parser.build_call_graph(granularity=GRANULARITY)
                
                # Convert and Save
                call_graph_obj = convert_nx_to_callgraph(G)
                out_json = folder_path / "call_graph.json"
                save_to_json(call_graph_obj, str(out_json))
                
                # Optional visualization
                if VISUALIZE:
                    plot_title = f"{folder_path.name} Call Graph"
                    save_prefix = str(folder_path / "call_graph_plot")
                    plot_json_graph(str(out_json), plot_title, save_prefix)
                
                extracted_graphs += 1
            except Exception as e:
                logger.error(f"Failed to extract graph for {traces_csv}: {e}", exc_info=True)

    logger.info(f"Extraction Summary:")
    logger.info(f"Trace files found: {trace_files_found}")
    logger.info(f"Graphs successfully extracted: {extracted_graphs}")
