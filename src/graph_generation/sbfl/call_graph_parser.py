import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Set
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CallEdge:
    caller: str
    callee: str
    frequency: int

@dataclass
class GraphMetadata:
    total_nodes: int
    total_edges: int

@dataclass
class CallGraph:
    metadata: GraphMetadata
    nodes: List[str]
    edges: List[CallEdge]

    def to_dict(self) -> dict:
        return asdict(self)

def parse_raw_call_graph(raw_graph_path: str) -> CallGraph:
    path = Path(raw_graph_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Call graph file not found: {raw_graph_path}")
        
    edges: List[CallEdge] = []
    unique_nodes: Set[str] = set()
    
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            if "->" not in line or ":" not in line:
                raise ValueError(f"Malformed edge at line {line_num}: {line}")
                
            edge_part, count_part = line.rsplit(":", 1)
            if "->" not in edge_part:
                raise ValueError(f"Missing '->' in edge part at line {line_num}: {edge_part}")
                
            caller, callee = edge_part.split("->")
            caller = caller.strip()
            callee = callee.strip()
            
            if not caller or not callee:
                raise ValueError(f"Empty caller or callee at line {line_num}: {line}")
            
            if "test" in caller.lower() or "test" in callee.lower():
                continue
            
            try:
                count = int(count_part.strip())
            except ValueError:
                raise ValueError(f"Invalid frequency count at line {line_num}: {count_part}")
                
            unique_nodes.add(caller)
            unique_nodes.add(callee)
            edges.append(CallEdge(caller=caller, callee=callee, frequency=count))
            
    return CallGraph(
        metadata=GraphMetadata(total_nodes=len(unique_nodes), total_edges=len(edges)),
        nodes=list(unique_nodes),
        edges=edges
    )

def save_to_json(graph: CallGraph, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(graph.to_dict(), f, indent=2)
    logger.info(f"Saved parsed call graph to {output_path}")
