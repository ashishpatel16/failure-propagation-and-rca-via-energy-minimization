import os
import sys
import shutil
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv

# Ensure we can import from src
ROOT_DIR = Path(os.getcwd())
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from benchmarking.d4j_manager import D4JManager
from benchmarking.gzoltar_manager import GZoltarRunner
from graph_generation.sbfl.tracer_runner import TracerRunner
from graph_generation.sbfl.call_graph_parser import parse_raw_call_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BenchmarkExtraction")

# Global configuration variables
TARGET_PROJECTS: Dict[str, str] = {
    # "Chart": "org.jfree",
    # "Cli": "org.apache.commons.cli",
    # "Closure": "com.google.javascript",
    # "Codec": "org.apache.commons.codec",
    # "Collections": "org.apache.commons.collections",
    # "Compress": "org.apache.commons.compress",
    # "Csv": "org.apache.commons.csv",
    # "Gson": "com.google.gson",
    # "JacksonCore": "com.fasterxml.jackson.core",
    # "JacksonDatabind": "com.fasterxml.jackson.databind",
    # "JacksonXml": "com.fasterxml.jackson.dataformat.xml",
    "Jsoup": "org.jsoup",
    # "JxPath": "org.apache.commons.jxpath",
    # "Lang": "org.apache.commons.lang",
    # "Math": "org.apache.commons.math",
    # "Mockito": "org.mockito",
    # "Time": "org.joda.time"
}

NUM_BUGS_PER_PROJECT: Optional[int] = 2  # Set to None to run all bugs
OUTPUT_CSV: str = "benchmark_extraction_results.csv"

@dataclass
class BenchmarkMetrics:
    project: str
    bug_id: str
    status: str
    checkout_compile_time_sec: float
    list_tests_time_sec: float
    collect_coverage_time_sec: float
    export_csv_time_sec: float
    call_graph_agent_time_sec: float
    call_graph_parse_time_sec: float
    total_pipeline_time_sec: float
    total_tests: int
    coverage_matrix_rows: int
    coverage_matrix_cols: int
    call_graph_nodes: int
    call_graph_edges: int

class BenchmarkExtractor:
    def __init__(self, d4j_path: str, java_home: str):
        self.d4j_manager = D4JManager(d4j_path, java_home)
        
        current_dir = ROOT_DIR / "src" / "graph_generation" / "sbfl"
        
        gzoltar_path = current_dir / "gzoltar"
        if not gzoltar_path.exists():
            raise FileNotFoundError(f"GZoltar directory missing: {gzoltar_path}")
        self.gzoltar_runner = GZoltarRunner(str(gzoltar_path), java_home)
        
        tracer_path = current_dir / "tracer_agent"
        if not tracer_path.exists():
            raise FileNotFoundError(f"Tracer directory missing: {tracer_path}")
        self.tracer_runner = TracerRunner(str(tracer_path))
        
    def benchmark_bug(self, project: str, bug_id: str, output_dir: str, package_prefix: str) -> BenchmarkMetrics:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        work_dir = output_path / "workspace"
        
        logger.info(f"Benchmarking {project}-{bug_id}")
        
        start_pipeline = time.perf_counter()
        
        metrics = BenchmarkMetrics(
            project=project,
            bug_id=bug_id,
            status="Failed",
            checkout_compile_time_sec=0.0,
            list_tests_time_sec=0.0,
            collect_coverage_time_sec=0.0,
            export_csv_time_sec=0.0,
            call_graph_agent_time_sec=0.0,
            call_graph_parse_time_sec=0.0,
            total_pipeline_time_sec=0.0,
            total_tests=0,
            coverage_matrix_rows=0,
            coverage_matrix_cols=0,
            call_graph_nodes=0,
            call_graph_edges=0
        )
        
        try:
            agent_jar = self.tracer_runner.compile_agent()
            
            # 1. Checkout & Compile
            start_time = time.perf_counter()
            if not work_dir.exists():
                self.d4j_manager.checkout(project, bug_id, str(work_dir))
                self.d4j_manager.compile(str(work_dir))
            metrics.checkout_compile_time_sec = time.perf_counter() - start_time
                
            props = self.d4j_manager.get_properties(str(work_dir))
            if "cp.test" not in props:
                raise ValueError(f"Failed to extract properties for {project}-{bug_id}")
                
            cp = props["cp.test"]
            
            if 'dir.bin.classes' not in props:
                raise ValueError(f"Property 'dir.bin.classes' not found for {project}-{bug_id}")
            if 'dir.bin.tests' not in props:
                raise ValueError(f"Property 'dir.bin.tests' not found for {project}-{bug_id}")
                
            src_classes = work_dir / props['dir.bin.classes']
            test_bin = work_dir / props['dir.bin.tests']
            
            tests_file = output_path / "tests.txt"
            gzoltar_output = output_path / "gzoltar-out"
            coverage_csv = output_path / "coverage.csv"
            
            # 2. List Tests
            start_time = time.perf_counter()
            if not tests_file.exists():
                self.gzoltar_runner.list_tests(str(test_bin), cp, str(tests_file))
            metrics.list_tests_time_sec = time.perf_counter() - start_time
            
            if not tests_file.exists():
                raise FileNotFoundError("Tests file was not generated by GZoltar.")
                
            with open(tests_file, 'r') as f:
                metrics.total_tests = sum(1 for line in f if line.strip())
                
            # 3. Collect Coverage
            start_time = time.perf_counter()
            if not coverage_csv.exists():
                self.gzoltar_runner.collect_coverage(str(work_dir), str(src_classes), cp, str(tests_file), str(gzoltar_output), "*", "*Test*:*junit*")
            metrics.collect_coverage_time_sec = time.perf_counter() - start_time
            
            # 4. Export CSV
            start_time = time.perf_counter()
            if not coverage_csv.exists():
                df = self.gzoltar_runner.export_to_csv(str(gzoltar_output), str(coverage_csv))
                metrics.coverage_matrix_rows = df.shape[0]
                metrics.coverage_matrix_cols = df.shape[1]
            else:
                df = pd.read_csv(coverage_csv)
                metrics.coverage_matrix_rows = df.shape[0]
                metrics.coverage_matrix_cols = df.shape[1]
            metrics.export_csv_time_sec = time.perf_counter() - start_time
                
            if not coverage_csv.exists():
                raise FileNotFoundError("Coverage CSV was not generated by GZoltar.")
                
            raw_graph_txt = output_path / "call_graph.txt"
            if not package_prefix:
                raise ValueError("package_prefix must be explicitly provided and cannot be empty.")
                
            # 5. Call Graph Agent Tracing
            start_time = time.perf_counter()
            if not raw_graph_txt.exists():
                self.d4j_manager.run_test_with_agent(str(work_dir), str(agent_jar), f"{raw_graph_txt};{package_prefix}")
            metrics.call_graph_agent_time_sec = time.perf_counter() - start_time
                
            if not raw_graph_txt.exists():
                raise FileNotFoundError("Dynamic call graph was not generated by TracerAgent.")
                
            # 6. Call Graph Parsing
            start_time = time.perf_counter()
            graph = parse_raw_call_graph(str(raw_graph_txt))
            
            # Filter out <clinit> and Test classes
            graph.nodes = [n for n in graph.nodes if "<clinit>" not in n and "Test" not in n]
            graph.edges = [e for e in graph.edges if "<clinit>" not in e.caller and "<clinit>" not in e.callee
                           and "Test" not in e.caller and "Test" not in e.callee]
            
            metrics.call_graph_nodes = len(graph.nodes)
            metrics.call_graph_edges = len(graph.edges)
            metrics.call_graph_parse_time_sec = time.perf_counter() - start_time
            
            metrics.status = "Success"
            
        except Exception as e:
            logger.error(f"Failed to benchmark {project}-{bug_id}: {e}", exc_info=True)
            metrics.status = f"Failed: {type(e).__name__} - {str(e)}"
        finally:
            metrics.total_pipeline_time_sec = time.perf_counter() - start_pipeline
            
            # Clean up the workspace to save space
            if work_dir.exists():
                logger.info(f"Cleaning up workspace: {work_dir}")
                shutil.rmtree(work_dir, ignore_errors=True)
                
            # Clean up raw gzoltar output
            if gzoltar_output.exists():
                logger.info(f"Cleaning up raw gzoltar output: {gzoltar_output}")
                shutil.rmtree(gzoltar_output, ignore_errors=True)
                
        return metrics

def main() -> None:
    load_dotenv()
    
    JAVA_HOME = os.environ.get("JAVA_HOME_PATH")
    D4J_PATH = os.environ.get("D4J_PATH")
    
    if not JAVA_HOME:
        raise ValueError("JAVA_HOME_PATH invalid or missing in environment variables.")
        
    if not D4J_PATH:
        raise ValueError("D4J_PATH invalid or missing in environment variables.")
        
    if not Path(JAVA_HOME).exists():
        raise FileNotFoundError(f"JAVA_HOME_PATH does not exist: {JAVA_HOME}")
        
    if not Path(D4J_PATH).exists():
        raise FileNotFoundError(f"D4J_PATH does not exist: {D4J_PATH}")
        
    extractor = BenchmarkExtractor(d4j_path=D4J_PATH, java_home=JAVA_HOME)
    
    targets: List[tuple[str, str, str]] = []
    
    for proj, prefix in TARGET_PROJECTS.items():
        try:
            bugs = extractor.d4j_manager.get_bug_ids(proj)
            if NUM_BUGS_PER_PROJECT is not None:
                bugs = bugs[:NUM_BUGS_PER_PROJECT]
            for b in bugs:
                targets.append((proj, b, prefix))
        except Exception as e:
            logger.error(f"Failed to fetch bugs for project {proj}: {e}")
            raise e

    logger.info(f"Total targets to benchmark: {len(targets)}")
    
    all_metrics: List[BenchmarkMetrics] = []
    
    for project, bug_id, package_prefix in targets:
        output_dir = ROOT_DIR / "data" / "benchmark_extraction" / f"{project}_{bug_id}"
        metrics = extractor.benchmark_bug(project, bug_id, str(output_dir), package_prefix)
        all_metrics.append(metrics)
        
        # Save intermediate results
        df = pd.DataFrame([asdict(m) for m in all_metrics])
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Saved intermediate benchmark results to {OUTPUT_CSV}")

    logger.info("Benchmarking completed successfully.")

if __name__ == "__main__":
    main()
