import logging
import sys
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger: logging.Logger = logging.getLogger(__name__)

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from evaluation.config import BENCHMARK_TARGET, GROUND_TRUTH_CSV
from evaluation.benchmark_reader import get_reader_for_lambda, BenchmarkReader, InstanceResult
from evaluation.analyzer import GroundTruthManager, BenchmarkAnalyzer

if __name__ == "__main__":
    LAMBDAS_TO_ANALYZE: List[float] = [0.1, 1.0]
    
    logger.info(f"Analysis Configuration:")
    logger.info(f"  TARGET: {BENCHMARK_TARGET}")
    logger.info(f"  GROUND TRUTH: {GROUND_TRUTH_CSV}")
    logger.info("-" * 40)
    
    try:
        gt_manager: GroundTruthManager = GroundTruthManager(GROUND_TRUTH_CSV)
        analyzer: BenchmarkAnalyzer = BenchmarkAnalyzer(gt_manager)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
        
    for lambd in LAMBDAS_TO_ANALYZE:
        logger.info(f"\nAnalyzing lambda = {lambd}")
        try:
            reader: BenchmarkReader = get_reader_for_lambda(BENCHMARK_TARGET, lambd)
            results: List[InstanceResult] = reader.read_all_instances()
            logger.info(f"Found {len(results)} instances to analyze.")
            
            for result in results:
                analyzer.analyze_instance(result)
                
        except FileNotFoundError:
            logger.warning(f"No evaluation results found for lambda {lambd} in target {BENCHMARK_TARGET}")

    logger.info("\nAnalysis skeleton completed.")
