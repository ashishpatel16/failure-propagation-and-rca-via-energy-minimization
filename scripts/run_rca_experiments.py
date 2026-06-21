import logging
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.rca_baselines import BaselineAlgo, run_baseline

# User Configurable Global Variables
TARGET_INSTANCE: str = "data/RE2/RE2-OB/checkoutservice_cpu/1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_all_baselines(instance_dir: str) -> None:
    """
    Computes and prints the Top-5 root cause predictions for all supported RCAEval baselines
    for a specific dataset instance.
    """
    data_dir_path: Path = Path(instance_dir)
    if not data_dir_path.exists():
        logger.error(f"Directory {instance_dir} does not exist.")
        raise FileNotFoundError(f"Directory {instance_dir} does not exist.")

    inject_time_file: Path = data_dir_path / "inject_time.txt"
    if not inject_time_file.exists():
        logger.error(f"inject_time.txt not found in {instance_dir}")
        raise FileNotFoundError(f"inject_time.txt not found in {instance_dir}")

    inject_time_str: str = inject_time_file.read_text().strip()
    inject_time: float = float(inject_time_str)

    logger.info(f"Target instance: {instance_dir}")
    logger.info(f"Inject time: {inject_time}")

    for algo in BaselineAlgo:
        logger.info(f"--- Running Baseline: {algo.value} ---")
        ranks: list[str] = run_baseline(algo, instance_dir, inject_time)
        
        if len(ranks) > 0:
            logger.info(f"Top 5 candidates for {algo.value}:")
            for i in range(min(5, len(ranks))):
                logger.info(f"  {i+1}. {ranks[i]}")
        else:
            logger.warning(f"Algorithm {algo.value} returned no ranks or failed.")

if __name__ == "__main__":
    compute_all_baselines(TARGET_INSTANCE)
