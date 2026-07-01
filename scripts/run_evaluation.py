import os
import sys
import logging
from pathlib import Path

# Configure logging to see output from processes
logging.basicConfig(level=logging.INFO, format='%(message)s')

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from evaluation.config import BENCHMARK_TARGET, OUTPUT_DIR, SAMPLE_SIZE, RANDOM_SEED, LAMBDAS_TO_ABLATE, MAX_WORKERS
from evaluation.benchmark_runner import run_evaluation_batch

if __name__ == "__main__":
    print(f"Configuration:")
    print(f"TARGET: {BENCHMARK_TARGET}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"SAMPLE_SIZE: {SAMPLE_SIZE}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print(f"LAMBDAS_TO_ABLATE: {LAMBDAS_TO_ABLATE}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print("-" * 40)
    
    run_evaluation_batch(
        target=BENCHMARK_TARGET,
        lambdas=LAMBDAS_TO_ABLATE,
        max_workers=MAX_WORKERS,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED,
        output_dir=OUTPUT_DIR
    )
