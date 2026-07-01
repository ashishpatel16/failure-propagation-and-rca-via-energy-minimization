import os
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent

# Global configurations
BENCHMARK_TARGET: str = "d4j"

# Base outputs directory
OUTPUT_DIR: Path = ROOT_DIR / "outputs_complete_directional_full_benchmark"

# Sampling
SAMPLE_SIZE: float = 1.0
if not (0.0 < SAMPLE_SIZE <= 1.0):
    raise ValueError(f"SAMPLE_SIZE must be between 0.0 and 1.0, got {SAMPLE_SIZE}")

RANDOM_SEED: int = 40
GROUND_TRUTH_CSV: Path = ROOT_DIR / "ground_truth.csv"

# Reduced grid for faster iteration: 0.0 is the baseline control; 0.01/0.1 cover
# the useful coupling band. lambda >= 1.0 over-smooths (documented in outputs_full).
LAMBDAS_TO_ABLATE: list[float] = [0.01, 0.1, 1]
MAX_WORKERS: int = 5