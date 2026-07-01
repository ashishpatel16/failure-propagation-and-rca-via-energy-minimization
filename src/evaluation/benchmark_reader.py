import os
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from pathlib import Path
from typing import List
import sys

ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from evaluation.config import OUTPUT_DIR, BENCHMARK_TARGET

class SbflMetric(Enum):
    TARANTULA = "tarantula"
    OCHIAI = "ochiai"
    DSTAR = "dstar"
    TARANTULA_GC = "tarantula_gc"
    OCHIAI_GC = "ochiai_gc"
    DSTAR_GC = "dstar_gc"

@dataclass
class InstanceResult:
    project: str
    bug_id: int
    lambd: float
    total_methods: int
    metrics_df: pd.DataFrame

class BenchmarkReader:
    def __init__(self, folder_path: Path):
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        self.folder_path: Path = folder_path

    def read_all_instances(self) -> List[InstanceResult]:
        instances: List[InstanceResult] = []
        for file_path in self.folder_path.iterdir():
            if file_path.is_file() and file_path.suffix == ".csv":
                instances.append(self.read_instance(file_path))
        return instances

    def read_instance(self, file_path: Path) -> InstanceResult:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_stem: str = file_path.stem
        parts: List[str] = file_stem.split("_")
        if len(parts) != 3:
            raise ValueError(f"Invalid filename format, expected Project_BugId_Lambd.csv, got: {file_stem}")
            
        project: str = parts[0]
        bug_id: int = int(parts[1])
        lambd: float = float(parts[2])
        
        df: pd.DataFrame = pd.read_csv(file_path)
        total_methods: int = len(df)
        
        return InstanceResult(
            project=project,
            bug_id=bug_id,
            lambd=lambd,
            total_methods=total_methods,
            metrics_df=df
        )



def get_reader_for_lambda(target: str, lambd: float) -> BenchmarkReader:
    folder_path: Path = OUTPUT_DIR / target / f"eval_lambd_{lambd}"
    return BenchmarkReader(folder_path)

if __name__ == "__main__":
    # Test reading the newly generated output for lambda 0.1
    test_lambd: float = 0.1
    print(f"Reading {BENCHMARK_TARGET} outputs for lambda={test_lambd}")
    
    try:
        reader: BenchmarkReader = get_reader_for_lambda(BENCHMARK_TARGET, test_lambd)
        results: List[InstanceResult] = reader.read_all_instances()
        print(f"Read {len(results)} instances.")
        
        for result in results:
            print(f"Project: {result.project}, Bug ID: {result.bug_id}, Lambda: {result.lambd}, Total Methods: {result.total_methods}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
