import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

MODULE_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = MODULE_DIR.parent.parent
RCA_EVAL_DIR: Path = ROOT_DIR / "rca-eval"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(RCA_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RCA_EVAL_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from analysis import RCASensitivityAnalyzer
from data_loader import load_ground_truth_dataset
from evaluator import RCASensitivityEvaluator
from rca_models import (
    RCAInstanceMetadata,
    RCASampleTrialResult,
    RCASensitivityConfig,
    RCASensitivitySummaryRow,
    RCASensitivityVerdict,
    Suite,
    UnaryPriorStrategy,
)

DATA_DIR: Path = ROOT_DIR / "notebooks" / "data"
OUTPUT_DIR: Path = MODULE_DIR / "outputs"

SUITES: List[Suite] = [Suite.RE2, Suite.RE3]
WINDOW_MINUTES_LIST: List[int] = [1, 2, 5, 10, 20]
METRIC_FRACTIONS: List[float] = [0.25, 0.50, 0.75, 1.00]
N_METRIC_TRIALS: int = 10
LAMBDAS: List[float] = [0.0, 0.1, 1.0, 10.0]
RHO: float = 1.0
STRATEGY: UnaryPriorStrategy = UnaryPriorStrategy.RANK_BASED
EPS: float = 1e-4
TDELTA: int = 0

RANDOM_SEED: int = 42
MAX_WORKERS: int = 3

SELECTED_INSTANCES: List[str] = []

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger: logging.Logger = logging.getLogger("RCASensitivityRunner")


def _run_instance_worker(
    instance: RCAInstanceMetadata, config: RCASensitivityConfig
) -> Tuple[List[RCASampleTrialResult], List[RCASensitivitySummaryRow]]:
    evaluator: RCASensitivityEvaluator = RCASensitivityEvaluator()
    return evaluator.evaluate_instance_sensitivity(instance=instance, config=config)


def main() -> None:
    """Main pipeline execution for RCA sensitivity analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    instance_out_dir: Path = OUTPUT_DIR / "instances"
    instance_out_dir.mkdir(parents=True, exist_ok=True)

    config: RCASensitivityConfig = RCASensitivityConfig(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        suites=SUITES,
        window_minutes_list=WINDOW_MINUTES_LIST,
        metric_fractions=METRIC_FRACTIONS,
        n_metric_trials=N_METRIC_TRIALS,
        lambdas=LAMBDAS,
        rho=RHO,
        strategy=STRATEGY,
        eps=EPS,
        tdelta=TDELTA,
        random_seed=RANDOM_SEED,
        max_workers=MAX_WORKERS,
    )

    all_instances: List[RCAInstanceMetadata] = load_ground_truth_dataset(
        data_dir=config.data_dir, suites=config.suites
    )

    if len(SELECTED_INSTANCES) > 0:
        filter_set: set = set(SELECTED_INSTANCES)
        target_instances: List[RCAInstanceMetadata] = [
            inst
            for inst in all_instances
            if f"{inst.fault}_{inst.instance_id}" in filter_set
            or inst.fault in filter_set
        ]
    else:
        target_instances = all_instances

    if len(target_instances) == 0:
        raise ValueError("No target instances matched the selection criteria.")

    logger.info(
        f"Starting RCA sensitivity analysis on {len(target_instances)} instances across windows {config.window_minutes_list}m"
    )

    all_trials: List[RCASampleTrialResult] = []
    all_summaries: List[RCASensitivitySummaryRow] = []
    analyzer: RCASensitivityAnalyzer = RCASensitivityAnalyzer()

    if config.max_workers > 1 and len(target_instances) > 1:
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_inst = {
                executor.submit(_run_instance_worker, inst, config): inst
                for inst in target_instances
            }
            with tqdm(
                total=len(target_instances), desc="Evaluating RCA Instances", unit="instance"
            ) as pbar:
                for future in as_completed(future_to_inst):
                    inst_meta: RCAInstanceMetadata = future_to_inst[future]
                    inst_name: str = f"{inst_meta.suite.value}_{inst_meta.dataset}_{inst_meta.fault}_{inst_meta.instance_id}"
                    try:
                        trials, summaries = future.result()
                        all_trials.extend(trials)
                        all_summaries.extend(summaries)

                        df_inst_trials: pd.DataFrame = analyzer.trial_results_to_dataframe(trials)
                        df_inst_trials.to_csv(instance_out_dir / f"{inst_name}_trials.csv", index=False)

                        pbar.set_postfix_str(f"Finished {inst_name}")
                        pbar.update(1)
                    except Exception as exc:
                        logger.error(f"Error processing {inst_name}: {exc}")
                        pbar.update(1)
    else:
        with tqdm(
            total=len(target_instances), desc="Evaluating RCA Instances", unit="instance"
        ) as pbar:
            for inst_meta in target_instances:
                inst_name = f"{inst_meta.suite.value}_{inst_meta.dataset}_{inst_meta.fault}_{inst_meta.instance_id}"
                trials, summaries = _run_instance_worker(inst_meta, config)
                all_trials.extend(trials)
                all_summaries.extend(summaries)
                df_inst_trials = analyzer.trial_results_to_dataframe(trials)
                df_inst_trials.to_csv(instance_out_dir / f"{inst_name}_trials.csv", index=False)
                pbar.set_postfix_str(f"Finished {inst_name}")
                pbar.update(1)

    if len(all_trials) == 0:
        raise ValueError("No successful RCA trials were recorded.")

    df_trials: pd.DataFrame = analyzer.trial_results_to_dataframe(all_trials)
    df_trials.to_csv(OUTPUT_DIR / "rca_trials_master.csv", index=False)
    logger.info(f"Saved master trials dataset: {OUTPUT_DIR / 'rca_trials_master.csv'} ({len(df_trials)} rows)")

    df_summaries: pd.DataFrame = analyzer.summary_rows_to_dataframe(all_summaries)
    df_summaries.to_csv(OUTPUT_DIR / "rca_summary.csv", index=False)
    logger.info(f"Saved summary dataset: {OUTPUT_DIR / 'rca_summary.csv'} ({len(df_summaries)} rows)")

    verdicts: List[RCASensitivityVerdict] = analyzer.compute_window_verdicts(df_trials)
    if len(verdicts) > 0:
        df_verdicts: pd.DataFrame = analyzer.verdicts_to_dataframe(verdicts)
        df_verdicts.to_csv(OUTPUT_DIR / "rca_verdicts.csv", index=False)
        logger.info(f"Saved verdicts dataset: {OUTPUT_DIR / 'rca_verdicts.csv'}")

    df_aggregate: pd.DataFrame = analyzer.build_window_aggregate_report(df_trials)
    df_aggregate.to_csv(OUTPUT_DIR / "rca_window_aggregate_report.csv", index=False)
    logger.info(f"Saved window aggregate report: {OUTPUT_DIR / 'rca_window_aggregate_report.csv'}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 15)
    print(df_aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
