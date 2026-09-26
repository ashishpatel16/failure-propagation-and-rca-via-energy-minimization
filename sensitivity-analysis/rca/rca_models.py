"""Data structures, enums, and types for microservice RCA sensitivity analysis."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List


class Suite(Enum):
    RE2 = "RE2"
    RE3 = "RE3"


class UnaryPriorStrategy(Enum):
    SIGMOID = "sigmoid"
    MINMAX = "minmax"
    RANK_BASED = "rank_based"


class RCASamplingDimension(Enum):
    TIME_WINDOW = "time_window"
    METRIC_DROPOUT = "metric_dropout"


class VerdictType(Enum):
    DAMPENS = "dampens"
    AMPLIFIES = "amplifies"
    NO_EFFECT = "no_effect"


@dataclass(frozen=True)
class RCAInstanceMetadata:
    suite: Suite
    dataset: str
    fault: str
    instance_id: int
    root_cause: str
    fault_type: str
    inject_time: int
    instance_dir: Path


@dataclass
class RCASampleTrialResult:
    instance_name: str
    suite: str
    dataset: str
    fault: str
    root_cause: str
    dimension_type: str
    dimension_value: float
    trial_idx: int
    lambda_val: float
    rho_val: float
    strategy: str
    arm_name: str
    predicted_rank: int
    is_top_1: bool
    is_top_3: bool
    is_top_5: bool
    mrr: float
    full_window_baseline_rank: int
    rank_deviation: float
    total_services: int


@dataclass
class RCASensitivitySummaryRow:
    instance_name: str
    dimension_type: str
    dimension_value: float
    lambda_val: float
    arm_name: str
    n_trials: int
    full_window_baseline_rank: int
    mean_rank: float
    median_rank: float
    std_rank: float
    var_rank: float
    mean_deviation_from_reference: float
    top_1_rate: float
    top_3_rate: float
    top_5_rate: float
    mean_mrr: float
    total_services: int


@dataclass
class RCASensitivityVerdict:
    instance_name: str
    dimension_type: str
    dimension_value: float
    pairwise_lambda: float
    baseline_std: float
    pairwise_std: float
    std_ratio_base_over_pairwise: float
    baseline_top_1: float
    pairwise_top_1: float
    baseline_mrr: float
    pairwise_mrr: float
    verdict: str


@dataclass
class RCASensitivityConfig:
    data_dir: Path
    output_dir: Path
    suites: List[Suite]
    window_minutes_list: List[int]
    metric_fractions: List[float]
    n_metric_trials: int
    lambdas: List[float]
    rho: float
    strategy: UnaryPriorStrategy
    eps: float
    tdelta: int
    random_seed: int
    max_workers: int
