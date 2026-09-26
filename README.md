   # Failure Propagation and Root-Cause Analysis via Energy Minimization

Locating the root cause of a failure is the same problem at every scale: a defect in one component contaminates the observable behavior of its neighbors, yet traditional Spectrum-Based Fault Localization (SBFL) and microservice Root-Cause Analysis (RCA) score each component in isolation, often ranking the true fault alongside its propagation symptoms. 

This work reformulates root-cause localization as a binary labeling problem on a system dependency graph `G`, where each node carries an anomaly score `S_i` (from any existing baseline such as Tarantula, Ochiai, DStar, or BARO) and each edge encodes structural propagation strength; following the graph-cut formulation of Boykov and Jolly, we minimize an energy that fuses this regional evidence with a pairwise structural prior, which under submodularity is solvable exactly in polynomial time via an s-t min-cut. We finally recover a continuous ranking from min-marginal energies. Setting the structural weight `λ = 0` provably recovers the original baseline ranking, so the method strictly generalizes SBFL rather than replacing it. We evaluate on two domains: SBFL on Defects4J and microservice RCA on RCA-Eval (RE2, RE3).

## Installation

### Prerequisites
- [uv](https://docs.astral.sh/uv/) for fast dependency management
- **Python 3.12+**
- **JDK 8+** on your `PATH` (required for the Defects4J Java instrumentation agent)

#### Setup Defects4J (for SBFL Evaluation)
```bash
git clone https://github.com/rjust/defects4j.git
cd defects4j
cpanm --installdeps .
./init.sh
export PATH=$PATH:"$(pwd)/framework/bin"
```

#### Setup RCA-Eval (for Microservice Evaluation)
```bash
git clone https://github.com/phamquiluan/RCAEval.git rca-eval
# Extract the relevant evaluation datasets 
```

### Project Setup
```bash
git clone https://anonymous.4open.science/status/failure-propagation-and-rca-via-energy-minimization-8DDA
cd failure-propagation-and-rca-via-energy-minimization

# uv automatically creates a virtual environment and installs dependencies from pyproject.toml
uv sync

# Activate the virtual environment
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Configuration (.env)
Create a `.env` file in the root of the project to configure paths for the Defects4J evaluation. You can copy the structure from `.env.example`:

```bash
# Example .env configuration
JAVA_HOME_PATH=
D4J_PATH=
CALLGRAPH_DIR=
```

## Reproducing the results

The repository uses standalone scripts in the `scripts/` and `src/` directories to run the evaluations. Configuration (like target lambdas and datasets) is managed in `src/evaluation/config.py`.

```bash
# 1. Run the RCA (Microservices) Evaluation Pipeline
uv run python src/evaluation/rca_eval.py

# 2. Run the SBFL (Defects4J) Extraction and Experiments
uv run python scripts/run_sbfl_experiments.py

# 3. Run the complete benchmark evaluation (controlled via config.py)
uv run python scripts/run_evaluation.py

# 4. Analyze Results (Aggregating metrics like Top-K, MRR)
uv run python scripts/analyze_results.py
uv run python scripts/analyze_benchmark_results.py
```

---

## Sensitivity & Robustness Analysis

We provide an extensive empirical evaluation assessing the robustness and noise-dampening capabilities of **Topological Energy Minimization via $s\text{--}t$ Graph Cuts** under practical real-world data constraints:

1. **Microservice Telemetry Sparsity & Window Truncation (RCA-Eval, 253 Instances):**
   - **Metric Withholding (25% to 75% telemetry availability):** When metric counters are withheld or dropped, graph cuts consistently suppress within-instance root-cause ranking variance by **22.3% to 46.7%** ($p = 1.70 \times 10^{-17}$ via paired $t$-test, $p = 9.30 \times 10^{-36}$ via Wilcoxon signed-rank test).
   - **Time-Window Truncation (1m to 20m triage windows):** In standard 5–20 minute incident triage windows, BARO-GC outranks unregularized BARO by **+1.06 to +1.35 mean rank positions** ($p < 0.05$), achieving equivalent diagnostic accuracy with up to $10\times$ less observation time.
2. **Defects4J Test Budget Subsampling (SBFL, 10 Draws per Level):**
   - **Passing Test Subsampling (25%, 50%, 75% test budgets):** Pairwise regularization at $\lambda = 1.0$ reduces mean rank variance by **63.4%** at a 25% test budget ($265.25 \to 97.15$) and **54.2%** at a 50% test budget ($108.27 \to 49.60$).
   - Up to **45.6%** of software defects experience significant variance dampening, acting as an isotropic low-pass filter over the static call graph.


### Reproducing Sensitivity Ablations

#### 1. Microservice RCA Sensitivity Sweep & Significance
```bash
# Run the metric sparsity and window duration factorial sweep across all 253 instances
uv run python sensitivity-analysis/rca/run_rca_sensitivity.py

# Compute paired t-tests, Wilcoxon tests, and p-values
uv run python sensitivity-analysis/rca/statistical_significance_rca.py

# Generate sensitivity plots and charts
uv run python sensitivity-analysis/rca/plot_rca_sensitivity.py
```

#### 2. Defects4J SBFL Test Subsampling Sweep & Significance
```bash
# Run the passing test subsampling sweep (10 draws per budget level across instances)
uv run python sensitivity-analysis/defects4j/run_sensitivity.py

# Compute paired t-tests, Wilcoxon signed-rank tests, and Cohen's d effect sizes
uv run python sensitivity-analysis/defects4j/statistical_significance_test.py

# Generate sensitivity plots and variance distributions
uv run python sensitivity-analysis/defects4j/plot_sensitivity.py
```

---

**Notes**
- Configuration for the evaluation sweeps (such as `LAMBDAS_TO_ABLATE`, `SAMPLE_SIZE`, and `BENCHMARK_TARGET`) is located in `src/evaluation/config.py`. Adjust this file before running `run_evaluation.py`.
- Outputs from the evaluations are saved as CSVs to directories like `outputs_complete_directional_full_benchmark/`, `outputs_rca/`, `sensitivity-analysis/defects4j/outputs/`, or `sensitivity-analysis/rca/outputs/`.