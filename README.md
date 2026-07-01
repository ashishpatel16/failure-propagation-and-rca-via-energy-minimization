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
git clone https://github.com/phamquiluan/RCAEval.git
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

**Notes**
- Configuration for the evaluation sweeps (such as `LAMBDAS_TO_ABLATE`, `SAMPLE_SIZE`, and `BENCHMARK_TARGET`) is located in `src/evaluation/config.py`. Adjust this file before running `run_evaluation.py`.
- Outputs from the evaluations are saved as CSVs to directories like `outputs_complete_directional_full_benchmark/` or `data/`.