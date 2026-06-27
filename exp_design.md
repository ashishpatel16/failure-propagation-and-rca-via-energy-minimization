# Experimental Design
We design our evaluation to assess whether integrating a structural graph-cut prior improves root-cause ranking compared to using regional evidence independently. The evaluation spans two distinct domains: source-level Spectrum-Based Fault Localization (SBFL) and distributed microservice Root-Cause Analysis (RCA). 

## Research Questions
We organize our empirical investigation around five research questions:
* **RQ2.1 (Effectiveness):** Does the graph-cut structural prior significantly improve localization accuracy over the underlying regional baseline heuristics?
* **RQ2.2 (Ambiguity Resolution):** To what extent does the framework resolve diagnostic ties and ambiguity groups produced by the baseline heuristics?
* **RQ2.3 (Design & Sensitivity):** How sensitive is the framework to the choice of edge weighting strategies and the structural smoothing hyperparameter $\lambda$?
* **RQ3.1 (Failure Modes):** Under what topological conditions does the structural prior degrade ranking performance?
* **RQ3.2 (Computational Cost):** Do graph construction, max-flow computation, and min-marginal recovery scale practically to industrial benchmark sizes?

## Datasets
We evaluate our approach using established benchmarks in both domains to ensure reproducibility and ground-truth reliability.

**Defects4J (SBFL):** We utilize the Defects4J benchmark [just2014defects4j]. A bug is included if the buggy version builds successfully, at least one test fails, coverage and call traces are successfully instrumented, and the developer patch maps to at least one executable method. The ground truth is defined as the set of patched methods. 

**RCA-Eval (Microservice RCA):** We utilize the RE2 and RE3 subsets of the RCA-Eval benchmark [pham2025rcaeval]. These subsets provide trace-based microservice failure scenarios containing multi-source telemetry, which are strictly required for reliable service dependency graph construction. We explicitly exclude the RE1 subset, as it relies exclusively on aggregated time-series metrics without the distributed traces necessary to infer structural topologies.

## Baselines and Configurations
Our evaluation employs a strictly paired design to isolate the impact of the structural prior. For SBFL, we evaluate three standard closed-form heuristics: Tarantula [jones2002tarantula], Ochiai [abreu2007ochiai], and DStar [wong2014dstar]. We compare the independent ranking of each heuristic against its graph-cut counterpart (e.g., Tarantula vs. GC-Tarantula). Both versions receive the exact same regional evidence; any variance in performance is exclusively attributable to the pairwise term. For RCA-Eval, we mirror this paired comparison against the benchmark's baseline service-level anomaly scores.

## Graph Construction and Edge Weighting
For Defects4J, nodes represent project methods and edges represent dynamic method calls. For RCA-Eval, nodes represent microservices and edges represent RPC interactions reconstructed from distributed traces.

To mitigate the confounding influence of heavily executed but innocent utility components, we introduce a fail-pass contrast weighting strategy. Let $|T_f|$ and $|T_p|$ denote the total number of failing and passing executions, respectively. 

## Hyperparameter Protocol
The smoothing parameter $\lambda$ dictates the influence of the graph topology. We evaluate $\lambda$ across a logarithmic grid: $\lambda \in \{0, 0.01, 0.1, 1, 10\}$. Our primary evaluation reports results using a fixed $\lambda$ selected on an isolated validation set. 

## Evaluation Metrics
We assess localization effectiveness using standard software engineering metrics: 
* Top-$k$ accuracy, $k \in \{1, 3, 5\}$ * EXAM score * Wasted Effort When baseline heuristics produce ambiguity groups (ties), we enforce the *average rank* (expected value of random tie-breaking) as the primary evaluation policy to prevent artificial inflation of baseline metrics. Best- and worst-rank scenarios are reported exclusively as sensitivity checks.