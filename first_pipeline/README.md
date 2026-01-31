
# MaxCut QAOA Transformer Benchmark

This repository contains a **fully reproducible benchmark pipeline** for studying
**parameter transfer in QAOA** using analytically solvable surrogate Hamiltonians.

## Core Idea

We study whether QAOA parameters trained on a transformed Hamiltonian \(J'\)
can be reused (warm-started) for the original MaxCut problem \(J\).

The transformation uses a compatible family:
\[
J'_{ij} = P_i + P_j
\]
which admits an **exact analytic ground state**.

## Pipeline Overview

1. **Benchmark Generation**
   - Erdős–Rényi (unweighted)
   - Erdős–Rényi (weighted)
   - 3-regular graphs
   - \(N \le 20\)

2. **Transformer (J → J')**
   - Power-law rank model for \(P_i\)
   - Lightweight proxies: row correlation, eigen overlap, frustration correlation

3. **Analytic Solver**
   - Closed-form solution for \(J'\) (used diagnostically only)

4. **QAOA Experiments**
   - Baseline: multi-start QAOA on \(J\)
   - Source: QAOA on dense \(J'\)
   - Transfer-only: evaluate \(\theta^*_{J'}\) on \(J\)
   - Warm-finetune: local optimization on \(J\)

5. **Evaluation**
   - Exact brute-force MaxCut (for \(N \le 20\))
   - Approximation ratios
   - Gain vs baseline
   - Correlation diagnostics

## Key Result

Despite capturing **classical structure**, the transformer does **not consistently
improve QAOA performance** when used for parameter transfer.

This demonstrates a clear gap between:
- Classical similarity of Hamiltonians
- Quantum variational behavior under QAOA

## Repository Structure

- `first_classicalTrans_pipeline(nonQAOA_AWARE).ipynb` – complete experimental pipeline
- `results_all.json` – raw results
- `summary.csv` – aggregated metrics


## Reproducibility

- Deterministic seeds
- N ≤ 20 (exact brute-force evaluation)
- PennyLane backend (statevector simulator)

## Citation

If you use this benchmark, please cite appropriately once the paper is available.

