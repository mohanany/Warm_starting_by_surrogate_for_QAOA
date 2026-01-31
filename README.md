# Behavioral Warm-Starting for QAOA (Two-Pipeline Research Repo)

 This repository contains **two pipelines** we developed to study **warm-starting / parameter transfer for QAOA** on MaxCut.

- **Stage I (Pipeline 1)**: an *analytic-structure transformer* that maps a target coupling matrix `J` to a solvable surrogate `J'` constrained to the family `J'_{ij} = P_i + P_j`.
- **Stage II (Pipeline 2 / v7.7)**: a *behavior-preserving surrogate selection* pipeline that chooses a donor surrogate from structured families using a **QAOA-aware fingerprint** (value ranking + directional-derivative alignment + hit@k), then transfers and fine-tunes.

 The `WRITEUP_*` documents focus mostly on Stage II (the final pipeline), while Stage I notes live in `first_pipeline/`.

## Repository structure

- `behavior_surrogate_pipeline_v7_7_gpu.py`
  - Stage II runnable script (CPU via NumPy, optional GPU via PyTorch).
- `FINAL_RUN_OF_7.7_GPU.ipynb`
  - Example notebook to run Stage II sweeps.
- `transformers_development_firststage.ipynb`
  - Development notebook used during earlier iterations.
- `first_pipeline/`
  - Stage I notebook + notes + figures.
- `WRITEUP_*.md`
  - Detailed technical writeups for Stage II.

## Paper artifacts (figure reproduction)

 We include a minimal, paper-facing artifact bundle under `paper_artifacts/`:

- `paper_artifacts/er_clean_results.csv` (ER-only reconstructed results used for paper plots)
- `paper_artifacts/plot_paper_figs_from_csv.py` (regenerates the paper figures from the CSV)

 Example:

```bash
 python paper_artifacts/plot_paper_figs_from_csv.py \
   --csv paper_artifacts/er_clean_results.csv \
   --out paper_artifacts/figures
```

## Citation

 This repository includes `CITATION.cff` so GitHub can display a “Cite this repository” entry.

## Installation

 Create a virtual environment and install requirements:

```bash
 python -m venv .venv
 source .venv/bin/activate
 pip install -r requirements.txt
```

 Notes:

- `torch` is optional (used for GPU acceleration when available). If you do not want it, remove it from `requirements.txt`.
- Stage II is intended for **small n** due to exact statevector simulation and brute-force optima.

## Stage II (v7.7) — Behavior-Surrogate Pipeline

### Quick run (single setting)

```bash
 python behavior_surrogate_pipeline_v7_7_gpu.py \
   --task maxcut \
   --n 12 \
   --p_target 3 \
   --p_match 3 \
   --families ER_dense_p05_weighted,RR_3regular \
   --seeds_per_family 2 \
   --save_dir ./run_v7_7
```

### Sweep over `(n,p)` (matches the notebook)

```bash
 python behavior_surrogate_pipeline_v7_7_gpu.py \
   --backend auto --device auto \
   --n_values 12,14,18,20 \
   --p_values 2,3,4,5 \
   --save_dir ./run_v7_7_auto
```

### Outputs

 For each run directory:

- `results_all.json`: per-instance records (opt/base/tr/warm/randFT + fingerprint + gating + accounting)
- `summary.csv`: aggregated metrics per family

## Stage I (Pipeline 1) — Analytic Transformer `J -> J'`

 Stage I is implemented as a notebook pipeline:

- `first_pipeline/first_classicalTrans_pipeline(nonQAOA_AWARE).ipynb`

 It benchmarks parameter transfer when the surrogate is constrained to the analytic family:

 \[
 J'_{ij} = P_i + P_j.
 \]

 It includes:

- learning/constructing `P` from `J` using classical proxies
- analytic ground-state solver for the surrogate family
- QAOA training on `J'` and transfer to `J`
- empirical finding: **classical similarity does not reliably yield QAOA transfer gains**

## Notes on reproducibility / scope

- Both pipelines are **research benchmarks**.
- Stage II computes exact energies (statevector) and exact optima (brute force), so runtime scales exponentially with `n`.
- For NISQ/hardware, the evaluator should be replaced with sampling-based estimation.
