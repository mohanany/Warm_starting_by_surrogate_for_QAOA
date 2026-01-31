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
- `transformers_baseline.ipynb`
  - Development notebook used during earlier iterations.
- `first_pipeline/`
  - Stage I notebook + notes + figures.
- `WRITEUP_*.md`
  - Detailed technical writeups for Stage II.
    ## Paper artifacts (figure reproduction)
