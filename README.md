# Behavioral Warm-Starting for QAOA (Two-Pipeline Research Repo)

 This repository contains **two pipelines** we developed to study **warm-starting / parameter transfer for QAOA** on MaxCut.

- **Stage I (Pipeline 1)**: an *analytic-structure transformer* that maps a target coupling matrix `J` to a solvable surrogate `J'` constrained to the family `J'_{ij} = P_i + P_j`.
- **Stage II (Pipeline 2 / v7.7)**: a *behavior-preserving surrogate selection* pipeline that chooses a donor surrogate from structured families using a **QAOA-aware fingerprint** (value ranking + directional-derivative alignment + hit@k), then transfers and fine-tunes.

 The `WRITEUP_*` documents focus mostly on Stage II (the final pipeline), while Stage I notes live in `first_pipeline/`.

