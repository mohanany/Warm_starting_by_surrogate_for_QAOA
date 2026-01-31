# Behavior-Surrogate Pipeline v7.7 - Usage Guide and Summary

## Part 5: Usage, Hyperparameters, and Interpretation

### 5.1 Quick Start

#### Minimal Example

```bash
python behavior_surrogate_pipeline_v7_7_gpu.py \
  --n 12 \
  --p_target 3 \
  --p_match 3 \
  --families ER_dense_p05_weighted \
  --seeds_per_family 2 \
  --save_dir ./results_quick
```

**Output:**

- `results_quick/results_all.json`: Per-instance results
- `results_quick/summary.csv`: Aggregated statistics

**Runtime:** ~2-5 minutes per instance (n=12, p=3, single core)

#### Recommended Configuration (Paper-Grade)

```bash
python behavior_surrogate_pipeline_v7_7_gpu.py \
  --n 12 \
  --p_target 3 \
  --p_match 3 \
  --families ER_dense_p05_weighted,RR_3regular \
  --seeds_per_family 10 \
  --iters 150 \
  --finetune_iters 160 \
  --restarts_baseline 8 \
  --fp_points 15 \
  --fp_cands 50 \
  --save_traces \
  --save_dir ./results_paper
```

**Changes from default:**

- More instances (10 seeds/family)
- Longer optimization (150/160 iters)
- More restarts (8 baseline)
- More fingerprint probes (15) and candidates (50)
- Save convergence traces for analysis

**Runtime:** ~50-80 minutes for 20 instances

---

### 5.2 Hyperparameter Reference

#### Problem Configuration

| **Parameter**    | **Default** | **Range** | **Effect**                                              |
| ---------------------- | ----------------- | --------------- | ------------------------------------------------------------- |
| `--n`                | 12                | 4-14            | Problem size (qubits). Larger → exponentially slower.        |
| `--p_target`         | 3                 | 1-10            | Target QAOA depth. Larger → better quality, more parameters. |
| `--p_match`          | 3                 | 1-10            | Fingerprint depth. Match p_target for consistency.            |
| `--families`         | ER_dense...       | *               | Target graph families (comma-separated).                      |
| `--seeds_per_family` | 2                 | 1-100           | Instances per family. More → better statistics.              |

**Target families:**

- `ER_dense_p05_weighted`: Erdős-Rényi (p=0.5, weighted)
- `RR_3regular`: Random 3-regular (unweighted)
- `Complete_weighted_dense`: Complete graph (weighted)

#### Optimization Settings

| **Parameter**       | **Default** | **Range** | **Effect**                                         |
| ------------------------- | ----------------- | --------------- | -------------------------------------------------------- |
| `--iters`               | 120               | 50-300          | Baseline SPSA iterations. More → better convergence.    |
| `--finetune_iters`      | 140               | 50-300          | Warm fine-tune iterations. More → better final quality. |
| `--src_iters`           | 120               | 50-300          | Donor optimization iterations.                           |
| `--restarts_baseline`   | 6                 | 1-20            | Multi-start count. More → better global search.         |
| `--restarts_source`     | 4                 | 1-10            | Donor multi-start count.                                 |
| `--probe_rand_restarts` | 3                 | 1-10            | Random fine-tune restarts (baseline comparison).         |

**Budget implications:**

- Baseline evals ≈ (2·iters + 1) · restarts_baseline
- Warm evals ≈ (2·finetune_iters + 1)
- Total target evals ≈ baseline + warm + overhead (~2500-3000)

#### SPSA Tuning

| **Parameter** | **Default** | **Range** | **Effect**                                              |
| ------------------- | ----------------- | --------------- | ------------------------------------------------------------- |
| `--spsa_a`        | 0.25              | 0.05-1.0        | Step size. Larger → faster initial progress, less stability. |
| `--spsa_c`        | 0.12              | 0.01-0.5        | Perturbation size. Larger → coarser gradient estimate.       |
| `--spsa_a_ft`     | 0.20              | 0.05-0.5        | Fine-tune step size (smaller = more refined).                 |
| `--spsa_c_ft`     | 0.10              | 0.01-0.3        | Fine-tune perturbation size.                                  |
| `--spsa_alpha`    | 0.602             | 0.5-1.0         | Step size decay (Spall recommendation: 0.602).                |
| `--spsa_gamma`    | 0.101             | 0.05-0.2        | Perturbation decay (Spall: 0.101).                            |
| `--spsa_A`        | 10.0              | 0-50            | Offset for stability (prevents initial steps too large).      |
| `--grad_clip`     | 5.0               | 0-20            | Max gradient norm. Larger → allows big steps.                |

**Tuning heuristics:**

- **If baseline converges slowly:** Increase `spsa_a` or `restarts_baseline`
- **If optimization is noisy:** Decrease `spsa_a`, increase `spsa_c`
- **If warm start diverges:** Decrease `spsa_a_ft`, increase `grad_clip`

#### Fingerprint Configuration

| **Parameter**  | **Default** | **Range** | **Effect**                                                 |
| -------------------- | ----------------- | --------------- | ---------------------------------------------------------------- |
| `--fp_points`      | 12                | 5-30            | Full probe count. More → better correlation estimate, costlier. |
| `--fp_pre_points`  | 4                 | 2-10            | Pre-selection probes. Fewer → faster screening.                 |
| `--fp_cands`       | 40                | 10-100          | Candidate count. More → better chance of match, costlier.       |
| `--fp_preselect`   | 12                | 5-40            | Top candidates for full fingerprint.                             |
| `--fp_dirs`        | 3                 | 1-10            | Directional derivative count. More → better gradient estimate.  |
| `--fp_eps`         | 2e-3              | 1e-4-1e-2       | Finite difference epsilon. Too small → numerical noise.         |
| `--fp_local_frac`  | 0.5               | 0.0-1.0         | Local probe fraction. 0 = all global, 1 = all local.             |
| `--fp_noise_sigma` | 0.05              | 0.01-0.2        | Global probe noise (radians). Larger → broader coverage.        |

**Fingerprint weights:**

| **Parameter** | **Default** | **Range** | **Effect**           |
| ------------------- | ----------------- | --------------- | -------------------------- |
| `--fp_w_rho`      | 0.45              | 0.0-1.0         | Value correlation weight.  |
| `--fp_w_grad`     | 0.45              | 0.0-1.0         | Gradient alignment weight. |
| `--fp_w_hit`      | 0.10              | 0.0-1.0         | Hit@k weight.              |

**Total must sum to 1.0**

**Tuning advice:**

- **Emphasize value matching:** Increase `fp_w_rho` (e.g., 0.6/0.3/0.1)
- **Emphasize dynamics:** Increase `fp_w_grad` (e.g., 0.3/0.6/0.1)
- **More local exploration:** Increase `fp_local_frac` (e.g., 0.7)

#### Gate Thresholds

| **Parameter**   | **Default** | **Range** | **Effect**                                         |
| --------------------- | ----------------- | --------------- | -------------------------------------------------------- |
| `--gate_rho`        | 0.55              | 0.0-1.0         | Minimum Spearman ρ for eligibility. Higher → stricter. |
| `--gate_grad`       | 0.25              | 0.0-1.0         | Minimum gradient cosine. Higher → stricter.             |
| `--gate_hit`        | 1.0               | 0.0-1.0         | Minimum hit@k (0 or 1). 1 → requires hit, 0 → ignores. |
| `--transfer_margin` | 0.0               | 0.0-0.5         | Transfer sanity margin. tr must be ≥ base - margin.     |

**Effect on pass rate:**

- **Strict gates** (ρ ≥ 0.7, grad ≥ 0.4, hit = 1): ~40% pass rate, high quality
- **Moderate gates** (default): ~60-70% pass rate, balanced
- **Loose gates** (ρ ≥ 0.3, grad ≥ 0.1, hit = 0): ~90% pass rate, more noise

#### Advanced Options

| **Parameter**          | **Type** | **Default** | **Effect**                                                    |
| ---------------------------- | -------------- | ----------------- | ------------------------------------------------------------------- |
| `--try_topk_donors`        | int            | 0                 | Try K best fingerprint candidates with probe optimization. 0 = off. |
| `--src_probe_iters`        | int            | 35                | Donor probe iterations (if try_topk > 0).                           |
| `--src_probe_restarts`     | int            | 2                 | Donor probe restarts.                                               |
| `--use_2d_scaling_grid`    | flag           | False             | Enable 2D (β,γ) scaling grid. Default: 1D γ-only.                |
| `--beta_mults`             | str            | "0.9,1.0,1.1"     | Beta multipliers for 2D grid.                                       |
| `--gamma_mults`            | str            | "0.9,1.0,1.1"     | Gamma multipliers for 2D grid.                                      |
| `--warm_gamma_mults`       | str            | "1.0,0.95,1.05"   | Gamma multipliers for 1D grid (default).                            |
| `--layer_selective_steps`  | int            | 40                | Warmup iterations (last layers only).                               |
| `--layer_selective_layers` | int            | 1                 | Number of last layers to optimize in warmup.                        |
| `--micro_warm`             | flag           | False             | Enable micro-warm rescue for borderline transfers.                  |
| `--micro_warm_margin`      | float          | 0.25              | Trigger micro-warm if base - tr ≤ margin.                          |
| `--micro_warm_iters`       | int            | 30                | Micro-warm iterations.                                              |
| `--save_traces`            | flag           | False             | Save optimization traces in results_all.json.                       |

#### Surrogate Families

| **Parameter**      | **Default**                          | **Options**    |
| ------------------------ | ------------------------------------------ | -------------------- |
| `--surrogate_families` | "powlaw,strength,block,circulant,lowrank2" | Comma-separated list |

**Available families:**

- `powlaw`: Power-law additive (heterogeneous nodes)
- `strength`: Target strength-based (degree-preserving)
- `block`: Block/community structure
- `circulant`: Translation-invariant (ring-like)
- `lowrank2`: Low-rank (latent 2D embedding)

---

### 5.3 Interpreting Results

#### Understanding results_all.json

**Key fields per instance:**

```json
{
  "opt": 15.234,      // Optimal MaxCut value (brute-force)
  "base": 13.456,     // Baseline (multi-start SPSA from random)
  "tr": 13.789,       // Transfer initialization quality
  "warm": 14.123,     // Final warm-started result
  "randFT": 13.567,   // Random fine-tune baseline (fair comparison)
  
  "gain_warm": 0.667,     // warm - base (improvement)
  "gain_randFT": 0.111,   // randFT - base
  "gain_cond": 0.667,     // Conditional gain (NaN if gate skipped)
  "gain_uncond": 0.667,   // Unconditional gain (0 if skipped)
  
  "gate": "PASS",         // or "SKIP"
  "gate_reason": "pass",  // or "fp_gate_fail", "transfer_worse_than_base"
  
  "fp_rho": 0.78,         // Fingerprint value correlation
  "fp_grad": 0.64,        // Fingerprint gradient alignment
  "fp_hit": 1.0,          // Hit@k (0 or 1)
  "fp_score": 0.739,      // Composite score
  
  "donor": "circulant",   // Selected surrogate family
  "scale": 1.05,          // Edge weight scaling factor
  
  "evals_total": 11036,   // Total evaluations (all stages)
  "evals_target_base": 1446,    // Baseline only
  "evals_target_overhead": 14,  // Fingerprint + transfer
  "evals_target_warm": 281,     // Warm fine-tune
  "evals_donor_opt": 964,       // Donor optimization
  
  "time_total_s": 23.456
}
```

**Quality assessment:**

- **Approximation ratio:** `base / opt` (baseline quality)
- **Improvement:** `gain_warm / (opt - base)` (fraction of gap closed)
- **Random comparison:** `gain_warm > gain_randFT` (warm beats random?)
- **Efficiency:** `gain_warm / evals_target_overhead` (gain per extra eval)

#### Understanding summary.csv

**Aggregated per family:**

```csv
family,fp_rho,fp_grad,fp_hit,fp_score,pass_rate,gain_warm_uncond,gain_warm_cond,gain_randFT,ratio_base,ratio_warm_cond,evals_total
ER_dense_p05_weighted,0.723,0.612,0.95,0.698,0.85,0.543,0.639,0.123,0.882,0.927,11023
```

**Key metrics:**

- **pass_rate:** Fraction passing both gates (reliability)
- **gain_warm_uncond:** Average gain (including skips as 0)
- **gain_warm_cond:** Average gain (passes only)
- **gain_randFT:** Baseline comparison
- **ratio_base:** base / opt (baseline quality)
- **ratio_warm_cond:** warm / opt (final quality, passes only)

**Success criteria:**

1. **pass_rate ≥ 0.7:** Method is reliable
2. **gain_warm_cond > gain_randFT:** Warm beats random fine-tune
3. **gain_warm_uncond > 0:** Net positive (even with skips)
4. **ratio_warm_cond > ratio_base:** Final quality better than baseline

#### Common Patterns

**Successful transfer:**

```
fp_rho ≥ 0.75, fp_grad ≥ 0.6, fp_hit = 1
gate = PASS
gain_warm ≈ 0.5-1.0
donor = circulant or strength (often)
```

**Failed fingerprint:**

```
fp_rho < 0.4, fp_grad < 0.2
gate = SKIP
gate_reason = "fp_gate_fail"
```

**Failed transfer sanity:**

```
fp_rho ≥ 0.55, fp_grad ≥ 0.25 (passed Gate 1)
tr < base
gate = SKIP
gate_reason = "transfer_worse_than_base"
```

**Borderline (micro-warm could help):**

```
tr slightly < base (e.g., base - tr ≈ 0.15)
Enable --micro_warm to test short rescue
```

---

### 5.4 Experimental Design Recommendations

#### For Paper/Publication

**Goal:** Rigorous evaluation with statistical significance.

**Configuration:**

```bash
python behavior_surrogate_pipeline_v7_7_gpu.py \
  --n 12 --p_target 3 --p_match 3 \
  --families ER_dense_p05_weighted,RR_3regular,Complete_weighted_dense \
  --seeds_per_family 20 \
  --iters 150 --finetune_iters 160 \
  --restarts_baseline 8 --restarts_source 6 \
  --fp_points 15 --fp_cands 60 --fp_preselect 15 \
  --save_traces
```

**Statistical analysis:**

1. **t-test:** Compare gain_warm_cond vs gain_randFT (paired)
2. **Effect size:** Cohen's d = (mean_warm - mean_rand) / pooled_std
3. **Pass rate:** Report with 95% confidence interval (Wilson score)
4. **Boxplots:** gain_warm per family, overlay randFT
5. **Correlation:** fp_score vs gain_warm (validate fingerprint)

**Report:**

- Mean ± std for all metrics
- Median + quartiles (robust to outliers)
- Pass rate per family
- Eval breakdown (target vs donor vs overhead)

#### For Exploration/Debugging

**Goal:** Quick iteration, diagnose issues.

**Configuration:**

```bash
python behavior_surrogate_pipeline_v7_7_gpu.py \
  --n 10 --p_target 2 --p_match 2  # Smaller, faster
  --seeds_per_family 2
  --iters 80 --finetune_iters 80
  --restarts_baseline 3
  --fp_points 8 --fp_cands 20
  --save_traces  # Enable to see convergence
```

**Diagnostics:**

- **Check traces:** Is baseline converging? (trace_base plateau?)
- **Fingerprint quality:** Are fp_rho/fp_grad reasonable (≥0.5/0.3)?
- **Transfer gap:** tr vs base (if too negative, tune gates or gamma_clip)
- **Donor diversity:** Which families selected? (check "donor" field)

#### For Scaling Studies

**Goal:** Vary n, p, measure trends.

**Sweep n:**

```bash
for n in 8 10 12 14; do
  python behavior_surrogate_pipeline_v7_7_gpu.py \
    --n $n --p_target 3 --p_match 3 \
    --families ER_dense_p05_weighted,RR_3regular \
    --seeds_per_family 5 \
    --save_dir ./run_n${n}
done
```

**Sweep p:**

```bash
for p in 2 3 4 5; do
  python behavior_surrogate_pipeline_v7_7_gpu.py \
    --n 12 --p_target $p --p_match $p \
    --families ER_dense_p05_weighted,RR_3regular \
    --seeds_per_family 5 \
    --save_dir ./run_p${p}
done
```

**Analyze:**

- Plot eval_total vs n (exponential scaling?)
- Plot pass_rate vs p (harder at higher p?)
- Plot gain_warm vs p (benefit increases with p?)

---

### 5.5 Troubleshooting

#### Low pass rate (< 50%)

**Possible causes:**

1. **Gates too strict:** Lower `gate_rho` (e.g., 0.45), `gate_grad` (e.g., 0.15)
2. **Poor fingerprint coverage:** Increase `fp_points` (e.g., 20), `fp_cands` (e.g., 60)
3. **Mismatched p_match:** Ensure `p_match == p_target` for consistency
4. **Target too irregular:** Try more diverse surrogate families

**Fix:**

```bash
--gate_rho 0.45 --gate_grad 0.15 --fp_points 20 --fp_cands 60
```

#### Negative gain_warm_uncond

**Possible causes:**

1. **Transfer sanity not working:** Increase `transfer_margin` (e.g., 0.1)
2. **Warm start diverging:** Decrease `spsa_a_ft` (e.g., 0.15), increase `grad_clip` (e.g., 8)
3. **Noisy evaluations:** Increase `spsa_c_ft` for more stable gradients

**Fix:**

```bash
--transfer_margin 0.1 --spsa_a_ft 0.15 --grad_clip 8.0 --micro_warm
```

#### Slow runtime

**Bottleneck identification:**

- Check `wall_*` fields in results (which stage dominates?)
- **Fingerprint candidates:** 80% of time → reduce `fp_cands` or `fp_points`
- **Baseline:** 50% of time → reduce `restarts_baseline` or `iters`

**Speed optimizations:**

1. Reduce problem size: `--n 10` instead of 12 (4x faster)
2. Reduce fingerprint: `--fp_points 8 --fp_cands 20`
3. Fewer restarts: `--restarts_baseline 4 --restarts_source 2`
4. Disable traces: Remove `--save_traces`

#### Poor fingerprint quality (fp_rho < 0.5)

**Possible causes:**

1. **Unsuitable surrogates:** Target has unique structure not captured
2. **Too few probes:** Increase `fp_points`
3. **Probe distribution mismatch:** Adjust `fp_local_frac`

**Fixes:**

1. Add custom surrogate family (see extensibility guide)
2. Increase probe diversity: `--fp_points 20 --fp_local_frac 0.3`
3. Try donor probing: `--try_topk_donors 3`

---

### 5.6 Output Analysis Scripts

#### Python: Load and Analyze Results

```python
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results_all.json') as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Filter passes only
passes = df[df['gate'] == 'PASS']

# Statistics
print("Pass rate:", df['gate_pass'].mean())
print("Mean gain (cond):", passes['gain_cond'].mean())
print("Mean gain (uncond):", df['gain_uncond'].mean())
print("Random baseline:", df['gain_randFT'].mean())

# Paired t-test
from scipy.stats import ttest_rel
t_stat, p_val = ttest_rel(passes['gain_warm'], passes['gain_randFT'])
print(f"t-test: t={t_stat:.3f}, p={p_val:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gains comparison
axes[0].boxplot([passes['gain_warm'], passes['gain_randFT']], 
                labels=['Warm', 'RandFT'])
axes[0].set_ylabel('Gain over Baseline')
axes[0].set_title('Warm vs Random Fine-Tune')

# Fingerprint correlation
axes[1].scatter(df['fp_score'], df['gain_uncond'], alpha=0.6)
axes[1].set_xlabel('Fingerprint Score')
axes[1].set_ylabel('Gain (Unconditional)')
axes[1].set_title('Fingerprint Quality vs Gain')

plt.tight_layout()
plt.savefig('analysis.png', dpi=150)
print("Saved analysis.png")
```

#### Convergence Traces

```python
# Requires --save_traces

for i, res in enumerate(results[:3]):  # First 3 instances
    if res['trace_base'] and res['trace_warm']:
        plt.figure(figsize=(8, 4))
      
        plt.plot(res['trace_base'], label='Baseline', alpha=0.7)
        if res['trace_randFT']:
            plt.plot(res['trace_randFT'], label='RandFT', alpha=0.7)
        if res['trace_warm']:
            offset = len(res['trace_base'])
            iters = range(offset, offset + len(res['trace_warm']))
            plt.plot(iters, res['trace_warm'], label='Warm', alpha=0.7)
      
        plt.axhline(res['opt'], color='k', linestyle='--', label='Optimal')
        plt.xlabel('Iteration')
        plt.ylabel('Expected Cut Value')
        plt.title(f"{res['family']}, seed={res['seed']}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'trace_{i}.png', dpi=150)
```

---

## Complete Workflow Summary

### Research Workflow (End-to-End)

1. **Define research question:**

   - Does behavior-based surrogate matching improve QAOA optimization?
   - What is the computational overhead?
   - How does it compare to random fine-tuning?
2. **Configure experiment:**

   ```bash
   python behavior_surrogate_pipeline_v7_7_gpu.py \
     --n 12 --p_target 3 --p_match 3 \
     --families ER_dense_p05_weighted,RR_3regular \
     --seeds_per_family 15 \
     --iters 150 --finetune_iters 160 \
     --restarts_baseline 8 \
     --fp_points 15 --fp_cands 50 \
     --save_traces \
     --save_dir ./paper_run
   ```
3. **Run pipeline:**

   - Runtime: ~1-2 hours for 30 instances
   - Monitor: Check printed progress (gate status, gains)
4. **Analyze results:**

   ```python
   # Load and compute statistics (see scripts above)
   # Generate plots (boxplots, scatter, convergence)
   ```
5. **Interpretation:**

   - **Success:** pass_rate ≥ 0.7, gain_warm_cond > gain_randFT, p < 0.05
   - **Neutral:** pass_rate ≈ 0.5, marginal gain
   - **Failure:** pass_rate < 0.3 or negative gain
6. **Iterate:**

   - Tune hyperparameters (gates, fingerprint, SPSA)
   - Extend surrogate families
   - Test on larger n (14, 16) if feasible

---

This completes the comprehensive documentation. You now have:

1. **Overview** (research context, architecture)
2. **Mathematical Foundations** (QAOA, SPSA, fingerprint formulas)
3. **Surrogate Families** (5 parametric generators)
4. **Implementation** (pipeline flow, eval accounting, code details)
5. **Usage Guide** (hyperparameters, interpretation, troubleshooting)
