# Behavior-Surrogate Pipeline v7.7 - Implementation and Workflow

## Part 4: Pipeline Implementation and Workflow

### 4.1 Complete Pipeline Flow

The pipeline executes 6 stages for each problem instance. Here's the complete flow with evaluation accounting:

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Target graph (family, n, seed), depths (p_T, p_M)   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 0: Optimal Value (Brute Force)                       │
│ • Enumerate all 2^n configurations                          │
│ • opt = max Cut(s) for s ∈ {±1}^n                           │
│ • Cost: O(2^n) classical (no quantum evals)                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Baseline Optimization                             │
│ • Multi-start SPSA on target at p_target                    │
│ • R = 6 restarts, I = 120 iters/restart                     │
│ • base = max over restarts                                  │
│ • EVALS: evals_target_base = (2*I + 1)*R ≈ 1446            │
│ • TIME: wall_baseline_s                                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Random Fine-Tune Baseline (Fair Comparison)       │
│ • R_rand = 3 random inits                                   │
│ • Scheduled SPSA, I_ft = 140 iters                          │
│ • randFT = max over random restarts                         │
│ • EVALS: evals_target_randFT = (2*I_ft + 1)*R_rand ≈ 843   │
│ • TIME: wall_randFT_s                                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Fingerprint Construction (Target)                 │
│ • Generate K_pre=4, K=12 probes at p_match                  │
│ • Mixed: 50% local (around base_θ), 50% global              │
│ • Evaluate values: f_T(θ_k) for k=1..K                      │
│ • Evaluate M=3 directional derivatives per probe            │
│ • EVALS: evals_fp_target = K_pre + K + 2*K*M                │
│ •       = 4 + 12 + 2*12*3 = 88                              │
│ • TIME: wall_fp_target_s                                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4A: Candidate Generation & Pre-Selection             │
│ • Generate N=40 candidates (5 families × 8 each)            │
│ • Quick ρ-based pre-screen on K_pre=4 probes                │
│ • Keep top N_pre=12 candidates                              │
│ • EVALS (candidates): K_pre * N = 4*40 = 160 per cand       │
│ • TIME: wall_fp_cands_s (part 1)                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4B: Full Fingerprint (Top Candidates)                │
│ • For each of top N_pre=12:                                 │
│   - Evaluate K=12 values                                    │
│   - Evaluate 2*K*M = 72 directional derivatives             │
│   - Compute score S = 0.45ρ + 0.45cos(∇) + 0.1hit           │
│ • Sort by score, select best                                │
│ • EVALS (candidates): (K + 2*K*M) * N_pre = 84*12 = 1008    │
│ • TOTAL fp_evals_cands = 160*40 + 1008 = 7408               │
│ • TIME: wall_fp_cands_s (part 2)                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
          ┌────────────────┴────────────────┐
          │ GATE 1: Fingerprint Eligibility │
          │ • ρ ≥ 0.55                       │
          │ • cos(∇) ≥ 0.25                  │
          │ • hit@2 = 1                      │
          └────────────────┬────────────────┘
                           │
              FAIL ←───────┴───────→ PASS
               │                       │
               ↓                       ↓
        ┌─────────────┐      ┌─────────────────────────────┐
        │ gate=SKIP   │      │ STAGE 5A: Donor Selection   │
        │ warm=NaN    │      │ • try_topk_donors=0 (default)│
        │ gain=0      │      │ • Use best fingerprint match │
        └─────────────┘      │ • If try_topk>0: probe opt   │
                             │   on top-K donors, pick best │
                             │ EVALS (if probing):          │
                             │ • Donor: I_probe*R_probe*K   │
                             │ • Target overhead: scaling   │
                             └──────────────┬───────────────┘
                                            ↓
                             ┌──────────────────────────────┐
                             │ STAGE 5B: Donor Optimization │
                             │ • Multi-start SPSA on donor  │
                             │ • R_src=4 restarts, I_src=120│
                             │ • Depth: p_match             │
                             │ EVALS: evals_donor_opt       │
                             │       ≈ (2*120+1)*4 = 964    │
                             │ TIME: wall_source_s          │
                             └──────────────┬───────────────┘
                                            ↓
                             ┌──────────────────────────────┐
                             │ STAGE 5C: Transfer Mapping   │
                             │ • Scale gammas by W ratio    │
                             │ • Lift p_match → p_target    │
                             │ • Eval-only scaling grid     │
                             │ • tr = best init value       │
                             │ EVALS: evals_target_overhead │
                             │       ≈ 1 + 3-9 (grid)       │
                             │ TIME: included in wall_warm  │
                             └──────────────┬───────────────┘
                                            │
                        ┌───────────────────┴────────────────┐
                        │ GATE 2: Transfer Sanity            │
                        │ • tr ≥ base - margin (margin=0)    │
                        └───────────────────┬────────────────┘
                                            │
                              FAIL ←────────┴────────→ PASS
                               │                       │
                               ↓                       ↓
                        ┌─────────────┐      ┌─────────────────────────┐
                        │ gate=SKIP   │      │ STAGE 6: Warm Fine-Tune │
                        │ trace back  │      │ • Scheduled SPSA        │
                        │ warm=NaN    │      │ • I_warm = 140 iters    │
                        │ gain=0      │      │ • Layer-selective start │
                        └─────────────┘      │ EVALS: evals_target_warm│
                                             │       ≈ 2*140+1 = 281   │
                                             │ TIME: wall_warm_s       │
                                             └─────────┬───────────────┘
                                                       ↓
                                             ┌─────────────────────────┐
                                             │ warm = final value      │
                                             │ gain = warm - base      │
                                             │ gate = PASS             │
                                             └─────────────────────────┘
                                                       ↓
┌───────────────────────────────────────────────────────────────────────┐
│ OUTPUT: InstanceResult                                                │
│ • Values: opt, base, tr, warm, randFT                                 │
│ • Gains: gain_warm, gain_randFT, gain_cond, gain_uncond               │
│ • Fingerprint: fp_rho, fp_grad, fp_hit, fp_score                      │
│ • Evals: breakdown by stage (target_base, target_overhead, donor_opt, │
│          target_warm, total)                                          │
│ • Wall-time: breakdown by stage                                       │
│ • Traces (optional): convergence history                              │
└───────────────────────────────────────────────────────────────────────┘
```

---

### 4.2 Evaluation Budget Breakdown

**Default configuration (n=12, p_target=3, p_match=3):**

| **Stage** | **Component** | **Evals** | **Notes** |
|-----------|---------------|-----------|-----------|
| **0** | Optimal | 0 | Classical enumeration (2^12 configs) |
| **1** | Baseline | 1446 | Target: (2·120+1)·6 restarts |
| **2** | RandFT | 843 | Target: (2·140+1)·3 restarts |
| **3** | FP (target) | 88 | Target: 4 pre + 12 + 2·12·3 dir-derivs |
| **4** | FP (cands) | 7408 | Candidates: 4·40 pre + 84·12 full |
| **5a** | Donor probe | 0* | *If try_topk_donors=0 (default) |
| **5b** | Donor opt | 964 | Donor: (2·120+1)·4 restarts |
| **5c** | Transfer eval | 4-10 | Target overhead: 1 raw + 3-9 scaling grid |
| **6** | Warm FT | 281 | Target: 2·140+1 (if gate passes) |
| | | | |
| **TOTAL (target)** | | **~2666** | base + randFT + fp + overhead + warm |
| **TOTAL (donor)** | | **~964** | Separate budget (simpler problem) |
| **TOTAL (candidates)** | | **~7408** | Fingerprint screening only |

**Grand total (all evals):** ~11000 evaluations (target + donor + candidates)

**BUT scientifically honest accounting:**
- **Target problem evals:** ~2666 (what matters for target)
- **Donor evals:** ~964 (cheaper: pre-optimized or simpler)
- **Candidate evals:** ~7408 (one-time cost, reusable across instances)

**Comparison:**
- **Baseline only:** 1446 evals → base
- **Baseline + warm:** 1446 + 964 (donor) + 10 (transfer) + 281 (warm) ≈ 2701 evals → warm
- **Overhead:** ~87% over baseline (1255 extra evals)
- **Gain needed:** Must improve by >0 to justify overhead

---

### 4.3 Key Implementation Details

#### 4.3.1 Statevector Evaluation (lines 264-283)

**Critical optimization: Pre-cached basis energies**

**Problem:** Evaluating QAOA requires computing expectation over 2^n basis states.

**Naive approach (too slow):**
```python
def expected_cut(theta):
    psi = evolve_qaoa(theta)  # O(2^n * p)
    expectation = 0
    for z in all_basis_states:  # O(2^n) loop
        s = spin_config(z)
        E = compute_ising_energy(s, W)  # O(n^2) per state
        expectation += |psi[z]|^2 * Cut(E)
    return expectation
```
**Cost:** O(2^n · n^2) per evaluation (prohibitive for n≥12)

**Optimized approach (lines 136-142):**
```python
class WeightedGraph:
    def build_cache(self):
        # Pre-compute ALL 2^n spin configs (O(2^n * n))
        self.spins = all_spins(n)  # [2^n, n] matrix
        
        # Pre-compute ALL Ising energies (O(2^n * n^2))
        v = self.spins @ self.W  # Matrix-vector products
        self.ising_energy = 0.5 * sum(v * self.spins, axis=1)  # [2^n]
        
        # Pre-compute ALL cut values (O(2^n))
        self.cut_value = 0.5*W_sum - 0.5*self.ising_energy  # [2^n]
```

**Then evaluation is O(1) memory lookup:**
```python
def expected_cut(theta):
    psi = evolve_qaoa(theta)  # O(2^n * p)
    probs = |psi|^2  # O(2^n)
    return dot(probs, self.cut_value)  # O(2^n), cached lookup
```

**Benefit:** ~n^2 speedup (from O(2^n · n^2) to O(2^n)) per evaluation.

#### 4.3.2 Unitary Application (lines 223-242)

**Challenge:** Apply R_x(2β) to all n qubits efficiently.

**Naive approach:** Build full 2^n × 2^n matrix (memory explosion).

**In-place bitwise approach:**
```python
def apply_rx_all(psi, beta, n):
    c, s = cos(beta), -1j*sin(beta)
    out = psi.copy()
    
    for q in range(n):  # For each qubit
        step = 2^q
        period = 2*step
        
        for base in range(0, 2^n, period):  # Strided loop
            i0, i1 = base, base + step
            
            # CRITICAL: copy() prevents aliasing
            a = out[i0:i0+step].copy()
            b = out[i1:i1+step].copy()
            
            # R_x rotation on qubit q
            out[i0:i0+step] = c*a + s*b  # |0⟩ component
            out[i1:i1+step] = s*a + c*b  # |1⟩ component
    
    return out
```

**Key insight:**
- Qubit q affects bits at positions {..., base, base+step, ...}
- Stride pattern: 2^q (e.g., q=0 → alternating, q=1 → pairs, etc.)
- **copy()** essential: Without it, modifying out[i0] corrupts out[i1] mid-update

**Complexity:** O(2^n · n) time, O(2^n) space.

#### 4.3.3 SPSA Update Mask (Layer-Selective)

**Goal:** Freeze certain parameters during updates.

**Implementation (lines 430-435):**
```python
# In scheduled_spsa (lines 492-495):
mask = zeros(2p)
mask[p-ls_layers:p] = 1        # Last ls_layers betas
mask[2p-ls_layers:2p] = 1      # Last ls_layers gammas

# In spsa_optimize (lines 430-435):
delta = rng.choice([-1,+1], size=2p)
if update_mask is not None:
    delta = delta * update_mask  # Zero out frozen coords
    
    # Fallback if mask is all-zero
    if all(delta == 0):
        delta = rng.choice([-1,+1], size=2p)  # Full random
```

**Effect:**
- Only active coordinates perturbed in SPSA
- Frozen coordinates unchanged
- Saves evaluations (same delta scaling applied to all coords)

---

### 4.4 Fingerprint Probe Generation

#### 4.4.1 Mixed Local+Global Strategy (lines 710-758)

**Motivation:** Pure random probes may miss target-specific structure.

**Algorithm:**
```python
def make_theta_probes_mixed(p, K, seed, theta_center, local_frac=0.5):
    K_local = round(K * local_frac)
    K_global = K - K_local
    
    probes = []
    
    # Local probes (Gaussian around center)
    for _ in range(K_local):
        beta_c, gamma_c = split_theta(theta_center, p)
        
        beta = beta_c + normal(0, sigma_beta, p)    # sigma ≈ 0.08π
        gamma = gamma_c + normal(0, sigma_gamma, p)  # sigma ≈ 0.08π
        
        probes.append(clip_theta(concat([beta, gamma]), p))
    
    # Global probes (uniform in plausible range)
    for _ in range(K_global):
        beta = uniform(0, 0.4π, p) + normal(0, 0.05, p)
        gamma = uniform(0, 0.7π, p) + normal(0, 0.05, p)
        
        probes.append(clip_theta(concat([beta, gamma]), p))
    
    shuffle(probes)
    return probes
```

**Default (v7.7):**
- **local_frac = 0.5:** 50/50 split
- **K = 12:** 6 local + 6 global
- **theta_center:** Baseline parameters (at p_target) projected to p_match

**Effect:**
- Local probes: Sensitive to fine structure near optimum
- Global probes: Broad coverage, avoid overfitting
- **Reduces false positives** from random-only matching

#### 4.4.2 Probe Center: Baseline Projection (lines 938-941)

```python
# After baseline optimization at p_target
theta_base_fp = lift_theta(base_theta, p_target, p_match, mode='repeat_last')
```

**Why project to p_match?**
- Fingerprint evaluated at p_match (cheaper, depth-agnostic matching)
- Target baseline at p_target may not exist at p_match
- **Lift** provides consistent initialization

**Example (p_target=5, p_match=3):**
```
base_theta (p=5): β = [0.3, 0.5, 0.7, 0.6, 0.5]
                  γ = [0.8, 1.0, 1.2, 1.1, 1.0]

θ_center (p=3):   β = [0.3, 0.5, 0.7] (truncate)
                  γ = [0.8, 1.0, 1.2] (truncate)

Local probes centered at [0.3, 0.5, 0.7, 0.8, 1.0, 1.2] ± noise
```

---

### 4.5 Donor Selection (Optional Top-K Probing)

**New in v7.7:** Try multiple top fingerprint candidates before committing (lines 1103-1212).

**Algorithm (if `try_topk_donors > 0`):**
```
eligible = [cands passing gate-1]

# Diversify: at least one per family
best_per_family = {}
for cand in eligible:
    if cand.family not in best_per_family or cand.score > best_per_family[cand.family].score:
        best_per_family[cand.family] = cand

try_set = top_K_from(best_per_family.values(), by='score')

best_probe_tr = -inf
best_probe_cand = None

for cand in try_set:
    # Cheap donor optimization (probe)
    theta_donor = multistart_spsa(
        evaluator=cand_evaluator,
        restarts=2, iters=35,  # Much cheaper than full
    )
    
    # Transfer to target
    theta_tr = transfer_mapping(theta_donor, cand.W, target.W)
    
    # Eval-only scaling grid on target
    tr_val = select_best_scaled_init_eval_only(theta_tr)
    
    if tr_val > best_probe_tr:
        best_probe_tr = tr_val
        best_probe_cand = cand
    
    # Early accept if clearly above baseline
    if best_probe_tr >= base + margin:
        break

# Use best probe candidate as donor
donor = best_probe_cand
```

**Cost (K=3 candidates):**
- Donor probes: (2·35+1)·2·3 = 426 evals (on donors)
- Target overhead: 3·(1+3) = 12 evals (transfer scoring)

**Default:** try_topk_donors = 0 (disabled for fairness in v7.7)

**When to enable:**
- Recall more important than overhead
- Expected heterogeneous behavior across families
- Willing to pay extra ~400 donor evals

---

### 4.6 Gate Logic and Scientific Honesty

#### Gate 1: Fingerprint Eligibility (lines 1086-1091)

```python
pass_records = [
    cand for cand in fp_records
    if (cand.fp.rho_val >= gate_rho)          # Default: 0.55
    and (cand.fp.grad_dir_cos >= gate_grad)   # Default: 0.25
    and (cand.fp.hit >= gate_hit)             # Default: 1.0
]

gate_pass_fp = len(pass_records) > 0
```

**Rationale:**
- ρ ≥ 0.55: Moderate-to-strong rank correlation
- cos(∇) ≥ 0.25: Positive gradient alignment (not anti-correlated)
- hit = 1: Target's best probe in candidate's top-2

**Effect:**
- **Reject poor surrogates** before expensive donor optimization
- **Pass rate:** ~50-80% (empirical, depends on target family)

#### Gate 2: Transfer Sanity (lines 1304, 1361)

```python
if tr_val < base - transfer_margin:  # margin = 0 default
    # Negative transfer detected
    gate = "SKIP"
    gate_reason = "transfer_worse_than_base"
    warm_val = NaN
else:
    # Proceed to warm fine-tune
    gate = "PASS"
    warm_val = run_warm_finetune(tr_val)
```

**Rationale:**
- **Prevent negative transfer:** If transferred init is worse than baseline, skip warm start
- **No harm:** At worst, equivalent to cold start
- **Scientific honesty:** Report skips (don't cherry-pick passes only)

#### Micro-Warm Rescue (Optional, lines 1306-1353)

**Problem:** Transfer may be slightly worse (e.g., tr = base - 0.1) due to noise, but recoverable.

**Solution (if `--micro_warm`):**
```python
if tr_val < base - transfer_margin:
    if (base - tr_val) <= micro_warm_margin:  # e.g., 0.25
        # Try short rescue optimization
        theta_rescued = spsa(theta_tr, iters=30)
        tr_val = eval(theta_rescued)
        
        if tr_val >= base - transfer_margin:
            # Rescue succeeded, proceed to full warm
            gate = "PASS"
            warm_val = run_warm_finetune(theta_rescued)
        else:
            # Still worse, skip
            gate = "SKIP"
    else:
        # Too far below baseline, skip immediately
        gate = "SKIP"
```

**Default:** Disabled (--micro_warm not set)

**When to enable:**
- Noisy evaluations (e.g., NISQ hardware)
- Tight budget (avoid wasted warm evals)

---

### 4.7 Conditional vs Unconditional Gains

**Definitions (lines 1389-1392):**
```python
gain_cond = (warm - base) if gate == "PASS" else NaN
gain_uncond = (warm - base) if gate == "PASS" else 0.0
```

**Interpretation:**

| **Metric** | **Includes Skips** | **Penalizes Failure** | **Use Case** |
|------------|--------------------|-----------------------|--------------|
| **gain_cond** | No (NaN if skip) | No | "Given a good surrogate, how much gain?" |
| **gain_uncond** | Yes (0 if skip) | Yes | "On average (including failures), how much gain?" |

**Example:**
```
Instance 1: gate=PASS, warm=10, base=8 → gain_cond=+2, gain_uncond=+2
Instance 2: gate=SKIP, warm=NaN, base=7 → gain_cond=NaN, gain_uncond=0
Instance 3: gate=PASS, warm=6, base=5  → gain_cond=+1, gain_uncond=+1

mean(gain_cond) = (2 + NaN + 1) / 2 = 1.5   (avg over passes only)
mean(gain_uncond) = (2 + 0 + 1) / 3 = 1.0   (avg over all, skip=0)
```

**Reporting (line 1474-1475):**
- **summary.csv includes both:** gain_warm_uncond, gain_warm_cond
- **pass_rate:** Fraction of instances passing both gates

**Scientific honesty requirement:**
- **Always report pass_rate** alongside gains
- **High gain_cond with low pass_rate** → method is selective (not universally beneficial)
- **Moderate gain_uncond with high pass_rate** → robust improvement

---

### 4.8 Output Files and Data Structure

#### results_all.json (line 1648-1650)

**Format:** JSON array of InstanceResult dicts

**Schema (dataclass lines 835-904):**
```json
[
  {
    "family": "ER_dense_p05_weighted",
    "seed": 1042,
    "n": 12,
    "p_target": 3,
    
    "opt": 15.234,
    "base": 13.456,
    "tr": 13.789,
    "warm": 14.123,
    "randFT": 13.567,
    
    "gain_warm": 0.667,
    "gain_randFT": 0.111,
    "gain_cond": 0.667,
    "gain_uncond": 0.667,
    
    "gate_pass": 1,
    "gate": "PASS",
    "gate_reason": "pass",
    
    "fp_rho": 0.78,
    "fp_grad": 0.64,
    "fp_hit": 1.0,
    "fp_score": 0.739,
    
    "donor": "circulant",
    "scale": 1.05,
    
    "beta_mult_best": 1.0,
    "gamma_mult_best": 0.95,
    
    "evals_baseline": 1446,
    "evals_randFT": 843,
    "evals_fp_target": 88,
    "evals_fp_cands": 7408,
    "evals_source": 964,
    "evals_warm": 295,
    
    "evals_target_base": 1446,
    "evals_target_randFT": 843,
    "evals_target_overhead": 14,
    "evals_target_warm": 281,
    "evals_donor_opt": 964,
    "evals_total": 11036,
    
    "time_total_s": 23.456,
    
    "trace_base": [9.1, 10.2, ..., 13.456],
    "trace_randFT": [8.5, 9.7, ..., 13.567],
    "trace_warm": [13.789, 13.9, ..., 14.123],
    
    "wall_baseline_s": 8.2,
    "wall_randFT_s": 5.1,
    "wall_fp_target_s": 0.3,
    "wall_fp_cands_s": 3.2,
    "wall_source_s": 4.5,
    "wall_warm_s": 2.1
  },
  ...
]
```

**Key fields:**
- **opt, base, warm, randFT:** Primary performance metrics
- **gate, gate_reason:** Transparency (why skipped if applicable)
- **evals_*:** Full accounting (no hidden costs)
- **trace_*:** Convergence curves (if --save_traces)

#### summary.csv (line 1652-1653)

**Aggregates per graph family:**

```csv
family,fp_rho,fp_grad,fp_hit,fp_score,pass_rate,gain_warm_uncond,gain_warm_cond,gain_randFT,ratio_base,ratio_warm_cond,evals_total,evals_target_overhead,evals_target_warm,evals_donor_opt,time_total_s
ER_dense_p05_weighted,0.723,0.612,0.95,0.698,0.85,0.543,0.639,0.123,0.882,0.927,11023,102,278,961,23.4
RR_3regular,0.801,0.701,1.00,0.767,0.95,0.678,0.714,0.089,0.854,0.921,11045,98,280,959,24.1
```

**Columns:**
- **fp_*:** Average fingerprint quality
- **pass_rate:** Fraction passing gates
- **gain_warm_uncond/cond:** Average gains (unconditional/conditional)
- **gain_randFT:** Baseline comparison
- **ratio_base:** base / opt (baseline quality)
- **ratio_warm_cond:** warm / opt (warm quality, passes only)
- **evals_*:** Average evaluation counts
- **time_total_s:** Average wall-clock time

**Analysis use:**
- Compare gain_warm_cond vs gain_randFT (is warm better than random?)
- Check pass_rate (is method reliable?)
- Audit evals_total (what's the cost?)

---

### 4.9 Command-Line Interface

**Basic usage (line 32-36):**
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

**Key hyperparameters:**

| **Category** | **Flag** | **Default** | **Meaning** |
|--------------|----------|-------------|-------------|
| **Problem** | --n | 12 | Qubits |
| | --p_target | 3 | QAOA depth for target |
| | --p_match | 3 | QAOA depth for fingerprint |
| | --families | ER_dense... | Target graph families |
| **Optimization** | --iters | 120 | Baseline SPSA iterations |
| | --restarts_baseline | 6 | Baseline multi-start count |
| | --src_iters | 120 | Donor SPSA iterations |
| | --finetune_iters | 140 | Warm fine-tune iterations |
| **SPSA** | --spsa_a | 0.25 | Step size a |
| | --spsa_c | 0.12 | Perturbation size c |
| | --grad_clip | 5.0 | Gradient norm clipping |
| **Fingerprint** | --fp_points | 12 | Full probe count |
| | --fp_cands | 40 | Candidate count |
| | --fp_local_frac | 0.5 | Local probe fraction |
| | --fp_w_rho | 0.45 | ρ weight in score |
| | --fp_w_grad | 0.45 | Gradient weight |
| | --fp_w_hit | 0.10 | Hit weight |
| **Gates** | --gate_rho | 0.55 | Min ρ threshold |
| | --gate_grad | 0.25 | Min cos(∇) threshold |
| | --transfer_margin | 0.0 | Transfer sanity margin |
| **Advanced** | --try_topk_donors | 0 | Donor probing (0=off) |
| | --use_2d_scaling_grid | flag | 2D vs 1D scaling |
| | --micro_warm | flag | Enable micro rescue |
| | --save_traces | flag | Save convergence curves |

**Output:**
```
[ER_dense_p05_weighted|seed=1042|p=3] opt=15.234 base=13.456 tr=13.789 warm=14.123 randFT=13.567 gain(warm-base)=+0.667 fp=(rho=+0.78,grad=+0.64,hit=1) gate=PASS donor=circulant scale=1.05 time=23.46s
...

✅ Suite done. Time(s) = 125.34
Saved: ./run_v7_7/results_all.json
Saved: ./run_v7_7/summary.csv
```

---

This completes Part 4. Next: **Part 5 - Experimental Design and Hyperparameter Tuning.**
