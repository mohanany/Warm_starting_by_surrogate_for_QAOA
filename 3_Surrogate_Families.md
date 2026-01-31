# Behavior-Surrogate Pipeline v7.7 - Surrogate Families

## Part 3: Surrogate Families and Generation

### 3.1 Design Philosophy

**Core Question:** What makes a good surrogate for QAOA behavior matching?

**Desiderata:**
1. **Diverse topologies:** Cover different graph structures (dense, sparse, regular, hierarchical)
2. **Tunable complexity:** Simple parametric forms (not just random graphs)
3. **Same size:** n vertices (enables direct parameter transfer)
4. **Controllable:** Deterministic generation from seed

**Non-Structural Families:**
Instead of graph-isomorphic variants (guaranteed poor diversity), we use **parametric weight matrix generators** that:
- Impose different *structural priors* (e.g., community, circulant, low-rank)
- Are **not** graph-theoretically similar to target
- Rely on **behavioral fingerprinting** to find matches

---

### 3.2 Scaling to Match Target

**Problem:** Surrogate and target have different edge weight scales.

**Solution (lines 578-584):**
```python
def scale_to_match(target_W, cand_W, eps=1e-12):
    t_mean = mean(|target_W[upper_triangular]|)
    c_mean = mean(|cand_W[upper_triangular]|)
    
    if c_mean < eps:
        return cand_W, 1.0
    
    scale = t_mean / c_mean
    return cand_W * scale, scale
```

**Effect:**
- **Rescale candidate weights** to have similar magnitude to target
- **Preserves structure:** Relative weight ratios unchanged
- **Applied before fingerprinting** (line 679, 1047)

---

### 3.3 Surrogate Family: Power-Law Additive (`powlaw`)

**Motivation:** Many real-world problems have heterogeneous edge weights with heavy-tailed distributions.

**Generation (lines 587-594):**
```python
def make_additive_powlaw(n, seed, alpha=1.2):
    rng = Generator(seed)
    
    # Node weights: power-law decay
    idx = [1, 2, ..., n]
    P[i] = i^(-alpha)  # P[1] = 1, P[n] ≈ n^(-1.2) ≈ 0.08 for n=12
    
    shuffle(P)  # Break correlation with index
    
    # Additive model: W[i,j] = P[i] + P[j]
    W = outer(P, 1) + outer(1, P)  # Broadcasting
    W[diagonal] = 0
    
    return max(W, 0)  # Ensure non-negative
```

**Properties:**
- **Complete graph:** All edges present
- **Heterogeneous:** High-degree nodes (large P[i]) → heavy edges
- **Deterministic structure:** W[i,j] determined by P[i], P[j]
- **Additive decomposition:** W = PP^T (rank-2 structure)

**Parameter:**
- **α = 1.2:** Moderate heterogeneity (α↑ → more uniform)

**Example (n=4, α=1.2, seed=42):**
```
P (before shuffle) = [1.000, 0.435, 0.284, 0.212]
P (after shuffle)  = [0.284, 1.000, 0.212, 0.435]

W = [[0.000, 1.284, 0.496, 0.719],
     [1.284, 0.000, 1.212, 1.435],
     [0.496, 1.212, 0.000, 0.647],
     [0.719, 1.435, 0.647, 0.000]]
```

**Use case:** Targets with hub nodes (e.g., scale-free networks).

---

### 3.4 Surrogate Family: Strength-Based (`strength`)

**Motivation:** Match target's *node degree distribution* (node strength = Σⱼ Wᵢⱼ).

**Generation (lines 597-607):**
```python
def make_additive_strength(target_W, seed, noise=0.02):
    rng = Generator(seed)
    
    # Extract node strengths from target
    strength = sum(target_W, axis=1)  # [n]
    
    # Normalize to [0, 1]
    strength = (strength - min(strength)) / (max(strength) + eps)
    
    # Add small noise (break exact structure)
    strength += noise * rng.normal(0, 1, n)
    strength = max(strength, 0)
    
    # Additive model
    W = outer(strength, 1) + outer(1, strength)
    W[diagonal] = 0
    
    return W
```

**Properties:**
- **Target-aware:** Uses target's strength profile
- **Noisy:** noise=0.02 prevents exact copy (diversity)
- **Preserves heterogeneity:** High-strength nodes → heavy edges
- **Simple structure:** Additive (rank-2)

**Example:**
```
Target strengths = [3.2, 5.1, 2.8, 4.5]
After normalize  = [0.20, 1.00, 0.00, 0.74]
After noise      = [0.18, 1.02, -0.01, 0.75] → [0.18, 1.02, 0.00, 0.75]

W_surr[1,2] = 0.18 + 1.02 = 1.20
```

**Use case:** Targets where node centrality drives behavior.

---

### 3.5 Surrogate Family: Block Dense (`block`)

**Motivation:** Community/modular structure (e.g., planted partition, SBM).

**Generation (lines 610-622):**
```python
def make_block_dense(n, seed, B=4):
    rng = Generator(seed)
    
    # Assign nodes to B blocks uniformly
    groups = rng.integers(0, B, size=n)
    
    # Block affinity matrix (symmetric)
    M = rng.random(B, B)
    M = 0.5 * (M + M.T)  # Symmetrize
    
    # Edge weights from block membership
    W = zeros(n, n)
    for i < j:
        W[i,j] = W[j,i] = M[groups[i], groups[j]]
    
    return W
```

**Properties:**
- **B blocks:** Default B=4 for n≤16, B=6 for n>16
- **Stochastic Block Model (SBM):** P(edge | blocks) ~ M[b1, b2]
- **Intra-block:** M[b, b] (within-community)
- **Inter-block:** M[b1, b2], b1≠b2 (between-community)

**Example (n=6, B=2, seed=123):**
```
groups = [0, 1, 0, 1, 0, 1]  # Alternating assignment
M = [[0.8, 0.3],
     [0.3, 0.7]]

W[0,2] = M[0,0] = 0.8  (same block)
W[0,1] = M[0,1] = 0.3  (different blocks)
```

**Use case:** Targets with community structure (e.g., social networks, circuit partitioning).

---

### 3.6 Surrogate Family: Circulant (`circulant`)

**Motivation:** Spatially structured problems (e.g., ring, grid with periodic BC).

**Generation (lines 625-644):**
```python
def make_circulant_dense(n, seed, k_freq=3):
    rng = Generator(seed)
    
    # Fourier basis for circulant pattern
    t = [0, 1, ..., n-1]
    c = zeros(n)
    
    for k in 1..k_freq:
        amp = rng.random() * 0.5
        phase = rng.random() * 2π
        c += amp * cos(2πkt/n + phase)  # k-th Fourier mode
    
    # Normalize to [0, 1]
    c = (c - min(c)) / (max(c) + eps)
    
    # Circulant: W[i,j] depends on |i-j| mod n
    W = zeros(n, n)
    for i < j:
        d = (j - i) mod n
        W[i,j] = W[j,i] = c[d]
    
    return W
```

**Properties:**
- **Circulant:** W[i,j] = f((j-i) mod n)
- **Translation-invariant:** Shift all indices → same matrix
- **Fourier decomposition:** c[d] = Σₖ aₖ cos(2πkd/n)
- **k_freq modes:** Controls smoothness (k↑ → more oscillations)

**Example (n=8, k_freq=2, seed=99):**
```
c = [1.00, 0.85, 0.45, 0.10, 0.00, 0.10, 0.45, 0.85]  (distance weights)

W[0,1] = c[1] = 0.85  (distance 1)
W[0,2] = c[2] = 0.45  (distance 2)
W[0,7] = c[7] = 0.85  (distance 1, wrapping)
```

**Use case:** Ring-like or grid problems (e.g., 1D quantum spin chains, TSP on circle).

---

### 3.7 Surrogate Family: Low-Rank (`lowrank2`)

**Motivation:** Problems with latent low-dimensional structure.

**Generation (lines 647-655):**
```python
def make_lowrank_dense(n, seed, rank=2):
    rng = Generator(seed)
    
    # Random Gaussian embedding
    U = rng.normal(0, 1, (n, rank))
    
    # Gram matrix (rank-2)
    W = U @ U.T
    W[diagonal] = 0
    
    # Normalize to [0, 1]
    W = (W - min(W)) / (max(W) + eps)
    
    return max(W, 0)
```

**Properties:**
- **rank(W) = 2:** W = UU^T where U ∈ ℝ^(n×2)
- **Geometric interpretation:** Nodes embedded in 2D space, W[i,j] = ⟨u_i, u_j⟩
- **Smooth:** High correlation structure (low-rank)
- **Dense:** All entries non-zero (after normalization)

**Example (n=3, rank=2, seed=7):**
```
U = [[0.5, 1.2],
     [0.8, 0.3],
     [1.1, 0.7]]

W_raw = U @ U.T = [[1.69, 0.76, 1.39],
                    [0.76, 0.73, 1.09],
                    [1.39, 1.09, 1.70]]

After diagonal removal and normalization.
```

**Use case:** Targets with strong correlations (e.g., covariance-based Ising, QML problems).

---

### 3.8 Candidate Generation Pipeline

**Top-level function (lines 658-684):**
```python
def generate_candidates(target, n_cands, seed, families):
    rng = Generator(seed)
    cands = []
    
    for i in range(n_cands):
        # Round-robin family selection
        fam = families[i % len(families)]
        
        # Unique seed per candidate
        s = rng.integers(0, 1e9)
        
        # Generate unscaled matrix
        if fam == 'powlaw':
            W_raw = make_additive_powlaw(target.n, s)
        elif fam == 'strength':
            W_raw = make_additive_strength(target.W, s)
        elif fam == 'block':
            B = 4 if target.n <= 16 else 6
            W_raw = make_block_dense(target.n, s, B)
        elif fam == 'circulant':
            W_raw = make_circulant_dense(target.n, s)
        elif fam == 'lowrank2':
            W_raw = make_lowrank_dense(target.n, s, rank=2)
        
        # Scale to match target
        W, scale = scale_to_match(target.W, W_raw)
        W = max(W, 0)  # Ensure non-negative
        W[diagonal] = 0
        
        cands.append(SurrogateCandidate(name=fam, W=W))
    
    return cands
```

**Default parameters (line 1593):**
- **families = "powlaw,strength,block,circulant,lowrank2"**
- **n_cands = 40** (lines 1548)
- **Round-robin:** 40 cands → 8 of each family

**Diversity strategy:**
- **Inter-family:** 5 structural types
- **Intra-family:** Different seeds → different instances

---

### 3.9 Candidate Pre-Selection

**Motivation:** Full fingerprint (with directional derivatives) is expensive (72 evals/candidate). Pre-filter to top candidates.

**Two-stage screening (lines 1027-1075):**

#### Stage 1: Quick pre-screen (lines 1027-1040)
```python
pre_scores = []
for cand in candidates:
    # Evaluate on small probe set (fp_pre_points = 4)
    cand_vals = eval_values(cand_evaluator, thetas_pre)
    
    # Rank correlation only (cheap)
    rho = spearman_r(target_vals_pre, cand_vals)
    
    pre_scores.append((rho, cand_index))

# Sort by rho descending
pre_scores.sort(reverse=True)

# Keep top fp_preselect candidates
top_candidates = pre_scores[:fp_preselect]
```

**Cost:** K_pre · N_cands evals (default: 4 · 40 = 160 on target, 160 on each candidate)

#### Stage 2: Full fingerprint (lines 1045-1075)
```python
for cand in top_candidates:
    # Scale to match target
    W_scaled, scale = scale_to_match(target.W, cand.W)
    
    # Full probe set (fp_points = 12)
    cand_vals = eval_values(cand_evaluator, thetas_full)
    
    # Directional derivatives (fp_dirs = 3)
    cand_dd = eval_dir_derivs(cand_evaluator, thetas_full, dirs, eps)
    
    # Composite fingerprint
    fp = fingerprint_score(
        target_vals, cand_vals,
        target_dd, cand_dd,
        topk=fp_topk,
        w_rho=0.45, w_grad=0.45, w_hit=0.1
    )
    
    fp_records.append((fp.score, cand, scale))

# Sort by score descending
fp_records.sort(reverse=True)
```

**Cost (per candidate):**
- Values: K_full = 12 evals
- Dir derivs: 2 · K_full · M = 2 · 12 · 3 = 72 evals
- **Total: 84 evals** per top candidate

**Breakdown (default: 40 cands, keep top 12):**
1. Pre-screen: 4 · 40 = 160 target evals, 160 · 40 = 6400 candidate evals
2. Full fingerprint: 
   - Target (once): 12 values + 72 dir-derivs = 84 evals
   - Top 12 candidates: 12 · 84 = 1008 evals

**Total fingerprint cost:**
- Target: 160 + 84 = 244 evals
- Candidates: 6400 + 1008 = 7408 evals

**Note:** Candidate evals are on *surrogates* (same n, different W), counted separately.

---

### 3.10 Why These Families?

| **Family** | **Structural Prior** | **Best For** | **QAOA Behavior Hypothesis** |
|------------|----------------------|--------------|------------------------------|
| **powlaw** | Heterogeneous hubs | Scale-free, star-like | Heavy edges → strong cost phase, localized optima |
| **strength** | Degree-preserving | Degree-constrained | Node centrality → parameter sensitivity to high-degree layers |
| **block** | Community structure | Modular, clustered | Block boundaries → mixing barrier, multi-scale dynamics |
| **circulant** | Translation-invariant | Ring, grid | Symmetry → degenerate landscapes, periodic optima |
| **lowrank2** | Latent factors | Low-rank signals | Smooth landscapes, global structure |

**Empirical observation (not proven):**
- No single family dominates across all targets
- Fingerprint scoring correctly identifies best match for each instance
- Diverse pool increases recall (at least one good match)

---

### 3.11 Comparison with Target Graph Families

**Target families (lines 186-197):**

#### ER_dense_p05_weighted (Erdős-Rényi)
```python
def er_dense_p05_weighted(n, seed, p=0.5):
    rng = Generator(seed)
    
    # Edge mask (each edge present with prob p)
    mask = rng.random(n, n) < p
    mask = upper_triangular(mask, k=1)
    
    # Random weights
    w = rng.random(n, n)
    
    W = mask * w
    W = W + W.T
    W[diagonal] = 0
    
    return W
```

**Properties:**
- **p = 0.5:** ~50% edge density (expected m ≈ n(n-1)/4 edges)
- **Random weights:** Uniform [0, 1]
- **No structure:** Null model (baseline)

#### RR_3regular (Random Regular)
```python
def rr_k_regular(n, k, seed):
    # Configuration model: n*k stubs, pair randomly
    assert (n * k) % 2 == 0
    
    stubs = repeat([0,1,...,n-1], k)
    shuffle(stubs)
    edges = reshape(stubs, (-1, 2))
    
    # Reject self-loops and multi-edges
    ...
    
    W = adjacency(edges)  # Unweighted (W[i,j] ∈ {0,1})
    return W
```

**Properties:**
- **k = 3:** Every node has degree 3 (unweighted)
- **Hard constraint:** Not all n admit k-regular graphs
- **Structured:** Homogeneous degree → different QAOA behavior than ER

#### Complete_weighted_dense
```python
def complete_weighted_dense(n, seed):
    W = rng.random(n, n)
    W = upper_triangular(W, k=1)
    W = W + W.T
    W[diagonal] = 0
    return W
```

**Properties:**
- **Complete:** All edges present (m = n(n-1)/2)
- **Random weights:** Fully dense

**Observation:**
- Surrogates are **not** isomorphic to targets
- Surrogates are parametric (structured), targets are random
- **Behavioral matching bridges the gap**

---

### 3.12 Extensibility: Adding Custom Families

**To add a new surrogate family:**

1. **Define generator function:**
```python
def make_my_family(n, seed, **params):
    rng = np.random.default_rng(seed)
    # ... generate W (n×n symmetric, non-negative, zero diagonal)
    return W
```

2. **Register in `generate_candidates` (line 658):**
```python
elif fam == 'my_family':
    W_raw = make_my_family(target.n, s, **params)
```

3. **Add to CLI (line 1593):**
```
--surrogate_families "powlaw,strength,block,circulant,lowrank2,my_family"
```

**Design guidelines:**
- **Deterministic:** Same seed → same W
- **Diverse:** Should differ from existing families
- **Tunable:** Expose 1-2 key parameters (e.g., rank, blocks)
- **Non-negative:** W[i,j] ≥ 0 (MaxCut assumes non-negative weights)

---

## Summary Table

| **Family** | **Generator** | **Params** | **Rank** | **Sparsity** | **Lines** |
|------------|---------------|------------|----------|--------------|-----------|
| powlaw | `make_additive_powlaw` | α=1.2 | 2 | Dense | 587-594 |
| strength | `make_additive_strength` | noise=0.02 | 2 | Dense | 597-607 |
| block | `make_block_dense` | B=4/6 | ≤ B | Dense | 610-622 |
| circulant | `make_circulant_dense` | k_freq=3 | n | Dense | 625-644 |
| lowrank2 | `make_lowrank_dense` | rank=2 | 2 | Dense | 647-655 |

**All families:**
- Same size n as target
- Deterministic from seed
- Scaled to match target edge weight magnitude
- Evaluated at same p_match as target fingerprint

---

This completes Part 3. Next: **Part 4 - Pipeline Implementation and Workflow.**
