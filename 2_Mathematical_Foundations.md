# Behavior-Surrogate Pipeline v7.7 - Mathematical Foundations

## Part 2: Mathematical Foundations

### 2.1 QAOA Circuit Evolution

#### Basis and State Representation
- **Computational basis:** |z⟩ for z ∈ {0,1}ⁿ
- **Spin mapping:** z → s where sᵢ = +1 if zᵢ = 0, else sᵢ = -1
- **Statevector:** |ψ⟩ = Σz ψz |z⟩ ∈ ℂ^(2ⁿ)
- **Normalization:** ⟨ψ|ψ⟩ = Σz |ψz|² = 1

#### Cost Hamiltonian (Ising Model)
For MaxCut on weighted graph W:

**H_cost = Σᵢ<ⱼ Wᵢⱼ ZᵢZⱼ**

In the Z-basis:
- H_cost |z⟩ = E(s(z)) |z⟩
- where **E(s) = Σᵢⱼ Wᵢⱼ sᵢsⱼ** (Ising energy)

**Implementation (code lines 219-220):**
```python
def apply_cost_phase(psi, gamma, ising_energy):
    return psi * np.exp(-1j * gamma * ising_energy)
```

**Matrix-free evaluation:**
- Pre-compute E(z) for all z (line 141): `self.ising_energy = 0.5 * np.sum((spins @ W) * spins, axis=1)`
- Phase application: ψz → ψz exp(-i γ E(z))

#### Mixer Hamiltonian (Transverse Field)
**H_mix = Σᵢ Xᵢ**

where Xᵢ is the Pauli-X on qubit i.

**Rotation interpretation:**
- exp(-i β H_mix) = exp(-i β Σᵢ Xᵢ) = ⊗ᵢ exp(-i β Xᵢ) = ⊗ᵢ Rx(2β)
- Rx(θ) = cos(θ/2) I - i sin(θ/2) X

**Implementation (code lines 223-242):**
```python
def apply_rx_all(psi, beta, n):
    c = cos(beta)
    s = -1j * sin(beta)
    # Apply Rx(2β) = c*I + s*X bitwise
    for q in range(n):
        step = 1 << q
        for base in range(0, len(psi), 2*step):
            i0, i1 = base, base + step
            a, b = psi[i0:i0+step].copy(), psi[i1:i1+step].copy()
            psi[i0:i0+step] = c*a + s*b
            psi[i1:i1+step] = s*a + c*b
```

**Note:** Uses `copy()` to prevent aliasing (critical for unitarity).

#### Full QAOA Circuit
**Parameters:** θ = (β₁,...,βₚ, γ₁,...,γₚ) ∈ ℝ^(2p)

**Evolution:**
|ψ(θ)⟩ = **U_p(θ)** |+⟩⊗ⁿ

where:
- |+⟩⊗ⁿ = (1/√(2ⁿ)) Σz |z⟩ (uniform superposition)
- **U_p(θ) = ∏ₗ₌₁ᵖ U_M(βₗ) U_C(γₗ)** (applied right to left)
- U_C(γ) = exp(-i γ H_cost)
- U_M(β) = exp(-i β H_mix)

**Objective function:**
**f(θ) = ⟨ψ(θ)| H_cost |ψ(θ)⟩ = Σz |ψz(θ)|² E(z)**

**MaxCut expectation:**
**⟨C(θ)⟩ = (½ W_sum) - (½ f(θ))**

**Implementation (code lines 264-283):**
```python
def expected_cut(theta):
    betas, gammas = theta[:p], theta[p:]
    psi = plus_state(n)  # (1/sqrt(N)) * ones(2^n)
    
    for l in range(p):
        psi = apply_cost_phase(psi, gammas[l], ising_energy)
        psi = apply_rx_all(psi, betas[l], n)
    
    probs = |psi|^2
    return dot(probs, cut_values)
```

---

### 2.2 Parameter Constraints and Clipping

**Physical parameter ranges:**
- **β ∈ [0, π/2]:** Rotation angle for Rx (up to π for full flip)
- **γ ∈ [0, π]:** Cost phase (periodic with 2π for Ising)

**Rationale:**
- β > π/2 redundant (symmetric under β ↔ π - β)
- γ ∈ [π, 2π] equivalent to [-π, 0] by periodicity

**Clipping function (lines 292-296):**
```python
def clip_theta(theta, p):
    betas = clip(theta[:p], 0, π/2)
    gammas = mod(theta[p:], π)  # Wrap to [0, π)
    return concat([betas, gammas])
```

---

### 2.3 SPSA Optimization (Simultaneous Perturbation Stochastic Approximation)

#### Why SPSA for QAOA?
- **Gradient-free:** No analytic gradients for general QAOA
- **Efficient:** 2 evals/iter (vs. 2p+1 for finite differences)
- **Robust:** Works with noisy evaluations (NISQ-friendly)

#### Algorithm (Maximization Variant)

**Goal:** max f(θ)

**Hyperparameters:**
- a, c: Step sizes (tuned empirically)
- α = 0.602, γ = 0.101: Decay exponents (Spall 1998 recommendations)
- A = 10: Offset for stability

**Iteration k:**

1. **Step sizes:**
   - aₖ = a / (k + A)^α
   - cₖ = c / k^γ

2. **Rademacher perturbation:**
   - Δₖ ~ Uniform({-1, +1}^(2p)) (coordinate-wise independent)

3. **Function evaluations:**
   - f₊ = f(θₖ + cₖΔₖ)
   - f₋ = f(θₖ - cₖΔₖ)

4. **Gradient estimate (simultaneous perturbation):**
   - **ĝₖ = (f₊ - f₋) / (2cₖ) · Δₖ**
   
   **Key insight:** Single random direction Δ estimates gradient in ALL coordinates simultaneously.

5. **Update (gradient ascent):**
   - θₖ₊₁ = θₖ + aₖ ĝₖ
   - θₖ₊₁ ← clip(θₖ₊₁)  # Enforce constraints

6. **Best tracking:**
   - best = max(best, f₊, f₋)
   - best_θ ← argmax{f(θₖ), f₊, f₋}

**Implementation (lines 399-467):**
```python
def spsa_optimize(evaluator, theta0, iters, seed, a, c, alpha, gamma, A, clip_norm):
    rng = random.Generator(seed)
    theta = clip_theta(theta0, p)
    best, best_theta = -inf, theta
    
    for k in 1..iters:
        ak = a / (k + A)^alpha
        ck = c / k^gamma
        
        delta = rng.choice([-1,+1], size=2p)
        f_plus = f(theta + ck*delta)
        f_minus = f(theta - ck*delta)
        
        ghat = (f_plus - f_minus) / (2*ck) * delta
        if clip_norm:
            ghat = clip_norm_to(ghat, clip_norm)
        
        theta = clip_theta(theta + ak*ghat, p)
        
        if max(f_plus, f_minus) > best:
            best = max(f_plus, f_minus)
            best_theta = argmax_theta
    
    return (best, best_theta, history)
```

**Gradient clipping (optional):**
- Prevents large updates from noisy estimates
- **ĝ ← ĝ · min(1, clip_norm / ||ĝ||)**

---

### 2.4 Multi-Start and Layer-Selective Strategies

#### Multi-Start SPSA (lines 528-565)
**Motivation:** QAOA landscape is non-convex with local optima.

**Algorithm:**
```
for r in 1..R:
    θ₀(r) ~ Uniform([0,π/2]ᵖ × [0,π]ᵖ)
    (fᵣ, θᵣ) = SPSA(θ₀(r), iters)
    
return argmax_r fᵣ
```

**Default:** R = 6 restarts for baseline (target), R = 4 for surrogate.

#### Layer-Selective Warmup (lines 470-525)
**Hypothesis:** Last layers (l = p) have strongest influence on final state.

**Two-stage schedule:**
1. **Stage 1 (ls_steps iters):** Update only last `ls_layers` parameters
   - Mask: Update (βₚ₋ₗₛ₊₁,...,βₚ, γₚ₋ₗₛ₊₁,...,γₚ)
   - Freeze earlier layers

2. **Stage 2 (remaining iters):** Update all parameters
   - Full gradient estimate

**Implementation:**
```python
def scheduled_spsa(evaluator, theta0, iters, ls_steps, ls_layers):
    if ls_steps <= 0 or ls_steps >= iters:
        return spsa_optimize(theta0, iters)  # No schedule
    
    # Stage 1: last layers only
    mask = zeros(2p)
    mask[p-ls_layers:p] = 1  # Last ls_layers betas
    mask[2p-ls_layers:2p] = 1  # Last ls_layers gammas
    
    stage1 = spsa_optimize(theta0, ls_steps, update_mask=mask)
    
    # Stage 2: all layers
    stage2 = spsa_optimize(stage1.best_theta, iters-ls_steps, update_mask=None)
    
    return stage2
```

**Default:** ls_steps = 40, ls_layers = 1 for warm fine-tuning.

---

### 2.5 Parameter Lifting (Depth Interpolation)

**Problem:** Fingerprint evaluated at depth p_match, but target runs at p_target.

**Need:** Lift θ ∈ ℝ^(2·p_match) → θ' ∈ ℝ^(2·p_target)

#### Modes (lines 348-386)

**1. repeat_last (default):**
```
β' = [β₁,...,βₚ_ₘ, βₚ_ₘ,...,βₚ_ₘ]  (repeat last)
       └─ p_match ─┘ └─ extra ──┘
γ' similarly
```

**2. repeat (tiling):**
```
β' = tile([β₁,...,βₚ_ₘ], ceil(p_target/p_match))[:p_target]
```

**3. pad_zero:**
```
β' = [β₁,...,βₚ_ₘ, 0,...,0]
```

**Implementation:**
```python
def lift_theta(theta, p_from, p_to, mode='repeat_last'):
    betas, gammas = theta[:p_from], theta[p_from:]
    
    if p_to < p_from:  # Truncate
        return concat([betas[:p_to], gammas[:p_to]])
    
    extra = p_to - p_from
    if mode == 'repeat_last':
        betas2 = concat([betas, full(extra, betas[-1])])
        gammas2 = concat([gammas, full(extra, gammas[-1])])
    # ... other modes
    
    return clip_theta(concat([betas2, gammas2]), p_to)
```

---

### 2.6 Fingerprint Metrics

#### 2.6.1 Spearman Rank Correlation (ρ)

**Goal:** Measure ordinal agreement of QAOA values across probes.

**Given:**
- Target values: **v_T = {f_T(θₖ)}_{k=1}^K**
- Candidate values: **v_C = {f_C(θₖ)}_{k=1}^K**

**Algorithm (lines 95-107):**
```
1. Rank values: r_T[k] = rank(v_T[k]) ∈ {1,...,K}
2. Center ranks: r̃_T = r_T - mean(r_T)
3. Spearman ρ = ⟨r̃_T, r̃_C⟩ / (||r̃_T|| · ||r̃_C||)
```

**Formula:**
**ρ = Σₖ(rank_T[k] - K̄)(rank_C[k] - K̄) / √(Σₖ(rank_T[k] - K̄)² · Σₖ(rank_C[k] - K̄)²)**

where K̄ = (K+1)/2.

**Interpretation:**
- ρ = +1: Perfect agreement (same ordering)
- ρ = 0: No monotonic relationship
- ρ = -1: Perfect inverse relationship

**Why Spearman not Pearson?**
- Robust to outliers and scale differences
- Captures behavioral similarity (which parameters are "good" or "bad"), not absolute values

---

#### 2.6.2 Directional Derivative Alignment

**Goal:** Match gradient structure without computing full gradients.

**Directional derivative at θ in direction v:**
**∇ᵥf(θ) = lim_{ε→0} [f(θ + εv) - f(θ - εv)] / (2ε)**

**Finite difference approximation:**
**D_v(θ) ≈ [f(θ + εv) - f(θ - εv)] / (2ε)**

**For K probes and M random directions:**
- Generate M unit vectors: **{vⱼ}_{j=1}^M** with vⱼ ~ Rademacher, ||vⱼ|| = 1
- Compute D_T[k,j] = ∇_{vⱼ}f_T(θₖ) for target
- Compute D_C[k,j] = ∇_{vⱼ}f_C(θₖ) for candidate

**Flatten to vectors:**
- d_T = flatten(D_T) ∈ ℝ^(K·M)
- d_C = flatten(D_C) ∈ ℝ^(K·M)

**Cosine similarity (lines 110-117, 817):**
**cos_grad = ⟨d_T, d_C⟩ / (||d_T|| · ||d_C||)**

**Implementation (lines 777-791):**
```python
def eval_dir_derivs(evaluator, thetas, dirs, eps):
    # thetas: K probe points
    # dirs: M×(2p) matrix of unit directions
    out = zeros(K, M)
    
    for k, theta in enumerate(thetas):
        for j, v in enumerate(dirs):
            f_plus = evaluator.expected_cut(theta + eps*v)
            f_minus = evaluator.expected_cut(theta - eps*v)
            out[k,j] = (f_plus - f_minus) / (2*eps)
    
    return out  # K × M
```

**Directional derivative cost:**
- 2·K·M evaluations (default: K=12, M=3 → 72 evals per candidate)

**Why not full gradients?**
- Full gradient: 2p evaluations per probe (e.g., 2·6 = 12 for p=3)
- Directional: 2M evaluations (M ≪ p often suffices)
- Captures dominant gradient structure

---

#### 2.6.3 Hit@k Metric

**Goal:** Does the target's best probe rank high in the candidate?

**Algorithm (lines 819-821):**
```
1. t_best = argmax_k f_T(θₖ)
2. Sort candidate values: rank_C = argsort(-f_C)
3. hit@k = 1 if t_best ∈ rank_C[0:k], else 0
```

**Interpretation:**
- **hit = 1:** Candidate's top-k includes target's global best
- **Captures peak alignment:** Even if overall correlation is moderate, matching peaks matters

**Default:** k = 2

---

#### 2.6.4 Composite Fingerprint Score

**Weighted sum (lines 802-828):**
**S = w_ρ · ρ + w_grad · cos_grad + w_hit · hit**

**Default weights (line 1563-1565):**
- w_ρ = 0.45
- w_grad = 0.45
- w_hit = 0.10

**Total = 1.0** (normalized score)

**Handling NaN:**
- If ρ or cos_grad is NaN (e.g., constant values), replace with 0

**Dataclass (lines 794-799):**
```python
@dataclass
class Fingerprint:
    rho_val: float        # Spearman ρ
    grad_dir_cos: float   # Cosine similarity of directional derivatives
    hit: float            # Hit@k (0 or 1)
    score: float          # Weighted sum
```

---

### 2.7 Transfer Mapping: Gamma Scaling

**Problem:** Candidate Hamiltonian H_C has different edge weights than target H_T.

**Strategy:** Scale γ parameters by edge weight ratio.

**Ratio computation (lines 1258-1260, 1165-1167):**
```python
W_T_mean = mean(|W_T[upper_triangular]|)
W_C_mean = mean(|W_C[upper_triangular]|)
scale = W_T_mean / (W_C_mean + ε)
```

**Transfer (lines 1261-1263):**
```python
betas_tr = betas_src  # Keep betas (mixer independent of weights)
gammas_tr = clip(gammas_src * scale, 0, π)
theta_tr = concat([betas_tr, gammas_tr])
```

**Safety clipping (line 1608):**
- **scale ∈ [1/γ_clip, γ_clip]** where γ_clip = 2.3 (default)
- Prevents extreme rescaling from outlier candidates

**Intuition:**
- Larger W → stronger cost Hamiltonian → need smaller γ for same phase
- γ ∝ 1/W preserves "effective strength" of cost phase

---

### 2.8 Evaluation-Only Scaling Grid

**Motivation:** Transferred init may benefit from small refinement.

**Strategy:** Try small grid of (β_mult, γ_mult) scaling factors (eval-only, no optimization).

**Algorithm (lines 299-337):**
```python
def select_best_scaled_init(evaluator, theta_in, beta_mults, gamma_mults):
    betas_0, gammas_0 = split_theta(theta_in, p)
    best_theta, best_val = theta_in, evaluator.expected_cut(theta_in)
    
    for bm in beta_mults:
        for gm in gamma_mults:
            if (bm, gm) == (1, 1): continue
            
            betas = clip(betas_0 * bm, 0, π/2)
            gammas = clip(gammas_0 * gm, 0, π)
            theta = concat([betas, gammas])
            
            val = evaluator.expected_cut(theta)
            if val > best_val:
                best_val = val
                best_theta = theta
                best_bm, best_gm = bm, gm
    
    return best_theta, best_val, best_bm, best_gm
```

**Default grids:**
- **2D grid (--use_2d_scaling_grid):** beta_mults = [0.9, 1.0, 1.1], gamma_mults = [0.9, 1.0, 1.1]
  - 9 evaluations (3×3 grid)
- **1D gamma-only (default):** warm_gamma_mults = [1.0, 0.95, 1.05]
  - 3 evaluations

**Cost:** Small (9-3 evals) compared to optimization (100+ evals).

**Scientific honesty:** All scaling evals counted in `evals_target_overhead`.

---

## Summary of Key Formulas

| **Component** | **Formula** | **Code** |
|---------------|-------------|----------|
| QAOA state | \|ψ(θ)⟩ = ∏ₗ e^(-iβₗΣXᵢ) e^(-iγₗH_cost) \|+⟩⊗ⁿ | lines 268-272 |
| Objective | f(θ) = ⟨ψ\|H_cost\|ψ⟩ = Σz \|ψz\|² E(z) | line 280 |
| SPSA gradient | ĝ = [(f(θ+cΔ) - f(θ-cΔ))/(2c)] · Δ | line 443 |
| Spearman ρ | Corr(rank(v_T), rank(v_C)) | lines 95-107 |
| Dir deriv | D_v(θ) = [f(θ+εv) - f(θ-εv)]/(2ε) | line 790 |
| Fingerprint | S = 0.45ρ + 0.45cos(∇) + 0.1hit | lines 823-826 |
| Gamma scale | γ_tr = γ_src · (mean\|W_T\| / mean\|W_C\|) | lines 1258-1261 |

---

This completes Part 2. Next: **Part 3 - Surrogate Families and Generation.**
