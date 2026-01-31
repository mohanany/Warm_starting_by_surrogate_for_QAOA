# Behavior-Surrogate Pipeline v7.7 - Technical Documentation

## Part 1: Overview and Research Context

### 1.1 Core Research Question

This pipeline implements and evaluates a novel approach to optimizing **QAOA (Quantum Approximate Optimization Algorithm)** parameters for combinatorial optimization problems, specifically **MaxCut**.

**The Central Idea:**

> Instead of optimizing QAOA parameters directly on the target problem (which is expensive), can we find a **simpler surrogate problem** whose QAOA **behavior** (not just structure) matches the target well enough that parameters optimized on the surrogate transfer effectively to the target?

### 1.2 Problem Context: QAOA for MaxCut

#### The MaxCut Problem

Given a weighted graph with adjacency matrix **W**, find a partition of vertices into two sets to maximize the total weight of edges crossing the partition.

Formally:

- Let **s** ∈ {+1, -1}ⁿ represent a partition (spin configuration)
- Cut value: **C(s) = ½ Σᵢⱼ Wᵢⱼ (1 - sᵢsⱼ)/2**
- Equivalent Ising energy: **H(s) = Σᵢⱼ Wᵢⱼ sᵢsⱼ**
- Relationship: **C(s) = ½ W_sum - ½ H(s)** where W_sum = Σᵢⱼ Wᵢⱼ

#### QAOA for MaxCut

QAOA is a variational quantum algorithm with depth **p** characterized by:

- **2p parameters:** β = (β₁, ..., βₚ) and γ = (γ₁, ..., γₚ)
- **Initial state:** |+⟩⊗ⁿ (equal superposition)
- **Circuit:** Apply p layers of (Cost-Phase, Mixer):
  - Cost phase: **U_C(γ) = exp(-i γ H_cost)** where H_cost is the Ising Hamiltonian
  - Mixer: **U_M(β) = exp(-i β Σᵢ Xᵢ)** (transverse field)

**Circuit structure:**

```
|ψ(β,γ)⟩ = U_M(βₚ) U_C(γₚ) ... U_M(β₁) U_C(γ₁) |+⟩⊗ⁿ
```

**Objective:** Maximize **f(β,γ) = ⟨ψ(β,γ)|H_cost|ψ(β,γ)⟩**

### 1.3 The Parameter Transfer Challenge

**Current Challenge:**

- For depth p ≥ 3 and moderate problem sizes (n ~ 12-14), finding good QAOA parameters requires **many costly evaluations**
- Each evaluation requires either:
  - Quantum hardware (noisy, expensive)
  - Classical simulation (exponential scaling: O(2ⁿ))

**Transfer Learning Hypothesis:**
If we can find a **surrogate Hamiltonian H'** (simpler or already solved) where QAOA exhibits similar **behavioral dynamics** to our target H, then:

1. Optimize parameters θ* on H' (cheap/pre-computed)
2. Transfer θ* → H with minimal fine-tuning (warm start)
3. Achieve better final quality than cold starts with same budget

### 1.4 Why "Behavior" Not Just "Structure"?

**Key Insight:** Graph isomorphism or similar spectral properties are **insufficient**. We need:

- Similar QAOA **landscape topology** in parameter space
- Aligned **gradient directions** at key points
- Matching **value rankings** across parameter probes

This requires a **dynamics-aware fingerprint** (detailed in Part 2).

### 1.5 Scientific Scope and Limitations

**Honest Scoping:**

- **Small-n regime:** n ≤ 14 (exact statevector simulation)
- **Research benchmark:** Not production-ready for NISQ hardware
- **Proof of concept:** Validates the behavior-matching idea
- **Extensibility:** Framework designed to replace evaluator with hardware/sampling

**Key Claims:**
✅ Behavior-based surrogate matching can improve QAOA optimization
✅ Fingerprint-based selection outperforms random or structural matching
✅ Transfer gating prevents negative transfer (scientifically honest)


## 1.6 Architecture Overview

The pipeline consists of 6 major stages:

### Stage 1: Baseline Optimization (Target)

- Multi-start SPSA on target problem at depth p_target
- Establishes baseline performance: **base = max f(θ) over random inits**

### Stage 2: Random Fine-Tune Baseline (RandFT)

- Same budget as warm fine-tune, but from random init(s)
- Fair comparison: **randFT** vs **warm** with equal budget

### Stage 3: Fingerprint Construction

- Generate K probe points in parameter space (mixed local+global)
- Evaluate both:
  - **Values:** f(θₖ) for k = 1...K
  - **Directional derivatives:** ∇ᵥf(θₖ) for random directions v

### Stage 4: Candidate Screening

- Generate N surrogate candidates from parametric families
- **Pre-selection:** Quick rank-correlation (ρ) filter
- **Full fingerprint:** Multi-component score on top candidates

### Stage 5: Donor Selection & Transfer

- **Gate 1 (Eligibility):** Fingerprint thresholds (ρ, gradient alignment, hit@k)
- **Optional:** Try top-K donors with cheap probe optimization
- **Transfer mapping:** Scale γ parameters by edge weight ratio
- **Gate 2 (Sanity):** Reject if transferred init worse than baseline

### Stage 6: Warm Fine-Tuning

- If gates pass: Run SPSA from transferred init
- **Layer-selective warmup:** Optimize last layers first, then all
- Report: **warm** (final value), **gain** (warm - base)

---

## 1.7 Key Innovations in v7.7

### 1. Gradient-Free Optimization (SPSA)

- **2 evaluations per iteration** (vs. 2p+1 for finite differences)
- Scales to high depth without gradient explosion
- Industry-standard for QAOA (used in IBM, Rigetti workflows)

### 2. Dynamics-Aware Fingerprint

- **Value correlation (ρ):** Spearman rank correlation of f(θₖ)
- **Gradient alignment:** Cosine similarity of directional derivatives
- **Hit@k:** Does target's best probe rank in candidate's top-k?

### 3. Mixed (Local+Global) Probes

- **Local (50%):** Centered around target baseline (at p_match)
- **Global (50%):** Broad exploration of parameter space
- Reduces false positives from overfitting to global random probes

### 4. Transfer Sanity Gate

- **Prevention of negative transfer:** Skip warm start if transferred init is worse than baseline
- Scientific honesty: Report both conditional (gate-pass only) and unconditional gains

### 5. Comprehensive Eval Accounting

- **Paper-grade breakdown:**
  - evals_target_base: Baseline optimization
  - evals_target_overhead: Fingerprint + transfer evaluation
  - evals_donor_opt: Surrogate optimization
  - evals_target_warm: Warm fine-tuning
- **Total accountability:** No hidden costs

---

## 1.8 Performance Metrics

For each instance, we track:

### Primary Outcomes

- **opt:** Brute-force optimal MaxCut value
- **base:** Baseline QAOA (multi-start from random)
- **tr:** Transferred initialization quality
- **warm:** Final warm-started result
- **randFT:** Random fine-tune baseline (fair comparison)

### Gains

- **gain_warm = warm - base:** Improvement over baseline
- **gain_randFT = randFT - base:** Random baseline improvement
- **gain_cond:** Conditional gain (NaN if gate skipped)
- **gain_uncond:** Unconditional gain (0 if gate skipped)

### Fingerprint Quality

- **fp_rho:** Value correlation
- **fp_grad:** Gradient alignment
- **fp_hit:** Hit@k indicator
- **fp_score:** Weighted sum (default: 0.45ρ + 0.45grad + 0.1hit)

### Gate Status

- **gate_pass:** 1 if warm start executed, 0 if skipped
- **gate_reason:** "pass" | "fp_gate_fail" | "transfer_worse_than_base"

---

## Next Sections Preview

- **Part 2:** Mathematical Foundations (QAOA circuit, SPSA, fingerprint metrics)
- **Part 3:** Surrogate Families (powlaw, strength, block, circulant, lowrank)
- **Part 4:** Implementation Details (code walkthrough)
- **Part 5:** Experimental Design (hyperparameters, eval budget)
- **Part 6:** Usage and Interpretation
